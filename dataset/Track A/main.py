#!/usr/bin/env python
# -*-coding:utf-8 -*-

import argparse
import json
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Optional
import pandas as pd
import httpx
import requests
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIError

from _types import ToolCall
from logger import init_logger
from utils import (
    print_model_response,
    print_tool_call,
    print_tool_result,
    extract_answer,
    extract_answer_all,
    compute_score,
)
from skills.answer_parser import parse_answer, iou_score
from rag_retriever import RAGRetriever

os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ.setdefault('AGENT_API_KEY', 'sk-XXXXXXXXXXXXX')

API_KEY = os.environ.get("AGENT_API_KEY", "dummy")

# ------------------------------------------------------------------------------
# System Prompt
# ------------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a 5G network optimization expert tasked with diagnosing wireless network \
performance faults from drive-test data.

## Mandatory Diagnostic Protocol

Follow these steps IN ORDER for every scenario:

**STEP 1** — Call `get_throughput_logs` (no arguments)
  → Scan the timestamps. Identify the "bad window": the period where throughput \
drops significantly below 100 Mbps.

**STEP 2** — Call `get_serving_cell_pci` with `time` set to the worst timestamp \
from Step 1
  → This tells you which cell the UE is camped on during the fault.

**STEP 3** — Call ALL of the following SIMULTANEOUSLY (in one response):
  - `get_cell_info(pci=<serving_pci>)` → full cell config: A3 offset, neighbor list, \
power, tilt, PDCCH, HO thresholds
  - `get_serving_cell_rsrp(time=<worst_t>)` → signal strength at the fault
  - `get_serving_cell_sinr(time=<worst_t>)` → interference level at the fault
  - `get_neighboring_cells_pci(time=<worst_t>)` → which neighbor cells are visible

**STEP 4** — For EACH neighbor PCI from Step 3, call `get_neighboring_cell_rsrp` \
(time=<worst_t>, pci=<neighbor_pci>)
  → Find the strongest neighbor's RSRP and compare it to the serving cell.

**STEP 5 (optional, only if ambiguous after Steps 1–4)** — Call any of:
  - `get_signaling_plane_event_log(time=<worst_t>)` → check for HO attempt failures
  - `get_kpi_data()` → check if serving cell is overloaded (high PRB utilization)
  - `get_mr_data()` → aggregate MR data for additional neighbor signal evidence
  - `judge_mainlobe_or_not(time, pci)` → check beam alignment
  - `calculate_overlap_ratio(pci_serving, pci_neighbor)` → coverage overlap

## Root Cause → Action Mapping

After collecting evidence, apply these rules:

| Observed pattern | Root cause | Recommended action |
|---|---|---|
| Neighbor RSRP − Serving RSRP > A3Offset×0.5 + A3Hyst×0.5 (dB), neighbor IS in config | Late handover (A3 threshold too high) | Decrease A3 Offset for serving cell |
| Strong neighbor NOT in `PCell Neighbor Cell` list | Missing neighbor relation | Add neighbor relationship |
| SINR < 0 dB, RSRP > −110 dBm | Co-channel interference | Tilt down or reduce power of serving/interfering cell |
| RSRP < −110 dBm | Coverage hole | Increase Tx power or lift antenna tilt |
| RF metrics healthy (RSRP > −100, SINR > 5) but throughput low | Backhaul / server issue | Check test server and transmission |
| Cell PRB utilization > 80% in KPI data | Cell overloaded | Decrease CovInterFreqA2/A5 thresholds to offload to neighbor band |

## Important Notes

- IntraFreqHoA3Offset is stored in 0.5 dB units (e.g., value=10 → 5.0 dB)
- IntraFreqHoA3Hyst is stored in 0.5 dB units (e.g., value=2 → 1.0 dB)
- The A3 condition is STRICT: neighbor must be STRICTLY GREATER than serving + offset + hyst
- The neighbor list format is "[gNodeBID_ARFCN_PCI, ...]"; check the LAST segment for PCI
- For multiple-answer questions, select 2–4 options; for single-answer, select exactly 1

## Answer Format (MANDATORY)

Always conclude with your answer enclosed in \\boxed{}:
- Single answer:   \\boxed{C9}
- Multiple answers: \\boxed{C3|C7}  (pipe-separated, ASCENDING numeric order)
"""


# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------

class Environment:
    """
    Responsible for:
    - discovering tool descriptors from FastAPI `/tools`
    - executing tool calls requested by the LLM
    - applying per-scenario context via X-Scenario-Id header
    """

    endpoint_mapper = {
        "get_all_scenario": "/scenario/all",
        "get_config_data": "/config-data",
        "get_user_plane_data": "/user-plane-data",
        "get_throughput_logs": "/throughput-logs",
        "get_cell_info": "/cell-info",
        "get_gnodeb_location": "/gnodeb-location",
        "get_user_location": "/user-location",
        "get_serving_cell_pci": "/serving-cell-pci",
        "get_serving_cell_rsrp": "/serving-cell-rsrp",
        "get_serving_cell_sinr": "/serving-cell-sinr",
        "get_rbs_allocated_to_user": "/rbs-allocated-to-user",
        "get_neighboring_cells_pci": "/neighboring-cells-pci",
        "get_neighboring_cell_rsrp": "/neighboring-cell-rsrp",
        "get_signaling_plane_event_log": "/signaling-plane-event-log",
        "get_all_cells_pci": "/all-cells-pci",
        "get_available_tools": "/tools",
        "health": "/health",
        "judge_mainlobe_or_not": "/judge_mainlobe",
        "calculate_horizontal_angle": "/calculate_horizontal_angle",
        "calculate_tilt_angle": "/calculate_tilt_angle",
        "calculate_pathloss": "/calculate_pathloss",
        "calculate_overlap_ratio": "/calculate_overlap_ratio",
        "get_kpi_data": "/get_kpi_data",
        "get_mr_data": "/get_mr_data",
        "optimize_antenna_gain": "/optimize_antenna_gain"
    }

    def __init__(self, server_url: str, verbose: bool = False, log_file: Optional[str] = None,
                 timeout: float = 15.0, logger: logging.Logger = None):
        self.server_url = server_url.rstrip("/")
        self.verbose = verbose
        self.timeout = timeout
        self.logger = logger if logger is not None else init_logger()

    def _headers(self, scenario_id: Optional[str] = None) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if scenario_id:
            headers["X-Scenario-Id"] = scenario_id
        return headers

    def _call_api(self, function_name: str, scenario_id: Optional[str] = None, **params: Any) -> Dict[str, Any]:
        endpoint = self.endpoint_mapper.get(function_name)
        if endpoint is None:
            return {"error": f"Unknown tool '{function_name}'"}

        url = f"{self.server_url}{endpoint}"
        headers = self._headers(scenario_id=scenario_id)

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            if self.verbose:
                self.logger.info(f"[Tools API] GET {endpoint} params={params}")
            return resp.json()
        except requests.exceptions.HTTPError:
            try:
                detail = resp.json().get("detail", str(resp.text))
            except Exception:
                detail = str(resp.text)
            if self.verbose:
                self.logger.info(f"[Tools API] GET {endpoint} -> HTTPError: {detail}")
            return {"error": detail}
        except Exception as e:
            if self.verbose:
                self.logger.info(f"[Tools API] GET {endpoint} -> ERROR: {e}")
            return {"error": str(e)}

    def get_tools(self) -> List[Dict[str, Any]]:
        tools = self._call_api("get_available_tools")
        if isinstance(tools, dict) and "error" in tools:
            return []
        if not isinstance(tools, list):
            return []
        return tools

    def get_scenarios(self) -> List[Dict[str, Any]]:
        scenarios = self._call_api("get_all_scenario")
        if isinstance(scenarios, dict) and "error" in scenarios:
            return []
        if not isinstance(scenarios, list):
            return []
        return scenarios

    def execute(self, tool_call: ToolCall, scenario_id: Optional[str] = None) -> str:
        try:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")
            result = self._call_api(function_name=function_name, scenario_id=scenario_id, **arguments)
            return json.dumps(result, ensure_ascii=False)
        except json.JSONDecodeError:
            error_msg = f"Tool parameter parsing failed: {tool_call.function.arguments}"
            if self.verbose:
                self.logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg}, ensure_ascii=False)
        except Exception as e:
            error_msg = f"Tool invocation execution failed: {str(e)}"
            if self.verbose:
                self.logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg}, ensure_ascii=False)


# ------------------------------------------------------------------------------
# LLM Agent Runner
# ------------------------------------------------------------------------------

class AgentsRunner:
    """
    Owns:
    - OpenAI client
    - RAG retriever (optional)
    - solve() loop (tool calling)
    - benchmark() across scenarios and attempts
    """

    def __init__(
            self,
            environment: Environment,
            model_url: str,
            model_name: str,
            model_provider: Optional[str] = None,
            max_tokens: int = 16000,
            max_retries: int = 3,
            max_iterations: int = 8,
            verbose: bool = False,
            logger: logging.Logger = None,
            rag_retriever: Optional[RAGRetriever] = None,
            rag_top_k: int = 2,
            use_system_prompt: bool = True,
    ):
        self.environment = environment
        self.model_url = model_url
        self.model_name = model_name
        self.model_provider = model_provider
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.logger = logger if logger is not None else init_logger()
        self.running_metrics = {}
        self.rag_retriever = rag_retriever
        self.rag_top_k = rag_top_k
        self.use_system_prompt = use_system_prompt

        self.client = OpenAI(
            base_url=model_url,
            api_key=API_KEY,
            http_client=httpx.Client(verify=False),
        )

    def _call_model(self, messages: List[Dict[str, Any]], functions: List[Dict[str, Any]], **kwargs):
        base_wait_time = 1.0

        call_kwargs = {
            "model": f"{self.model_provider}/{self.model_name}" if self.model_provider else self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            **kwargs
        }

        if functions:
            call_kwargs["tools"] = functions
            call_kwargs["tool_choice"] = "auto"

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**call_kwargs)
                return response.choices[0].message

            except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as exc:
                if self.verbose:
                    self.logger.error(traceback.format_exc())

                if hasattr(exc, "status_code") and 400 <= exc.status_code < 500 and exc.status_code != 429:
                    if self.verbose:
                        self.logger.info("Non-retriable exception: %s", exc)
                    return None

                if attempt == self.max_retries:
                    if self.verbose:
                        self.logger.info("Final failure after %s attempts: %s", self.max_retries, exc)
                    return None

                wait = base_wait_time * (2 ** (attempt - 1))
                if self.verbose:
                    self.logger.info("Retry %s/%s after %.1fs due to: %s", attempt, self.max_retries, wait, exc)
                time.sleep(wait)

            except Exception as exc:
                if self.verbose:
                    self.logger.info("Unhandled exception: %s", exc)
                return None

        return None

    def _build_system_message(self, scenario: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build the system message combining the diagnostic protocol with
        RAG few-shot examples (if retriever is available).
        """
        if not self.use_system_prompt:
            return None

        system_content = SYSTEM_PROMPT

        if self.rag_retriever is not None:
            try:
                few_shots = self.rag_retriever.retrieve_and_format(scenario, top_k=self.rag_top_k)
                if few_shots:
                    system_content = system_content + "\n\n" + few_shots
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"[RAG] Retrieval failed: {e}")

        return {"role": "system", "content": system_content}

    def run(self, scenario: Dict[str, Any], free_mode: bool = True) -> Dict[str, Any]:
        scenario_id = scenario.get("scenario_id")
        task = scenario.get("task", {})
        tag = scenario.get("tag", "single-answer")

        root_causes = "".join([f"{item['id']}:{item['label']}\n" for item in task.get("options", [])])

        # tools from server
        tool_defs = self.environment.get_tools()
        if not tool_defs:
            return {"scenario_id": scenario_id, "status": "unresolved", "reason": "No tools available"}

        question = task.get("description", "") + f"\nOptions:\n{root_causes}"

        # Build conversation — include system prompt if enabled
        messages: List[Dict[str, Any]] = []
        system_msg = self._build_system_message(scenario)
        if system_msg:
            messages.append(system_msg)
        messages.append({"role": "user", "content": question})

        num_tool_calls = 0
        list_tool_calls = []
        status = None
        reason = None
        last_msg = None

        for i in range(self.max_iterations):
            self.logger.info(f"\n[Scenario: {scenario_id}] Round {i + 1} conversation, calling tools:")

            msg = self._call_model(messages, functions=tool_defs)
            if msg is None:
                continue

            last_msg = msg
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

            if self.verbose:
                print_model_response(msg, logger=self.logger, minimize=False)

            # Execute ALL tool calls in this turn (supports parallel batching)
            if msg.tool_calls:
                num_tool_calls += len(msg.tool_calls)

                for j, tool_call in enumerate(msg.tool_calls):
                    if self.verbose:
                        print_tool_call(tool_call, logger=self.logger)

                    tool_result = self.environment.execute(tool_call, scenario_id=scenario_id)

                    messages.append({"role": "tool", "content": tool_result, "tool_call_id": tool_call.id})

                    if self.verbose:
                        print_tool_result(tool_result, logger=self.logger)

                    has_failed = "error" in tool_result
                    list_tool_calls.append({
                        "function_name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                        "turn": i + 1,
                        "has_failed": has_failed,
                        "order": j + 1,
                        "results": tool_result
                    })

            elif msg.content:
                status = "solved"
                break

            else:
                status = "unresolved"
                reason = "Unable to answer this question."
                break

        if status is None:
            status = "unresolved"
            reason = "The maximum number of iterations has been reached."

        # Final constraint prompt: force the model to emit \boxed{} if it hasn't
        if free_mode:
            current_answer = getattr(last_msg, "content", "") or getattr(last_msg, "reasoning_content", "") if last_msg else ""
            current_traces = getattr(last_msg, "reasoning_content", "") if last_msg else ""
            agent_answer = parse_answer(current_answer) or parse_answer(current_traces)

            if not agent_answer:
                self.logger.info(f"\n[Scenario: {scenario_id}] Forcing final answer extraction...")
                status = "solved"

                is_multiple = tag == "multiple-answer" or "two to four" in question.lower()

                if not is_multiple:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Based on your analysis above, select the SINGLE most appropriate "
                            "optimization solution. Enclose its ID in \\boxed{} in your final answer. "
                            f"Example: \\boxed{{C3}}\n\nOptions:\n{root_causes}"
                        ),
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Based on your analysis above, select TWO to FOUR possible optimization "
                            "solutions. Enclose their IDs in \\boxed{} separated by |, in ascending "
                            "numeric order. Example: \\boxed{{C3|C7}} or \\boxed{{C5|C9|C11|C20}}\n\n"
                            f"Options:\n{root_causes}"
                        ),
                    })

                msg2 = self._call_model(messages, functions=[])
                if msg2 is not None:
                    last_msg = msg2

        return {
            "scenario_id": scenario_id,
            "num_iterations": (i + 1),
            "tool_calls": list_tool_calls,
            "num_tool_calls": num_tool_calls,
            "status": status,
            "traces": getattr(last_msg, "reasoning_content", "") if last_msg else "",
            "answer": getattr(last_msg, "content", "") or getattr(last_msg, "reasoning_content", "") if last_msg else "",
            "messages": messages,
            "reason": reason,
        }

    def benchmark(
            self,
            num_attempts: int,
            save_dir: str,
            save_freq: int = 10,
            max_samples: int = None,
            free_mode: bool = True,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)

        completions: List[Dict[str, Any]] = []
        save_result: List[Dict[str, Any]] = []

        scenarios = self.environment.get_scenarios()

        if max_samples is not None:
            scenarios = scenarios[:min(max_samples, len(scenarios))]

        for idx, scenario in enumerate(scenarios):
            scenario_id = scenario.get("scenario_id")
            start_time = time.time()

            n_success = 0.0
            agent_answers: List[str] = []
            sample_response: Optional[Dict[str, Any]] = {}

            for attempt in range(num_attempts):
                self.logger.info(f"[Scenario {scenario_id}] attempt {attempt + 1}/{num_attempts}")

                response = self.run(scenario=scenario, free_mode=free_mode)
                sample_response = response

                if response.get("status") == "solved":
                    raw_answer = response.get("answer", "")
                    raw_traces = response.get("traces", "")

                    # Use the robust parser first, fall back to legacy extractor
                    agent_answer = (
                        parse_answer(raw_answer)
                        or parse_answer(raw_traces)
                        or extract_answer_all(raw_answer)
                        or extract_answer_all(raw_traces)
                    )

                    ground_truth = scenario.get("answer")
                    if ground_truth:
                        score = iou_score(agent_answer, ground_truth)
                        n_success += score
                    agent_answers.append(agent_answer)

                    pink = "\033[95m"
                    reset = "\033[0m"
                    gt_str = ground_truth if ground_truth else "N/A"
                    self.logger.info(
                        f"{pink}\n[Scenario: {scenario_id}] Agent answer: {agent_answer} | "
                        f"Ground truth: {gt_str}{reset}"
                    )

            acc = n_success / float(num_attempts)
            latency = round((time.time() - start_time) / float(num_attempts), 2)

            completions.append({
                "scenario_id": scenario_id,
                "free_mode": free_mode,
                "response": sample_response.get("answer", ""),
                "traces": sample_response.get("traces", ""),
                "num_iterations": sample_response.get("num_iterations", 0),
                "num_tool_calls": (sample_response or {}).get("num_tool_calls", 0),
                "tool_calls": sample_response.get("tool_calls", []),
                "answers": agent_answers,
                "ground_truth": scenario.get("answer"),
                "accuracy": acc,
                "latency": latency,
            })

            final_answer = agent_answers[0] if agent_answers else ""
            save_result.append({
                "scenario_id": scenario_id,
                "answers": final_answer,
            })

            if ((idx + 1) % save_freq == 0) or ((idx + 1) == len(scenarios)):
                df = pd.DataFrame(save_result)
                df.to_csv(os.path.join(save_dir, "result.csv"), index=False)
                self.logger.info(f"[Benchmark] Saved {idx + 1}/{len(scenarios)} results.")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track A Agent Benchmark")
    parser.add_argument("--server_url", type=str, default="http://localhost:7860")
    parser.add_argument("--model_url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--model_name", type=str, default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--model_provider", type=str, default=None)
    parser.add_argument("--num_attempts", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=16000)
    parser.add_argument("--max_iterations", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--log_file", type=str, default="./log.log")
    parser.add_argument("--free_mode", action="store_false",
                        help="Disable free_mode (free_mode=True by default)")
    parser.add_argument("--no_system_prompt", action="store_true",
                        help="Disable the system prompt (for ablation)")
    parser.add_argument("--no_rag", action="store_true",
                        help="Disable RAG retrieval (for ablation)")
    parser.add_argument("--rag_train_json", type=str, default="data/Phase_1/train.json",
                        help="Path to training JSON for RAG index")
    parser.add_argument("--rag_top_k", type=int, default=2,
                        help="Number of RAG few-shot examples to retrieve")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = init_logger(log_file=args.log_file)

    # Build RAG retriever (unless disabled)
    rag = None
    if not args.no_rag and os.path.exists(args.rag_train_json):
        try:
            rag = RAGRetriever(train_json_path=args.rag_train_json)
            rag.build_index()
            logger.info("[RAG] Retriever ready.")
        except ImportError:
            logger.warning("[RAG] sentence-transformers or faiss-cpu not installed. Skipping RAG.")
            rag = None
    elif args.no_rag:
        logger.info("[RAG] Disabled by --no_rag flag.")
    else:
        logger.warning(f"[RAG] Training JSON not found at {args.rag_train_json}. Skipping RAG.")

    env = Environment(server_url=args.server_url, verbose=args.verbose, logger=logger)

    runner = AgentsRunner(
        environment=env,
        model_url=args.model_url,
        model_name=args.model_name,
        model_provider=args.model_provider,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        logger=logger,
        rag_retriever=rag,
        rag_top_k=args.rag_top_k,
        use_system_prompt=not args.no_system_prompt,
    )

    runner.benchmark(
        max_samples=args.max_samples,
        num_attempts=args.num_attempts,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        free_mode=not args.free_mode,
    )
