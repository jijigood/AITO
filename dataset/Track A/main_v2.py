#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
V2 Orchestrator — Track A Telco Troubleshooting

Implements the structured pipeline from ARCHITECTURE_V2.md:
  1. parse_scenario()    → ParsedScenario   (local, no HTTP)
  2. route()             → Family
  3. run_all_skills()    → ScenarioFeatures
  4. generate_candidates()→ CandidateOutput
  5. RAG V2 retrieval    → few-shot context string
  6. Single Qwen call    → \boxed{answer}
  7. Verifier (optional) → targeted tool call + re-ask if low confidence
  8. Fallback            → top_candidate() from CandidateOutput

Key difference from main.py (V1): Qwen receives structured evidence + a
pre-ranked candidate list and selects from it — no free-form tool loop.
"""

import argparse
import json
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
import requests
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIError

from logger import init_logger
from skills.answer_parser import iou_score, parse_answer
from skills.candidate_generator import generate_candidates
from skills.feature_engine import ScenarioFeatures, run_all_skills
from skills.scenario_parser import parse_scenario
from skills.scenario_router import route
from rag_retriever_v2 import StructuredCaseRAG

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ.setdefault("AGENT_API_KEY", "sk-XXXXXXXXXXXXX")

API_KEY = os.environ.get("AGENT_API_KEY", "dummy")

# ──────────────────────────────────────────────────────────────────────────────
# System prompt (short — Qwen selects, does not re-diagnose)
# ──────────────────────────────────────────────────────────────────────────────

QWEN_V2_SYSTEM_PROMPT = """\
You are a 5G network optimization expert. The diagnostic analysis has already
been performed. Your task is to select the single best answer (or the best
combination of answers for multiple-answer questions) from the pre-ranked
candidate list provided. Base your choice on the diagnostic evidence and the
retrieved similar cases. Do not re-run analysis — just choose and output the
answer.

Output format (mandatory, nothing else after the boxed answer):
- Single-answer:   \\boxed{C9}
- Multiple-answer: \\boxed{C3|C7}  (pipe-separated, ascending numeric order, 2–4 codes)
"""


# ──────────────────────────────────────────────────────────────────────────────
# Feature vector bridge (ScenarioFeatures → 13-dim for RAG V2)
# ──────────────────────────────────────────────────────────────────────────────

def _sf_to_feature_vec(sf: ScenarioFeatures) -> np.ndarray:
    """
    Map ScenarioFeatures → 13-dim float32 vector matching rag_retriever_v2.py's
    feature order exactly so that cosine similarity is computed in the same space
    as the indexed training vectors.

    Feature index map:
      0  worst_tp / 500
      1  drop_magnitude / 500
      2  (rsrp + 140) / 60
      3  (sinr + 20) / 40
      4  neighbor_advantage / 20
      5  (a3_margin + 10) / 20
      6  max_prb_util / 100
      7  num_neighbors / 5
      8  late_handover_score
      9  missing_neighbor_score
      10 interference_score
      11 coverage_hole_score
      12 inter_freq_score
    """
    tp = sf.throughput
    mob = sf.mobility
    ifreq = sf.inter_freq

    worst_tp       = (tp.worst_throughput_mbps or 0.0)
    drop_mag       = (tp.throughput_drop_magnitude or 0.0)
    rsrp           = (mob.serving_rsrp_dbm or -140.0)
    sinr           = (mob.serving_sinr_db or -20.0)
    adv            = (mob.neighbor_advantage_db or 0.0)
    a3_margin      = (mob.a3_margin_db or -10.0)
    prb_util       = (ifreq.max_dl_prb_util or 0.0)
    num_nb         = float(len(mob.visible_neighbor_pcis))

    vec = np.array([
        worst_tp / 500.0,
        drop_mag / 500.0,
        (rsrp + 140.0) / 60.0,
        (sinr + 20.0) / 40.0,
        adv / 20.0,
        (a3_margin + 10.0) / 20.0,
        prb_util / 100.0,
        num_nb / 5.0,
        sf.score_late_handover,
        sf.score_missing_neighbor,
        sf.score_interference,
        sf.score_coverage_hole,
        sf.score_inter_freq_load,
    ], dtype=np.float32)

    return np.clip(vec, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_qwen_prompt(
    scenario: Dict[str, Any],
    family_name: str,
    sf: ScenarioFeatures,
    candidates_text: str,
    rag_ctx: str,
) -> str:
    """
    Build the structured user-message for a single Qwen call.

    Sections:
      ## Scenario
      ## Diagnostic Evidence
      ## Candidate Options
      ## Similar Solved Cases
      ## All Options
      [selection instruction]
    """
    task = scenario.get("task", {})
    tag = scenario.get("tag", "single-answer")
    options = task.get("options", [])

    # Family description
    family_desc = {
        "F1": "4 BS · single-answer",
        "F2": "4 BS · multiple-answer (2–4 codes)",
        "F3": "5 BS · multiple-answer (exactly 4 codes)",
        "F4": "13 BS · single-answer (inter-frequency focus)",
    }.get(family_name, family_name)

    parts: List[str] = []

    parts.append(f"## Scenario\nFamily: {family_name} ({family_desc})  |  tag: {tag}\n")

    evidence = sf.as_evidence_text()
    parts.append(f"## Diagnostic Evidence\n{evidence}\n")

    if candidates_text:
        parts.append(candidates_text)

    if rag_ctx:
        parts.append(rag_ctx)

    # Full option list so Qwen can verify exact C-code → label mapping
    opt_lines = "\n".join(f"  {o['id']}: {o['label']}" for o in options)
    parts.append(f"## All Options\n{opt_lines}\n")

    is_multiple = "multiple" in tag.lower()
    if is_multiple:
        parts.append(
            "Select the BEST combination of 2–4 option codes that together fix the issue.\n"
            "Output **only** the boxed answer (no other text):\n"
            "  \\boxed{C_N|C_M}   ← pipe-separated, ascending numeric order"
        )
    else:
        parts.append(
            "Select the SINGLE best option code that fixes the issue.\n"
            "Output **only** the boxed answer (no other text):\n"
            "  \\boxed{C_N}"
        )

    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Environment (reused from main.py — HTTP calls for verifier only)
# ──────────────────────────────────────────────────────────────────────────────

class Environment:
    """HTTP wrapper for the FastAPI tool server (used only by the verifier)."""

    endpoint_mapper = {
        "get_all_scenario":          "/scenario/all",
        "get_kpi_data":              "/get_kpi_data",
        "get_mr_data":               "/get_mr_data",
        "get_signaling_plane_event_log": "/signaling-plane-event-log",
        "judge_mainlobe_or_not":     "/judge_mainlobe",
        "calculate_overlap_ratio":   "/calculate_overlap_ratio",
        "calculate_horizontal_angle":"/calculate_horizontal_angle",
        "calculate_tilt_angle":      "/calculate_tilt_angle",
        "get_available_tools":       "/tools",
    }

    def __init__(self, server_url: str, timeout: float = 20.0, logger: logging.Logger = None):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.logger = logger or init_logger()

    def _headers(self, scenario_id: Optional[str] = None) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if scenario_id:
            h["X-Scenario-Id"] = scenario_id
        return h

    def call(self, function_name: str, scenario_id: Optional[str] = None, **params) -> Dict[str, Any]:
        endpoint = self.endpoint_mapper.get(function_name)
        if endpoint is None:
            return {"error": f"Unknown tool '{function_name}'"}
        url = f"{self.server_url}{endpoint}"
        try:
            resp = requests.get(url, params=params,
                                headers=self._headers(scenario_id), timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            return {"error": str(exc)}

    def get_scenarios(self) -> List[Dict[str, Any]]:
        result = self.call("get_all_scenario")
        if isinstance(result, list):
            return result
        return []


# ──────────────────────────────────────────────────────────────────────────────
# V2 Runner
# ──────────────────────────────────────────────────────────────────────────────

class V2Runner:
    """
    Orchestrates the full V2 pipeline for every scenario.

    Parameters
    ----------
    environment : Environment
        HTTP client — used only by the verifier.
    model_url, model_name : str
        OpenAI-compatible endpoint for Qwen.
    rag_v2 : StructuredCaseRAG | None
        Pre-built structured RAG index.  If None, RAG context is omitted.
    max_tokens : int
        Max token budget for each Qwen call.
    max_retries : int
        Retry count on transient API errors.
    confidence_high_threshold : float
        Top candidate score above which we trust Qwen without verification.
    confidence_low_threshold : float
        Top candidate score below which we trigger the verifier.
    use_verifier : bool
        Whether to enable the verifier (server tool calls for low confidence).
    """

    def __init__(
        self,
        environment: Environment,
        model_url: str,
        model_name: str,
        model_provider: Optional[str] = None,
        rag_v2: Optional[StructuredCaseRAG] = None,
        rag_top_k: int = 3,
        max_tokens: int = 4096,
        max_retries: int = 3,
        confidence_high_threshold: float = 0.7,
        confidence_low_threshold: float = 0.2,
        use_verifier: bool = True,
        verbose: bool = False,
        logger: logging.Logger = None,
    ):
        self.environment = environment
        self.model_url = model_url
        self.model_name = model_name
        self.model_provider = model_provider
        self.rag_v2 = rag_v2
        self.rag_top_k = rag_top_k
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.confidence_high_threshold = confidence_high_threshold
        self.confidence_low_threshold = confidence_low_threshold
        self.use_verifier = use_verifier
        self.verbose = verbose
        self.logger = logger or init_logger()

        self.client = OpenAI(
            base_url=model_url,
            api_key=API_KEY,
            http_client=httpx.Client(verify=False),
        )

    # ── Qwen call ──────────────────────────────────────────────────────────────

    def _call_qwen(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Single model call (no tools). Returns content string or None."""
        model_id = (
            f"{self.model_provider}/{self.model_name}"
            if self.model_provider
            else self.model_name
        )
        base_wait = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=self.max_tokens,
                )
                msg = resp.choices[0].message
                return msg.content or getattr(msg, "reasoning_content", None) or ""
            except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as exc:
                if self.verbose:
                    self.logger.error(traceback.format_exc())
                is_client_err = (
                    hasattr(exc, "status_code")
                    and 400 <= exc.status_code < 500
                    and exc.status_code != 429
                )
                if is_client_err or attempt == self.max_retries:
                    return None
                wait = base_wait * (2 ** (attempt - 1))
                self.logger.info("[Qwen] Retry %d/%d after %.1fs: %s", attempt, self.max_retries, wait, exc)
                time.sleep(wait)
            except Exception as exc:
                self.logger.error("[Qwen] Unhandled: %s", exc)
                return None
        return None

    # ── Verifier ───────────────────────────────────────────────────────────────

    def _run_verifier(
        self,
        scenario_id: str,
        sf: ScenarioFeatures,
        messages: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Call 1–2 server tools whose results are relevant to the top issue,
        append them to the conversation, then re-ask Qwen.
        """
        top_issue, _ = sf.top_issue()
        tool_results: List[str] = []

        # Choose tools based on top issue
        if top_issue in ("late_handover", "missing_neighbor"):
            r = self.environment.call("get_signaling_plane_event_log", scenario_id=scenario_id,
                                      time=sf.throughput.worst_time or "")
            tool_results.append(f"[signaling_events]\n{json.dumps(r, ensure_ascii=False)[:800]}")

        elif top_issue in ("inter_freq_load",):
            r = self.environment.call("get_kpi_data", scenario_id=scenario_id)
            tool_results.append(f"[kpi_data]\n{json.dumps(r, ensure_ascii=False)[:800]}")

        elif top_issue in ("overshoot_overlap", "interference"):
            pci_s = sf.mobility.serving_pci_at_worst or 0
            pci_n = sf.mobility.best_neighbor_pci or 0
            if pci_s and pci_n:
                r = self.environment.call("calculate_overlap_ratio",
                                          scenario_id=scenario_id,
                                          pci_serving=pci_s,
                                          pci_neighbor=pci_n)
                tool_results.append(f"[overlap_ratio]\n{json.dumps(r, ensure_ascii=False)[:400]}")

        elif top_issue in ("coverage_hole",):
            r = self.environment.call("get_mr_data", scenario_id=scenario_id)
            tool_results.append(f"[mr_data]\n{json.dumps(r, ensure_ascii=False)[:800]}")

        if not tool_results:
            return None

        # Append tool evidence as a user message, then re-ask
        extra = "Additional verification data:\n" + "\n\n".join(tool_results)
        extra += "\n\nGiven this additional evidence, confirm or revise your answer. Output **only** \\boxed{...}."
        new_messages = list(messages) + [{"role": "user", "content": extra}]
        return self._call_qwen(new_messages)

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def solve(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the full V2 pipeline for one scenario.

        Returns a dict with:
          scenario_id, answer, status, latency_s, pipeline details
        """
        t0 = time.time()
        scenario_id = scenario.get("scenario_id", "?")
        tag = scenario.get("tag", "single-answer")

        # ── Step 1: Parse ──────────────────────────────────────────────────────
        try:
            ps = parse_scenario(scenario)
        except Exception as exc:
            self.logger.error("[%s] parse_scenario failed: %s", scenario_id, exc)
            return self._error_result(scenario_id, "parse_failed")

        # ── Step 2: Route ──────────────────────────────────────────────────────
        try:
            family = route(ps.tag, ps.num_base_stations)
        except Exception as exc:
            self.logger.error("[%s] route() failed: %s", scenario_id, exc)
            return self._error_result(scenario_id, "route_failed")

        if self.verbose:
            self.logger.info("[%s] Family: %s (BS=%d, tag=%s)",
                             scenario_id, family.name, family.num_bs, family.is_multiple)

        # ── Step 3: Feature extraction ─────────────────────────────────────────
        try:
            sf = run_all_skills(ps)
        except Exception as exc:
            self.logger.error("[%s] run_all_skills() failed: %s", scenario_id, exc)
            return self._error_result(scenario_id, "skills_failed")

        if self.verbose:
            top_issue, top_score = sf.top_issue()
            self.logger.info("[%s] Top issue: %s (%.2f)", scenario_id, top_issue, top_score)

        # ── Step 4: Candidate generation ───────────────────────────────────────
        try:
            candidates = generate_candidates(ps, sf, family)
        except Exception as exc:
            self.logger.error("[%s] generate_candidates() failed: %s", scenario_id, exc)
            candidates = None

        candidates_text = candidates.format_for_prompt() if candidates else ""
        top_cand_score = 0.0
        top_cand_answer = None
        if candidates:
            top_cand_answer = candidates.top_candidate()
            # Extract top score from first candidate
            if not family.is_multiple and candidates.single_candidates:
                top_cand_score = candidates.single_candidates[0].score
            elif family.is_multiple and candidates.multi_candidates:
                top_cand_score = candidates.multi_candidates[0].score

        # ── Step 5: Structured RAG retrieval ───────────────────────────────────
        rag_ctx = ""
        if self.rag_v2 is not None:
            try:
                fvec = _sf_to_feature_vec(sf)
                rag_ctx = self.rag_v2.retrieve_and_format(fvec, family.name, top_k=self.rag_top_k)
            except Exception as exc:
                self.logger.warning("[%s] RAG retrieval failed: %s", scenario_id, exc)

        # ── Step 6: Build Qwen prompt ──────────────────────────────────────────
        user_msg = _build_qwen_prompt(scenario, family.name, sf, candidates_text, rag_ctx)

        if self.verbose:
            self.logger.info("[%s] Prompt (first 300 chars):\n%s...", scenario_id, user_msg[:300])

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": QWEN_V2_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # ── Step 7: Single Qwen call ───────────────────────────────────────────
        raw_response = self._call_qwen(messages)
        answer = parse_answer(raw_response or "")

        if self.verbose:
            self.logger.info("[%s] Qwen response: %s  →  parsed: %r",
                             scenario_id, (raw_response or "")[:200], answer)

        # ── Step 8: Confidence check + optional verifier ───────────────────────
        is_low_confidence = (not answer) or (top_cand_score < self.confidence_low_threshold)

        if is_low_confidence and self.use_verifier:
            self.logger.info("[%s] Low confidence (score=%.2f, answer=%r) — running verifier",
                             scenario_id, top_cand_score, answer)
            verifier_response = self._run_verifier(scenario_id, sf, messages)
            if verifier_response:
                verifier_answer = parse_answer(verifier_response)
                if verifier_answer:
                    answer = verifier_answer
                    if self.verbose:
                        self.logger.info("[%s] Verifier revised answer to: %s", scenario_id, answer)

        # ── Step 9: Deterministic fallback ─────────────────────────────────────
        if not answer and top_cand_answer:
            answer = top_cand_answer
            self.logger.info("[%s] Fallback to top candidate: %s", scenario_id, answer)

        latency = round(time.time() - t0, 2)
        self.logger.info("[%s] Final answer: %s  (%.1fs)", scenario_id, answer, latency)

        return {
            "scenario_id": scenario_id,
            "answer": answer,
            "status": "solved" if answer else "unresolved",
            "family": family.name,
            "top_issue": sf.top_issue()[0],
            "top_issue_score": round(sf.top_issue()[1], 3),
            "top_candidate_score": round(top_cand_score, 3),
            "latency_s": latency,
            "qwen_raw": raw_response or "",
        }

    def _error_result(self, scenario_id: str, reason: str) -> Dict[str, Any]:
        return {
            "scenario_id": scenario_id,
            "answer": "",
            "status": "unresolved",
            "family": "?",
            "top_issue": reason,
            "top_issue_score": 0.0,
            "top_candidate_score": 0.0,
            "latency_s": 0.0,
            "qwen_raw": "",
        }

    # ── Benchmark ──────────────────────────────────────────────────────────────

    def benchmark(
        self,
        save_dir: str,
        max_samples: Optional[int] = None,
        save_freq: int = 10,
    ) -> None:
        """
        Run V2 pipeline on all scenarios fetched from the server and write result.csv.
        """
        os.makedirs(save_dir, exist_ok=True)

        scenarios = self.environment.get_scenarios()
        if not scenarios:
            self.logger.error("No scenarios fetched from server. Check --server_url.")
            return

        if max_samples is not None:
            scenarios = scenarios[: min(max_samples, len(scenarios))]

        self.logger.info("[V2] Running on %d scenarios...", len(scenarios))

        save_result: List[Dict[str, Any]] = []
        completions: List[Dict[str, Any]] = []
        total_iou = 0.0
        n_evaluated = 0

        for idx, scenario in enumerate(scenarios):
            result = self.solve(scenario)
            scenario_id = result["scenario_id"]
            agent_answer = result["answer"]

            # IOU scoring against ground truth (if available — training mode)
            ground_truth = scenario.get("answer")
            iou = None
            if ground_truth:
                iou = iou_score(agent_answer, ground_truth)
                total_iou += iou
                n_evaluated += 1
                pink = "\033[95m"
                reset = "\033[0m"
                self.logger.info(
                    "%s[%s] pred=%r  gt=%r  iou=%.3f%s",
                    pink, scenario_id, agent_answer, ground_truth, iou, reset
                )

            save_result.append({"scenario_id": scenario_id, "answers": agent_answer})
            completions.append({**result, "ground_truth": ground_truth, "iou": iou})

            # Periodic save
            if ((idx + 1) % save_freq == 0) or ((idx + 1) == len(scenarios)):
                df = pd.DataFrame(save_result)
                df.to_csv(os.path.join(save_dir, "result.csv"), index=False)
                mean_iou = (total_iou / n_evaluated) if n_evaluated else float("nan")
                self.logger.info(
                    "[V2] Saved %d/%d  mean_iou=%.3f",
                    idx + 1, len(scenarios), mean_iou
                )

        self.logger.info("[V2] Done. Final mean IOU: %.3f", total_iou / max(n_evaluated, 1))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track A V2 Agent Benchmark")
    parser.add_argument("--server_url",    type=str, default="http://localhost:7860")
    parser.add_argument("--model_url",     type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--model_name",    type=str, default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--model_provider",type=str, default=None)
    parser.add_argument("--max_samples",   type=int, default=None)
    parser.add_argument("--save_freq",     type=int, default=10)
    parser.add_argument("--max_tokens",    type=int, default=4096)
    parser.add_argument("--save_dir",      type=str, default="./results_v2")
    parser.add_argument("--log_file",      type=str, default="./log_v2.log")
    parser.add_argument("--rag_train_json",type=str,
                        default="data/Phase_1/train.json",
                        help="Path to train.json for structured RAG V2 index")
    parser.add_argument("--rag_top_k",     type=int, default=3,
                        help="Number of RAG few-shot examples to retrieve")
    parser.add_argument("--no_rag",        action="store_true",
                        help="Disable structured RAG retrieval")
    parser.add_argument("--no_verifier",   action="store_true",
                        help="Disable verifier (no server tool calls)")
    parser.add_argument("--confidence_high", type=float, default=0.7,
                        help="Top candidate score above which verifier is skipped")
    parser.add_argument("--confidence_low",  type=float, default=0.2,
                        help="Top candidate score below which verifier is triggered")
    parser.add_argument("--verbose",       action="store_true")
    args = parser.parse_args()

    logger = init_logger(log_file=args.log_file)

    # Build RAG V2 index (unless disabled)
    rag_v2 = None
    if not args.no_rag:
        if os.path.exists(args.rag_train_json):
            try:
                rag_v2 = StructuredCaseRAG(train_json_path=args.rag_train_json)
                rag_v2.build_index()
                logger.info("[RAG V2] Structured index ready.")
            except Exception as exc:
                logger.warning("[RAG V2] Failed to build index: %s. Skipping RAG.", exc)
                rag_v2 = None
        else:
            logger.warning("[RAG V2] train.json not found at %s. Skipping RAG.", args.rag_train_json)
    else:
        logger.info("[RAG V2] Disabled by --no_rag flag.")

    env = Environment(server_url=args.server_url, logger=logger)

    runner = V2Runner(
        environment=env,
        model_url=args.model_url,
        model_name=args.model_name,
        model_provider=args.model_provider,
        rag_v2=rag_v2,
        rag_top_k=args.rag_top_k,
        max_tokens=args.max_tokens,
        confidence_high_threshold=args.confidence_high,
        confidence_low_threshold=args.confidence_low,
        use_verifier=not args.no_verifier,
        verbose=args.verbose,
        logger=logger,
    )

    runner.benchmark(
        save_dir=args.save_dir,
        max_samples=args.max_samples,
        save_freq=args.save_freq,
    )
