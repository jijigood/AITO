#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
V3 Orchestrator — Track A Telco Troubleshooting

Key improvements over V2:
  1. F2/F3/F4 bypass Qwen — deterministic candidates are 100% accurate on
     training data, so we skip the model entirely for these families.
  2. Feature engine fixes — inter_freq capped for 4BS, SINR-based grouping,
     serving-cell-targeted candidates.
  3. Server tool calls — targeted tools for F1 ambiguous cases.
  4. Better Qwen prompt — only for F1, with tool evidence and focused candidate
     set.

Pipeline:
  1. parse_scenario()      → ParsedScenario   (local)
  2. route()               → Family
  3. run_all_skills()      → ScenarioFeatures  (local, improved)
  4. generate_candidates() → CandidateOutput
  5. DECISION POINT:
     a. F2/F3/F4 → deterministic top_candidate() — NO Qwen
     b. F1 → tool calls → Qwen selection → fallback
  6. Output

────────────────────────────────────────────────────────────────────────────
Team optimization targets:
  [Teammate 1] Feature engine rules → skills/feature_engine.py
               Focus: lift_tilt vs increase_power, insufficient_data detection
               Eval:  python eval_v3.py --full --per-action --family F1
  [Teammate 2] Server tool integration → skills/tool_caller.py
               Focus: judge_mainlobe for coverage, calculate_tilt_angle
               Eval:  run main_v3.py with --use-tools on training server
  [Teammate 3] Evaluation & RAG → eval_v3.py, rag_retriever_v2.py
               Focus: error analysis, RAG feature vector tuning
               Eval:  python eval_v3.py --full --per-action --errors errs.csv
  [You]        Fine-tuning & submission
               Focus: SFT data regeneration, LoRA tuning, result packaging
────────────────────────────────────────────────────────────────────────────

Usage:
    # With local Qwen server:
    python main_v3.py --server_url http://localhost:7860 \
                      --model_url http://localhost:8001/v1 \
                      --model_name qwen

    # Deterministic-only mode (no Qwen, no server tools):
    python main_v3.py --server_url http://localhost:7860 --deterministic

    # With server tools:
    python main_v3.py --server_url http://localhost:7860 \
                      --model_url http://localhost:8001/v1 \
                      --use-tools
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
from skills.tool_caller import ToolCaller, ToolEvidence
from rag_retriever_v2 import StructuredCaseRAG

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ.setdefault("AGENT_API_KEY", "sk-XXXXXXXXXXXXX")

API_KEY = os.environ.get("AGENT_API_KEY", "dummy")


# ──────────────────────────────────────────────────────────────────────────────
# System prompt — only used for F1 (Qwen selects from focused candidate set)
# ──────────────────────────────────────────────────────────────────────────────

QWEN_V3_SYSTEM_PROMPT = """\
You are a 5G network optimization expert. The diagnostic analysis has already
been performed. Your task is to select the single best answer from the pre-ranked
candidate list provided.

Key decision guidelines:
- If the user is OUTSIDE the mainlobe of the serving cell → prefer lift_tilt or adjust_azimuth
- If RSRP < -110 dBm → coverage hole → prefer increase_power or lift_tilt
- If SINR < 0 dB with strong neighbor → interference → prefer decrease_a3 (serving cell) or add_neighbor
- If radio KPIs are healthy (RSRP > -100, SINR > 5) but throughput is low:
  - High PRB (>70%): consider lift_tilt or insufficient_data
  - Low PRB (<30%): consider pdcch_2sym or check_server
- If DL CCE utilization > 0.5% → prefer pdcch_2sym
- If no clear issue found → prefer check_server or insufficient_data

Base your choice on the diagnostic evidence and similar solved cases below.
Output format (mandatory, nothing else after):
  \\boxed{C_N}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Feature vector bridge (ScenarioFeatures → 13-dim for RAG V2)
# ──────────────────────────────────────────────────────────────────────────────

def _sf_to_feature_vec(sf: ScenarioFeatures) -> np.ndarray:
    tp = sf.throughput
    mob = sf.mobility
    ifreq = sf.inter_freq

    vec = np.array([
        (tp.worst_throughput_mbps or 0.0) / 500.0,
        (tp.throughput_drop_magnitude or 0.0) / 500.0,
        ((mob.serving_rsrp_dbm or -140.0) + 140.0) / 60.0,
        ((mob.serving_sinr_db or -20.0) + 20.0) / 40.0,
        (mob.neighbor_advantage_db or 0.0) / 20.0,
        ((mob.a3_margin_db or -10.0) + 10.0) / 20.0,
        (ifreq.max_dl_prb_util or 0.0) / 100.0,
        float(len(mob.visible_neighbor_pcis)) / 5.0,
        sf.score_late_handover,
        sf.score_missing_neighbor,
        sf.score_interference,
        sf.score_coverage_hole,
        sf.score_inter_freq_load,
    ], dtype=np.float32)

    return np.clip(vec, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder — F1 only
# ──────────────────────────────────────────────────────────────────────────────

def _build_f1_prompt(
    scenario: Dict[str, Any],
    sf: ScenarioFeatures,
    candidates_text: str,
    rag_ctx: str,
    tool_evidence_text: str,
) -> str:
    """Build the user message for F1 Qwen selection."""
    task = scenario.get("task", {})
    options = task.get("options", [])

    parts: List[str] = []

    # Evidence
    evidence = sf.as_evidence_text()
    parts.append(f"## Diagnostic Evidence\n{evidence}")

    # Tool evidence (if available)
    if tool_evidence_text:
        parts.append(tool_evidence_text)

    # Candidates
    if candidates_text:
        parts.append(candidates_text)

    # RAG context
    if rag_ctx:
        parts.append(rag_ctx)

    # Full option list
    opt_lines = "\n".join(f"  {o['id']}: {o['label']}" for o in options)
    parts.append(f"## All Options\n{opt_lines}")

    parts.append(
        "Select the SINGLE best option code.\n"
        "Output **only** the boxed answer:\n"
        "  \\boxed{C_N}"
    )

    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Server environment (for fetching scenarios)
# ──────────────────────────────────────────────────────────────────────────────

class Environment:
    """HTTP wrapper for the FastAPI tool server."""

    def __init__(self, server_url: str, timeout: float = 20.0, logger=None):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)

    def _headers(self, scenario_id: Optional[str] = None):
        h = {"Content-Type": "application/json"}
        if scenario_id:
            h["X-Scenario-Id"] = scenario_id
        return h

    def get_scenarios(self) -> List[Dict[str, Any]]:
        try:
            resp = requests.get(
                f"{self.server_url}/scenario/all",
                headers=self._headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            return result if isinstance(result, list) else []
        except Exception as exc:
            self.logger.error("Failed to fetch scenarios: %s", exc)
            return []


# ──────────────────────────────────────────────────────────────────────────────
# V3 Runner
# ──────────────────────────────────────────────────────────────────────────────

class V3Runner:
    """
    V3 pipeline orchestrator.

    Key difference from V2: F2/F3/F4 are fully deterministic (no Qwen call).
    Qwen is only used for F1 where the deterministic pipeline narrows to 2-5
    candidates that need intelligent selection.
    """

    def __init__(
        self,
        environment: Environment,
        model_url: str = "",
        model_name: str = "qwen",
        model_provider: Optional[str] = None,
        rag_v2: Optional[StructuredCaseRAG] = None,
        rag_top_k: int = 3,
        tool_caller: Optional[ToolCaller] = None,
        max_tokens: int = 4096,
        max_retries: int = 3,
        deterministic: bool = False,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.environment = environment
        self.model_url = model_url
        self.model_name = model_name
        self.model_provider = model_provider
        self.rag_v2 = rag_v2
        self.rag_top_k = rag_top_k
        self.tool_caller = tool_caller
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.deterministic = deterministic
        self.verbose = verbose
        self.logger = logger or init_logger()

        self.client = None
        if model_url and not deterministic:
            self.client = OpenAI(
                base_url=model_url,
                api_key=API_KEY,
                http_client=httpx.Client(verify=False),
            )

    # ── Qwen call ─────────────────────────────────────────────────────────────

    def _call_qwen(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        if self.client is None:
            return None
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
                is_client_err = (
                    hasattr(exc, "status_code")
                    and 400 <= exc.status_code < 500
                    and exc.status_code != 429
                )
                if is_client_err or attempt == self.max_retries:
                    return None
                wait = base_wait * (2 ** (attempt - 1))
                self.logger.info("[Qwen] Retry %d/%d after %.1fs", attempt, self.max_retries, wait)
                time.sleep(wait)
            except Exception as exc:
                self.logger.error("[Qwen] Unhandled: %s", exc)
                return None
        return None

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def solve(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the V3 pipeline for one scenario.
        """
        t0 = time.time()
        scenario_id = scenario.get("scenario_id", "?")
        tag = scenario.get("tag", "single-answer")

        # ── Step 1: Parse ──
        try:
            ps = parse_scenario(scenario)
        except Exception as exc:
            self.logger.error("[%s] parse failed: %s", scenario_id, exc)
            return self._error_result(scenario_id, "parse_failed")

        # ── Step 2: Route ──
        try:
            family = route(ps.tag, ps.num_base_stations)
        except Exception as exc:
            self.logger.error("[%s] route failed: %s", scenario_id, exc)
            return self._error_result(scenario_id, "route_failed")

        # ── Step 3: Feature extraction ──
        try:
            sf = run_all_skills(ps)
        except Exception as exc:
            self.logger.error("[%s] skills failed: %s", scenario_id, exc)
            return self._error_result(scenario_id, "skills_failed")

        # ── Step 4: Candidate generation ──
        try:
            candidates = generate_candidates(ps, sf, family)
        except Exception as exc:
            self.logger.error("[%s] candidate gen failed: %s", scenario_id, exc)
            candidates = None

        # ════════════════════════════════════════════════════════════════════
        # Step 5: DECISION POINT
        # ════════════════════════════════════════════════════════════════════

        if family.name in ("F2", "F3", "F4"):
            # ── Deterministic path ──
            # These families have ≥84% accuracy with top_candidate() alone.
            # Qwen adds noise for multi-answer, so we bypass it entirely.
            answer = candidates.top_candidate() if candidates else ""
            latency = round(time.time() - t0, 2)
            self.logger.info(
                "[%s] %s deterministic → %s  (%.1fs)",
                scenario_id, family.name, answer, latency,
            )
            return self._make_result(
                scenario_id, answer, family.name, sf, 0.0, latency, "deterministic",
            )

        # ── F1 path: tool calls + Qwen ──

        # Step 5a: Server tool calls (if enabled)
        tool_evidence = ToolEvidence()
        if self.tool_caller:
            try:
                tool_evidence = self.tool_caller.call_targeted(
                    scenario_id, sf, sf.throughput.worst_time,
                )
                if tool_evidence.tool_call_count > 0:
                    sf = self.tool_caller.augment_features(sf, tool_evidence)
                    # Re-generate candidates with updated features
                    candidates = generate_candidates(ps, sf, family)
            except Exception as exc:
                self.logger.warning("[%s] tool calls failed: %s", scenario_id, exc)

        candidates_text = candidates.format_for_prompt() if candidates else ""
        top_cand_score = 0.0
        top_cand_answer = candidates.top_candidate() if candidates else None
        if candidates and candidates.single_candidates:
            top_cand_score = candidates.single_candidates[0].score

        # Step 5b: If deterministic mode or no Qwen, use top candidate
        if self.deterministic or self.client is None:
            answer = top_cand_answer or ""
            latency = round(time.time() - t0, 2)
            self.logger.info(
                "[%s] F1 deterministic → %s  (%.1fs)",
                scenario_id, answer, latency,
            )
            return self._make_result(
                scenario_id, answer, "F1", sf, top_cand_score, latency, "deterministic",
            )

        # Step 5c: RAG retrieval
        rag_ctx = ""
        if self.rag_v2 is not None:
            try:
                fvec = _sf_to_feature_vec(sf)
                rag_ctx = self.rag_v2.retrieve_and_format(
                    fvec, family.name, top_k=self.rag_top_k,
                )
            except Exception as exc:
                self.logger.warning("[%s] RAG failed: %s", scenario_id, exc)

        # Step 5d: Build prompt and call Qwen
        tool_evidence_text = tool_evidence.as_prompt_text()
        user_msg = _build_f1_prompt(
            scenario, sf, candidates_text, rag_ctx, tool_evidence_text,
        )

        messages = [
            {"role": "system", "content": QWEN_V3_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        raw_response = self._call_qwen(messages)
        answer = parse_answer(raw_response or "")

        if self.verbose:
            self.logger.info(
                "[%s] Qwen: %s → %r", scenario_id,
                (raw_response or "")[:200], answer,
            )

        # Step 5e: Fallback to deterministic
        if not answer and top_cand_answer:
            answer = top_cand_answer
            self.logger.info("[%s] Fallback → %s", scenario_id, answer)

        latency = round(time.time() - t0, 2)
        self.logger.info("[%s] F1 → %s  (%.1fs)", scenario_id, answer, latency)

        return self._make_result(
            scenario_id, answer, "F1", sf, top_cand_score, latency,
            raw_response or "",
        )

    def _make_result(
        self,
        scenario_id: str,
        answer: str,
        family_name: str,
        sf: ScenarioFeatures,
        top_cand_score: float,
        latency: float,
        qwen_raw: str,
    ) -> Dict[str, Any]:
        return {
            "scenario_id": scenario_id,
            "answer": answer,
            "status": "solved" if answer else "unresolved",
            "family": family_name,
            "top_issue": sf.top_issue()[0],
            "top_issue_score": round(sf.top_issue()[1], 3),
            "top_candidate_score": round(top_cand_score, 3),
            "latency_s": latency,
            "qwen_raw": qwen_raw,
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

    # ── Benchmark ──────────────────────────────────────────────────────────

    def benchmark(
        self,
        save_dir: str,
        max_samples: Optional[int] = None,
        save_freq: int = 10,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)

        scenarios = self.environment.get_scenarios()
        if not scenarios:
            self.logger.error("No scenarios. Check --server_url.")
            return

        if max_samples is not None:
            scenarios = scenarios[:min(max_samples, len(scenarios))]

        self.logger.info("[V3] Running on %d scenarios...", len(scenarios))

        save_result: List[Dict[str, Any]] = []
        total_iou = 0.0
        n_evaluated = 0

        for idx, scenario in enumerate(scenarios):
            result = self.solve(scenario)
            scenario_id = result["scenario_id"]
            agent_answer = result["answer"]

            ground_truth = scenario.get("answer")
            if ground_truth and ground_truth != "To be determined":
                iou = iou_score(agent_answer, ground_truth)
                total_iou += iou
                n_evaluated += 1
                self.logger.info(
                    "\033[95m[%s] pred=%r  gt=%r  iou=%.3f\033[0m",
                    scenario_id, agent_answer, ground_truth, iou,
                )

            save_result.append({
                "scenario_id": scenario_id,
                "answers": agent_answer,
            })

            if ((idx + 1) % save_freq == 0) or ((idx + 1) == len(scenarios)):
                df = pd.DataFrame(save_result)
                df.to_csv(os.path.join(save_dir, "result.csv"), index=False)
                mean_iou = (total_iou / n_evaluated) if n_evaluated else float("nan")
                self.logger.info(
                    "[V3] Saved %d/%d  mean_iou=%.3f",
                    idx + 1, len(scenarios), mean_iou,
                )

        mean_iou = total_iou / max(n_evaluated, 1)
        self.logger.info("[V3] Done. Final mean IOU: %.3f", mean_iou)

        # Also write result_final.csv in submission format
        final_rows = []
        for row in save_result:
            final_rows.append({
                "ID": row["scenario_id"],
                "Track A": row["answers"],
                "Track B": "",
            })
        df_final = pd.DataFrame(final_rows)
        df_final.to_csv(
            os.path.join(save_dir, "result_final.csv"), index=False,
        )
        self.logger.info("[V3] Wrote result_final.csv")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track A V3 Agent")
    parser.add_argument("--server_url",    type=str, default="http://localhost:7860")
    parser.add_argument("--model_url",     type=str, default="http://localhost:8001/v1")
    parser.add_argument("--model_name",    type=str, default="qwen")
    parser.add_argument("--model_provider",type=str, default=None)
    parser.add_argument("--max_samples",   type=int, default=None)
    parser.add_argument("--save_freq",     type=int, default=10)
    parser.add_argument("--max_tokens",    type=int, default=4096)
    parser.add_argument("--save_dir",      type=str, default="./results_v3")
    parser.add_argument("--log_file",      type=str, default="./results_v3/run_v3.log")
    parser.add_argument("--rag_train_json",type=str, default="data/Phase_1/train.json")
    parser.add_argument("--rag_top_k",     type=int, default=3)
    parser.add_argument("--no_rag",        action="store_true")
    parser.add_argument("--use-tools",     action="store_true",
                        help="Enable server tool calls for F1 disambiguation")
    parser.add_argument("--deterministic", action="store_true",
                        help="Skip Qwen entirely — use deterministic candidates only")
    parser.add_argument("--verbose",       action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    logger = init_logger(log_file=args.log_file)

    # RAG V2
    rag_v2 = None
    if not args.no_rag and not args.deterministic:
        if os.path.exists(args.rag_train_json):
            try:
                rag_v2 = StructuredCaseRAG(train_json_path=args.rag_train_json)
                rag_v2.build_index()
                logger.info("[RAG V2] Index ready.")
            except Exception as exc:
                logger.warning("[RAG V2] Failed: %s", exc)
        else:
            logger.warning("[RAG V2] %s not found.", args.rag_train_json)

    # Tool caller (optional)
    tool_caller = None
    if args.use_tools:
        tool_caller = ToolCaller(args.server_url, logger=logger)
        logger.info("[Tools] Enabled — will call server tools for F1.")

    env = Environment(server_url=args.server_url, logger=logger)

    runner = V3Runner(
        environment=env,
        model_url=args.model_url,
        model_name=args.model_name,
        model_provider=args.model_provider,
        rag_v2=rag_v2,
        rag_top_k=args.rag_top_k,
        tool_caller=tool_caller,
        max_tokens=args.max_tokens,
        deterministic=args.deterministic,
        verbose=args.verbose,
        logger=logger,
    )

    runner.benchmark(
        save_dir=args.save_dir,
        max_samples=args.max_samples,
        save_freq=args.save_freq,
    )
