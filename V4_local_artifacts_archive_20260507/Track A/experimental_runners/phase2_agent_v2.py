#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
phase2_agent_v2.py — Phase 2 agent for Track A  (LLM-restored).

ROOT CAUSE OF <0.2 SCORE IN PHASE 2
====================================
phase2_run.py removed the Qwen LLM and relied only on deterministic rules.
But in Phase 1, the LLM was the critical component that recovered the ~29%
of F1 scenarios where the correct answer has gt_rank=-1 (i.e., the rules'
top candidate is simply wrong and the correct C-code is not even in the
top-5 list). Without the LLM to look across ALL options, every one of those
cases fails. This is the primary structural cause of the Phase 2 collapse.

FIX IN THIS FILE
================
Restore the LLM (Qwen3.5-35B-A3B) for the families that need it, while
keeping the deterministic path for families where rules are near-perfect:

  F1 / F1_5BS   (single-answer, 4–5 BS)  → skills + tools + Qwen + RAG
  F2 / F4M      (multi-answer,  4 / 13 BS) → skills + tools + Qwen
  F3            (5-BS, 4-answer template)  → deterministic (100% in Phase 1)
  F4            (13-BS, single decrease_interfreq) → deterministic (100%)

The data-gathering layer (api_data_gatherer.py) and all Phase 2 tooling
(judge_mainlobe, calculate_overlap_ratio, etc.) are unchanged.

USAGE
=====
    python phase2_agent_v2.py \\
        --server_url https://120.46.145.77/no \\
        --token      <BEARER_TOKEN> \\
        --model_url  http://127.0.0.1:8001/v1 \\
        --model_name qwen \\
        --test_json  data/Phase_2/test.json \\
        --out        results/phase2/result_v2.csv

    # Without RAG (faster, slightly lower quality):
    python phase2_agent_v2.py ... --no_rag

    # Without LLM (debug: pure deterministic, same as phase2_run.py):
    python phase2_agent_v2.py ... --deterministic
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

import httpx
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from main import Environment
from skills.scenario_parser import parse_scenario
from skills.scenario_router import FAMILIES, Family
from skills.feature_engine import run_all_skills, ScenarioFeatures
from skills.candidate_generator import (
    generate_candidates,
    CandidateOutput,
    generate_single_candidates,
)
from skills.answer_parser import parse_answer
from logger import init_logger

# Reuse Phase 2 helpers that are already well-tested
from phase2_run import (
    _phase2_family,
    _data_shapes,
    _feature_scores,
    _candidate_summary,
    _format_answer,
    _apply_phase2_tool_checks,
    _attach_agent_trace,
    _apply_fine_tuned_verifier,
    Phase2FineTunedVerifier,
    _save_results,
    _ordered_results,
    _append_diagnostic,
)


# ──────────────────────────────────────────────────────────────────────────────
# System prompts
# (single-answer is identical to main_v3.py's QWEN_V3_SYSTEM_PROMPT;
#  multi-answer adds F2/F4M-specific guidance)
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_SINGLE = """\
You are a 5G network optimization expert. The diagnostic analysis has already
been performed. Your task is to select the single best answer from the options.

Key decision rules:
- Outside_Mainlobe_Flag=True           → lift_tilt or adjust_azimuth (coverage hole)
- RSRP < -110 dBm + low throughput     → increase_power or lift_tilt
- SINR < 0 dB + neighbor advantage > 3 dB → decrease_a3 (serving cell)
- Best neighbor NOT in config + advantage > 3 dB → add_neighbor
- High PRB utilisation (>70%) + inter-freq cells → decrease_interfreq threshold
- DL CCE utilisation > 0.5%            → pdcch_2sym
- MR count < 7 OR inside mainlobe + low SINR → insufficient_data
- Healthy RF (RSRP > -100, SINR > 5) + low PRB → check_server
- High overlap ratio + moderate SINR   → press_down_tilt (secondary cell)

Base your choice on the diagnostic evidence and similar solved cases.
Output ONLY: \\boxed{CN}  (e.g. \\boxed{C5})
"""

_SYSTEM_MULTI = """\
You are a 5G network optimization expert. The diagnostic analysis has already
been performed. Select the best combination of options from the candidate sets.

F2 (4 BS, 2 answers): the answer is always one of two complementary pairs:
  - Overshoot/interference → adjust_azimuth (secondary) + increase_power (secondary)
  - Overshoot/tilt fix     → decrease_power (primary)   + press_down_tilt (primary)
  Pay close attention to which specific cells are involved; same action types
  on different target cells yield different C-codes.

F4M (13 BS, 2 answers): both selected options must be decrease_interfreq,
  typically on the secondary cell and a tertiary cell.

Output ONLY: \\boxed{CN|CM}  in ascending numeric order, e.g. \\boxed{C3|C17}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_all_options(ps) -> str:
    lines: List[str] = []
    for cid, opt in sorted(
        ps.parsed_options.items(),
        key=lambda kv: int(str(kv[0]).lstrip("C")) if str(kv[0]).startswith("C") else 999,
    ):
        lines.append(f"  {cid}: {opt.get('raw', '')}")
    return "\n".join(lines)


def _fmt_single_candidates(cand_out: CandidateOutput) -> str:
    lines = ["## Ranked Candidates\n"]
    for c in cand_out.single_candidates[:6]:
        lines.append(
            f"  {c.c_code}  score={c.score:.3f}  "
            f"action={c.action_type}  cell={c.target_cell}  reason={c.reason}"
        )
    return "\n".join(lines)


def _fmt_multi_candidates(cand_out: CandidateOutput) -> str:
    lines = ["## Ranked Candidate Sets\n"]
    for s in cand_out.multi_candidates[:5]:
        codes = "|".join(s.c_codes)
        lines.append(f"  {codes}  score={s.score:.3f}  reason={s.reason}")
    return "\n".join(lines)


def _fmt_rag_hits(hits: List[Dict]) -> str:
    if not hits:
        return ""
    lines = ["## Similar Solved Cases (from training data)\n"]
    for h in hits[:4]:
        sim = float(h.get("_similarity", h.get("similarity", 0.0)))
        answer = h.get("answer_labels") or h.get("gt_c_code") or h.get("answer", "?")
        action = h.get("gt_action_type", "")
        snippet = h.get("prompt_snippet", "")
        lines.append(f"  sim={sim:.3f} | answer={answer} | {action}")
        if snippet:
            lines.append(f"    {snippet}")
    return "\n".join(lines)


def _build_single_prompt(ps, sf: ScenarioFeatures, cand_out: CandidateOutput,
                          rag_hits: List[Dict]) -> str:
    """F1 / F1_5BS prompt — mirrors main_v3.py's _build_f1_prompt."""
    parts: List[str] = []

    # Diagnostic evidence
    parts.append(f"## Diagnostic Evidence\n{sf.as_evidence_text()}")

    # Tool evidence (mainlobe / overlap results)
    tool_ev = getattr(sf, "tool_evidence", None)
    if tool_ev is not None:
        te_text = (
            tool_ev.as_prompt_text()
            if hasattr(tool_ev, "as_prompt_text")
            else str(tool_ev).strip()
        )
        if te_text:
            parts.append(te_text)

    # Pre-ranked candidates (give the LLM the rules' best guesses)
    parts.append(_fmt_single_candidates(cand_out))

    # RAG hits (similar training examples)
    rag_text = _fmt_rag_hits(rag_hits)
    if rag_text:
        parts.append(rag_text)

    # Full option list so LLM can pick GT even when it has gt_rank=-1
    parts.append(f"## All Options\n{_fmt_all_options(ps)}")

    parts.append(
        "Select the SINGLE best option code.\n"
        "Output ONLY the boxed answer: \\boxed{CN}"
    )
    return "\n\n".join(parts)


def _build_multi_prompt(ps, sf: ScenarioFeatures, cand_out: CandidateOutput,
                         family: Family) -> str:
    """F2 / F4M prompt — multi-answer selection."""
    expected = int(getattr(family, "expected_answer_count", 2))
    parts: List[str] = []

    parts.append(f"## Diagnostic Evidence\n{sf.as_evidence_text()}")

    tool_ev = getattr(sf, "tool_evidence", None)
    if tool_ev is not None:
        te_text = (
            tool_ev.as_prompt_text()
            if hasattr(tool_ev, "as_prompt_text")
            else str(tool_ev).strip()
        )
        if te_text:
            parts.append(te_text)

    parts.append(_fmt_multi_candidates(cand_out))
    parts.append(f"## All Options\n{_fmt_all_options(ps)}")
    parts.append(
        f"Select exactly {expected} option codes that together address the issue.\n"
        "Output ONLY the boxed answer in ascending order: \\boxed{CN|CM}"
    )
    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Risk router
#
# Decides whether a scenario needs LLM arbitration or can be resolved by the
# deterministic rules alone.  The routing table is derived from the Phase 1
# LoRA heldout evaluation:
#
#   check_server      100 % rule accuracy  → low risk  → deterministic
#   pdcch_2sym        100 %                → low risk  → deterministic
#   decrease_a3        94 %                → low-medium risk
#   add_neighbor       68 %                → medium risk → LLM
#   lift_tilt           0 %                → high risk → always LLM
#   insufficient_data   0 %                → high risk → always LLM
#   uncertain          n/a                 → always LLM
#
# For multi-answer families (F2 / F4M), cell-pair selection has independent
# errors regardless of action type, so LLM is always used.
#
# F3 / F4 are template-based (100 % in Phase 1) and are never sent to LLM.
# ──────────────────────────────────────────────────────────────────────────────

# Issues where the rule-based pipeline alone is 0 % accurate on Phase 1 heldout.
# The LLM MUST be consulted whenever the predicted top issue falls here.
_HIGH_RISK_ISSUES = frozenset({
    "coverage_hole",     # confused with lift_tilt / insufficient_data
    "insufficient_data", # 0 % rule IOU
    "uncertain",         # no dominant signal at all
})

# Issues with near-perfect rule accuracy.  Only use deterministic when the
# top-candidate score also clears the confidence threshold below.
_LOW_RISK_ISSUES = frozenset({
    "backhaul_server",   # → check_server: 100 % rule IOU
    "pdcch_issue",       # → pdcch_2sym:   100 % rule IOU
})

# Minimum score for a "low-risk" deterministic decision.
# Below this threshold even nominally low-risk issues are sent to the LLM.
_DETERMINISTIC_SCORE_FLOOR = 0.60

# Gap required between the top issue and the second-best issue.
# If the second issue is within this fraction of the top, the diagnosis is
# considered ambiguous and goes to the LLM.
_AMBIGUITY_GAP_RATIO = 0.85   # second / top < 0.85 → unambiguous


def _risk_router(sf: ScenarioFeatures, family: Family,
                 candidate_output: CandidateOutput) -> Tuple[bool, str]:
    """
    Return (use_llm, reason).

    use_llm=True  → route to LLM + RAG arbitration.
    use_llm=False → keep the deterministic top candidate.
    """
    # Templates are deterministic ceilings — LLM adds noise.
    if family.name in ("F3", "F4"):
        return False, "template_family"

    # Multi-answer families: cell-pair selection is error-prone regardless of
    # action type.  Always use LLM.
    if family.is_multiple:
        return True, "multi_answer_cell_pair_selection"

    top_issue, top_score = sf.top_issue()

    # No dominant signal → LLM must reason from scratch.
    if top_issue == "uncertain":
        return True, "uncertain_diagnosis"

    # Known zero-accuracy issues → LLM mandatory.
    if top_issue in _HIGH_RISK_ISSUES:
        return True, f"high_risk_issue:{top_issue}"

    # Low confidence on any issue → LLM.
    if top_score < _DETERMINISTIC_SCORE_FLOOR:
        return True, f"low_confidence:{top_score:.3f}"

    # Ambiguous: second-best issue is too close to the top.
    all_scores = {
        "late_handover":    sf.score_late_handover,
        "missing_neighbor": sf.score_missing_neighbor,
        "inter_freq_load":  sf.score_inter_freq_load,
        "interference":     sf.score_interference,
        "overshoot_overlap": sf.score_overshoot_overlap,
        "backhaul_server":  sf.score_backhaul_server,
        "pdcch_issue":      sf.score_pdcch_issue,
    }
    if sf.score_insufficient_data is not None:
        all_scores["insufficient_data"] = sf.score_insufficient_data

    sorted_vals = sorted(all_scores.values(), reverse=True)
    if len(sorted_vals) >= 2 and sorted_vals[0] > 0:
        gap_ratio = sorted_vals[1] / sorted_vals[0]
        if gap_ratio >= _AMBIGUITY_GAP_RATIO:
            return True, f"ambiguous_gap_ratio:{gap_ratio:.3f}"

    # High-precision low-risk issues with clear signal: stay deterministic.
    if top_issue in _LOW_RISK_ISSUES:
        return False, f"low_risk_issue:{top_issue}:score={top_score:.3f}"

    # Medium-risk issues (e.g. decrease_a3 at 94%, but still not 100%).
    # Deterministic is likely correct here, but use LLM for any borderline score.
    if top_score >= 0.70:
        return False, f"medium_risk_high_score:{top_issue}:{top_score:.3f}"

    return True, f"medium_risk_low_score:{top_issue}:{top_score:.3f}"


# ──────────────────────────────────────────────────────────────────────────────
# Qwen caller
# ──────────────────────────────────────────────────────────────────────────────

class QwenCaller:
    """Thin wrapper around the OpenAI-compatible Qwen endpoint."""

    def __init__(
        self,
        model_url: str,
        model_name: str,
        max_tokens: int = 512,
        max_retries: int = 3,
        timeout: float = 60.0,
        enable_thinking: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        self.logger = logger or logging.getLogger(__name__)
        self.client = OpenAI(
            base_url=model_url,
            api_key=os.environ.get("AGENT_API_KEY", "dummy"),
            timeout=timeout,
            http_client=httpx.Client(verify=False, timeout=timeout),
        )

    def call(self, system: str, user: str) -> Optional[str]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0,
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": self.enable_thinking,
                        }
                    },
                )
                msg = resp.choices[0].message
                return (
                    msg.content
                    or getattr(msg, "reasoning_content", None)
                    or getattr(msg, "reasoning", None)
                    or ""
                )
            except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
                if attempt == self.max_retries:
                    self.logger.warning("[Qwen] final failure: %s", exc)
                    return None
                wait = float(attempt)
                self.logger.info("[Qwen] retry %d/%d after %.0fs: %s",
                                 attempt, self.max_retries, wait, exc)
                time.sleep(wait)
            except Exception as exc:
                self.logger.error("[Qwen] unhandled: %s", exc)
                return None
        return None


# ──────────────────────────────────────────────────────────────────────────────
# RAG helper
# ──────────────────────────────────────────────────────────────────────────────

def _build_rag(train_json: Optional[str], logger: logging.Logger):
    """Build the Phase 1 training RAG index.  Returns None if unavailable."""
    if not train_json:
        return None
    if not os.path.exists(train_json):
        logger.warning("[RAG] training JSON not found: %s", train_json)
        return None
    try:
        from rag_retriever_v2 import StructuredCaseRAG
        rag = StructuredCaseRAG(train_json_path=train_json)
        rag.build_index()
        logger.info("[RAG] Index ready (%s).", train_json)
        return rag
    except Exception as exc:
        logger.warning("[RAG] Failed to build index: %s", exc)
        return None


def _retrieve_rag(rag, sf: ScenarioFeatures, family_name: str, top_k: int = 4) -> List[Dict]:
    """Query the RAG and normalise hits to plain dicts."""
    if rag is None:
        return []
    try:
        # Map Phase 2 override families back to the closest Phase 1 family
        query_family = family_name
        if family_name in ("F1_5BS",):
            query_family = "F1"
        elif family_name in ("F4M",):
            query_family = "F4"

        hits = rag.retrieve(sf, k=top_k, family_name=query_family)
        result: List[Dict] = []
        for h in hits[:top_k]:
            result.append(h if isinstance(h, dict) else vars(h))
        return result
    except Exception as exc:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Answer validation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _validate_answer(parsed: str, ps, family: Family) -> bool:
    """Return True iff `parsed` is a valid answer for this family."""
    if not parsed:
        return False
    codes = [c.strip() for c in parsed.split("|") if c.strip()]
    valid = set(ps.parsed_options.keys())
    if not all(c in valid for c in codes):
        return False
    expected = int(getattr(family, "expected_answer_count", 1 if not family.is_multiple else 2))
    return len(codes) == expected


# ──────────────────────────────────────────────────────────────────────────────
# Per-scenario solver
# ──────────────────────────────────────────────────────────────────────────────

def _solve_scenario(
    scenario: Dict[str, Any],
    env: Environment,
    logger: logging.Logger,
    qwen: Optional[QwenCaller],
    rag,
    fetch_signaling: bool = False,
    deterministic: bool = False,
    llm_verifier: Optional[Phase2FineTunedVerifier] = None,
    llm_policy: str = "audit",
) -> Tuple[str, Dict[str, Any]]:
    """
    Full Phase 2 agent pipeline for one scenario.

    Decision routing:
      F1 / F1_5BS  → deterministic skills + tools → Qwen LLM (with RAG fallback)
      F2 / F4M     → deterministic skills + tools → Qwen LLM (multi-answer prompt)
      F3 / F4      → deterministic skills only    (these families are 100% template-based)
    """
    sid  = scenario.get("scenario_id", "?")
    tag  = scenario.get("tag", "single-answer")
    ctx  = scenario.get("context", {})
    wi   = ctx.get("wireless_network_information", {})
    num_bs = int(str(wi.get("num_base_stations", "4")).strip())

    routed_family, family, family_override = _phase2_family(tag, num_bs)

    diagnostics: Dict[str, Any] = {
        "scenario_id":    sid,
        "tag":            tag,
        "num_base_stations": num_bs,
        "routed_family":  routed_family.name,
        "family_override": family_override,
        "family":         family.name,
        "is_multiple":    bool(family.is_multiple),
        "answer":         "",
        "api_calls":      [],
        "candidates":     [],
        "error":          None,
        "llm_used":       False,
        "llm_answer":     None,
        "llm_reject":     None,
    }
    if family_override:
        logger.info("[%s] family override: %s (tag=%s bs=%d)", sid, family.name, tag, num_bs)

    # ── Step 1: fetch data from cloud API ────────────────────────────────────
    from skills.api_data_gatherer import APIDataGatherer
    gatherer = APIDataGatherer(
        env, scenario, fetch_signaling=fetch_signaling, logger=logger,
    )
    raw = gatherer.gather()
    diagnostics["api_calls"]      = getattr(gatherer, "api_calls", [])
    diagnostics["data_available"] = getattr(gatherer, "data_available", {})
    diagnostics["worst_time"]     = gatherer.worst_time
    diagnostics["serving_pci"]    = gatherer.serving_pci

    # ── Step 2: parse ────────────────────────────────────────────────────────
    ps = parse_scenario(raw)
    diagnostics["data_shapes"] = _data_shapes(ps)
    if not ps.parsed_options:
        diagnostics["error"] = "no_parsed_options"
        return "", diagnostics

    # ── Step 3: feature extraction ───────────────────────────────────────────
    sf = run_all_skills(ps)
    diagnostics["feature_scores"] = _feature_scores(sf)
    top_issue, top_score = sf.top_issue()
    diagnostics["top_issue"] = {"name": top_issue, "score": float(top_score)}

    # ── Step 4: conditional tool calls ───────────────────────────────────────
    _apply_phase2_tool_checks(gatherer, ps, sf, family, diagnostics)
    diagnostics["api_calls"]      = getattr(gatherer, "api_calls", [])
    diagnostics["feature_scores"] = _feature_scores(sf)
    top_issue, top_score = sf.top_issue()
    diagnostics["top_issue"] = {"name": top_issue, "score": float(top_score)}

    # ── Step 5: candidate generation ────────────────────────────────────────
    candidate_output = generate_candidates(ps, sf, family)
    diagnostics["candidates"] = _candidate_summary(candidate_output)
    if not diagnostics["candidates"]:
        diagnostics["error"] = "no_candidates"
        return "", diagnostics

    deterministic_answer = _format_answer(candidate_output, tag)

    # ════════════════════════════════════════════════════════════════════════
    # Step 6: DECISION POINT — risk_router
    #
    # Route each scenario to deterministic OR LLM+RAG based on:
    #   1. Family template type (F3/F4 → always deterministic)
    #   2. Predicted top issue risk level (0%-accuracy issues → always LLM)
    #   3. Candidate score confidence (< 0.60 → LLM)
    #   4. Diagnosis ambiguity (two close issues → LLM)
    #
    # This avoids LLM hallucination on already-correct deterministic answers
    # while still rescuing the ~29% of F1 cases where gt_rank=-1.
    # ════════════════════════════════════════════════════════════════════════

    if qwen is not None and not deterministic:
        use_llm, risk_reason = _risk_router(sf, family, candidate_output)
    else:
        use_llm, risk_reason = False, "deterministic_mode" if deterministic else "no_llm"

    diagnostics["llm_used"]    = use_llm
    diagnostics["risk_reason"] = risk_reason

    answer = deterministic_answer   # default: rules win if LLM fails/skipped

    if use_llm:
        t_llm = time.time()
        if family.is_multiple:
            # F2 / F4M — multi-answer
            user_msg   = _build_multi_prompt(ps, sf, candidate_output, family)
            system_msg = _SYSTEM_MULTI
        else:
            # F1 / F1_5BS — single-answer with RAG context
            rag_hits  = _retrieve_rag(rag, sf, family.name)
            user_msg  = _build_single_prompt(ps, sf, candidate_output, rag_hits)
            system_msg = _SYSTEM_SINGLE

        raw_resp = qwen.call(system_msg, user_msg)
        diagnostics["llm_latency_s"] = round(time.time() - t_llm, 2)

        if raw_resp:
            parsed = parse_answer(raw_resp)
            diagnostics["llm_answer"] = parsed
            diagnostics["llm_raw"]    = raw_resp[:400]

            if _validate_answer(parsed, ps, family):
                answer = parsed
                logger.info("[%s] LLM accepted: %r (was: %r)", sid, parsed, deterministic_answer)
            else:
                diagnostics["llm_reject"] = (
                    f"invalid: parsed={parsed!r} "
                    f"valid_codes={set(ps.parsed_options.keys())} "
                    f"expected_count={getattr(family, 'expected_answer_count', '?')}"
                )
                logger.info(
                    "[%s] LLM rejected (%s), using deterministic: %r",
                    sid, diagnostics["llm_reject"], deterministic_answer,
                )
        else:
            diagnostics["llm_reject"] = "no_response"
            logger.info("[%s] LLM returned None, using deterministic: %r",
                        sid, deterministic_answer)

    # ── Step 7: optional fine-tuned verifier ─────────────────────────────────
    answer = _apply_fine_tuned_verifier(
        llm_verifier, llm_policy, ps, sf, family, diagnostics, answer,
    )

    diagnostics["answer"] = answer
    _attach_agent_trace(diagnostics, answer)

    top_cand = diagnostics["candidates"][0] if diagnostics["candidates"] else {}
    logger.info(
        "[%s] family=%s llm=%s risk=%s answer=%r top_cand=%s",
        sid, family.name, use_llm, risk_reason, answer,
        f"{top_cand.get('answer')}(score={top_cand.get('score'):.3f})"
        if top_cand else "none",
    )
    return answer, diagnostics


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 2 Track A agent — LLM-restored V2"
    )
    ap.add_argument("--server_url",   default="https://120.46.145.77/no")
    ap.add_argument("--token",        default=None,
                    help="Bearer token (or set AGENT_API_KEY env var)")

    # LLM
    ap.add_argument("--model_url",    default=None,
                    help="OpenAI-compatible Qwen endpoint, e.g. http://127.0.0.1:8001/v1")
    ap.add_argument("--model_name",   default="qwen")
    ap.add_argument("--model_provider", default=None)
    ap.add_argument("--max_tokens",   type=int, default=512)
    ap.add_argument("--llm_timeout",  type=float, default=60.0)
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Enable Qwen3 extended thinking (CoT); higher quality, slower")
    ap.add_argument("--deterministic", action="store_true",
                    help="Skip LLM entirely — pure rules (debug/baseline mode)")

    # RAG
    ap.add_argument("--rag_train_json", default="data/Phase_1/train.json",
                    help="Path to Phase 1 train.json for RAG index")
    ap.add_argument("--no_rag", action="store_true")

    # Optional fine-tuned verifier (LoRA)
    ap.add_argument("--lora_url",     default=None,
                    help="LoRA verifier endpoint (leave unset to disable)")
    ap.add_argument("--lora_name",    default="v4")
    ap.add_argument("--lora_policy",
                    choices=["off", "audit", "gate", "always"], default="audit")

    # I/O
    ap.add_argument("--test_json",    type=Path,
                    default=Path(__file__).parent / "data/Phase_2/test.json")
    ap.add_argument("--out",          type=Path,
                    default=Path("results/phase2/result_v2.csv"))
    ap.add_argument("--diagnostics_jsonl", type=Path, default=None)
    ap.add_argument("--save_freq",    type=int, default=10)
    ap.add_argument("--timeout",      type=float, default=15.0,
                    help="Per-tool HTTP timeout (seconds)")
    ap.add_argument("--fetch_signaling", action="store_true")
    ap.add_argument("--max_samples",  type=int, default=None)
    ap.add_argument("--sample_per_group", type=int, default=None)
    ap.add_argument("--scenario_ids", type=str, default=None)
    ap.add_argument("--scenario_ids_file", type=Path, default=None)
    ap.add_argument("--resume",       action="store_true")
    ap.add_argument("--log_file",     type=str, default=None)
    ap.add_argument("--verbose",      action="store_true")
    args = ap.parse_args()

    logger = init_logger(log_file=args.log_file)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.token:
        os.environ["AGENT_API_KEY"] = args.token

    # ── Load scenarios ────────────────────────────────────────────────────────
    if not args.test_json.exists():
        logger.error("test.json not found: %s", args.test_json)
        sys.exit(1)
    with open(args.test_json, encoding="utf-8") as f:
        scenarios: List[Dict[str, Any]] = json.load(f)

    # Filtering (same logic as phase2_run.py)
    selected_ids: set = set()
    if args.scenario_ids:
        selected_ids.update(s.strip() for s in args.scenario_ids.split(",") if s.strip())
    if args.scenario_ids_file and args.scenario_ids_file.exists():
        selected_ids.update(
            line.strip()
            for line in args.scenario_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if selected_ids:
        scenarios = [s for s in scenarios if str(s.get("scenario_id", "")).strip() in selected_ids]
        logger.info("Scenario filter: %d selected", len(scenarios))

    if args.sample_per_group is not None:
        grouped: Dict[Tuple[str, int], int] = {}
        sampled: List[Dict[str, Any]] = []
        for sc in scenarios:
            t = str(sc.get("tag", "single-answer"))
            try:
                bs = int(str(sc.get("context", {}).get("wireless_network_information", {})
                              .get("num_base_stations", "4")).strip())
            except ValueError:
                bs = 4
            key = (t, bs)
            if grouped.get(key, 0) < args.sample_per_group:
                sampled.append(sc)
                grouped[key] = grouped.get(key, 0) + 1
        scenarios = sampled

    if args.max_samples is not None:
        scenarios = scenarios[: args.max_samples]

    logger.info("Loaded %d scenarios from %s", len(scenarios), args.test_json)

    # ── Build components ──────────────────────────────────────────────────────
    env = Environment(
        server_url=args.server_url,
        verbose=args.verbose,
        logger=logger,
        api_key=args.token,
        timeout=args.timeout,
    )

    # Qwen LLM caller
    qwen: Optional[QwenCaller] = None
    if args.model_url and not args.deterministic:
        model_name = (
            f"{args.model_provider}/{args.model_name}"
            if args.model_provider else args.model_name
        )
        qwen = QwenCaller(
            model_url=args.model_url,
            model_name=model_name,
            max_tokens=args.max_tokens,
            timeout=args.llm_timeout,
            enable_thinking=args.enable_thinking,
            logger=logger,
        )
        logger.info("[Qwen] Enabled: model=%s url=%s thinking=%s",
                    model_name, args.model_url, args.enable_thinking)
    elif args.deterministic:
        logger.info("[Qwen] Disabled (--deterministic)")
    else:
        logger.warning("[Qwen] No --model_url supplied; running deterministic fallback")

    # RAG index (Phase 1 training data)
    rag = None
    if not args.no_rag and qwen is not None:
        rag = _build_rag(args.rag_train_json, logger)
    else:
        logger.info("[RAG] Disabled (no_rag=%s qwen=%s)", args.no_rag, qwen is not None)

    # Optional LoRA fine-tuned verifier
    llm_verifier: Optional[Phase2FineTunedVerifier] = None
    if args.lora_url and args.lora_policy != "off":
        try:
            llm_verifier = Phase2FineTunedVerifier(
                model_url=args.lora_url,
                model_name=args.lora_name,
                logger=logger,
            )
            logger.info("[LoRA] Enabled: model=%s policy=%s", args.lora_name, args.lora_policy)
        except Exception as exc:
            logger.warning("[LoRA] Init failed: %s", exc)

    # ── Output paths ─────────────────────────────────────────────────────────
    diag_path = args.diagnostics_jsonl or args.out.with_suffix(".diagnostics.jsonl")
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and diag_path.exists():
        diag_path.unlink()

    # ── Resume support ────────────────────────────────────────────────────────
    results_by_id: Dict[str, Dict[str, str]] = {}
    if args.resume and args.out.exists():
        with open(args.out, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = str(row.get("ID", "")).strip()
                if sid:
                    results_by_id[sid] = {
                        "ID": sid,
                        "Track A": str(row.get("Track A", "") or ""),
                        "Track B": str(row.get("Track B", "") or ""),
                    }
        logger.info("[Resume] loaded %d existing rows", len(results_by_id))

    # ── Main loop ─────────────────────────────────────────────────────────────
    start_total = time.time()

    for idx, scenario in enumerate(scenarios):
        sid = scenario.get("scenario_id", f"unknown_{idx}")
        if sid in results_by_id and results_by_id[sid].get("Track A"):
            logger.info("[%s] resume skip (answer=%r)", sid, results_by_id[sid]["Track A"])
            continue

        t0 = time.time()
        try:
            answer, diag = _solve_scenario(
                scenario,
                env,
                logger,
                qwen=qwen,
                rag=rag,
                fetch_signaling=args.fetch_signaling,
                deterministic=args.deterministic,
                llm_verifier=llm_verifier,
                llm_policy=args.lora_policy,
            )
        except Exception:
            tb = traceback.format_exc()
            logger.error("[%s] EXCEPTION:\n%s", sid, tb)
            answer = ""
            diag = {"scenario_id": sid, "tag": scenario.get("tag", ""), "answer": "", "error": tb}

        elapsed = time.time() - t0
        diag["elapsed_seconds"] = elapsed
        logger.info("[%s] done in %.1fs — answer: %r", sid, elapsed, answer)

        results_by_id[sid] = {"ID": sid, "Track A": answer, "Track B": ""}
        _append_diagnostic(diag, diag_path)

        if (idx + 1) % args.save_freq == 0 or (idx + 1) == len(scenarios):
            _save_results(_ordered_results(scenarios, results_by_id), args.out, logger)

    # ── Final save & summary ──────────────────────────────────────────────────
    total = time.time() - start_total
    results = _ordered_results(scenarios, results_by_id)
    _save_results(results, args.out, logger)
    answered = sum(1 for r in results if r["Track A"])
    logger.info(
        "Done. %d/%d answered in %.0fs (avg %.1fs/scenario)",
        answered, len(results), total, total / max(len(results), 1),
    )


if __name__ == "__main__":
    main()
