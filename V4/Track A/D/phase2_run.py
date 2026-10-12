#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
phase2_run.py — Phase 2 agent for Track A.

For each Phase 2 test scenario:
  1. Parse option labels from JSON (no API calls needed).
  2. Route to family (F1/F2/F3/F4) by tag + num_base_stations.
  3. Fetch raw data from Phase 2 cloud server (6 API calls).
  4. Run the V3 deterministic pipeline (feature_engine → candidate_generator).
  5. Optionally call judge_mainlobe for lift_tilt/insufficient_data disambiguation.
  6. Optionally call a Qwen-compatible model as either:
     - an audited/gated fine-tuned verifier, or
     - a RAG-backed arbitration agent for high-risk Phase 2 cases.
  7. Write result.csv for submission.

Usage:
    python phase2_run.py \\
        --server_url https://120.46.145.77/no \\
        --token      <your-bearer-token> \\
        --test_json  data/Phase_2/test.json \\
        --out        result_phase2.csv \\
        --save_freq  10

The output CSV has platform-ready columns: ID, Track A, Track B.
This runner is Track A-only. Track B is intentionally left blank because this
team is not participating in Track B.
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
from urllib.parse import urlparse

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Ensure skills/ is importable even when run from a different cwd
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from main import Environment
from skills.scenario_parser import parse_scenario
from skills.scenario_router import route, FAMILIES, Family
from skills.feature_engine import run_all_skills
from skills.candidate_generator import generate_candidates
from skills.answer_parser import parse_answer
from logger import init_logger


# ---------------------------------------------------------------------------
# Fine-tuned verifier (Qwen3.5-A3B LoRA via OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

def _is_local_model_url(model_url: str) -> bool:
    host = (urlparse(model_url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


class Phase2FineTunedVerifier:
    """
    OpenAI-compatible model component for Phase 2.

    In audit/gate/always modes it behaves as the calibrated LoRA verifier. In
    arbitrate mode the same thin client is used as a general Qwen-compatible
    final decision maker over structured evidence and RAG context.
    """

    def __init__(
        self,
        model_url: str,
        model_name: str = "v4",
        model_provider: Optional[str] = None,
        max_tokens: int = 64,
        timeout: float = 30.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        from openai import OpenAI
        import httpx

        self.model_url = model_url
        self.model_name = model_name
        self.model_provider = model_provider
        self.max_tokens = max_tokens
        self.logger = logger or logging.getLogger(__name__)
        self.bypass_env_proxy = _is_local_model_url(model_url)
        self.client = OpenAI(
            base_url=model_url,
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
            timeout=timeout,
            http_client=httpx.Client(
                verify=False,
                timeout=timeout,
                trust_env=not self.bypass_env_proxy,
            ),
        )

    @property
    def model_id(self) -> str:
        if self.model_provider:
            return f"{self.model_provider}/{self.model_name}"
        return self.model_name

    def call(
        self,
        messages: List[Dict[str, str]],
        enable_thinking: bool = False,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Optional[str]]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                temperature=0,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": enable_thinking,
                    },
                },
            )
            msg = resp.choices[0].message
            raw = (
                msg.content
                or getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or ""
            )
            return raw, None
        except Exception as exc:
            self.logger.warning("[LLM] call failed: %s", exc)
            return "", str(exc)


def _valid_option_codes(ps) -> List[str]:
    return sorted(
        [str(c) for c in getattr(ps, "parsed_options", {}).keys()],
        key=lambda x: int(x[1:]) if x.startswith("C") and x[1:].isdigit() else 999,
    )


def _format_options_for_verifier(ps) -> str:
    lines: List[str] = []
    for cid in _valid_option_codes(ps):
        opt = ps.parsed_options.get(cid, {})
        cells = opt.get("cells") or []
        relation = f" cells={cells}" if cells else ""
        lines.append(
            f"{cid}: action={opt.get('action_type', '?')} "
            f"cell={opt.get('target_cell') or '?'}{relation} | "
            f"{opt.get('raw', '')}"
        )
    return "\n".join(lines)


def _build_verifier_messages(
    ps,
    sf,
    family: Family,
    diagnostics: Dict[str, Any],
    deterministic_answer: str,
) -> List[Dict[str, str]]:
    candidate_lines = []
    for idx, row in enumerate(diagnostics.get("candidates", [])[:8], 1):
        candidate_lines.append(
            f"{idx}. {row.get('answer')} score={row.get('score')} "
            f"action={row.get('action_type', '')} target={row.get('target_cell', '')} "
            f"reason={row.get('reason', '')}"
        )

    tool_lines = []
    for row in diagnostics.get("tool_adjustments", []):
        tool_lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))

    public_trace = {
        "family": family.name,
        "is_multiple": family.is_multiple,
        "top_issue": diagnostics.get("top_issue"),
        "feature_scores": diagnostics.get("feature_scores"),
        "data_available": diagnostics.get("data_available"),
        "tool_evidence": tool_lines or ["no conditional tool evidence"],
        "deterministic_answer": deterministic_answer,
    }

    system = (
        "You are a calibrated 5G troubleshooting verifier. "
        "Use the diagnostic evidence, tool observations, and candidate list to "
        "select the best option code. Do not provide reasoning text. "
        "Output only a boxed answer like \\boxed{C7} or \\boxed{C3|C8}."
    )
    user = (
        "## Public Agent Trace\n"
        f"{json.dumps(public_trace, ensure_ascii=False, indent=2)}\n\n"
        "## Diagnostic Evidence\n"
        f"{sf.as_evidence_text()}\n\n"
        "## Ranked Candidates\n"
        f"{chr(10).join(candidate_lines)}\n\n"
        "## All Options\n"
        f"{_format_options_for_verifier(ps)}\n\n"
        "Select the best final Track A answer. Output only the boxed answer."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _same_cardinality(answer: str, family: Family) -> bool:
    if not answer:
        return False
    n = len([c for c in answer.split("|") if c])
    if family.is_multiple:
        if family.name == "F2":
            return n in (2, 4)
        return n == int(family.expected_answer_count)
    return n == 1


def _apply_fine_tuned_verifier(
    verifier: Optional[Phase2FineTunedVerifier],
    policy: str,
    ps,
    sf,
    family: Family,
    diagnostics: Dict[str, Any],
    deterministic_answer: str,
) -> str:
    diagnostics["fine_tuned_verifier"] = {
        "enabled": bool(verifier and policy != "off"),
        "policy": policy,
        "model_answer": "",
        "used_for_final": False,
        "reason": "",
        "raw_preview": "",
    }
    if verifier is None or policy == "off":
        diagnostics["fine_tuned_verifier"]["reason"] = "disabled"
        return deterministic_answer

    # This LoRA was trained and validated as a single-answer F1 selector.  For
    # multi-answer families we still keep the skills/tools component as authority
    # until a multi-answer checkpoint passes heldout gates.
    if family.is_multiple:
        diagnostics["fine_tuned_verifier"]["reason"] = "skipped_multi_answer_family"
        return deterministic_answer

    messages = _build_verifier_messages(ps, sf, family, diagnostics, deterministic_answer)
    raw, error = verifier.call(messages)
    parsed = parse_answer(raw)
    valid_codes = set(_valid_option_codes(ps))
    parsed_codes = set(c for c in parsed.split("|") if c)
    info = diagnostics["fine_tuned_verifier"]
    info.update({
        "model_name": verifier.model_id,
        "model_answer": parsed,
        "raw_preview": raw[:240],
        "error": error,
    })

    if error:
        info["reason"] = "model_call_failed"
        return deterministic_answer
    if not parsed or not parsed_codes <= valid_codes or not _same_cardinality(parsed, family):
        info["reason"] = "invalid_or_wrong_cardinality"
        return deterministic_answer

    if policy == "audit":
        info["reason"] = "audited_only"
        return deterministic_answer

    if policy == "always":
        info["used_for_final"] = parsed != deterministic_answer
        info["reason"] = "always_override_policy"
        return parsed

    # Gate mode: use agreement as confidence evidence, but only allow overrides
    # outside the two known failed heldout gates for v4_fast_balanced.
    if parsed == deterministic_answer:
        info["reason"] = "agreement_with_deterministic"
        return deterministic_answer

    top_issue = (diagnostics.get("top_issue") or {}).get("name", "")
    risky_issues = {"coverage_hole", "insufficient_data", "uncertain"}
    if top_issue in risky_issues:
        info["reason"] = f"blocked_override_known_risky_issue:{top_issue}"
        return deterministic_answer

    top_score = 0.0
    if diagnostics.get("candidates"):
        top_score = float(diagnostics["candidates"][0].get("score") or 0.0)
    candidate_answers = {row.get("answer") for row in diagnostics.get("candidates", [])[:3]}
    if parsed in candidate_answers and top_score < 0.55:
        info["used_for_final"] = True
        info["reason"] = "gated_override_low_confidence_top3"
        return parsed

    info["reason"] = "blocked_by_gate"
    return deterministic_answer


# ---------------------------------------------------------------------------
# RAG-backed LLM arbitration
# ---------------------------------------------------------------------------

_ARBITRATION_HIGH_RISK_ISSUES = {
    "coverage_hole",
    "insufficient_data",
    "uncertain",
}

_ARBITRATION_PAIR_ACTIONS = {"lift_tilt", "insufficient_data"}


def _rag_family_name(family: Family) -> str:
    """Map Phase 2 override families back to their closest Phase 1 RAG bucket."""
    if family.name == "F1_5BS":
        return "F1"
    if family.name == "F4M":
        return "F4"
    return family.name


def _build_rag_index(train_json: Optional[Path], logger: logging.Logger):
    """Build/load the structured Phase 1 RAG index for arbitration prompts."""
    if train_json is None:
        return None
    path = Path(train_json)
    if not path.exists():
        logger.warning("[RAG] train json not found: %s", path)
        return None
    try:
        from rag_retriever_v2 import StructuredCaseRAG

        rag = StructuredCaseRAG(str(path))
        rag.build_index()
        logger.info("[RAG] Index ready: %s", path)
        return rag
    except Exception as exc:
        logger.warning("[RAG] Disabled after build failure: %s", exc)
        return None


def _retrieve_rag_hits(rag, sf, family: Family, top_k: int, logger: logging.Logger) -> List[Any]:
    if rag is None or top_k <= 0:
        return []
    query_family = _rag_family_name(family)
    branch = None
    try:
        scores = _feature_scores(sf)
        risk = compute_risk_level(sf, family, scores)
        branch = risk.get("branch")
    except Exception:
        branch = None
    try:
        return rag.retrieve(
            sf,
            k=top_k,
            family_name=query_family,
            stratify_by_action=(query_family in ("F1", "F2")),
            include_confusion_pairs=(query_family == "F1"),
            branch=branch,
        )
    except TypeError:
        # Older retriever builds expose only (sf, k, family_name).
        try:
            return rag.retrieve(sf, k=top_k, family_name=query_family)
        except Exception as exc:
            logger.warning("[RAG] retrieve failed: %s", exc)
            return []
    except Exception as exc:
        logger.warning("[RAG] retrieve failed: %s", exc)
        return []


def _format_rag_hits_for_prompt(hits: List[Any]) -> str:
    if not hits:
        return "no similar solved cases available"

    lines: List[str] = []
    for idx, hit in enumerate(hits[:8], 1):
        data = hit if isinstance(hit, dict) else vars(hit)
        features = data.get("features") or {}
        compact_features = {
            k: round(float(v), 3)
            for k, v in features.items()
            if isinstance(v, (int, float))
        }
        lines.append(
            f"{idx}. scenario={data.get('scenario_id', '?')} "
            f"sim={float(data.get('similarity', data.get('_similarity', 0.0)) or 0.0):.3f} "
            f"branch={data.get('branch', data.get('root_cause', '?'))} "
            f"train_action={data.get('gt_action_type', data.get('action_type', '?'))} "
            f"target_role={data.get('target_role', '') or 'null'} "
            f"confidence={data.get('confidence', '') or 'unknown'} "
            f"features={json.dumps(compact_features, ensure_ascii=False, sort_keys=True)} "
            f"confusion_pair={bool(data.get('is_confusion_pair', False))}"
        )
        key_evidence = data.get("key_evidence") or []
        tool_sequence = data.get("tool_sequence") or []
        if key_evidence:
            lines.append(
                f"   evidence={'; '.join(str(x) for x in key_evidence[:2])}"
            )
        if tool_sequence:
            lines.append(
                f"   tools={', '.join(str(x) for x in tool_sequence[:3])}"
            )
    return "\n".join(lines)


def _rag_hit_public_dict(hit: Any) -> Dict[str, Any]:
    data = hit if isinstance(hit, dict) else vars(hit)
    features = data.get("features") or {}
    compact_features = {
        str(k): round(float(v), 4)
        for k, v in features.items()
        if isinstance(v, (int, float))
    }
    return {
        "scenario_id": data.get("scenario_id", ""),
        "similarity": float(data.get("similarity", data.get("_similarity", 0.0)) or 0.0),
        "gt_action_type": data.get("gt_action_type", data.get("action_type", "")),
        "prompt_snippet": (data.get("prompt_snippet", "") or "")[:200],
        "branch": data.get("branch", ""),
        "root_cause": data.get("root_cause", ""),
        "target_role": data.get("target_role", None),
        "confidence": data.get("confidence", ""),
        "tool_sequence": list(data.get("tool_sequence") or []),
        "key_evidence": list(data.get("key_evidence") or []),
        "is_confusion_pair": bool(data.get("is_confusion_pair", False)),
        "features": compact_features,
    }


def _candidate_score_gap(diagnostics: Dict[str, Any]) -> float:
    candidates = diagnostics.get("candidates") or []
    if len(candidates) < 2:
        return 1.0
    try:
        return float(candidates[0].get("score") or 0.0) - float(candidates[1].get("score") or 0.0)
    except Exception:
        return 1.0


def _top_candidate_score(diagnostics: Dict[str, Any]) -> float:
    candidates = diagnostics.get("candidates") or []
    if not candidates:
        return 0.0
    try:
        return float(candidates[0].get("score") or 0.0)
    except Exception:
        return 0.0


def _candidate_actions(diagnostics: Dict[str, Any], limit: int = 5) -> List[str]:
    return [
        str(row.get("action_type", ""))
        for row in (diagnostics.get("candidates") or [])[:limit]
    ]


def _arbitration_reasons(
    ps,
    sf,
    family: Family,
    diagnostics: Dict[str, Any],
    force: bool = False,
    margin_threshold: float = 0.25,
) -> List[str]:
    """Return reasons to call the LLM arbiter. Empty means keep deterministic."""
    # These two Phase 1 template families are deterministic ceilings; Qwen tends
    # to add noise, so even --llm_arbitrate_all leaves them on the rule path.
    if family.name in ("F3", "F4"):
        return []

    if force:
        return ["forced_by_cli"]

    reasons: List[str] = []
    top_issue = (diagnostics.get("top_issue") or {}).get("name", "")
    top_score = float((diagnostics.get("top_issue") or {}).get("score") or 0.0)
    actions = _candidate_actions(diagnostics)
    action_set = set(actions)

    if family.name == "F1_5BS":
        reasons.append("new_phase2_family:F1_5BS")
    if family.name == "F4M":
        reasons.append("new_phase2_family:F4M")
    if family.is_multiple and family.name != "F3":
        reasons.append("multi_answer_cell_pair_selection")
    if top_issue in _ARBITRATION_HIGH_RISK_ISSUES:
        reasons.append(f"high_risk_issue:{top_issue}")
    if top_score < 0.55:
        reasons.append(f"low_issue_score:{top_score:.3f}")
    top_candidate_score = _top_candidate_score(diagnostics)
    if top_candidate_score < 0.40:
        reasons.append(f"low_candidate_score:{top_candidate_score:.3f}")

    gap = _candidate_score_gap(diagnostics)
    if gap <= margin_threshold:
        reasons.append(f"close_candidate_gap:{gap:.3f}")

    if _ARBITRATION_PAIR_ACTIONS <= action_set:
        reasons.append("lift_tilt_vs_insufficient_data_boundary")

    # Phase 1 F1 labels are intentionally narrow. If the deterministic top
    # option escaped that action family, ask the LLM to inspect all options.
    if not family.is_multiple and ps.num_base_stations <= 5 and actions:
        preferred = {
            "add_neighbor",
            "check_server",
            "decrease_a3",
            "insufficient_data",
            "lift_tilt",
            "pdcch_2sym",
        }
        if actions[0] and actions[0] not in preferred:
            reasons.append(f"non_f1_primary_top_action:{actions[0]}")

    return reasons


def _format_tool_evidence_for_prompt(sf, diagnostics: Dict[str, Any]) -> str:
    tool_ev = getattr(sf, "tool_evidence", None)
    rendered = ""
    if tool_ev is not None and hasattr(tool_ev, "as_prompt_text"):
        rendered = tool_ev.as_prompt_text()
    elif tool_ev is not None and hasattr(tool_ev, "__dict__"):
        rendered = json.dumps(
            {
                k: v for k, v in vars(tool_ev).items()
                if not str(k).startswith("_")
            },
            ensure_ascii=False,
            default=str,
            sort_keys=True,
        )
    elif tool_ev is not None:
        rendered = str(tool_ev)

    adjustments = diagnostics.get("tool_adjustments") or []
    if adjustments:
        adjustment_text = "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True)
            for row in adjustments
        )
        rendered = (rendered + "\n" if rendered else "") + "Tool adjustments:\n" + adjustment_text
    return rendered or "no conditional tool evidence"


def _cardinality_instruction(family: Family) -> str:
    if family.name == "F2":
        return (
            "2 option codes, or 4 option codes only when the F2 comprehensive "
            "hint/raw evidence supports a broader 4-action fix"
        )
    if family.is_multiple:
        return f"exactly {int(family.expected_answer_count)} option codes"
    return "exactly 1 option code"


def _cardinality_retry_text(family: Family) -> str:
    if family.name == "F2":
        return "2 or 4 code(s)"
    if family.is_multiple:
        return f"exactly {int(family.expected_answer_count)} code(s)"
    return "exactly 1 code"


def _jsonable_value(value: Any) -> Any:
    """Convert pandas/numpy-ish scalar values into compact JSON-safe values."""
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return round(value, 4)
    if isinstance(value, (str, int, bool)):
        text = str(value).strip() if isinstance(value, str) else value
        return text[:240] if isinstance(text, str) else text
    return str(value).strip()[:240]


def _compact_row(row: Any, tokens: Optional[List[str]] = None, max_items: int = 24) -> Dict[str, Any]:
    if row is None:
        return {}
    try:
        data = row.to_dict()
    except Exception:
        try:
            data = dict(row)
        except Exception:
            return {}

    selected: Dict[str, Any] = {}
    lowered = [t.lower() for t in tokens or []]
    for key, value in data.items():
        key_str = str(key)
        if lowered and not any(t in key_str.lower() for t in lowered):
            continue
        val = _jsonable_value(value)
        if val is None or val == "":
            continue
        selected[key_str] = val
        if len(selected) >= max_items:
            break
    return selected


def _compact_cell_config(ps, pci: Optional[int]) -> Dict[str, Any]:
    if pci is None:
        return {}
    cfg = ps.get_cell_config(pci)
    if not cfg:
        return {}
    wanted = [
        "Cell ID",
        "Local Cell ID",
        "LocalCellId",
        "gNodeB ID",
        "PCI",
        "ARFCN",
        "Band",
        "TxPower",
        "Max Transmit",
        "PdcchOccupiedSymbolNum",
        "IntraFreqHoA3Offset",
        "IntraFreqHoA3Hyst",
        "Azimuth",
        "Tilt",
    ]
    out: Dict[str, Any] = {}
    for key, value in cfg.items():
        key_str = str(key)
        if any(token.lower() in key_str.lower() for token in wanted):
            val = _jsonable_value(value)
            if val is not None and val != "":
                out[key_str] = val
    return out


def _mr_neighbor_samples(ps, max_rows: int = 3) -> List[Dict[str, Any]]:
    if getattr(ps, "mr", None) is None or ps.mr.empty:
        return []
    tokens = ["timestamp", "time", "serving", "neighbor", "pci", "rsrp", "cell"]
    rows: List[Dict[str, Any]] = []
    for _, row in ps.mr.head(max_rows).iterrows():
        compact = _compact_row(row, tokens=tokens, max_items=24)
        if compact:
            rows.append(compact)
    return rows


def _raw_evidence_summary(ps, sf, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """Raw-ish measurement summary for LLM diagnosis without exposing prior answer."""
    worst_row = None
    if sf.throughput.worst_time:
        worst_row = ps.get_row_at_time(sf.throughput.worst_time)

    mob = sf.mobility
    best_neighbor_cell = ps.pci_to_cell_id.get(mob.best_neighbor_pci) if mob.best_neighbor_pci else None
    visible_neighbors = [
        {
            "pci": int(pci),
            "cell_id": ps.pci_to_cell_id.get(pci),
            "rsrp_dbm": _jsonable_value(rsrp),
            "configured_neighbor_of_serving": (
                ps.is_neighbor_configured(mob.serving_pci_at_worst, pci)
                if mob.serving_pci_at_worst
                else False
            ),
        }
        for pci, rsrp in sorted(
            (mob.neighbor_rsrp_map or {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )[:8]
    ]

    return {
        "scenario_id": ps.scenario_id,
        "tag": getattr(ps, "tag", ""),
        "family": diagnostics.get("family"),
        "data_available": diagnostics.get("data_available"),
        "data_shapes": {
            name: {
                "rows": shape.get("rows"),
                "cols": shape.get("cols"),
            }
            for name, shape in (diagnostics.get("data_shapes") or {}).items()
        },
        "throughput": {
            "worst_time": sf.throughput.worst_time,
            "worst_mbps": _jsonable_value(sf.throughput.worst_throughput_mbps),
            "drop_start_time": sf.throughput.drop_start_time,
            "drop_end_time": sf.throughput.drop_end_time,
            "recovery": bool(sf.throughput.recovery_flag),
        },
        "serving": {
            "pci": mob.serving_pci_at_worst,
            "cell_id": mob.serving_cell_id,
            "rsrp_dbm": _jsonable_value(mob.serving_rsrp_dbm),
            "sinr_db": _jsonable_value(mob.serving_sinr_db),
            "a3_threshold_db": _jsonable_value(mob.serving_a3_threshold_db),
        },
        "best_neighbor": {
            "pci": mob.best_neighbor_pci,
            "cell_id": best_neighbor_cell,
            "rsrp_dbm": _jsonable_value(mob.best_neighbor_rsrp_dbm),
            "advantage_db": _jsonable_value(mob.neighbor_advantage_db),
            "a3_margin_db": _jsonable_value(mob.a3_margin_db),
            "configured_neighbor_of_serving": bool(mob.neighbor_in_config),
        },
        "visible_neighbors_by_rsrp": visible_neighbors,
        "traffic": {
            "max_dl_prb_util": _jsonable_value(sf.inter_freq.max_dl_prb_util),
            "max_cce_util": _jsonable_value(sf.capacity.max_cce_util),
        },
        "cell_config": {
            "serving": _compact_cell_config(ps, mob.serving_pci_at_worst),
            "best_neighbor": _compact_cell_config(ps, mob.best_neighbor_pci),
        },
        "worst_user_plane_row": _compact_row(
            worst_row,
            tokens=["timestamp", "time", "throughput", "rsrp", "sinr", "pci", "prb", "cqi", "mcs"],
            max_items=28,
        ),
        "mr_neighbor_rows": _mr_neighbor_samples(ps),
    }


def _ranked_candidates_text(diagnostics: Dict[str, Any], limit: int = 10) -> str:
    lines = []
    for idx, row in enumerate((diagnostics.get("candidates") or [])[:limit], 1):
        lines.append(
            f"{idx}. answer={row.get('answer')} score={row.get('score')} "
            f"action={row.get('action_type', '')} target={row.get('target_cell', '')} "
            f"reason={row.get('reason', '')}"
        )
    return "\n".join(lines) or "no ranked candidates"


def _f2_four_action_hint(diagnostics: Dict[str, Any]) -> str:
    hints = []
    for row in diagnostics.get("candidates") or []:
        answer = str(row.get("answer") or "")
        if len([c for c in answer.split("|") if c.strip()]) == 4:
            hints.append(
                f"answer={answer} reason={row.get('reason', '')} "
                f"(only use if raw evidence supports a comprehensive fix)"
            )
    return "\n".join(hints[:3]) or "none"


def _build_arbitrator_messages(
    ps,
    sf,
    family: Family,
    diagnostics: Dict[str, Any],
    deterministic_answer: str,
    reasons: List[str],
    rag_hits: List[Any],
) -> List[Dict[str, str]]:
    trace = {
        "scenario_id": ps.scenario_id,
        "family": family.name,
        "is_multiple": family.is_multiple,
        "expected_answer_count": family.expected_answer_count,
        "arbitration_reasons": reasons,
        "data_available": diagnostics.get("data_available"),
        "cell_roles": ps.cell_roles,
        "pci_to_cell_id": {str(k): v for k, v in ps.pci_to_cell_id.items()},
    }

    cardinality = _cardinality_instruction(family)
    system = (
        "You are the final Track A 5G troubleshooting arbitration agent. "
        "First diagnose from raw measurements and server tool evidence; do not "
        "look for or assume a prior deterministic answer in the first pass. "
        "Then map the chosen action and target cell to the current option list. "
        "Option codes are scenario-local: never copy a C-code from a RAG example "
        "unless that same code is valid in this case. "
        "Follow the hierarchy: validate data availability, inspect RF quality, "
        "neighbor/config relation, load/capacity signals, and server/tool evidence; "
        "then choose the option code(s) that match the most likely root cause. "
        "For high-PRB healthy-radio F1 cases, explicitly arbitrate between "
        "insufficient_data and tertiary lift_tilt using tool evidence and RAG "
        "patterns; do not rely on raw mainlobe flag polarity alone. "
        "For F1_5BS single-answer cases, treat the case as a new Phase 2 "
        "single-answer variant of F1, not as an F3 multiple-answer template. "
        "Compare decrease_a3, add_neighbor, lift_tilt, insufficient_data, "
        "pdcch_2sym, check_server, and press_down_tilt; press_down_tilt is "
        "only a hypothesis and should not be selected merely because overlap "
        "is high. For F2 multiple-answer cases, prefer pairs whose two actions "
        "target the real serving/best-neighbor/interferer relationship rather "
        "than option-frequency primary/secondary labels alone; use a 4-code "
        "F2 answer only when raw evidence favors a comprehensive broader repair. "
        f"Return {cardinality}. The response must start with the boxed answer on "
        "the first line as "
        "\\boxed{C7}, \\boxed{C3|C11}, or \\boxed{C1|C4|C9|C12}. "
        "Do not begin with 'Based on', bullets, headings, or analysis. On the "
        "second line only, add one short Evidence sentence naming the primary "
        "evidence, winning action set, and matching cell(s)."
    )
    user = (
        "## Public Agent Trace\n"
        f"{json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
        "## Raw Measurement Evidence\n"
        f"{json.dumps(diagnostics.get('raw_evidence_summary') or {}, ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
        "## Server Tool Evidence\n"
        f"{_format_tool_evidence_for_prompt(sf, diagnostics)}\n\n"
        "## F2 Comprehensive Candidate Hint\n"
        f"{_f2_four_action_hint(diagnostics)}\n\n"
        "## Similar Solved Cases\n"
        f"{_format_rag_hits_for_prompt(rag_hits)}\n\n"
        "## All Current Options\n"
        f"{_format_options_for_verifier(ps)}\n\n"
        f"Select {cardinality} from the current options. Your first characters "
        "must be \\boxed{...}; any text before the boxed answer is invalid. "
        "Second line may be one short Evidence sentence."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _validate_model_answer(parsed: str, ps, family: Family) -> Tuple[bool, str]:
    if not parsed:
        return False, "empty_or_unparseable"
    codes = [c.strip() for c in parsed.split("|") if c.strip()]
    if len(codes) != len(set(codes)):
        return False, "duplicate_codes"
    valid_codes = set(_valid_option_codes(ps))
    bad = [c for c in codes if c not in valid_codes]
    if bad:
        return False, f"invalid_codes:{bad}"
    if not _same_cardinality(parsed, family):
        expected = "2_or_4" if family.name == "F2" else str(int(family.expected_answer_count))
        return False, (
            f"wrong_cardinality:got={len(codes)} "
            f"expected={expected}"
        )
    return True, "valid"


def _retry_llm_answer(
    arbiter: Phase2FineTunedVerifier,
    messages: List[Dict[str, str]],
    raw: str,
    reason: str,
    ps,
    family: Family,
) -> Tuple[str, str, Optional[str], str]:
    valid_codes = ", ".join(_valid_option_codes(ps))
    retry_messages = messages + [
        {"role": "assistant", "content": raw or ""},
        {
            "role": "user",
            "content": (
                f"The previous answer failed validation: {reason}. "
                f"You must output {_cardinality_retry_text(family)} from the "
                f"current option codes only: {valid_codes}. "
                "Do not copy C-codes from examples. Output only the final boxed "
                "answer, e.g. \\boxed{C7} or \\boxed{C3|C11}."
            ),
        },
    ]
    raw2, error2 = arbiter.call(retry_messages, enable_thinking=False)
    parsed2 = parse_answer(raw2)
    ok2, reason2 = _validate_model_answer(parsed2, ps, family)
    if error2 or not ok2:
        return parsed2, raw2, error2, reason2
    return parsed2, raw2, None, "valid"


def _confirm_conflicting_llm_answer(
    arbiter: Phase2FineTunedVerifier,
    messages: List[Dict[str, str]],
    raw: str,
    parsed: str,
    deterministic_answer: str,
    ps,
    family: Family,
    diagnostics: Dict[str, Any],
) -> Tuple[str, str, Optional[str], str]:
    """Second pass only when independent LLM diagnosis disagrees with pipeline."""
    confirm_messages = messages + [
        {"role": "assistant", "content": raw or f"\\boxed{{{parsed}}}"},
        {
            "role": "user",
            "content": (
                f"You independently selected \\boxed{{{parsed}}}. The deterministic "
                f"pipeline proposes \\boxed{{{deterministic_answer}}}. Compare both "
                "against the raw evidence again. Pipeline candidate hints below are "
                "prior hints, not ground truth; keep your answer if the measurements "
                "support it, otherwise switch to the pipeline answer or another valid "
                "option.\n\n"
                "## Pipeline Candidate Hints\n"
                f"{_ranked_candidates_text(diagnostics)}\n\n"
                f"Return {_cardinality_retry_text(family)} from the current option "
                f"codes only: {', '.join(_valid_option_codes(ps))}. "
                "Output only the final boxed answer."
            ),
        },
    ]
    raw2, error2 = arbiter.call(confirm_messages, enable_thinking=False)
    parsed2 = parse_answer(raw2)
    ok2, reason2 = _validate_model_answer(parsed2, ps, family)
    if error2 or not ok2:
        return parsed2, raw2, error2, reason2
    return parsed2, raw2, None, "valid"


def _apply_llm_arbitrator(
    arbiter: Optional[Phase2FineTunedVerifier],
    rag,
    rag_top_k: int,
    ps,
    sf,
    family: Family,
    diagnostics: Dict[str, Any],
    deterministic_answer: str,
    force: bool,
    margin_threshold: float,
    enable_thinking: bool,
    logger: logging.Logger,
) -> str:
    info = {
        "enabled": bool(arbiter),
        "policy": "arbitrate",
        "used_for_final": False,
        "called": False,
        "reasons": [],
        "model_answer": "",
        "raw_preview": "",
        "error": None,
    }
    diagnostics["llm_arbitrator"] = info
    diagnostics["fine_tuned_verifier"] = {
        "enabled": False,
        "policy": "arbitrate",
        "reason": "replaced_by_llm_arbitrator",
    }

    if arbiter is None:
        info["error"] = "no_model_endpoint"
        return deterministic_answer

    reasons = _arbitration_reasons(
        ps,
        sf,
        family,
        diagnostics,
        force=force,
        margin_threshold=margin_threshold,
    )
    info["reasons"] = reasons
    if not reasons:
        info["error"] = "not_high_risk"
        return deterministic_answer

    rag_hits = _retrieve_rag_hits(rag, sf, family, rag_top_k, logger)
    diagnostics["rag"] = {
        "enabled": rag is not None,
        "family": _rag_family_name(family),
        "top_k": rag_top_k,
        "hits": [
            _rag_hit_public_dict(hit)
            for hit in rag_hits[:rag_top_k]
        ],
    }

    messages = _build_arbitrator_messages(
        ps,
        sf,
        family,
        diagnostics,
        deterministic_answer,
        reasons,
        rag_hits,
    )
    info["called"] = True
    info["enable_thinking"] = bool(enable_thinking)
    raw, error = arbiter.call(messages, enable_thinking=enable_thinking)
    parsed = parse_answer(raw)
    ok, reason = _validate_model_answer(parsed, ps, family)
    info.update({
        "model_name": arbiter.model_id,
        "model_answer": parsed,
        "raw_preview": raw[:320],
        "error": error,
        "validation": reason,
        "initial_model_answer": parsed,
        "initial_raw_preview": raw[:320],
        "initial_error": error,
        "initial_validation": reason,
    })

    if error:
        return deterministic_answer
    if not ok:
        retryable = (
            reason.startswith("wrong_cardinality")
            or reason.startswith("invalid_codes")
            or reason in {"empty_or_unparseable", "duplicate_codes"}
        )
        if retryable:
            parsed2, raw2, error2, reason2 = _retry_llm_answer(
                arbiter,
                messages,
                raw,
                reason,
                ps,
                family,
            )
            info["retry_used"] = True
            info["retry_model_answer"] = parsed2
            info["retry_raw_preview"] = raw2[:320]
            info["retry_error"] = error2
            info["retry_validation"] = reason2
            if not error2 and reason2 == "valid":
                final_answer = parsed2
                final_raw = raw2
                if parsed2 != deterministic_answer:
                    c_parsed, c_raw, c_error, c_reason = _confirm_conflicting_llm_answer(
                        arbiter,
                        messages,
                        raw2,
                        parsed2,
                        deterministic_answer,
                        ps,
                        family,
                        diagnostics,
                    )
                    info["conflict_confirmation_used"] = True
                    info["conflict_model_answer"] = c_parsed
                    info["conflict_raw_preview"] = c_raw[:320]
                    info["conflict_error"] = c_error
                    info["conflict_validation"] = c_reason
                    if not c_error and c_reason == "valid":
                        final_answer = c_parsed
                        final_raw = c_raw
                info["model_answer"] = final_answer
                info["raw_preview"] = final_raw[:320]
                info["validation"] = "valid"
                info["used_for_final"] = final_answer != deterministic_answer
                logger.info(
                    "[%s] LLM arbitration retry accepted answer=%r deterministic=%r reasons=%s",
                    ps.scenario_id,
                    final_answer,
                    deterministic_answer,
                    ",".join(reasons),
                )
                return final_answer
        return deterministic_answer

    final_answer = parsed
    final_raw = raw
    if parsed != deterministic_answer:
        c_parsed, c_raw, c_error, c_reason = _confirm_conflicting_llm_answer(
            arbiter,
            messages,
            raw,
            parsed,
            deterministic_answer,
            ps,
            family,
            diagnostics,
        )
        info["conflict_confirmation_used"] = True
        info["conflict_model_answer"] = c_parsed
        info["conflict_raw_preview"] = c_raw[:320]
        info["conflict_error"] = c_error
        info["conflict_validation"] = c_reason
        if not c_error and c_reason == "valid":
            final_answer = c_parsed
            final_raw = c_raw

    info["model_answer"] = final_answer
    info["raw_preview"] = final_raw[:320]
    info["validation"] = "valid"
    info["used_for_final"] = final_answer != deterministic_answer
    logger.info(
        "[%s] LLM arbitration accepted answer=%r deterministic=%r reasons=%s",
        ps.scenario_id,
        final_answer,
        deterministic_answer,
        ",".join(reasons),
    )
    return final_answer


def _attach_agent_trace(
    diagnostics: Dict[str, Any],
    final_answer: str,
) -> None:
    """Public, audit-friendly trace of the four required agent components."""
    diagnostics["agent_trace"] = {
        "skills": {
            "used": True,
            "summary": "scenario_parser → scenario_router → feature_engine → candidate_generator",
            "top_issue": diagnostics.get("top_issue"),
        },
        "multi_turn_tool_use": {
            "used": bool(diagnostics.get("api_calls")),
            "api_call_count": len(diagnostics.get("api_calls", [])),
            "conditional_tools": diagnostics.get("tool_adjustments", []),
        },
        "reasoning": {
            "used": True,
            "summary": (
                f"family={diagnostics.get('family')} "
                f"top_issue={(diagnostics.get('top_issue') or {}).get('name')} "
                f"arbitration_triggered={bool((diagnostics.get('llm_arbitrator') or {}).get('called'))} "
                f"rag_hits={len((diagnostics.get('rag') or {}).get('hits', []))}"
            ),
        },
        "rag": diagnostics.get("rag", {"enabled": False}),
        "llm_arbitration": diagnostics.get("llm_arbitrator", {"enabled": False}),
        "fine_tuning": diagnostics.get("fine_tuned_verifier", {"enabled": False}),
        "final_answer": final_answer,
    }


# ---------------------------------------------------------------------------
# Answer formatting
# ---------------------------------------------------------------------------

def _format_answer(candidate_output, tag: str) -> str:
    """Format the top candidate/set as a submission-ready string."""
    answer = candidate_output.top_candidate()
    if not answer:
        return ""

    is_multiple = "multiple" in tag.lower()
    codes = [c.strip() for c in answer.split("|") if c.strip()]

    if not is_multiple:
        return codes[0] if codes else ""

    # Multiple-answer: generate_candidates() already returns ranked answer sets.
    # Sort only for stable submission formatting.
    unique_codes = sorted(set(codes), key=lambda x: int(x[1:]))
    return "|".join(unique_codes)


def _candidate_summary(candidate_output) -> List[Dict[str, Any]]:
    """Compact candidate list for diagnostics JSONL."""
    rows: List[Dict[str, Any]] = []
    if not candidate_output.is_multiple:
        for c in candidate_output.single_candidates:
            rows.append({
                "answer": c.c_code,
                "score": float(c.score),
                "action_type": c.action_type,
                "target_cell": c.target_cell,
                "reason": c.reason,
            })
    else:
        for c in candidate_output.multi_candidates:
            rows.append({
                "answer": "|".join(c.c_codes),
                "score": float(c.score),
                "reason": c.reason,
            })
    return rows


def _phase2_family(tag: str, num_bs: int) -> Tuple[Family, Family, Optional[str]]:
    """
    Phase 2 introduces tag/BS combinations that were absent from Phase 1.
    Preserve the actual answer cardinality from `tag`, while keeping the
    strongest BS-specific prior where it matters.
    """
    routed = route(tag, num_bs)
    is_multiple = "multiple" in tag.lower()

    if is_multiple:
        if num_bs == 5:
            return routed, FAMILIES["F3"], None if routed.name == "F3" else "F3"
        if num_bs == 13:
            family = Family(
                name="F4M",  # type: ignore[arg-type]
                num_bs=13,
                is_multiple=True,
                expected_answer_count=2,
                description="13 BS + multiple-answer (inter-frequency pair)",
                primary_skill_bundle="inter_freq_composite",
            )
            return routed, family, family.name
        return routed, FAMILIES["F2"], None if routed.name == "F2" else "F2"

    if num_bs == 13:
        return routed, FAMILIES["F4"], None if routed.name == "F4" else "F4"
    if num_bs == 5:
        family = Family(
            name="F1_5BS",  # type: ignore[arg-type]
            num_bs=5,
            is_multiple=False,
            expected_answer_count=1,
            description="5 BS + single-answer (Phase 2 generic single)",
            primary_skill_bundle="full",
        )
        return routed, family, family.name
    return routed, FAMILIES["F1"], None if routed.name == "F1" else "F1"


def _data_shapes(ps) -> Dict[str, Dict[str, int]]:
    """Small parser-health summary for diagnostics/audit."""
    frames = {
        "user_plane": ps.user_plane,
        "net_config": ps.net_config,
        "signaling": ps.signaling,
        "traffic": ps.traffic,
        "mr": ps.mr,
    }
    return {
        name: {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": [str(c) for c in list(df.columns)[:16]],
        }
        for name, df in frames.items()
    }


def _feature_scores(sf) -> Dict[str, float]:
    """Composite issue scores, rounded only for readable diagnostics."""
    scores = {
        "late_handover": sf.score_late_handover,
        "missing_neighbor": sf.score_missing_neighbor,
        "inter_freq_load": sf.score_inter_freq_load,
        "interference": sf.score_interference,
        "coverage_hole": sf.score_coverage_hole,
        "overshoot_overlap": sf.score_overshoot_overlap,
        "pdcch_issue": sf.score_pdcch_issue,
        "backhaul_server": sf.score_backhaul_server,
        "insufficient_data": getattr(sf, "score_insufficient_data", None) or 0.0,
    }
    return {k: round(float(v or 0.0), 4) for k, v in scores.items()}


def _phase2_mr_neighbor_rsrp_map(ps) -> Dict[int, float]:
    result: Dict[int, float] = {}
    if getattr(ps, "mr", None) is None or ps.mr.empty:
        return result
    for n in range(1, 7):
        pci_col = next((c for c in ps.mr.columns if f"Neighbor {n} PCI" in c), None)
        rsrp_col = next((c for c in ps.mr.columns if f"Neighbor {n} RSRP" in c), None)
        if not pci_col or not rsrp_col:
            continue
        for _, row in ps.mr.iterrows():
            try:
                pci = int(float(row.get(pci_col)))
                rsrp = float(row.get(rsrp_col))
            except (ValueError, TypeError):
                continue
            if pci <= 0 or rsrp != rsrp:
                continue
            result[pci] = max(result.get(pci, rsrp), rsrp)
    return result


def _apply_phase2_mr_config_fallback(ps, sf, family: Family, diagnostics: Dict[str, Any]) -> None:
    """Phase 2-only mobility recovery when signaling is absent but MR is present."""
    if family.name not in {"F1", "F1_5BS", "F2"}:
        return
    if diagnostics.get("data_available", {}).get("signaling"):
        return

    mr_map = _phase2_mr_neighbor_rsrp_map(ps)
    if not mr_map:
        return

    before_scores = _feature_scores(sf)
    before_best = sf.mobility.best_neighbor_pci

    merged = dict(sf.mobility.neighbor_rsrp_map or {})
    for pci, rsrp in mr_map.items():
        merged[pci] = max(merged.get(pci, rsrp), rsrp)
    if merged:
        sf.mobility.neighbor_rsrp_map = merged
        sf.mobility.visible_neighbor_pcis = list(merged.keys())
        sf.mobility.best_neighbor_pci, sf.mobility.best_neighbor_rsrp_dbm = max(
            merged.items(),
            key=lambda item: item[1],
        )

    mob = sf.mobility
    if (
        mob.best_neighbor_rsrp_dbm is not None
        and mob.serving_rsrp_dbm is not None
        and mob.serving_a3_threshold_db is not None
    ):
        mob.neighbor_advantage_db = round(mob.best_neighbor_rsrp_dbm - mob.serving_rsrp_dbm, 2)
        mob.a3_margin_db = round(mob.neighbor_advantage_db - mob.serving_a3_threshold_db, 2)
        mob.neighbor_in_config = (
            ps.is_neighbor_configured(mob.serving_pci_at_worst, mob.best_neighbor_pci)
            if mob.serving_pci_at_worst and mob.best_neighbor_pci
            else False
        )

    adv = mob.neighbor_advantage_db or 0.0
    if adv <= 3:
        return

    boost = min(0.85, 0.55 + max(0.0, adv - 3.0) * 0.04)
    if not mob.neighbor_in_config:
        sf.score_missing_neighbor = max(sf.score_missing_neighbor, boost)
        sf.score_late_handover = max(sf.score_late_handover, 0.25)
        issue = "missing_neighbor_from_mr_config"
    elif mob.a3_margin_db is None or mob.a3_margin_db <= 1.5 or adv >= 5:
        sf.score_late_handover = max(sf.score_late_handover, boost)
        sf.score_missing_neighbor = max(sf.score_missing_neighbor, 0.20)
        issue = "late_handover_from_mr_config"
    else:
        return

    diagnostics["feature_scores_before_phase2_mr_fallback"] = before_scores
    diagnostics.setdefault("phase2_feature_adjustments", []).append({
        "source": "mr_config_fallback",
        "issue": issue,
        "best_neighbor_before": before_best,
        "best_neighbor_after": mob.best_neighbor_pci,
        "neighbor_in_config": bool(mob.neighbor_in_config),
        "advantage_db": float(adv),
        "a3_margin_db": mob.a3_margin_db,
        "scores_after": _feature_scores(sf),
    })


def _apply_phase2_tool_checks(gatherer, ps, sf, family: Family, diagnostics: Dict[str, Any]) -> None:
    """Run targeted Phase 2 tool evidence and reuse C's optimized augmentation."""
    try:
        from skills.tool_caller import ToolCaller, ToolEvidence
    except Exception as exc:
        diagnostics.setdefault("tool_adjustments", []).append({
            "tool": "import_tool_caller",
            "error": str(exc),
        })
        return

    evidence = ToolEvidence()
    adjustments: List[Dict[str, Any]] = []
    top_issue, top_score = sf.top_issue()
    serving_pci = sf.mobility.serving_pci_at_worst or gatherer.serving_pci
    worst_time = sf.throughput.worst_time or gatherer.worst_time

    # Coverage/server/insufficient-data ambiguities are the expensive mistakes
    # in Phase 2, so call judge_mainlobe when the local scores make that branch
    # plausible.  The actual score polarity is delegated to ToolCaller's
    # validated augment_features().
    tool_sensitive_family = family.name in ("F1", "F1_5BS", "F2", "F4M")
    uncertain_single_rf = (
        tool_sensitive_family
        and not family.is_multiple
        and top_issue == "uncertain"
        and serving_pci is not None
        and worst_time is not None
    )
    cov = sf.score_coverage_hole or 0.0
    insuff = getattr(sf, "score_insufficient_data", None) or 0.0
    server = sf.score_backhaul_server or 0.0
    should_mainlobe = (
        tool_sensitive_family
        and not family.is_multiple
        and serving_pci is not None
        and worst_time is not None
        and (
            top_issue in ("coverage_hole", "backhaul_server", "insufficient_data")
            or family.name == "F1_5BS"
            or max(cov, insuff, server) >= 0.25
            or uncertain_single_rf
            or (
                sf.mobility.serving_sinr_db is not None
                and sf.mobility.serving_sinr_db > 5
                and sf.throughput.worst_throughput_mbps is not None
                and sf.throughput.worst_throughput_mbps < 80
            )
        )
    )
    if should_mainlobe:
        result = gatherer._call_tool("judge_mainlobe_or_not", time=worst_time, pci=serving_pci)
        if isinstance(result, dict) and "Outside_Mainlobe_Flag" in result:
            evidence.mainlobe_flags[int(serving_pci)] = bool(result["Outside_Mainlobe_Flag"])
            evidence.tool_call_count += 1
            adjustments.append({
                "tool": "judge_mainlobe_or_not",
                "pci": int(serving_pci),
                "outside": bool(result["Outside_Mainlobe_Flag"]),
            })

    # MR sparsity is the main signal for insufficient_data.  Use the already
    # fetched MR table so we do not spend an extra HTTP call.  Phase 2 has a
    # small mid-SINR gray zone where local scores remain all-zero; in single
    # answer families this is exactly where mainlobe/MR evidence is most useful.
    should_mr = (
        tool_sensitive_family
        and (
            top_issue in ("coverage_hole", "backhaul_server", "insufficient_data", "pdcch_issue")
            or family.name == "F1_5BS"
            or max(cov, insuff, server) >= 0.25
            or uncertain_single_rf
        )
    )
    if should_mr and getattr(gatherer, "data_available", {}).get("mr"):
        evidence.mr_sample_count = int(ps.mr.shape[0])
        if sf.mobility.serving_cell_id:
            evidence.mr_sample_counts[sf.mobility.serving_cell_id] = int(ps.mr.shape[0])
        evidence.tool_call_count += 1
        adjustments.append({
            "tool": "get_mr_data",
            "sample_count": int(ps.mr.shape[0]),
        })

    # High overlap between serving and best neighbor helps distinguish F2
    # overshoot/interference fixes from generic handover fixes.
    best_neighbor = sf.mobility.best_neighbor_pci
    should_overlap = (
        family.name in ("F1", "F1_5BS", "F2")
        and
        serving_pci is not None
        and best_neighbor is not None
        and (
            family.is_multiple
            or top_issue in ("interference", "overshoot_overlap", "late_handover", "missing_neighbor")
            or (sf.mobility.serving_sinr_db is not None and sf.mobility.serving_sinr_db < 5)
        )
    )
    if should_overlap:
        ratio = gatherer.get_overlap_ratio(int(serving_pci), int(best_neighbor))
        if ratio is not None:
            evidence.overlap_ratio = ratio
            evidence.tool_call_count += 1
            adjustments.append({
                "tool": "calculate_overlap_ratio",
                "serving_pci": int(serving_pci),
                "neighbor_pci": int(best_neighbor),
                "ratio": float(ratio),
            })

    diagnostics["tool_adjustments"] = adjustments
    if not evidence.tool_call_count:
        return

    before = _feature_scores(sf)
    sf.tool_evidence = evidence
    ToolCaller(getattr(gatherer.env, "server_url", ""), logger=gatherer.logger).augment_features(sf, evidence)
    diagnostics["feature_scores_before_tools"] = before
    diagnostics["feature_scores_after_tools"] = _feature_scores(sf)


# ---------------------------------------------------------------------------
# Per-scenario solver
# ---------------------------------------------------------------------------

def _solve_scenario(
    scenario: Dict[str, Any],
    env: Environment,
    logger: logging.Logger,
    fetch_signaling: bool = False,
    llm_verifier: Optional[Phase2FineTunedVerifier] = None,
    llm_policy: str = "audit",
    rag=None,
    rag_top_k: int = 6,
    llm_arbitrate_all: bool = False,
    llm_arbitrate_margin: float = 0.25,
    llm_enable_thinking: bool = True,
    agentic_policy: str = "off",
) -> Tuple[str, Dict[str, Any]]:
    """
    Run the full Phase 2 agent pipeline for one scenario.
    Returns a C-code answer string (e.g. "C5" or "C3|C7").
    Returns empty string on unrecoverable error.
    """
    sid = scenario.get("scenario_id", "?")
    tag = scenario.get("tag", "single-answer")
    ctx = scenario.get("context", {})
    wi = ctx.get("wireless_network_information", {})
    num_bs = int(str(wi.get("num_base_stations", "4")).strip())

    # Route to family
    routed_family, family, family_override = _phase2_family(tag, num_bs)
    diagnostics: Dict[str, Any] = {
        "scenario_id": sid,
        "tag": tag,
        "num_base_stations": num_bs,
        "routed_family": routed_family.name,
        "family_override": family_override,
        "answer": "",
        "api_calls": [],
        "candidates": [],
        "error": None,
    }
    if family_override:
        logger.info(f"[{sid}] Family override: {family.name} (tag={tag}, bs={num_bs})")
    diagnostics["family"] = family.name
    diagnostics["is_multiple"] = bool(family.is_multiple)

    logger.info(f"[{sid}] tag={tag}, bs={num_bs}, family={family.name}")

    # ------------------------------------------------------------------
    # Step 1: gather data via API
    # ------------------------------------------------------------------
    from skills.api_data_gatherer import APIDataGatherer
    gatherer = APIDataGatherer(env, scenario, fetch_signaling=fetch_signaling, logger=logger)
    raw = gatherer.gather()
    diagnostics["api_calls"] = getattr(gatherer, "api_calls", [])
    diagnostics["data_available"] = getattr(gatherer, "data_available", {})
    diagnostics["worst_time"] = gatherer.worst_time
    diagnostics["last_time"] = gatherer.last_time
    diagnostics["serving_pci"] = gatherer.serving_pci

    # ------------------------------------------------------------------
    # Step 2: parse scenario (option labels + DataFrames)
    # ------------------------------------------------------------------
    ps = parse_scenario(raw)
    diagnostics["data_shapes"] = _data_shapes(ps)

    if not ps.parsed_options:
        logger.warning(f"[{sid}] No parsed options found — skipping")
        diagnostics["error"] = "no_parsed_options"
        return "", diagnostics

    # ------------------------------------------------------------------
    # Step 3: feature extraction
    # ------------------------------------------------------------------
    sf = run_all_skills(ps)
    _apply_phase2_mr_config_fallback(ps, sf, family, diagnostics)
    diagnostics["feature_scores"] = _feature_scores(sf)
    top_issue, top_score = sf.top_issue()
    diagnostics["top_issue"] = {"name": top_issue, "score": float(top_score)}

    _apply_phase2_tool_checks(gatherer, ps, sf, family, diagnostics)
    diagnostics["api_calls"] = getattr(gatherer, "api_calls", [])
    diagnostics["feature_scores"] = _feature_scores(sf)
    top_issue, top_score = sf.top_issue()
    diagnostics["top_issue"] = {"name": top_issue, "score": float(top_score)}
    diagnostics["raw_evidence_summary"] = _raw_evidence_summary(ps, sf, diagnostics)

    # ------------------------------------------------------------------
    # Step 4: candidate generation
    # ------------------------------------------------------------------
    candidate_output = generate_candidates(ps, sf, family)
    diagnostics["candidates"] = _candidate_summary(candidate_output)
    if not diagnostics["candidates"]:
        logger.warning(f"[{sid}] No candidates produced")
        diagnostics["error"] = "no_candidates"
        return "", diagnostics

    answer = _format_answer(candidate_output, tag)

    # ------------------------------------------------------------------
    # Step 4b: controlled agentic overlay (high-risk cases only)
    # ------------------------------------------------------------------
    if agentic_policy == "high_risk" and llm_verifier is not None:
        from skills.agentic_overlay import run_agentic_overlay
        task_description = scenario.get("task", {}).get("description", "")
        answer = run_agentic_overlay(
            ps=ps,
            sf=sf,
            gatherer=gatherer,
            family=family,
            tag=tag,
            scores=_feature_scores(sf),
            deterministic_answer=answer,
            task_description=task_description,
            llm_verifier=llm_verifier,
            diagnostics=diagnostics,
            logger=logger,
        )

    if llm_policy == "arbitrate":
        answer = _apply_llm_arbitrator(
            llm_verifier,
            rag,
            rag_top_k,
            ps,
            sf,
            family,
            diagnostics,
            answer,
            force=llm_arbitrate_all,
            margin_threshold=llm_arbitrate_margin,
            enable_thinking=llm_enable_thinking,
            logger=logger,
        )
    else:
        answer = _apply_fine_tuned_verifier(
            llm_verifier,
            llm_policy,
            ps,
            sf,
            family,
            diagnostics,
            answer,
        )
    diagnostics["answer"] = answer
    _attach_agent_trace(diagnostics, answer)
    top = diagnostics["candidates"][0]
    logger.info(f"[{sid}] answer={answer!r} top_candidate={top}")
    return answer, diagnostics


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 2 Track A agent with skills, tools, reasoning trace, and optional model verifier/arbitrator")
    ap.add_argument("--server_url", default="https://120.46.145.77/no",
                    help="Phase 2 competition server URL")
    ap.add_argument("--token", default=None,
                    help="Bearer token for Phase 2 server (set AGENT_API_KEY env var if preferred)")
    ap.add_argument("--test_json", type=Path,
                    default=Path(__file__).parent / "data/Phase_2/test.json",
                    help="Path to Phase 2 test.json")
    ap.add_argument("--out", type=Path, default=Path("result_phase2.csv"),
                    help="Output CSV path (columns: ID, Track A, Track B; Track B stays blank)")
    ap.add_argument("--diagnostics_jsonl", type=Path, default=None,
                    help="Per-scenario diagnostics JSONL path")
    ap.add_argument("--save_freq", type=int, default=10,
                    help="Save result.csv every N scenarios")
    ap.add_argument("--timeout", type=float, default=15.0,
                    help="Per-tool HTTP timeout in seconds")
    ap.add_argument("--fetch_signaling", action="store_true",
                    help="Also call get_signaling_plane_event_log (live probe found it empty by default)")
    ap.add_argument("--model_url", type=str, default=None,
                    help="OpenAI-compatible model endpoint, e.g. http://127.0.0.1:8001/v1")
    ap.add_argument("--model_name", type=str, default="v4",
                    help="Model name exposed by the endpoint")
    ap.add_argument("--model_provider", type=str, default=None,
                    help="Optional provider prefix for hosted OpenAI-compatible endpoints")
    ap.add_argument("--llm_policy", choices=["off", "audit", "gate", "always", "arbitrate"], default="audit",
                    help="How the model participates: audit/gate/always as verifier, or arbitrate as final high-risk LLM+RAG judge")
    ap.add_argument("--agentic_policy", choices=["off", "high_risk"], default="off",
                    help="off: deterministic only; high_risk: controlled 2-call agentic overlay on ambiguous branches")
    ap.add_argument("--llm_max_tokens", type=int, default=384,
                    help="Max tokens for verifier/arbitrator brief reasoning plus boxed answer")
    ap.add_argument("--llm_timeout", type=float, default=30.0,
                    help="Model HTTP timeout")
    ap.add_argument("--rag_train_json", type=Path,
                    default=Path(__file__).parent / "data/Phase_1/train.json",
                    help="Phase 1 train.json used by structured RAG when --llm_policy arbitrate")
    ap.add_argument("--rag_top_k", type=int, default=6,
                    help="Number of structured RAG hits injected into arbitration prompts")
    ap.add_argument("--no_rag", action="store_true",
                    help="Disable RAG retrieval in arbitration mode")
    ap.add_argument("--llm_arbitrate_all", action="store_true",
                    help="Call the LLM arbiter for every non-template scenario instead of only high-risk ones")
    ap.add_argument("--llm_arbitrate_margin", type=float, default=0.25,
                    help="Candidate score gap at or below which arbitration is triggered")
    ap.add_argument("--no_llm_thinking", action="store_true",
                    help="Disable Qwen extended-thinking mode for the arbitration call")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from existing --out CSV and append diagnostics")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Limit number of scenarios (for testing)")
    ap.add_argument("--sample_per_group", type=int, default=None,
                    help="Run the first N scenarios from each (tag, num_base_stations) group")
    ap.add_argument("--scenario_ids", type=str, default=None,
                    help="Comma-separated scenario IDs to run")
    ap.add_argument("--scenario_ids_file", type=Path, default=None,
                    help="File with one scenario ID per line")
    ap.add_argument("--log_file", type=str, default=None,
                    help="Optional log file path")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logger = init_logger(log_file=args.log_file)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Apply bearer token if supplied via CLI (overrides env var)
    import os
    if args.token:
        os.environ["AGENT_API_KEY"] = args.token

    # ------------------------------------------------------------------
    # Load test scenarios
    # ------------------------------------------------------------------
    if not args.test_json.exists():
        logger.error(f"test.json not found: {args.test_json}")
        sys.exit(1)

    with open(args.test_json, encoding="utf-8") as f:
        scenarios: List[Dict[str, Any]] = json.load(f)

    selected_ids = set()
    if args.scenario_ids:
        selected_ids.update(s.strip() for s in args.scenario_ids.split(",") if s.strip())
    if args.scenario_ids_file:
        selected_ids.update(
            line.strip()
            for line in args.scenario_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if selected_ids:
        before = len(scenarios)
        scenarios = [s for s in scenarios if str(s.get("scenario_id", "")).strip() in selected_ids]
        missing = selected_ids - {str(s.get("scenario_id", "")).strip() for s in scenarios}
        if missing:
            logger.warning(f"Requested scenario IDs not found: {sorted(missing)[:10]}")
        logger.info(f"Scenario ID filter: {len(scenarios)}/{before} selected")

    if args.sample_per_group is not None:
        grouped: Dict[Tuple[str, int], int] = {}
        sampled: List[Dict[str, Any]] = []
        for scenario in scenarios:
            tag = str(scenario.get("tag", "single-answer"))
            wi = scenario.get("context", {}).get("wireless_network_information", {})
            try:
                bs = int(str(wi.get("num_base_stations", "4")).strip())
            except ValueError:
                bs = 4
            key = (tag, bs)
            if grouped.get(key, 0) < args.sample_per_group:
                sampled.append(scenario)
                grouped[key] = grouped.get(key, 0) + 1
        scenarios = sampled
        logger.info(f"Group sample filter: selected {len(scenarios)} scenarios from {len(grouped)} groups")

    if args.max_samples is not None:
        scenarios = scenarios[: args.max_samples]

    logger.info(f"Loaded {len(scenarios)} Phase 2 scenarios from {args.test_json}")
    diagnostics_path = args.diagnostics_jsonl or args.out.with_suffix(".diagnostics.jsonl")
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and diagnostics_path.exists():
        diagnostics_path.unlink()

    # ------------------------------------------------------------------
    # Build Environment (reuse main.py's Environment class)
    # ------------------------------------------------------------------
    env = Environment(
        server_url=args.server_url,
        verbose=args.verbose,
        logger=logger,
        api_key=args.token,
        timeout=args.timeout,
    )

    llm_verifier = None
    needs_model = args.llm_policy != "off" or args.agentic_policy != "off"
    if args.model_url and needs_model:
        try:
            llm_verifier = Phase2FineTunedVerifier(
                model_url=args.model_url,
                model_name=args.model_name,
                model_provider=args.model_provider,
                max_tokens=args.llm_max_tokens,
                timeout=args.llm_timeout,
                logger=logger,
            )
            logger.info(
                "[LLM] Enabled model=%s policy=%s url=%s bypass_env_proxy=%s",
                llm_verifier.model_id,
                args.llm_policy,
                args.model_url,
                llm_verifier.bypass_env_proxy,
            )
        except Exception as exc:
            logger.warning("[LLM] Disabled after init failure: %s", exc)
            llm_verifier = None
    elif needs_model:
        logger.info("[LLM] No --model_url supplied; diagnostics will mark model disabled.")

    rag = None
    if args.llm_policy == "arbitrate" and llm_verifier is not None and not args.no_rag:
        rag = _build_rag_index(args.rag_train_json, logger)
    elif args.llm_policy == "arbitrate":
        logger.info("[RAG] Disabled (no_rag=%s model_ready=%s)", args.no_rag, llm_verifier is not None)

    # ------------------------------------------------------------------
    # Solve each scenario
    # ------------------------------------------------------------------
    results_by_id: Dict[str, Dict[str, str]] = {}
    if args.resume and args.out.exists():
        with open(args.out, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid_existing = str(row.get("ID", "")).strip()
                if sid_existing:
                    results_by_id[sid_existing] = {
                        "ID": sid_existing,
                        "Track A": str(row.get("Track A", "") or ""),
                        "Track B": str(row.get("Track B", "") or ""),
                    }
        logger.info(f"Resume enabled: loaded {len(results_by_id)} existing rows from {args.out}")

    start_total = time.time()

    for idx, scenario in enumerate(scenarios):
        sid = scenario.get("scenario_id", f"unknown_{idx}")
        if sid in results_by_id and results_by_id[sid].get("Track A"):
            logger.info(f"[{sid}] resume skip existing answer={results_by_id[sid]['Track A']!r}")
            continue

        t0 = time.time()
        diagnostics: Dict[str, Any] = {"scenario_id": sid}

        try:
            answer, diagnostics = _solve_scenario(
                scenario,
                env,
                logger,
                fetch_signaling=args.fetch_signaling,
                llm_verifier=llm_verifier,
                llm_policy=args.llm_policy,
                rag=rag,
                rag_top_k=args.rag_top_k,
                llm_arbitrate_all=args.llm_arbitrate_all,
                llm_arbitrate_margin=args.llm_arbitrate_margin,
                llm_enable_thinking=not args.no_llm_thinking,
                agentic_policy=args.agentic_policy,
            )
        except Exception:
            logger.error(f"[{sid}] EXCEPTION:\n{traceback.format_exc()}")
            answer = ""
            diagnostics = {
                "scenario_id": sid,
                "tag": scenario.get("tag", ""),
                "answer": "",
                "error": traceback.format_exc(),
            }

        elapsed = time.time() - t0
        diagnostics["elapsed_seconds"] = elapsed
        logger.info(f"[{sid}] done in {elapsed:.1f}s — answer: {answer!r}")

        results_by_id[sid] = {"ID": sid, "Track A": answer, "Track B": ""}
        _append_diagnostic(diagnostics, diagnostics_path)

        # Periodic save
        if (idx + 1) % args.save_freq == 0 or (idx + 1) == len(scenarios):
            _save_results(_ordered_results(scenarios, results_by_id), args.out, logger)

    total_elapsed = time.time() - start_total
    results = _ordered_results(scenarios, results_by_id)
    _save_results(results, args.out, logger)
    answered = sum(1 for r in results if r["Track A"])
    logger.info(
        f"\nDone. {answered}/{len(results)} answered in {total_elapsed:.0f}s "
        f"(avg {total_elapsed / max(len(results), 1):.1f}s/scenario)"
    )


def _save_results(results: List[Dict[str, str]], out_path: Path, logger: logging.Logger) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Track A", "Track B"])
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Saved {len(results)} rows → {out_path}")


def _ordered_results(scenarios: List[Dict[str, Any]], rows_by_id: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    ordered: List[Dict[str, str]] = []
    for idx, scenario in enumerate(scenarios):
        sid = scenario.get("scenario_id", f"unknown_{idx}")
        if sid in rows_by_id:
            ordered.append(rows_by_id[sid])
    return ordered


def _append_diagnostic(row: Dict[str, Any], path: Path) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
