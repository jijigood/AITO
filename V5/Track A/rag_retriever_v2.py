#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Structured Case RAG — V2 Architecture

Unlike V1 (raw text embedding of question + option labels), V2 retrieval
operates on structured case summaries derived from RF features, enabling
family-filtered nearest-neighbour search.

Two retrieval stages (per ARCHITECTURE_V2.md §7.6):
  1. Hard filter by family name (F1/F2/F3/F4) and answer type
  2. Nearest-neighbour by structured feature vector

Each training example is summarised into a numeric feature vector:
  [worst_tp, drop_magnitude, serving_rsrp, serving_sinr,
   neighbor_advantage, a3_margin, max_prb_util, num_neighbors,
   late_handover_score, missing_neighbor_score, interference_score,
   coverage_hole_score, inter_freq_score]

Dependencies: sentence-transformers, faiss-cpu, numpy
"""

import json
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skills.agentic_overlay import compute_risk_level
from skills.feature_engine import run_all_skills
from skills.scenario_parser import parse_option_label, parse_scenario
from skills.scenario_router import route

try:
    import faiss as _faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _faiss = None
    _FAISS_AVAILABLE = False


_F1_ACTION_BUCKETS = [
    "lift_tilt",
    "insufficient_data",
    "add_neighbor",
    "pdcch_2sym",
    "check_server",
    "decrease_a3",
]

_CONFUSION_TRIGGER_ISSUES = frozenset({"coverage_hole", "backhaul_server"})

_ISSUE_DIM_NAMES = [
    "late_handover",
    "missing_neighbor",
    "interference",
    "coverage_hole",
    "inter_freq_load",
]

_ACTION_ROOT_CAUSE_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "add_neighbor": ("missing_neighbor",),
    "decrease_a3": ("late_handover",),
    "decrease_interfreq": ("inter_freq_load",),
    "press_down_tilt": ("overshoot_overlap", "interference"),
    "lift_tilt": ("coverage_hole",),
    "pdcch_2sym": ("pdcch_issue",),
    "adjust_azimuth": ("overshoot_overlap",),
    "increase_a3": ("late_handover",),
    "decrease_power": ("interference", "overshoot_overlap"),
    "increase_power": ("coverage_hole",),
    "check_server": ("backhaul_server",),
    "insufficient_data": ("insufficient_data",),
    "unknown": ("uncertain",),
}

_ROOT_CAUSE_TOOLS: Dict[str, List[str]] = {
    "missing_neighbor": ["get_signaling_plane_event_log", "calculate_overlap_ratio"],
    "late_handover": ["get_signaling_plane_event_log", "calculate_overlap_ratio"],
    "coverage_hole": ["judge_mainlobe_or_not", "get_mr_data"],
    "insufficient_data": ["get_mr_data", "judge_mainlobe_or_not"],
    "inter_freq_load": ["get_kpi_data"],
    "pdcch_issue": ["get_kpi_data"],
    "backhaul_server": ["get_kpi_data", "get_mr_data"],
    "overshoot_overlap": ["calculate_overlap_ratio", "judge_mainlobe_or_not"],
    "interference": ["calculate_overlap_ratio", "get_kpi_data"],
    "uncertain": [],
}

_PAIR_BRANCH_TOOLS: Dict[str, List[str]] = {
    "lift_tilt_vs_insufficient_data": ["judge_mainlobe_or_not", "get_mr_data"],
    "add_neighbor_vs_decrease_a3": ["get_signaling_plane_event_log", "calculate_overlap_ratio"],
}

_ROOT_CAUSE_PRIMARY_ACTION: Dict[str, str] = {
    "missing_neighbor": "add_neighbor",
    "late_handover": "decrease_a3",
    "coverage_hole": "lift_tilt",
    "insufficient_data": "insufficient_data",
    "inter_freq_load": "decrease_interfreq",
    "pdcch_issue": "pdcch_2sym",
    "backhaul_server": "check_server",
    "overshoot_overlap": "press_down_tilt",
    "interference": "decrease_power",
}

_TRAJECTORY_TOOL_ALLOWLIST = {
    "calculate_overlap_ratio",
    "calculate_tilt_angle",
    "get_kpi_data",
    "get_mr_data",
    "get_signaling_plane_event_log",
    "get_user_location",
    "judge_mainlobe_or_not",
}

_UNSAFE_TRAJECTORY_KEYS = {"answer", "answer_labels", "gt_c_code", "options"}
_UNSAFE_C_CODE_RE = re.compile(r"\bC(?:[1-9]|1\d|2[0-4])\b")
_PROBE_DIAGNOSTICS_CANDIDATES = [
    "results/phase2/agentic_overlay_tests/high_risk_agentic_probe.diagnostics.jsonl",
    "results/phase2/agentic_overlay_tests/probe_agentic_representative.diagnostics.jsonl",
    "results/phase2/current_submission/result_full_uncertainfix.diagnostics.jsonl",
]


def _get_top_issue_from_query_vec(query_vec: np.ndarray) -> str:
    issue_scores = query_vec[8:13]
    if len(issue_scores) < 5:
        return "unknown"
    return _ISSUE_DIM_NAMES[int(np.argmax(issue_scores))]


def _feature_scores_dict(sf) -> Dict[str, float]:
    scores = {
        "late_handover": float(getattr(sf, "score_late_handover", 0.0) or 0.0),
        "missing_neighbor": float(getattr(sf, "score_missing_neighbor", 0.0) or 0.0),
        "inter_freq_load": float(getattr(sf, "score_inter_freq_load", 0.0) or 0.0),
        "interference": float(getattr(sf, "score_interference", 0.0) or 0.0),
        "coverage_hole": float(getattr(sf, "score_coverage_hole", 0.0) or 0.0),
        "overshoot_overlap": float(getattr(sf, "score_overshoot_overlap", 0.0) or 0.0),
        "pdcch_issue": float(getattr(sf, "score_pdcch_issue", 0.0) or 0.0),
        "backhaul_server": float(getattr(sf, "score_backhaul_server", 0.0) or 0.0),
    }
    insuff = getattr(sf, "score_insufficient_data", None)
    if insuff is not None:
        scores["insufficient_data"] = float(insuff or 0.0)
    return scores


def _action_support_score(sf, action_type: str) -> float:
    scores = _feature_scores_dict(sf)
    candidates = _ACTION_ROOT_CAUSE_CANDIDATES.get(action_type, ("uncertain",))
    return max((scores.get(name, 0.0) for name in candidates), default=0.0)


def _choose_root_cause(sf, action_type: str) -> str:
    scores = _feature_scores_dict(sf)
    candidates = _ACTION_ROOT_CAUSE_CANDIDATES.get(action_type, ("uncertain",))
    ranked = sorted(
        ((name, scores.get(name, 0.0)) for name in candidates),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked:
        return "uncertain"
    best_name, best_score = ranked[0]
    if best_score <= 0.0:
        return candidates[0]
    return best_name


def _default_float(
    value: Optional[float],
    default: float,
    field_name: str,
    defaults_used: Dict[str, float],
) -> float:
    if value is None:
        defaults_used[field_name] = default
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        defaults_used[field_name] = default
        return float(default)


def _stable_unique(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _recommended_tool_sequence(branch: str, root_cause: str) -> List[str]:
    if branch in _PAIR_BRANCH_TOOLS:
        return list(_PAIR_BRANCH_TOOLS[branch])
    return list(_ROOT_CAUSE_TOOLS.get(root_cause, []))


def _probe_top_issue_name(diagnostic: Dict[str, Any]) -> str:
    top_issue = diagnostic.get("top_issue") or {}
    if isinstance(top_issue, dict) and top_issue.get("name"):
        return str(top_issue.get("name") or "").strip()
    nested = (((diagnostic.get("agent_trace") or {}).get("skills") or {}).get("top_issue") or {})
    if isinstance(nested, dict) and nested.get("name"):
        return str(nested.get("name") or "").strip()
    return ""


def _branch_from_root_action(root_cause: str, action: str) -> str:
    if root_cause in {"coverage_hole", "insufficient_data"} or action in {"lift_tilt", "insufficient_data"}:
        return "lift_tilt_vs_insufficient_data"
    if root_cause in {"missing_neighbor", "late_handover"} or action in {"add_neighbor", "decrease_a3"}:
        return "add_neighbor_vs_decrease_a3"
    return root_cause or action or "unknown"


def _normalize_probe_root_action(root_cause: str, action: str) -> Tuple[str, str]:
    root_cause = (root_cause or "uncertain").strip() or "uncertain"
    action = (action or "").strip()

    if action in _ROOT_CAUSE_PRIMARY_ACTION and action not in _ACTION_ROOT_CAUSE_CANDIDATES:
        action = _ROOT_CAUSE_PRIMARY_ACTION[action]
    if not action or action == root_cause:
        action = _ROOT_CAUSE_PRIMARY_ACTION.get(root_cause, action or "unknown")

    candidates = _ACTION_ROOT_CAUSE_CANDIDATES.get(action)
    if candidates and root_cause not in candidates:
        semantic_action = _ROOT_CAUSE_PRIMARY_ACTION.get(root_cause)
        if semantic_action:
            action = semantic_action
        elif root_cause == "uncertain":
            root_cause = candidates[0]

    return root_cause, action or "unknown"


def _safe_tool_sequence_from_probe(diagnostic: Dict[str, Any], branch: str, root_cause: str) -> List[str]:
    tool_names: List[str] = []
    sources = [
        diagnostic.get("agentic_tools_called") or [],
        diagnostic.get("tool_adjustments") or [],
        (((diagnostic.get("agent_trace") or {}).get("multi_turn_tool_use") or {}).get("conditional_tools") or []),
    ]
    for rows in sources:
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get("tool") or row.get("tool_name") or row.get("name") or row.get("api_name")
            if name:
                tool_names.append(str(name))

    filtered = [name for name in _stable_unique(tool_names) if name in _TRAJECTORY_TOOL_ALLOWLIST]
    if filtered:
        return filtered
    return _recommended_tool_sequence(branch, root_cause)


def _build_key_evidence(
    ps,
    sf,
    root_cause: str,
    action: str,
    mr_count: int,
    supporting_actions: List[str],
) -> List[str]:
    evidence: List[str] = []
    mob = sf.mobility
    tp = sf.throughput
    inter_freq = sf.inter_freq
    capacity = sf.capacity
    transport = sf.transport

    if mob.serving_sinr_db is not None:
        evidence.append(f"serving SINR {mob.serving_sinr_db:.1f} dB")
    if mob.serving_rsrp_dbm is not None:
        evidence.append(f"serving RSRP {mob.serving_rsrp_dbm:.1f} dBm")
    if (
        mob.best_neighbor_pci is not None
        and mob.best_neighbor_rsrp_dbm is not None
        and mob.neighbor_advantage_db is not None
    ):
        cfg = "configured" if mob.neighbor_in_config else "not configured"
        evidence.append(
            "best neighbor stronger by "
            f"{mob.neighbor_advantage_db:.1f} dB and {cfg}"
        )
    if mob.a3_margin_db is not None and root_cause in {"late_handover", "missing_neighbor"}:
        state = "handover threshold not met" if mob.a3_margin_db <= 0 else "handover threshold met"
        evidence.append(f"A3 margin {mob.a3_margin_db:+.1f} dB with {state}")
    if root_cause == "inter_freq_load" and inter_freq.max_dl_prb_util is not None:
        evidence.append(f"downlink PRB utilization peaks at {inter_freq.max_dl_prb_util:.0f}%")
    if root_cause == "pdcch_issue" and capacity.max_cce_util is not None:
        evidence.append(f"CCE utilization peaks at {capacity.max_cce_util:.0f}%")
    if root_cause == "backhaul_server" and transport.radio_healthy_flag and tp.worst_throughput_mbps is not None:
        evidence.append(
            "radio looks healthy while throughput still drops to "
            f"{tp.worst_throughput_mbps:.1f} Mbps"
        )
    if root_cause == "insufficient_data":
        evidence.append(f"MR sample count is only {mr_count}")
    if root_cause in {"coverage_hole", "overshoot_overlap"} and len(mob.visible_neighbor_pcis) >= 2:
        evidence.append(f"{len(mob.visible_neighbor_pcis)} neighbors are visible around the fault window")
    if root_cause == "interference" and mob.serving_sinr_db is not None and mob.serving_sinr_db < 0:
        evidence.append("poor SINR despite usable serving RSRP suggests interference")
    if supporting_actions:
        evidence.append("composite fix also includes " + ", ".join(supporting_actions))
    if not evidence:
        top_issue, top_score = sf.top_issue()
        evidence.append(f"top issue is {top_issue} with score {top_score:.2f}")
    return _stable_unique(evidence)[:5]


def _derive_confidence(sf, action_type: str, branch: str, mr_count: int) -> str:
    scores = sorted(_feature_scores_dict(sf).values(), reverse=True)
    top_score = scores[0] if scores else 0.0
    second_score = scores[1] if len(scores) > 1 else 0.0
    gap = top_score - second_score
    action_score = _action_support_score(sf, action_type)

    if branch in _PAIR_BRANCH_TOOLS:
        if action_score >= 0.65 and gap >= 0.05:
            return "medium"
        return "low"
    if action_type == "insufficient_data" and mr_count < 7 and action_score >= 0.65:
        return "high"
    if action_score >= 0.70 and gap >= 0.15:
        return "high"
    if action_score >= 0.35:
        return "medium"
    return "low"


def build_trajectory_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one Phase 1 training scenario into a prompt-safe trajectory record.

    The core schema follows docs/owner_d/trajectory_schema.{md,json}. Extra
    derived fields are included to help later indexing / debugging, but they
    never expose answer codes or answer label text.
    """
    ps = parse_scenario(raw)
    family = route(ps.tag, ps.num_base_stations)
    sf = run_all_skills(ps)
    scores = _feature_scores_dict(sf)
    risk = compute_risk_level(sf, family, scores)

    options = raw.get("task", {}).get("options", [])
    option_labels = {opt.get("id", ""): opt.get("label", "") for opt in options}
    answer_codes = [code.strip() for code in str(raw.get("answer", "")).split("|") if code.strip()]
    parsed_actions: List[Dict[str, Any]] = []
    for code in answer_codes:
        parsed = parse_option_label(option_labels.get(code, ""))
        parsed["c_code"] = code
        parsed["support_score"] = _action_support_score(sf, parsed.get("action_type", "unknown"))
        parsed_actions.append(parsed)

    parsed_actions.sort(
        key=lambda item: (
            -float(item.get("support_score", 0.0) or 0.0),
            str(item.get("action_type", "")),
            str(item.get("c_code", "")),
        )
    )
    primary = parsed_actions[0] if parsed_actions else {"action_type": "unknown", "target_cell": None}
    action = str(primary.get("action_type", "unknown") or "unknown")
    target_cell = primary.get("target_cell")
    target_role = ps.cell_roles.get(target_cell) if target_cell else None
    root_cause = _choose_root_cause(sf, action)
    branch = str(risk.get("branch") or root_cause or "unknown")

    defaults_used: Dict[str, float] = {}
    serving_sinr = _default_float(
        getattr(sf.mobility, "serving_sinr_db", None),
        -20.0,
        "serving_sinr",
        defaults_used,
    )
    serving_rsrp = _default_float(
        getattr(sf.mobility, "serving_rsrp_dbm", None),
        -140.0,
        "serving_rsrp",
        defaults_used,
    )
    best_neighbor_advantage = _default_float(
        getattr(sf.mobility, "neighbor_advantage_db", None),
        0.0,
        "best_neighbor_advantage",
        defaults_used,
    )
    mr_count = int(ps.mr.shape[0]) if getattr(ps, "mr", None) is not None and not ps.mr.empty else 0

    supporting_actions = _stable_unique([
        str(item.get("action_type", "unknown"))
        for item in parsed_actions[1:]
        if str(item.get("action_type", "unknown"))
    ])
    supporting_root_causes = _stable_unique([_choose_root_cause(sf, act) for act in supporting_actions])

    record = {
        "scenario_id": str(ps.scenario_id),
        "family": family.name,
        "tag": str(ps.tag),
        "branch": branch,
        "serving_sinr": round(serving_sinr, 4),
        "serving_rsrp": round(serving_rsrp, 4),
        "best_neighbor_advantage": round(best_neighbor_advantage, 4),
        "neighbor_in_config": bool(getattr(sf.mobility, "neighbor_in_config", False)),
        "mainlobe_outside": None,
        "mr_count": mr_count,
        "tool_sequence": _recommended_tool_sequence(branch, root_cause),
        "key_evidence": _build_key_evidence(ps, sf, root_cause, action, mr_count, supporting_actions),
        "root_cause": root_cause,
        "action": action,
        "target_role": target_role,
        "confidence": _derive_confidence(sf, action, branch, mr_count),
        "record_version": "trajectory_v1",
        "source_origin": "phase1_train",
        "answer_cardinality": len(answer_codes),
        "supporting_actions": supporting_actions,
        "supporting_root_causes": supporting_root_causes,
        "feature_scores": {k: round(v, 4) for k, v in sorted(scores.items())},
        "family_description": family.description,
        "risk_level": str(risk.get("level") or "low"),
        "risk_reasons": list(risk.get("reasons") or []),
        "field_defaults": defaults_used,
    }
    validate_trajectory_record(record)
    return record


def build_trajectory_records_from_train(
    train_json_path: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    with open(train_json_path, encoding="utf-8") as f:
        raw_examples = json.load(f)
    if limit is not None:
        raw_examples = raw_examples[:limit]
    return [build_trajectory_record(raw) for raw in raw_examples]


def validate_trajectory_record(record: Dict[str, Any]) -> None:
    required = {
        "scenario_id": str,
        "family": str,
        "tag": str,
        "branch": str,
        "serving_sinr": (float, int),
        "serving_rsrp": (float, int),
        "best_neighbor_advantage": (float, int),
        "neighbor_in_config": bool,
        "mr_count": int,
        "tool_sequence": list,
        "key_evidence": list,
        "root_cause": str,
        "action": str,
        "confidence": str,
    }
    for key, expected_type in required.items():
        if key not in record:
            raise ValueError(f"trajectory record missing required key: {key}")
        if not isinstance(record[key], expected_type):
            raise ValueError(f"trajectory record field {key} has invalid type: {type(record[key])}")
    if record["confidence"] not in {"high", "medium", "low"}:
        raise ValueError(f"invalid confidence value: {record['confidence']!r}")
    if record["mr_count"] < 0:
        raise ValueError("mr_count must be >= 0")
    if record["mainlobe_outside"] not in (True, False, None):
        raise ValueError("mainlobe_outside must be bool or null")
    if not record["key_evidence"]:
        raise ValueError("key_evidence must not be empty")
    if any(key in record for key in _UNSAFE_TRAJECTORY_KEYS):
        raise ValueError("unsafe answer-bearing keys leaked into trajectory record")
    dumped = json.dumps(record, ensure_ascii=False)
    if _UNSAFE_C_CODE_RE.search(dumped):
        raise ValueError("answer code leakage detected in trajectory record")


def build_probe_trajectory_record(diagnostic: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert one Phase 2 diagnostics row into a prompt-safe trajectory record.

    Returns None when the diagnostics row does not contain enough structured
    evidence to derive a stable trajectory memory item.
    """
    scenario_id = str(diagnostic.get("scenario_id", "")).strip()
    family = str(diagnostic.get("family", "")).strip()
    tag = str(diagnostic.get("tag", "")).strip()
    top_issue_name = _probe_top_issue_name(diagnostic)
    raw = diagnostic.get("raw_evidence_summary") or {}
    serving = raw.get("serving") or {}
    best_neighbor = raw.get("best_neighbor") or {}
    traffic = raw.get("traffic") or {}
    multi_turn = (diagnostic.get("agent_trace") or {}).get("multi_turn_tool_use") or {}
    agentic_diag = diagnostic.get("agentic_diagnosis") or {}
    prefetched = diagnostic.get("agentic_prefetched_evidence") or {}

    if not scenario_id or not family:
        return None

    llm_diag = diagnostic.get("llm_arbitrator", {}) or {}
    action = (
        str(agentic_diag.get("action") or "")
        or str(llm_diag.get("action") or "")
        or str(diagnostic.get("final_action_type") or "")
        or _ROOT_CAUSE_PRIMARY_ACTION.get(top_issue_name, "")
        or top_issue_name
        or "unknown"
    )
    root_cause = (
        str(agentic_diag.get("root_cause") or "")
        or str(llm_diag.get("root_cause") or "")
        or top_issue_name
        or "uncertain"
    )
    root_cause, action = _normalize_probe_root_action(root_cause, action)
    branch = (
        str(diagnostic.get("agentic_branch") or "")
        or str((diagnostic.get("agentic_risk") or {}).get("branch") or "")
        or str(llm_diag.get("branch") or "")
        or str((diagnostic.get("risk") or {}).get("branch") or "")
        or _branch_from_root_action(root_cause, action)
    )
    target_role = agentic_diag.get("target_role") or llm_diag.get("target_role")

    serving_sinr = (
        prefetched.get("serving_sinr_dB")
        if prefetched.get("serving_sinr_dB") not in (None, "")
        else serving.get("sinr_db")
    )
    serving_rsrp = (
        prefetched.get("serving_rsrp_dBm")
        if prefetched.get("serving_rsrp_dBm") not in (None, "")
        else serving.get("rsrp_dbm")
    )
    best_neighbor_advantage = (
        prefetched.get("neighbor_advantage_dB")
        if prefetched.get("neighbor_advantage_dB") not in (None, "")
        else best_neighbor.get("advantage_db")
    )
    if prefetched.get("neighbor_in_config") in (True, False):
        neighbor_in_config = bool(prefetched.get("neighbor_in_config"))
    else:
        neighbor_in_config = bool(best_neighbor.get("configured_neighbor_of_serving", False))

    mr_count = int(prefetched.get("mr_count") or 0)
    if mr_count == 0:
        mr_count = int(len(raw.get("mr_neighbor_rows") or []))
    if mr_count == 0:
        mr_count = int((raw.get("best_neighbor") or {}).get("mr_count", 0) or 0)

    tool_sequence = _safe_tool_sequence_from_probe(diagnostic, branch, root_cause)

    evidence: List[str] = []
    if serving_sinr not in (None, ""):
        evidence.append(f"serving SINR {float(serving_sinr):.1f} dB")
    if serving_rsrp not in (None, ""):
        evidence.append(f"serving RSRP {float(serving_rsrp):.1f} dBm")
    if best_neighbor_advantage not in (None, ""):
        cfg = "configured" if neighbor_in_config else "not configured"
        evidence.append(f"best neighbor stronger by {float(best_neighbor_advantage):.1f} dB and {cfg}")
    feature_scores = prefetched.get("feature_scores") or {}
    if root_cause in feature_scores:
        evidence.append(f"{root_cause} feature score {float(feature_scores[root_cause]):.2f}")
    max_prb = traffic.get("max_dl_prb_util")
    if max_prb not in (None, ""):
        evidence.append(f"downlink PRB utilization peaks at {float(max_prb):.0f}%")
    max_cce = traffic.get("max_cce_util")
    if max_cce not in (None, ""):
        evidence.append(f"CCE utilization peaks at {float(max_cce):.0f}%")
    if mr_count > 0:
        evidence.append(f"probe trace captured {mr_count} MR neighbor rows")
    if multi_turn.get("api_call_count"):
        evidence.append(f"probe trace used {int(multi_turn['api_call_count'])} tool calls")
    evidence = _stable_unique(evidence)
    if not evidence:
        return None

    confidence = str(agentic_diag.get("evidence_confidence") or llm_diag.get("confidence") or "").strip()
    if confidence not in {"high", "medium", "low"}:
        confidence = "high" if tool_sequence and mr_count > 0 else "medium"
        if len(evidence) <= 2:
            confidence = "low"

    record = {
        "scenario_id": scenario_id,
        "family": family,
        "tag": tag or ("multiple-answer" if family in {"F2", "F3"} else "single-answer"),
        "branch": branch,
        "serving_sinr": float(serving_sinr if serving_sinr not in (None, "") else -20.0),
        "serving_rsrp": float(serving_rsrp if serving_rsrp not in (None, "") else -140.0),
        "best_neighbor_advantage": float(best_neighbor_advantage if best_neighbor_advantage not in (None, "") else 0.0),
        "neighbor_in_config": neighbor_in_config,
        "mainlobe_outside": None,
        "mr_count": mr_count,
        "tool_sequence": tool_sequence,
        "key_evidence": evidence[:5],
        "root_cause": root_cause,
        "action": action,
        "target_role": target_role,
        "confidence": confidence,
        "record_version": "trajectory_v1_probe",
        "source_origin": "phase2_probe",
        "probe_source_file": str(diagnostic.get("_source_file", "")),
    }
    validate_trajectory_record(record)
    return record


def load_probe_trajectory_records(diagnostics_paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path_str in diagnostics_paths or _PROBE_DIAGNOSTICS_CANDIDATES:
        path = Path(path_str)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_source_file"] = str(path)
                record = build_probe_trajectory_record(row)
                if record is not None:
                    records.append(record)
    return records


@dataclass
class RAGHit:
    scenario_id: str
    family: str
    features: Dict[str, float]
    gt_c_code: str
    gt_action_type: str
    similarity: float
    prompt_snippet: str
    is_confusion_pair: bool = False
    branch: str = ""
    root_cause: str = ""
    target_role: Optional[str] = None
    confidence: str = ""
    tool_sequence: List[str] = field(default_factory=list)
    key_evidence: List[str] = field(default_factory=list)


class StructuredCaseRAG:
    """
    Builds and queries a structured case index over the 2000 training examples.

    Each entry stores:
      - family_name: F1/F2/F3/F4
      - answer: ground-truth C-code string
      - feature_vector: 13-dim numeric summary
      - text_summary: compact human-readable evidence string
      - option_labels: {c_code: label} for the correct answer codes
    """

    FEATURE_DIM = 16
    CACHE_DIR = ".rag_cache_v2"
    CACHE_VERSION = 2
    FAISS_EXPORT_PATH = "artifacts/owner_d/index/trajectory_index.faiss"

    def __init__(self, train_json_path: str, trajectory_jsonl_path: Optional[str] = None):
        self.train_json_path = train_json_path
        self.trajectory_jsonl_path = trajectory_jsonl_path or self._resolve_trajectory_jsonl_path(train_json_path)
        self.cases: List[Dict[str, Any]] = []
        self.family_indices: Dict[str, List[int]] = {}  # family_name → list of case indices
        self.branch_indices: Dict[str, List[int]] = {}
        self.root_cause_indices: Dict[str, List[int]] = {}
        self._action_indices: Dict[str, List[int]] = {}
        self._confusion_index: Dict[str, Dict[str, List[str]]] = {}
        self.feature_matrix: Optional[np.ndarray] = None  # shape (N, FEATURE_DIM)
        self._faiss_index = None

    @staticmethod
    def _resolve_trajectory_jsonl_path(train_json_path: str) -> str:
        train_path = Path(train_json_path).resolve()
        candidates = [
            Path(__file__).resolve().parent / "artifacts/owner_d/data/trajectory_memory_combined.jsonl",
            Path(__file__).resolve().parent / "artifacts/owner_d/data/trajectory_examples.jsonl",
            Path(__file__).resolve().parent / "trajectory_examples.jsonl",
            train_path.parent / "trajectory_memory_combined.jsonl",
            train_path.parent / "trajectory_examples.jsonl",
        ]
        try:
            candidates.append(train_path.parents[2] / "artifacts/owner_d/data/trajectory_memory_combined.jsonl")
            candidates.append(train_path.parents[2] / "artifacts/owner_d/data/trajectory_examples.jsonl")
            candidates.append(train_path.parents[2] / "trajectory_memory_combined.jsonl")
            candidates.append(train_path.parents[2] / "trajectory_examples.jsonl")
        except IndexError:
            pass
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0])

    def _source_hash(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        digest.update(self._sha256_file(self.train_json_path).encode("utf-8"))
        traj_path = Path(self.trajectory_jsonl_path)
        if traj_path.exists():
            digest.update(self._sha256_file(str(traj_path)).encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _family_stub(family_name: str):
        @dataclass(frozen=True)
        class _FamilyStub:
            name: str
            is_multiple: bool

        return _FamilyStub(
            name=family_name,
            is_multiple=family_name in {"F2", "F3"},
        )

    def _load_trajectory_records(self) -> Dict[str, Dict[str, Any]]:
        traj_path = Path(self.trajectory_jsonl_path)
        if traj_path.exists():
            records: Dict[str, Dict[str, Any]] = {}
            with traj_path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    validate_trajectory_record(record)
                    records[str(record["scenario_id"])] = record
            if records:
                return records

        with open(self.train_json_path, encoding="utf-8") as f:
            raw_examples = json.load(f)
        return {
            str(record["scenario_id"]): record
            for record in (build_trajectory_record(raw) for raw in raw_examples)
        }

    @staticmethod
    def _format_prompt_safe_snippet(case: Dict[str, Any], sim: float, is_confusion: bool = False) -> str:
        evidence = list(case.get("key_evidence") or [])
        evidence_text = "; ".join(evidence[:2]) if evidence else "no key evidence"
        tools = list(case.get("tool_sequence") or [])
        tools_text = ", ".join(tools[:2]) if tools else "no tools"
        branch = case.get("branch") or "unknown"
        root = case.get("root_cause") or "unknown"
        action = case.get("action") or case.get("action_type") or "unknown"
        confidence = case.get("confidence") or "unknown"
        target_role = case.get("target_role")
        role_text = f" target_role={target_role}" if target_role else ""
        confusion_text = " confusion_pair" if is_confusion else ""
        return (
            f"[{case.get('family', '?')}|{branch}] root={root} action={action}{role_text} "
            f"confidence={confidence} sim={sim:.3f}{confusion_text} "
            f"evidence={evidence_text} tools={tools_text}"
        )[:240]

    @staticmethod
    def _format_prompt_safe_case_block(case: Dict[str, Any], sim: float, rank: Optional[int] = None) -> str:
        title = f"Case {rank}" if rank is not None else "Case"
        branch = case.get("branch") or "unknown"
        family = case.get("family") or "?"
        root = case.get("root_cause") or "unknown"
        action = case.get("action") or case.get("action_type") or "unknown"
        target_role = case.get("target_role") or "null"
        confidence = case.get("confidence") or "unknown"
        evidence = list(case.get("key_evidence") or [])
        tools = list(case.get("tool_sequence") or [])
        evidence_text = "; ".join(str(x) for x in evidence[:3]) if evidence else "no key evidence"
        tools_text = ", ".join(str(x) for x in tools[:3]) if tools else "no tools"
        return (
            f"### {title} [{family}] similarity={sim:.3f}\n"
            f"- branch: {branch}\n"
            f"- root_cause: {root}\n"
            f"- action: {action}\n"
            f"- target_role: {target_role}\n"
            f"- confidence: {confidence}\n"
            f"- key_evidence: {evidence_text}\n"
            f"- tool_sequence: {tools_text}"
        )

    @staticmethod
    def _branch_match_rank(
        case: Dict[str, Any],
        query_branch: Optional[str],
        query_root_cause: Optional[str],
    ) -> int:
        if query_branch and case.get("branch") == query_branch:
            return 2
        if query_root_cause and case.get("root_cause") == query_root_cause:
            return 1
        return 0

    def _sort_case_candidates(
        self,
        candidates: List[Tuple[int, float]],
        query_branch: Optional[str] = None,
        query_root_cause: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        return sorted(
            candidates,
            key=lambda item: (
                -self._branch_match_rank(self.cases[item[0]], query_branch, query_root_cause),
                -round(float(item[1]), 12),
                str(self.cases[item[0]].get("scenario_id", "")),
            ),
        )

    def _infer_query_branch(self, sf, family_name: str, branch: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        if branch:
            if branch in _PAIR_BRANCH_TOOLS:
                if "insufficient" in branch:
                    return branch, "coverage_hole"
                if "decrease_a3" in branch:
                    return branch, "late_handover"
            return branch, branch

        scores = _feature_scores_dict(sf)
        risk = compute_risk_level(sf, self._family_stub(family_name), scores)
        inferred_branch = risk.get("branch")
        top_issue = sf.top_issue()[0] if hasattr(sf, "top_issue") and callable(sf.top_issue) else None
        if inferred_branch:
            return str(inferred_branch), str(top_issue or inferred_branch)
        return None, str(top_issue or "uncertain")

    def _select_family_indices(
        self,
        family_name: str,
        query_branch: Optional[str],
        query_root_cause: Optional[str],
    ) -> Tuple[List[int], str]:
        family_idxs = list(self.family_indices.get(family_name, list(range(len(self.cases)))))
        if not family_idxs:
            return list(range(len(self.cases))), "global_fallback"

        if query_branch:
            branch_idxs = [
                idx for idx in self.branch_indices.get(query_branch, [])
                if idx in set(family_idxs)
            ]
            if branch_idxs:
                return sorted(branch_idxs), "branch"

        if query_root_cause:
            root_idxs = [
                idx for idx in self.root_cause_indices.get(query_root_cause, [])
                if idx in set(family_idxs)
            ]
            if root_idxs:
                return sorted(root_idxs), "root_cause"

        return sorted(family_idxs), "family"

    def _build_case(self, raw: Dict[str, Any], trajectory_record: Dict[str, Any], vec: np.ndarray, idx: int) -> Dict[str, Any]:
        case = dict(trajectory_record)
        case.update({
            "idx": idx,
            "scenario_id": raw.get("scenario_id", trajectory_record.get("scenario_id", "")),
            "family": trajectory_record.get("family", self._get_family_name(raw)),
            "tag": trajectory_record.get("tag", raw.get("tag", "single-answer")),
            "feature_vector": vec,
            "action_type": trajectory_record.get("action", "unknown"),
            "answer": raw.get("answer", ""),
            "prompt_snippet_safe": self._format_prompt_safe_snippet(trajectory_record, 0.0),
        })
        return case

    # ──────────────────────────────────────────────────────────────────────────
    # Feature extraction (works on raw training JSON — no HTTP)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_feature_vector(raw: Dict[str, Any]) -> np.ndarray:
        """
        Extract a 16-dim numeric feature vector from a raw training example.
        Applies safe defaults (0.0) for missing values.
        """
        import io
        import pandas as pd

        data = raw.get("data", {})

        def _csv(key: str) -> pd.DataFrame:
            s = data.get(key, "")
            if not s:
                return pd.DataFrame()
            try:
                return pd.read_csv(io.StringIO(s), sep="|")
            except Exception:
                return pd.DataFrame()

        up = _csv("user_plane_data")
        nc = _csv("network_configuration_data")
        tr = _csv("traffic_data")

        # ── Throughput features ──
        worst_tp = 0.0
        drop_magnitude = 0.0
        if not up.empty:
            tp_col = next((c for c in up.columns if "DL Throughput" in c), None)
            if tp_col:
                tp = pd.to_numeric(up[tp_col], errors="coerce").dropna()
                if not tp.empty:
                    worst_tp = float(tp.min())
                    drop_magnitude = float(tp.max() - tp.min())

        # ── RF features at worst timestamp ──
        serving_rsrp = 0.0
        serving_sinr = 0.0
        neighbor_advantage = 0.0
        a3_margin = 0.0
        num_neighbors = 0

        if not up.empty:
            tp_col2 = next((c for c in up.columns if "DL Throughput" in c), None)
            if tp_col2:
                tp_vals = pd.to_numeric(up[tp_col2], errors="coerce")
                worst_idx = int(tp_vals.idxmin()) if not tp_vals.isna().all() else 0
                worst_row = up.iloc[worst_idx]

                rsrp_col = next((c for c in up.columns if "SS-RSRP" in c and "Serving" in c), None)
                sinr_col = next((c for c in up.columns if "SS-SINR" in c and "Serving" in c), None)
                if rsrp_col:
                    try:
                        serving_rsrp = float(worst_row[rsrp_col])
                    except (ValueError, TypeError):
                        pass
                if sinr_col:
                    try:
                        serving_sinr = float(worst_row[sinr_col])
                    except (ValueError, TypeError):
                        pass

                # Neighbour RSRP columns
                n_pci_cols = [c for c in up.columns if "Neighbor" in c and "PCI" in c and "Serving" not in c]
                n_rsrp_cols = [c for c in up.columns if "Neighbor" in c and "RSRP" in c and "Serving" not in c]
                num_neighbors = len(n_pci_cols)
                best_n_rsrp = -999.0
                for i, rc in enumerate(n_rsrp_cols):
                    try:
                        v = float(worst_row[rc])
                        if v > best_n_rsrp:
                            best_n_rsrp = v
                    except (ValueError, TypeError):
                        pass
                if best_n_rsrp > -999:
                    neighbor_advantage = best_n_rsrp - serving_rsrp

        # ── A3 margin from net_config ──
        if not nc.empty:
            pci_col = next((c for c in nc.columns if c.strip() == "PCI"), None)
            a3_col = next((c for c in nc.columns if "A3Offset" in c), None)
            hyst_col = next((c for c in nc.columns if "A3Hyst" in c), None)
            if a3_col and hyst_col and not nc.empty:
                try:
                    a3_offset = float(nc[a3_col].iloc[0]) * 0.5
                    a3_hyst = float(nc[hyst_col].iloc[0]) * 0.5
                    a3_margin = neighbor_advantage - (a3_offset + a3_hyst)
                except (ValueError, TypeError, IndexError):
                    pass

        # ── Traffic features ──
        max_prb_util = 0.0
        if not tr.empty:
            prb_col = next((c for c in tr.columns if "Downlink PRB" in c), None)
            if prb_col:
                vals = pd.to_numeric(tr[prb_col], errors="coerce").dropna()
                max_prb_util = float(vals.max()) if not vals.empty else 0.0

        # ── Issue scores (simplified, fast version of feature_engine) ──
        late_ho_score = 0.0
        missing_nb_score = 0.0
        interference_score = 0.0
        coverage_hole_score = 0.0
        inter_freq_score = 0.0

        if neighbor_advantage > 2 and a3_margin <= 0:
            late_ho_score = min(1.0, abs(a3_margin) / 5.0 + 0.3)
        if serving_sinr < 0 and serving_rsrp > -110:
            interference_score = min(1.0, abs(serving_sinr) / 10.0 + 0.3)
        if serving_rsrp < -110:
            coverage_hole_score = min(1.0, (-110 - serving_rsrp) / 20.0 + 0.5)
        if max_prb_util > 70:
            inter_freq_score = min(1.0, (max_prb_util - 70) / 30.0)

        # Proxy for missing_neighbor: count configured neighbors from net_config
        # and compare against visible MR neighbors; gap → missing neighbor signal.
        if not nc.empty and num_neighbors > 0:
            import re as _re
            nb_col = next(
                (c for c in nc.columns if "Neighbor Cell" in c and "gNodeB" in c), None
            )
            if nb_col:
                nb_str = str(nc[nb_col].iloc[0]) if len(nc) > 0 else ""
                num_configured = len(_re.findall(r'\[', nb_str))
                gap = max(0, num_neighbors - num_configured)
                if gap > 0:
                    missing_nb_score = min(1.0, gap / max(num_neighbors, 1))

        vec = np.array([
            worst_tp / 500.0,            # normalised throughput
            drop_magnitude / 500.0,      # normalised drop magnitude
            (serving_rsrp + 140) / 60.0, # normalised RSRP [-140, -80] → [0,1]
            (serving_sinr + 20) / 40.0,  # normalised SINR [-20, 20] → [0,1]
            neighbor_advantage / 20.0,   # normalised advantage
            (a3_margin + 10) / 20.0,     # normalised A3 margin [-10, 10] → [0,1]
            max_prb_util / 100.0,        # normalised PRB util
            num_neighbors / 5.0,         # normalised neighbor count
            late_ho_score,
            missing_nb_score,
            interference_score,
            coverage_hole_score,
            inter_freq_score,
            0.0,                         # outside_mainlobe_ratio (train data has no tool evidence)
            0.0,                         # tilt_angle_mean
            0.0,                         # overlap_ratio_scalar
        ], dtype=np.float32)

        # Clip to [0, 1]
        return np.clip(vec, 0.0, 1.0)

    @staticmethod
    def _coerce_feature_dim(vec: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Keep legacy 13-dim query vectors working against newer cached indices by
        zero-padding missing tail dimensions inside rag_retriever_v2.
        """
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.shape[0] == target_dim:
            return arr
        if arr.shape[0] < target_dim:
            return np.pad(arr, (0, target_dim - arr.shape[0]), mode="constant").astype(np.float32)
        return arr[:target_dim].astype(np.float32)

    @staticmethod
    def _get_family_name(raw: Dict[str, Any]) -> str:
        tag = raw.get("tag", "single-answer")
        ctx = raw.get("context", {})
        num_bs = int(str(ctx.get("wireless_network_information", {}).get("num_base_stations", "4")))
        is_multi = "multiple" in tag.lower()
        if num_bs == 5:
            return "F3"
        if num_bs == 13:
            return "F4"
        return "F2" if is_multi else "F1"

    @staticmethod
    def _sha256_file(path: str) -> str:
        import hashlib

        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _build_faiss_index(self, matrix: np.ndarray):
        """Build an optional FAISS cosine-similarity index."""
        if not _FAISS_AVAILABLE:
            return None
        norm_matrix = matrix.copy().astype(np.float32)
        _faiss.normalize_L2(norm_matrix)
        index = _faiss.IndexFlatIP(norm_matrix.shape[1])
        index.add(norm_matrix)
        return index

    def _cosine_sim_to_subset(self, query: np.ndarray, subset_idxs: List[int]) -> np.ndarray:
        """Return cosine similarities between query and feature_matrix[subset_idxs]."""
        if not subset_idxs:
            return np.array([], dtype=np.float32)
        sub = self.feature_matrix[subset_idxs].astype(np.float32)
        q = self._coerce_feature_dim(query, sub.shape[1]).reshape(1, -1)
        norms_sub = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8
        norm_q = float(np.linalg.norm(q)) + 1e-8
        return ((sub / norms_sub) @ (q / norm_q).T).flatten()

    @staticmethod
    def _sf_to_feature_vec(sf) -> np.ndarray:
        """
        Convert ScenarioFeatures-like input to the 16-dim float32 vector used by
        structured retrieval, with the last 3 dims reserved for tool evidence.
        """
        tp = sf.throughput
        mob = sf.mobility
        ifreq = sf.inter_freq
        base = np.array([
            (tp.worst_throughput_mbps or 0.0) / 500.0,
            (tp.throughput_drop_magnitude or 0.0) / 500.0,
            ((mob.serving_rsrp_dbm or -140.0) + 140.0) / 60.0,
            ((mob.serving_sinr_db or -20.0) + 20.0) / 40.0,
            (mob.neighbor_advantage_db or 0.0) / 20.0,
            ((mob.a3_margin_db or -10.0) + 10.0) / 20.0,
            (ifreq.max_dl_prb_util or 0.0) / 100.0,
            float(len(getattr(mob, "visible_neighbor_pcis", []) or [])) / 5.0,
            sf.score_late_handover,
            sf.score_missing_neighbor,
            sf.score_interference,
            sf.score_coverage_hole,
            sf.score_inter_freq_load,
        ], dtype=np.float32)

        tool_ev = getattr(sf, "tool_evidence", None)
        outside_ratio = 0.0
        tilt_mean = 0.0
        overlap_scalar = 0.0

        if tool_ev is not None:
            ml_flags = getattr(tool_ev, "mainlobe_flags", {}) or {}
            tilt_angles = getattr(tool_ev, "tilt_angles", {}) or {}
            raw_overlap = getattr(tool_ev, "overlap_ratio", None)

            n = max(len(ml_flags), 1)
            outside_ratio = sum(1 for v in ml_flags.values() if bool(v)) / n

            if tilt_angles:
                tilt_mean = sum(tilt_angles.values()) / len(tilt_angles) / 20.0

            if isinstance(raw_overlap, dict):
                overlap_scalar = float(max(raw_overlap.values(), default=0.0))
            elif isinstance(raw_overlap, (int, float)) and raw_overlap is not None:
                overlap_scalar = float(raw_overlap)

        tool_vec = np.array([outside_ratio, tilt_mean, overlap_scalar], dtype=np.float32)
        return np.clip(np.concatenate([base, tool_vec]), 0.0, 1.0)

    @staticmethod
    def _deduplicate(hits: List[dict], query_vec: np.ndarray, threshold: float = 0.98) -> List[dict]:
        """
        Drop hits whose feature_vector is too close to query_vec.
        This is mainly used to remove self-hits / near-duplicates.
        """
        kept = []
        for hit in hits:
            vec = hit["feature_vector"].astype(np.float32)
            aligned_query = StructuredCaseRAG._coerce_feature_dim(query_vec, vec.shape[0])
            q_norm = aligned_query / (np.linalg.norm(aligned_query) + 1e-8)
            v_norm = vec / (np.linalg.norm(vec) + 1e-8)
            if float(np.dot(q_norm, v_norm)) < threshold:
                kept.append(hit)
        return kept

    def _case_to_raghit(self, case: dict, sim: float, is_confusion: bool = False) -> "RAGHit":
        """
        Convert internal case dict → public RAGHit.
        Note: in the current codebase RAGHit.features is a compact dict, not a full
        ScenarioFeatures object.
        """
        vec = case["feature_vector"]
        features = {
            "worst_tp": float(vec[0]),
            "serving_rsrp": float(vec[2]),
            "serving_sinr": float(vec[3]),
            "neighbor_advantage": float(vec[4]),
            "max_prb_util": float(vec[6]),
            "score_late_handover": float(vec[8]),
            "score_missing_neighbor": float(vec[9]),
            "score_coverage_hole": float(vec[11]),
        }
        snippet = self._format_prompt_safe_snippet(case, sim, is_confusion=is_confusion)
        return RAGHit(
            scenario_id=case["scenario_id"],
            family=str(case.get("family", "")),
            features=features,
            gt_c_code=case.get("answer", ""),
            gt_action_type=case.get("action_type", "unknown"),
            similarity=sim,
            prompt_snippet=snippet,
            is_confusion_pair=is_confusion,
            branch=str(case.get("branch", "")),
            root_cause=str(case.get("root_cause", "")),
            target_role=case.get("target_role"),
            confidence=str(case.get("confidence", "")),
            tool_sequence=list(case.get("tool_sequence") or []),
            key_evidence=list(case.get("key_evidence") or []),
        )

    def _build_confusion_index(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Build a seed -> opposite-class-neighbor mapping for the two most confusing
        action buckets: lift_tilt and insufficient_data.

        Returns:
            {
                "lift_tilt": {seed_sid: [opp_sid1, opp_sid2, ...]},
                "insufficient_data": {seed_sid: [opp_sid1, opp_sid2, ...]},
            }
        """
        result = {"lift_tilt": {}, "insufficient_data": {}}

        lt_idxs = self._action_indices.get("lift_tilt", [])
        id_idxs = self._action_indices.get("insufficient_data", [])
        if not lt_idxs or not id_idxs:
            return result

        lt_matrix = self.feature_matrix[lt_idxs].astype(np.float32)
        id_matrix = self.feature_matrix[id_idxs].astype(np.float32)

        lt_norms = np.linalg.norm(lt_matrix, axis=1, keepdims=True) + 1e-8
        id_norms = np.linalg.norm(id_matrix, axis=1, keepdims=True) + 1e-8
        sims = (lt_matrix / lt_norms) @ (id_matrix / id_norms).T

        for row_i, row in enumerate(sims):
            top2 = np.argsort(-row)[:2]
            seed_sid = self.cases[lt_idxs[row_i]]["scenario_id"]
            result["lift_tilt"][seed_sid] = [self.cases[id_idxs[j]]["scenario_id"] for j in top2]

        for row_i, row in enumerate(sims.T):
            top2 = np.argsort(-row)[:2]
            seed_sid = self.cases[id_idxs[row_i]]["scenario_id"]
            result["insufficient_data"][seed_sid] = [self.cases[lt_idxs[j]]["scenario_id"] for j in top2]

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Index building
    # ──────────────────────────────────────────────────────────────────────────

    def build_index(self, force_rebuild: bool = False) -> None:
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(self.CACHE_DIR, "rag_v2_cases.pkl")
        matrix_path = os.path.join(self.CACHE_DIR, "rag_v2_matrix.npy")
        confuse_path = os.path.join(self.CACHE_DIR, "confusion_index.pkl")
        hash_path = os.path.join(self.CACHE_DIR, "data_hash.txt")
        faiss_path = os.path.join(self.CACHE_DIR, "rag_v2_faiss.index")
        export_faiss_path = self.FAISS_EXPORT_PATH

        current_hash = self._source_hash()
        cached_hash = ""
        if os.path.exists(hash_path):
            with open(hash_path, "r", encoding="utf-8") as f:
                cached_hash = f.read().strip()

        cache_valid = (
            not force_rebuild
            and current_hash == cached_hash
            and os.path.exists(cache_path)
            and os.path.exists(matrix_path)
        )

        if cache_valid:
            print("[RAG V2] Loading cached structured index...")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            if data.get("cache_version") != self.CACHE_VERSION:
                print("[RAG V2] Cache version mismatch, rebuilding...")
                return self.build_index(force_rebuild=True)
            self.cases = data["cases"]
            self.family_indices = data["family_indices"]
            self.branch_indices = data.get("branch_indices", {})
            self.root_cause_indices = data.get("root_cause_indices", {})
            self._action_indices = data.get("action_indices", {})
            self.feature_matrix = np.load(matrix_path)
            if self.feature_matrix.ndim != 2 or self.feature_matrix.shape[1] != self.FEATURE_DIM:
                print(
                    "[RAG V2] Feature dim mismatch "
                    f"({self.feature_matrix.shape} vs *x{self.FEATURE_DIM}), rebuilding..."
                )
                return self.build_index(force_rebuild=True)
            if os.path.exists(confuse_path):
                with open(confuse_path, "rb") as f:
                    self._confusion_index = pickle.load(f)
            else:
                self._confusion_index = {}
            if _FAISS_AVAILABLE and os.path.exists(faiss_path):
                self._faiss_index = _faiss.read_index(faiss_path)
            else:
                self._faiss_index = None
            if _FAISS_AVAILABLE and self._faiss_index is not None:
                try:
                    _faiss.write_index(self._faiss_index, export_faiss_path)
                except Exception:
                    pass
            print(f"[RAG V2] Loaded {len(self.cases)} cases.")
            return

        print("[RAG V2] Building structured feature index...")
        with open(self.train_json_path, encoding="utf-8") as f:
            raw_examples = json.load(f)
        trajectory_by_id = self._load_trajectory_records()

        self.cases = []
        self.family_indices = {}
        self.branch_indices = {}
        self.root_cause_indices = {}
        self._action_indices = {}
        self._confusion_index = {}
        self._faiss_index = None
        vectors = []

        for i, raw in enumerate(raw_examples):
            vec = self._extract_feature_vector(raw)
            vectors.append(vec)
            scenario_id = raw.get("scenario_id", "")
            trajectory_record = trajectory_by_id.get(str(scenario_id))
            if trajectory_record is None:
                trajectory_record = build_trajectory_record(raw)
            family_name = trajectory_record.get("family", self._get_family_name(raw))
            case = self._build_case(raw, trajectory_record, vec, i)
            global_idx = len(self.cases)
            action_type = case.get("action_type", "unknown")
            self._action_indices.setdefault(action_type, []).append(global_idx)
            self.cases.append(case)
            self.family_indices.setdefault(family_name, []).append(global_idx)
            self.branch_indices.setdefault(str(case.get("branch", "unknown")), []).append(global_idx)
            self.root_cause_indices.setdefault(str(case.get("root_cause", "uncertain")), []).append(global_idx)

            if (i + 1) % 500 == 0:
                print(f"[RAG V2] Indexed {i + 1}/{len(raw_examples)}...")

        self.feature_matrix = np.stack(vectors, axis=0).astype(np.float32)

        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "cache_version": self.CACHE_VERSION,
                    "cases": self.cases,
                    "family_indices": self.family_indices,
                    "branch_indices": self.branch_indices,
                    "root_cause_indices": self.root_cause_indices,
                    "action_indices": self._action_indices,
                },
                f,
            )
        np.save(matrix_path, self.feature_matrix)
        self._confusion_index = self._build_confusion_index()
        with open(confuse_path, "wb") as f:
            pickle.dump(self._confusion_index, f)
        self._faiss_index = self._build_faiss_index(self.feature_matrix)
        if self._faiss_index is not None:
            _faiss.write_index(self._faiss_index, faiss_path)
            try:
                Path(export_faiss_path).parent.mkdir(parents=True, exist_ok=True)
                _faiss.write_index(self._faiss_index, export_faiss_path)
            except Exception:
                pass
        with open(hash_path, "w", encoding="utf-8") as f:
            f.write(current_hash)
        print(f"[RAG V2] Index built: {len(self.cases)} cases, {self.FEATURE_DIM} features each.")

    # ──────────────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def _retrieve_raw(
        self,
        query_feature_vector: np.ndarray,
        family_name: str,
        top_k: int = 3,
        branch: Optional[str] = None,
        root_cause: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top_k structurally similar training cases within the same family.

        Args:
            query_feature_vector: 13-dim normalised feature vector for the test scenario
            family_name:          "F1" | "F2" | "F3" | "F4"
            top_k:                number of cases to retrieve

        Returns:
            List of case dicts ordered by similarity (most similar first).
        """
        if self.feature_matrix is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        family_idxs, _ = self._select_family_indices(family_name, branch, root_cause)
        if not family_idxs:
            family_idxs = list(range(len(self.cases)))

        sub_matrix = self.feature_matrix[family_idxs]   # (M, FEATURE_DIM)
        q = self._coerce_feature_dim(query_feature_vector, sub_matrix.shape[1]).reshape(1, -1)

        # Cosine similarity (vectors are already [0,1] so L2 is also reasonable)
        norms_sub = np.linalg.norm(sub_matrix, axis=1, keepdims=True) + 1e-8
        norms_q = np.linalg.norm(q) + 1e-8
        sims = (sub_matrix / norms_sub) @ (q / norms_q).T   # (M, 1)
        sims = sims.flatten()

        ranked = self._sort_case_candidates(
            [(family_idxs[local_idx], float(sims[local_idx])) for local_idx in range(len(family_idxs))],
            query_branch=branch,
            query_root_cause=root_cause,
        )
        results = []
        for global_idx, sim in ranked[:top_k]:
            case = dict(self.cases[global_idx])
            case["_similarity"] = float(sim)
            results.append(case)
        return results

    def _flat_retrieve(
        self,
        query_vec: np.ndarray,
        family_idxs: List[int],
        k: int,
        query_branch: Optional[str] = None,
        query_root_cause: Optional[str] = None,
    ) -> List[dict]:
        """Plain top-k cosine retrieval within the given family bucket."""
        sims = self._cosine_sim_to_subset(query_vec, family_idxs)
        ranked = self._sort_case_candidates(
            [(family_idxs[li], float(sims[li])) for li in range(len(family_idxs))],
            query_branch=query_branch,
            query_root_cause=query_root_cause,
        )
        out = []
        for gi, sim in ranked[:k]:
            case = dict(self.cases[gi])
            case["_similarity"] = float(sim)
            out.append(case)
        return out

    def _stratified_retrieve(
        self,
        query_vec: np.ndarray,
        family_idxs: List[int],
        k: int,
        query_branch: Optional[str] = None,
        query_root_cause: Optional[str] = None,
    ) -> List[dict]:
        """
        Try to draw at least one example from each major F1 action bucket, then
        backfill by overall similarity.
        """
        family_set = set(family_idxs)
        quota = max(2, k // max(len(_F1_ACTION_BUCKETS), 1))

        collected: List[dict] = []
        used: set[int] = set()

        for bucket in _F1_ACTION_BUCKETS:
            bucket_idxs = [gi for gi in self._action_indices.get(bucket, []) if gi in family_set]
            if not bucket_idxs:
                continue
            sims = self._cosine_sim_to_subset(query_vec, bucket_idxs)
            ranked = self._sort_case_candidates(
                [(bucket_idxs[li], float(sims[li])) for li in range(len(bucket_idxs))],
                query_branch=query_branch,
                query_root_cause=query_root_cause,
            )
            for gi, sim in ranked[:quota]:
                if gi in used:
                    continue
                case = dict(self.cases[gi])
                case["_similarity"] = float(sim)
                collected.append(case)
                used.add(gi)

        remaining = k - len(collected)
        if remaining > 0:
            all_sims = self._cosine_sim_to_subset(query_vec, family_idxs)
            ranked = self._sort_case_candidates(
                [(family_idxs[li], float(all_sims[li])) for li in range(len(family_idxs))],
                query_branch=query_branch,
                query_root_cause=query_root_cause,
            )
            for gi, sim in ranked:
                if gi in used:
                    continue
                case = dict(self.cases[gi])
                case["_similarity"] = float(sim)
                collected.append(case)
                used.add(gi)
                remaining -= 1
                if remaining == 0:
                    break

        return collected

    def _get_confusion_pairs(
        self,
        query_vec: np.ndarray,
        family_idxs: List[int],
        query_branch: Optional[str] = None,
        query_root_cause: Optional[str] = None,
    ) -> List["RAGHit"]:
        """
        Return up to 2 confusion exemplars, appended after the normal top-k hits.
        Each returned hit represents a seed case from one confusing class; its
        prompt_snippet can be enriched with one opposite-class sid from
        _confusion_index for boundary context.
        """
        family_set = set(family_idxs)
        out: List[RAGHit] = []

        for ctype in ("lift_tilt", "insufficient_data"):
            bucket_idxs = [gi for gi in self._action_indices.get(ctype, []) if gi in family_set]
            if not bucket_idxs:
                continue
            sims = self._cosine_sim_to_subset(query_vec, bucket_idxs)
            ranked = self._sort_case_candidates(
                [(bucket_idxs[li], float(sims[li])) for li in range(len(bucket_idxs))],
                query_branch=query_branch,
                query_root_cause=query_root_cause,
            )
            if not ranked:
                continue
            gi, best_sim = ranked[0]
            case = dict(self.cases[gi])
            hit = self._case_to_raghit(case, best_sim, is_confusion=True)

            seed_sid = case["scenario_id"]
            paired = self._confusion_index.get(ctype, {}).get(seed_sid, [])
            if paired:
                hit.prompt_snippet = f"{hit.prompt_snippet} contrast={paired[0]}"[:240]
            out.append(hit)

        out.sort(key=lambda hit: (-round(float(hit.similarity), 12), hit.scenario_id))
        return out

    def retrieve(
        self,
        sf,
        k: int = 6,
        stratify_by_action: bool = True,
        include_confusion_pairs: bool = True,
        family_name: str = "F1",
        branch: Optional[str] = None,
    ) -> List["RAGHit"]:
        """
        Frozen public interface for Owner A / SFT integration.
        """
        if sf is None:
            return []
        if self.feature_matrix is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_vec = self._sf_to_feature_vec(sf)
        query_branch, query_root_cause = self._infer_query_branch(sf, family_name, branch)
        family_idxs, filter_mode = self._select_family_indices(family_name, query_branch, query_root_cause)
        if not family_idxs:
            family_idxs = list(range(len(self.cases)))
            filter_mode = "global_fallback"

        use_stratified = stratify_by_action and family_name in {"F1", "F2"} and filter_mode == "family"
        if use_stratified:
            hits = self._stratified_retrieve(
                query_vec,
                family_idxs,
                k,
                query_branch=query_branch,
                query_root_cause=query_root_cause,
            )
        else:
            hits = self._flat_retrieve(
                query_vec,
                family_idxs,
                k,
                query_branch=query_branch,
                query_root_cause=query_root_cause,
            )

        hits = self._deduplicate(hits, query_vec, threshold=0.98)
        hits.sort(
            key=lambda h: (
                -self._branch_match_rank(h, query_branch, query_root_cause),
                -round(float(h["_similarity"]), 12),
                str(h.get("scenario_id", "")),
            )
        )
        normal_hits = [self._case_to_raghit(h, h["_similarity"]) for h in hits[:k]]

        if include_confusion_pairs:
            if hasattr(sf, "top_issue") and callable(sf.top_issue):
                top_issue = sf.top_issue()[0]
            else:
                top_issue = _get_top_issue_from_query_vec(query_vec)
            if top_issue in _CONFUSION_TRIGGER_ISSUES:
                confusion_hits = self._get_confusion_pairs(
                    query_vec,
                    family_idxs,
                    query_branch=query_branch,
                    query_root_cause=query_root_cause,
                )
                seen = {h.scenario_id for h in normal_hits}
                confusion_hits = [
                    h for h in confusion_hits
                    if h.scenario_id not in seen
                ]
                confusion_budget = min(len(confusion_hits), k)
                normal_budget = max(k - confusion_budget, 0)
                return (normal_hits[:normal_budget] + confusion_hits[:confusion_budget])[:k]

        return normal_hits[:k]

    def format_few_shots(self, retrieved: List[Dict[str, Any]]) -> str:
        """Format retrieved cases as few-shot context for the Qwen prompt."""
        if not retrieved:
            return ""
        lines = ["## Similar Solved Cases\n"]
        for i, case in enumerate(retrieved, 1):
            sim = float(case.get("_similarity", case.get("similarity", 0.0)) or 0.0)
            lines.append(self._format_prompt_safe_case_block(case, sim, rank=i))
        return "\n".join(lines)

    def retrieve_trajectory_context(
        self,
        sf,
        branch: Optional[str] = None,
        k: int = 3,
        family_name: str = "F1",
    ) -> str:
        hits = self.retrieve(
            sf,
            k=k,
            family_name=family_name,
            stratify_by_action=(family_name in {"F1", "F2"}),
            include_confusion_pairs=(family_name == "F1"),
            branch=branch,
        )
        if not hits:
            return "## Similar Solved Cases\nno similar solved cases available"
        lines = ["## Similar Solved Cases\n"]
        for idx, hit in enumerate(hits, 1):
            case = {
                "family": hit.family if not isinstance(hit, dict) else hit.get("family", ""),
                "branch": hit.branch if not isinstance(hit, dict) else hit.get("branch", ""),
                "root_cause": hit.root_cause if not isinstance(hit, dict) else hit.get("root_cause", ""),
                "action": hit.gt_action_type if not isinstance(hit, dict) else hit.get("gt_action_type", hit.get("action", "")),
                "target_role": hit.target_role if not isinstance(hit, dict) else hit.get("target_role", None),
                "confidence": hit.confidence if not isinstance(hit, dict) else hit.get("confidence", ""),
                "key_evidence": hit.key_evidence if not isinstance(hit, dict) else hit.get("key_evidence", []),
                "tool_sequence": hit.tool_sequence if not isinstance(hit, dict) else hit.get("tool_sequence", []),
            }
            sim = float(hit.similarity if not isinstance(hit, dict) else hit.get("similarity", 0.0) or 0.0)
            lines.append(self._format_prompt_safe_case_block(case, sim, rank=idx))
        return "\n".join(lines)

    def retrieve_and_format(
        self,
        query_feature_vector: np.ndarray,
        family_name: str,
        top_k: int = 3,
    ) -> str:
        retrieved = self._retrieve_raw(query_feature_vector, family_name, top_k=top_k)
        return self.format_few_shots(retrieved)
