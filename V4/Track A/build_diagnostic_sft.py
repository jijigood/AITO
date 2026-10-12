#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Build diagnostic-JSON SFT data for the Track A arbitration model.

The first training target is deliberately narrow:
  - Train split: Phase 1 F1 single-answer rows, excluding heldout IDs.
  - Val split: the 200 heldout F1 IDs, if available.
  - Plan split: non-F1 rows converted to diagnosis_plan JSON for later use.

The assistant never emits scenario-local C-codes.  It emits diagnosis JSON that
can be mapped back to C-codes by the existing agentic overlay mapper.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from main_v3 import _sf_to_feature_vec
from rag_retriever_v2 import StructuredCaseRAG
from skills.candidate_generator import CandidateOutput, generate_candidates
from skills.feature_engine import ScenarioFeatures, run_all_skills
from skills.scenario_parser import ParsedScenario, parse_scenario
from skills.scenario_router import Family, route


ROOT_CAUSE_BY_ACTION: Dict[str, str] = {
    "add_neighbor": "missing_neighbor",
    "decrease_a3": "late_handover",
    "lift_tilt": "coverage_hole",
    "pdcch_2sym": "pdcch_issue",
    "check_server": "backhaul_server",
    "insufficient_data": "insufficient_data",
    "decrease_interfreq": "inter_freq_load",
    "increase_power": "coverage_hole",
    "adjust_azimuth": "coverage_hole",
    "decrease_power": "interference",
    "press_down_tilt": "overshoot_overlap",
    "increase_a3": "late_handover",
}

F1_ACTIONS = {
    "add_neighbor",
    "check_server",
    "decrease_a3",
    "insufficient_data",
    "lift_tilt",
    "pdcch_2sym",
}

RESAMPLE_WEIGHTS: Dict[str, float] = {
    "lift_tilt": 3.0,
    "insufficient_data": 3.0,
    "add_neighbor": 2.0,
    "pdcch_2sym": 1.5,
    "decrease_a3": 1.0,
    "check_server": 1.0,
}

HARD_NEGATIVE_PAIRS = [
    ("lift_tilt", "insufficient_data"),
    ("insufficient_data", "lift_tilt"),
    ("add_neighbor", "decrease_a3"),
]

INSUFFICIENT_MR_RELIABILITY_FLOOR = 30
LIFT_TILT_HARD_EVIDENCE_MR_FLOOR = 50
LIFT_INSUFFICIENT_CLOSE_GAP = 0.25

C_CODE_RE = re.compile(r"\bC(?:[1-9]|1\d|2[0-4])\b")


SYSTEM_PROMPT = """\
You are a 5G troubleshooting diagnostic arbitration model.

Given structured evidence, all available Track A options, deterministic
candidate diagnoses, tool evidence, and similar case patterns, output only
valid diagnostic JSON. Track A option codes may appear in the prompt so you can
reason about the available option space, but your answer must not output option
codes or boxed answers.

Choose the diagnosis action, target cell, relation cell, evidence confidence,
must-have evidence, blocking evidence, and whether this diagnosis should
override the deterministic top-1 answer. The downstream deterministic mapper
will convert your diagnosis to the current scenario's option code.

For lift_tilt vs insufficient_data boundary cases, do not select lift_tilt
solely because a heuristic coverage_hole score is elevated. Use the candidate
subtype:
- Select lift_tilt when the top lift_tilt candidate is a coverage hypothesis
  with an isolated tertiary target cell.
- Select insufficient_data when lift_tilt candidates are only generic high-PRB
  healthy-radio or primary fallback candidates and no coverage-specific target
  can be isolated.
MR count affects confidence level only, not action selection. Insufficient_data
confidence must not exceed medium.

Set should_override=true only when the deterministic top-1 candidate should be
replaced by the diagnosis. Set should_override=false when the deterministic
top-1 candidate already matches the diagnosis.
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_default_heldout() -> Optional[Path]:
    candidates = [
        _repo_root() / "heldout_f1_val_ids.txt",
        Path("/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track A/heldout_f1_val_ids.txt"),
        Path("/home/huangxiaolin/AITO/dataset/Track A/v3/heldout_f1_val_ids.txt"),
    ]
    return next((path for path in candidates if path.exists()), None)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_heldout_ids(path: Optional[Path]) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _write_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _strip_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(v) for v in value]
    return value


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def _feature_scores(sf: ScenarioFeatures) -> Dict[str, float]:
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


def _first_lift_tilt_candidate(candidates: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return next((row for row in candidates if row.get("action") == "lift_tilt"), None)


def _is_generic_primary_lift_fallback(row: Optional[Dict[str, Any]]) -> bool:
    if not row:
        return False
    return (
        "high-PRB healthy-radio" in str(row.get("reason") or "")
        and row.get("target_role") == "primary"
    )


def _is_coverage_tertiary_lift(row: Optional[Dict[str, Any]]) -> bool:
    if not row:
        return False
    return (
        "F1 coverage hypothesis" in str(row.get("reason") or "")
        and row.get("target_role") == "tertiary"
    )


def _lift_insufficient_boundary_info(
    evidence: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    scores = evidence.get("feature_scores", {}) or {}
    measurements = evidence.get("measurements", {}) or {}
    coverage = float(scores.get("coverage_hole") or 0.0)
    insufficient = float(scores.get("insufficient_data") or 0.0)
    gap = round(abs(coverage - insufficient), 4)
    mr_count = int(measurements.get("mr_count") or 0)
    first_lt = _first_lift_tilt_candidate(candidates)
    lift_subtype = "unknown"
    if _is_coverage_tertiary_lift(first_lt):
        lift_subtype = "coverage_hypothesis_tertiary"
    elif _is_generic_primary_lift_fallback(first_lt):
        lift_subtype = "generic_high_prb_primary_fallback"

    flags: List[str] = []
    if mr_count < INSUFFICIENT_MR_RELIABILITY_FLOOR:
        flags.append("low_mr_count_affects_confidence_only")
    if gap <= LIFT_INSUFFICIENT_CLOSE_GAP:
        flags.append("coverage_insufficient_score_gap_is_not_decisive")
    if lift_subtype != "unknown":
        flags.append(f"top_lift_tilt_subtype={lift_subtype}")

    return {
        "candidate_actions": ["lift_tilt", "insufficient_data"],
        "coverage_hole_score": round(coverage, 4),
        "insufficient_data_score": round(insufficient, 4),
        "score_gap": gap,
        "mr_count": mr_count,
        "top_lift_tilt_candidate": _strip_none({
            "reason": first_lt.get("reason") if first_lt else None,
            "target_role": first_lt.get("target_role") if first_lt else None,
            "target_cell_id": first_lt.get("target_cell_id") if first_lt else None,
            "subtype": lift_subtype,
        }),
        "risk_flags": flags,
        "decision_rules": [
            "select lift_tilt for F1 coverage hypothesis with isolated tertiary target cell",
            "select insufficient_data for generic high-PRB healthy-radio/primary fallback only",
            "MR count and score proximity affect confidence only, not action selection",
            "insufficient_data is capped at medium confidence",
        ],
    }


def _is_lift_insufficient_boundary(
    candidates: Sequence[Dict[str, Any]],
    candidate_actions: Optional[Sequence[str]] = None,
) -> bool:
    actions = {str(row.get("action") or "") for row in candidates}
    actions.update(str(action) for action in (candidate_actions or []))
    return {"lift_tilt", "insufficient_data"}.issubset(actions)


def _mr_count(ps: ParsedScenario) -> int:
    if getattr(ps, "mr", None) is None or ps.mr.empty:
        return 0
    return int(ps.mr.shape[0])


def _max_cce(sf: ScenarioFeatures) -> Optional[float]:
    return _safe_float(getattr(sf.capacity, "max_cce_util", None))


def _evidence_json(ps: ParsedScenario, sf: ScenarioFeatures, family: Family) -> Dict[str, Any]:
    mob = sf.mobility
    serving_role = ps.cell_roles.get(mob.serving_cell_id) if mob.serving_cell_id else None
    best_neighbor_cell_id = (
        ps.pci_to_cell_id.get(int(mob.best_neighbor_pci))
        if mob.best_neighbor_pci is not None
        else None
    )
    best_neighbor_role = ps.cell_roles.get(best_neighbor_cell_id) if best_neighbor_cell_id else None
    return _strip_none({
        "scenario_family": family.name,
        "tag": ps.tag,
        "num_base_stations": ps.num_base_stations,
        "serving_cell": {
            "role": serving_role,
            "cell_id": mob.serving_cell_id,
            "pci": mob.serving_pci_at_worst,
            "rsrp_dbm": _safe_float(mob.serving_rsrp_dbm),
            "sinr_db": _safe_float(mob.serving_sinr_db),
            "a3_threshold_db": _safe_float(mob.serving_a3_threshold_db),
        },
        "best_neighbor": {
            "role": best_neighbor_role,
            "cell_id": best_neighbor_cell_id,
            "pci": mob.best_neighbor_pci,
            "rsrp_dbm": _safe_float(mob.best_neighbor_rsrp_dbm),
            "advantage_db": _safe_float(mob.neighbor_advantage_db),
            "a3_margin_db": _safe_float(mob.a3_margin_db),
            "configured_neighbor_of_serving": bool(mob.neighbor_in_config),
        },
        "traffic": {
            "worst_throughput_mbps": _safe_float(sf.throughput.worst_throughput_mbps),
            "worst_time": sf.throughput.worst_time,
            "max_dl_prb_util": _safe_float(sf.inter_freq.max_dl_prb_util),
            "max_cce_util": _max_cce(sf),
        },
        "measurements": {
            "mr_count": _mr_count(ps),
            "visible_neighbor_count": len(mob.visible_neighbor_pcis or []),
        },
        "feature_scores": _feature_scores(sf),
        "top_issue": {
            "name": sf.top_issue()[0],
            "score": round(float(sf.top_issue()[1] or 0.0), 4),
        },
    })


def _option_relation(
    ps: ParsedScenario,
    sf: ScenarioFeatures,
    opt: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    action = opt.get("action_type")
    cells = list(opt.get("cells") or [])
    target = opt.get("target_cell")
    relation = None
    if action == "add_neighbor":
        serving = sf.mobility.serving_cell_id
        if serving and serving in cells:
            relation = serving
            target = next((cell for cell in cells if cell != serving), target)
        elif len(cells) >= 2:
            target = target or cells[-1]
            relation = next((cell for cell in cells if cell != target), cells[0])
    return target, relation


def _available_option_rows(ps: ParsedScenario, sf: ScenarioFeatures) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for code, opt in sorted(
        ps.parsed_options.items(),
        key=lambda item: int(item[0][1:]) if item[0][1:].isdigit() else 999,
    ):
        target, relation = _option_relation(ps, sf, opt)
        rows.append(_strip_none({
            "option_code": code,
            "action": opt.get("action_type"),
            "target_role": ps.cell_roles.get(target) if target else None,
            "target_cell_id": target,
            "relation_role": ps.cell_roles.get(relation) if relation else None,
            "relation_cell_id": relation,
        }))
    return rows


def _format_available_options(rows: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for row in rows:
        parts = [
            f"{row.get('option_code')}:",
            f"action={row.get('action') or '?'}",
        ]
        target = row.get("target_cell_id")
        target_role = row.get("target_role")
        relation = row.get("relation_cell_id")
        relation_role = row.get("relation_role")
        if target:
            target_text = f"target={target}"
            if target_role:
                target_text += f"({target_role})"
            parts.append(target_text)
        if relation:
            relation_text = f"relation={relation}"
            if relation_role:
                relation_text += f"({relation_role})"
            parts.append(relation_text)
        lines.append(" ".join(str(part) for part in parts if part))
    return "\n".join(lines)


def _format_candidate_rows(rows: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, row in enumerate(rows[:8], 1):
        parts = [f"{idx}."]
        option_code = row.get("option_code")
        if option_code:
            parts.append(f"option={option_code}")
        parts.append(f"action={row.get('action') or '?'}")
        try:
            parts.append(f"score={float(row.get('score') or 0.0):.3f}")
        except (TypeError, ValueError):
            parts.append(f"score={row.get('score')}")
        target = row.get("target_cell_id")
        target_role = row.get("target_role")
        if target:
            target_text = f"target={target}"
            if target_role:
                target_text += f"({target_role})"
            parts.append(target_text)
        relation = row.get("relation_cell_id")
        relation_role = row.get("relation_role")
        if relation:
            relation_text = f"relation={relation}"
            if relation_role:
                relation_text += f"({relation_role})"
            parts.append(relation_text)
        if row.get("reason"):
            parts.append(f"reason={row.get('reason')}")
        lines.append(" ".join(str(part) for part in parts if part))
    return "\n".join(lines)


def _tool_evidence_json() -> Dict[str, Any]:
    return {
        "status": "not_available_offline_training",
        "available_tools": [
            "judge_mainlobe_or_not",
            "calculate_tilt_angle",
            "get_user_location",
        ],
        "note": (
            "Runtime agentic overlay may provide these tool results. Offline "
            "SFT rows include this placeholder to preserve prompt shape."
        ),
    }


def _candidate_rows(
    ps: ParsedScenario,
    sf: ScenarioFeatures,
    output: CandidateOutput,
    limit: int = 8,
    force_actions: Optional[Sequence[str]] = None,
    preferred_action: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, Optional[str], Optional[str]]] = set()

    def add_option(
        action: str,
        target: Optional[str],
        relation: Optional[str],
        score: float,
        reason: str,
        option_code: Optional[str] = None,
    ) -> None:
        key = (action, target, relation)
        if key in seen:
            return
        seen.add(key)
        rows.append(_strip_none({
            "option_code": option_code,
            "action": action,
            "target_role": ps.cell_roles.get(target) if target else None,
            "target_cell_id": target,
            "relation_role": ps.cell_roles.get(relation) if relation else None,
            "relation_cell_id": relation,
            "score": round(float(score or 0.0), 4),
            "reason": reason,
        }))

    for cand in output.single_candidates[:limit]:
        opt = ps.parsed_options.get(cand.c_code, {})
        target, relation = _option_relation(ps, sf, opt)
        add_option(cand.action_type, target, relation, cand.score, cand.reason, cand.c_code)

    forced = set(force_actions or [])
    if forced:
        for opt in ps.parsed_options.values():
            action = str(opt.get("action_type") or "")
            if action not in forced:
                continue
            target, relation = _option_relation(ps, sf, opt)
            add_option(action, target, relation, 0.0, "available current-option candidate", opt.get("id"))

    if preferred_action:
        preferred = [row for row in rows if row.get("action") == preferred_action]
        others = [row for row in rows if row.get("action") != preferred_action]
        rows = preferred + others

    return rows[: max(limit, len(rows))]


def _rag_rows(
    rag: Optional[StructuredCaseRAG],
    sf: ScenarioFeatures,
    family_name: str,
    scenario_id: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    if rag is None or top_k <= 0:
        return []
    try:
        hits = rag.retrieve(
            sf,
            k=max(top_k + 2, top_k),
            family_name=family_name,
            stratify_by_action=family_name in {"F1", "F2"},
            include_confusion_pairs=family_name == "F1",
        )
    except Exception:
        return []

    rows: List[Dict[str, Any]] = []
    for hit in hits:
        sid = getattr(hit, "scenario_id", "")
        if sid == scenario_id:
            continue
        rows.append(_strip_none({
            "similarity": round(float(getattr(hit, "similarity", 0.0) or 0.0), 4),
            "family": getattr(hit, "family", family_name),
            "branch": getattr(hit, "branch", ""),
            "root_cause": getattr(hit, "root_cause", ""),
            "action": getattr(hit, "gt_action_type", ""),
            "target_role": getattr(hit, "target_role", None),
            "confidence": getattr(hit, "confidence", ""),
            "key_evidence": list(getattr(hit, "key_evidence", []) or [])[:3],
            "tool_sequence": list(getattr(hit, "tool_sequence", []) or [])[:3],
        }))
        if len(rows) >= top_k:
            break
    return rows


def _must_have_and_blocking(
    root_cause: str,
    evidence: Dict[str, Any],
    candidate_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[str], List[str]]:
    serving = evidence.get("serving_cell", {})
    neighbor = evidence.get("best_neighbor", {})
    traffic = evidence.get("traffic", {})
    meas = evidence.get("measurements", {})

    sinr = serving.get("sinr_db")
    rsrp = serving.get("rsrp_dbm")
    prb = traffic.get("max_dl_prb_util")
    cce = traffic.get("max_cce_util")
    mr_count = meas.get("mr_count")
    adv = neighbor.get("advantage_db")
    margin = neighbor.get("a3_margin_db")
    configured = neighbor.get("configured_neighbor_of_serving")
    first_lt = _first_lift_tilt_candidate(candidate_rows or [])
    has_boundary_candidates = _is_lift_insufficient_boundary(candidate_rows or [])
    first_lt_is_coverage_tertiary = _is_coverage_tertiary_lift(first_lt)
    first_lt_is_generic_primary = _is_generic_primary_lift_fallback(first_lt)

    common = []
    if sinr is not None:
        common.append(f"serving SINR={sinr} dB")
    if rsrp is not None:
        common.append(f"serving RSRP={rsrp} dBm")
    if prb is not None:
        common.append(f"max DL PRB={prb}%")
    if mr_count is not None:
        common.append(f"MR count={mr_count}")

    if root_cause == "missing_neighbor":
        relation_text = (
            "best neighbor not configured for serving cell"
            if configured is False
            else "neighbor relation evidence is ambiguous but add_neighbor candidate matches the labeled repair"
        )
        must = [
            f"neighbor advantage={adv} dB",
            relation_text,
        ]
        blocking = [
            "strong configured neighbor with a handover threshold issue would favor late_handover"
        ]
    elif root_cause == "late_handover":
        must = [
            f"neighbor advantage={adv} dB",
            f"A3 margin={margin} dB",
            "best neighbor configured for serving cell" if configured else "handover relation evidence",
        ]
        blocking = ["unconfigured strong neighbor would favor missing_neighbor"]
    elif root_cause == "coverage_hole":
        if has_boundary_candidates and first_lt_is_coverage_tertiary:
            must = common + [
                "top lift_tilt candidate is F1 coverage hypothesis with isolated tertiary target cell",
                "coverage-shaping action is directly targeted at that specific cell",
            ]
            blocking = [
                "if top lift_tilt candidate is generic high-PRB healthy-radio/primary fallback, prefer insufficient_data",
                "MR count affects confidence only, not the lift_tilt vs insufficient_data decision",
            ]
        else:
            must = common + [
                "candidate coverage-shaping action is available",
                "coverage-specific target cell is isolated by candidate evidence",
            ]
            blocking = [
                "generic high-PRB healthy-radio/primary fallback alone would favor insufficient_data",
                "dominant CCE pressure would favor pdcch_issue",
            ]
    elif root_cause == "insufficient_data":
        if has_boundary_candidates and first_lt_is_generic_primary:
            must = common + [
                "top lift_tilt candidate is generic high-PRB healthy-radio/primary fallback only",
                "no coverage hypothesis with isolated tertiary target cell is available",
            ]
        else:
            must = common + [
                "no coverage hypothesis with isolated tertiary target cell is available",
                "evidence is ambiguous for a confident RF diagnosis",
            ]
        blocking = [
            "if a coverage hypothesis with isolated tertiary target exists in candidates, prefer lift_tilt",
            "MR count affects confidence only, not the lift_tilt vs insufficient_data decision",
        ]
    elif root_cause == "pdcch_issue":
        must = common + [f"CCE/PDCCH pressure={cce}"]
        blocking = ["no CCE pressure with healthy radio would favor backhaul_server"]
    elif root_cause == "backhaul_server":
        must = common + ["radio evidence does not isolate an RF optimization"]
        blocking = ["strong RF anomaly would favor mobility or coverage action"]
    elif root_cause == "inter_freq_load":
        must = common + ["13-BS/inter-frequency family prior"]
        blocking = ["4-BS F1/F2 should not default to inter-frequency action"]
    else:
        must = common or ["structured evidence supports the selected action"]
        blocking = ["contradictory stronger issue score would block this diagnosis"]

    def clean(items: Iterable[str]) -> List[str]:
        out: List[str] = []
        for item in items:
            if item and "None" not in item and item not in out:
                out.append(item)
        return out[:6]

    return clean(must), clean(blocking)


def _diagnosis_for_option(
    ps: ParsedScenario,
    sf: ScenarioFeatures,
    opt: Dict[str, Any],
    candidate_rows: Sequence[Dict[str, Any]],
    evidence: Dict[str, Any],
    should_override: Optional[bool] = None,
) -> Dict[str, Any]:
    action = str(opt.get("action_type") or "unknown")
    root_cause = ROOT_CAUSE_BY_ACTION.get(action, "uncertain")
    target, relation = _option_relation(ps, sf, opt)
    target_role = ps.cell_roles.get(target) if target else None
    relation_role = ps.cell_roles.get(relation) if relation else None
    must_have, blocking = _must_have_and_blocking(root_cause, evidence, candidate_rows)

    matching_rank = None
    matching_score = 0.0
    for idx, row in enumerate(candidate_rows, 1):
        if row.get("action") != action:
            continue
        if action in {"check_server", "insufficient_data"} or row.get("target_cell_id") == target:
            matching_rank = idx
            matching_score = float(row.get("score", 0.0) or 0.0)
            break

    action_rank = next(
        (idx for idx, row in enumerate(candidate_rows, 1) if row.get("action") == action),
        None,
    )

    # Confidence is intentionally conservative.  It is later used as an
    # override gate in Phase 2, so ambiguous boundary labels should teach the
    # model caution rather than blanket certainty.
    boundary_info = (
        _lift_insufficient_boundary_info(evidence, candidate_rows)
        if any(row.get("action") == "insufficient_data" for row in candidate_rows)
        and any(row.get("action") == "lift_tilt" for row in candidate_rows)
        else {}
    )
    mr_count = int((evidence.get("measurements") or {}).get("mr_count") or 0)

    if action == "insufficient_data":
        if action_rank is not None and action_rank <= 4:
            confidence = "medium"
        else:
            confidence = "low"
    elif action in {"lift_tilt", "add_neighbor", "decrease_a3"}:
        boundary_without_hard_evidence = (
            action == "lift_tilt"
            and bool(boundary_info)
            and mr_count <= LIFT_TILT_HARD_EVIDENCE_MR_FLOOR
        )
        if (
            matching_rank is not None
            and matching_rank <= 2
            and matching_score >= 0.65
            and not boundary_without_hard_evidence
        ):
            confidence = "high"
        elif action_rank is not None and action_rank <= 4:
            confidence = "medium"
        else:
            confidence = "low"
    elif action in {"pdcch_2sym", "check_server"}:
        if matching_rank is not None and matching_rank <= 2 and matching_score >= 0.45:
            confidence = "high"
        elif action_rank is not None and action_rank <= 5:
            confidence = "medium"
        else:
            confidence = "low"
    elif matching_rank is not None and matching_score >= 0.60:
        confidence = "high"
    elif action_rank is not None:
        confidence = "medium"
    else:
        confidence = "low"

    if action in {"check_server", "insufficient_data"}:
        target = relation = target_role = relation_role = None

    diagnosis = {
        "root_cause": root_cause,
        "action": action,
        "target_role": target_role,
        "target_cell_id": target,
        "relation_role": relation_role,
        "relation_cell_id": relation,
        "evidence_confidence": confidence,
        "must_have_evidence": must_have,
        "blocking_evidence": blocking,
    }
    if should_override is not None:
        diagnosis["should_override"] = bool(should_override)
    return diagnosis


def _diagnosis_plan(
    ps: ParsedScenario,
    sf: ScenarioFeatures,
    answer_codes: Sequence[str],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_rows: List[Dict[str, Any]] = []
    plan = []
    for code in answer_codes:
        opt = ps.parsed_options.get(code, {})
        if not opt:
            continue
        diagnosis = _diagnosis_for_option(ps, sf, opt, candidate_rows, evidence)
        diagnosis["evidence_confidence"] = "high"
        plan.append(diagnosis)
    return {"diagnosis_plan": plan}


def _user_prompt(
    ps: ParsedScenario,
    family: Family,
    evidence: Dict[str, Any],
    options: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    tool_evidence: Dict[str, Any],
    rag_rows: Sequence[Dict[str, Any]],
    candidate_actions: Optional[Sequence[str]] = None,
) -> str:
    boundary_info = (
        _lift_insufficient_boundary_info(evidence, candidates)
        if _is_lift_insufficient_boundary(candidates, candidate_actions)
        else None
    )
    parts = [
        "## Scenario",
        json.dumps({
            "scenario_id": ps.scenario_id,
            "family": family.name,
            "answer_cardinality": 1 if not family.is_multiple else family.expected_answer_count,
        }, ensure_ascii=False, indent=2, sort_keys=True),
        "## Available Options (all C-codes, compact)",
        _format_available_options(options),
        "## Deterministic Candidates (scored)",
        _format_candidate_rows(candidates),
        "## Structured Evidence",
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True),
        "## Tool Evidence",
        json.dumps(tool_evidence, ensure_ascii=False, indent=2, sort_keys=True),
    ]
    if candidate_actions:
        parts.extend([
            "## Boundary Candidate Actions",
            json.dumps(list(candidate_actions), ensure_ascii=False, indent=2),
        ])
    if boundary_info:
        parts.extend([
            "## Boundary Arbitration",
            json.dumps(boundary_info, ensure_ascii=False, indent=2, sort_keys=True),
        ])
    parts.extend([
        "## Similar Case Evidence Patterns",
        json.dumps(list(rag_rows), ensure_ascii=False, indent=2, sort_keys=True)
        if rag_rows else "[]",
        "## Output Contract",
        (
            "Output only one JSON object with keys: root_cause, action, "
            "target_role, target_cell_id, relation_role, relation_cell_id, "
            "evidence_confidence, should_override, must_have_evidence, "
            "blocking_evidence. Do not output Track A option codes."
        ),
    ])
    return "\n\n".join(parts)


def _make_record(
    ps: ParsedScenario,
    family: Family,
    sf: ScenarioFeatures,
    rag: Optional[StructuredCaseRAG],
    split: str,
    augmentation: str = "none",
    candidate_actions: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    answer_codes = [code.strip() for code in str(ps.ground_truth or "").split("|") if code.strip()]
    if not answer_codes:
        return None
    evidence = _evidence_json(ps, sf, family)
    options = _available_option_rows(ps, sf)
    tool_evidence = _tool_evidence_json()
    output = generate_candidates(ps, sf, family)
    deterministic_top1_code = output.top_candidate() or ""
    deterministic_top1_action = ""
    if deterministic_top1_code and "|" not in deterministic_top1_code:
        deterministic_top1_action = str(
            ps.parsed_options.get(deterministic_top1_code, {}).get("action_type", "")
        )
    gt_action = ps.parsed_options.get(answer_codes[0], {}).get("action_type", "")
    force_actions = list(candidate_actions or [])
    if gt_action:
        force_actions.append(gt_action)
    candidates_raw = _candidate_rows(ps, sf, output, limit=8, force_actions=force_actions)
    # Phase 3 keeps deployable deterministic ordering in both train and
    # validation.  The model learns when to override via should_override rather
    # than by seeing the label promoted to rank 1.
    preferred_action = None
    candidates = _candidate_rows(
        ps,
        sf,
        output,
        limit=8,
        force_actions=force_actions,
        preferred_action=preferred_action,
    )
    rag_hits = _rag_rows(rag, sf, family.name, ps.scenario_id, top_k=5)

    if family.name == "F1" and len(answer_codes) == 1:
        gt_opt = ps.parsed_options.get(answer_codes[0], {})
        should_override = bool(deterministic_top1_code and deterministic_top1_code != answer_codes[0])
        assistant_obj = _diagnosis_for_option(
            ps,
            sf,
            gt_opt,
            candidates,
            evidence,
            should_override=should_override,
        )
    else:
        should_override = bool(deterministic_top1_code and set(deterministic_top1_code.split("|")) != set(answer_codes))
        assistant_obj = _diagnosis_plan(ps, sf, answer_codes, evidence)
        assistant_obj["should_override"] = should_override

    user = _user_prompt(
        ps,
        family,
        evidence,
        options,
        candidates,
        tool_evidence,
        rag_hits,
        candidate_actions=candidate_actions,
    )

    assistant = json.dumps(assistant_obj, ensure_ascii=False, sort_keys=True)
    if C_CODE_RE.search(assistant):
        raise ValueError(f"C-code leakage in assistant for {ps.scenario_id}")
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {
            "scenario_id": ps.scenario_id,
            "family": family.name,
            "split": split,
            "augmentation": augmentation,
            "gt_action": gt_action,
            "gt_root_cause": ROOT_CAUSE_BY_ACTION.get(gt_action, "uncertain"),
            "target_role": assistant_obj.get("target_role") if isinstance(assistant_obj, dict) else None,
            "evidence_confidence": assistant_obj.get("evidence_confidence") if isinstance(assistant_obj, dict) else None,
            "deterministic_top1_code": deterministic_top1_code,
            "deterministic_top1_action": deterministic_top1_action,
            "gt_code": answer_codes[0] if len(answer_codes) == 1 else "|".join(answer_codes),
            "should_override": should_override,
            "should_override_action": bool(deterministic_top1_action and gt_action and deterministic_top1_action != gt_action),
            "boundary_pair": _is_lift_insufficient_boundary(candidates, candidate_actions),
        },
    }


@dataclass
class BaseExample:
    raw: Dict[str, Any]
    ps: ParsedScenario
    family: Family
    sf: ScenarioFeatures
    action: str
    feature_vec: np.ndarray


def _build_base_examples(raw_examples: Sequence[Dict[str, Any]]) -> List[BaseExample]:
    examples: List[BaseExample] = []
    for raw in raw_examples:
        ps = parse_scenario(raw)
        family = route(ps.tag, ps.num_base_stations)
        sf = run_all_skills(ps)
        answer_codes = [code.strip() for code in str(ps.ground_truth or "").split("|") if code.strip()]
        action = ""
        if answer_codes:
            action = str(ps.parsed_options.get(answer_codes[0], {}).get("action_type", ""))
        examples.append(BaseExample(raw=raw, ps=ps, family=family, sf=sf, action=action, feature_vec=_sf_to_feature_vec(sf)))
    return examples


def _resampled_records(
    base: Sequence[BaseExample],
    rag: Optional[StructuredCaseRAG],
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], Counter]:
    records: List[Dict[str, Any]] = []
    augment_counter: Counter = Counter()
    for ex in base:
        if ex.family.name != "F1":
            continue
        weight = RESAMPLE_WEIGHTS.get(ex.action, 1.0)
        repeats = int(weight)
        if rng.random() < (weight - repeats):
            repeats += 1
        repeats = max(repeats, 1)
        for idx in range(repeats):
            rec = _make_record(
                ex.ps,
                ex.family,
                ex.sf,
                rag,
                split="train",
                augmentation="upsample" if idx else "none",
            )
            if rec:
                records.append(rec)
                augment_counter[rec["meta"]["augmentation"]] += 1
    return records, augment_counter


def _hard_negative_records(
    train_base: Sequence[BaseExample],
    rag: Optional[StructuredCaseRAG],
    per_source: int = 1,
) -> Tuple[List[Dict[str, Any]], Counter]:
    by_action: Dict[str, List[BaseExample]] = defaultdict(list)
    for ex in train_base:
        if ex.family.name == "F1":
            by_action[ex.action].append(ex)

    out: List[Dict[str, Any]] = []
    counts: Counter = Counter()
    for src_action, neg_action in HARD_NEGATIVE_PAIRS:
        left = by_action.get(src_action, [])
        right = by_action.get(neg_action, [])
        if not left or not right:
            continue
        right_vecs = np.stack([ex.feature_vec for ex in right], axis=0)
        right_norm = right_vecs / (np.linalg.norm(right_vecs, axis=1, keepdims=True) + 1e-8)
        for ex in left:
            q = ex.feature_vec / (np.linalg.norm(ex.feature_vec) + 1e-8)
            sims = right_norm @ q
            nearest = np.argsort(-sims)[:per_source]
            for _idx in nearest:
                rec = _make_record(
                    ex.ps,
                    ex.family,
                    ex.sf,
                    rag,
                    split="train",
                    augmentation=f"hard_negative:{src_action}_vs_{neg_action}",
                    candidate_actions=[src_action, neg_action],
                )
                if rec:
                    out.append(rec)
                    counts[rec["meta"]["augmentation"]] += 1
    return out, counts


def _update_dataset_info(path: Path) -> None:
    info: Dict[str, Any] = {}
    if path.exists():
        info = json.loads(path.read_text(encoding="utf-8"))
    for name, filename in {
        "sft_diag_train": "sft_diag_train.jsonl",
        "sft_diag_val": "sft_diag_val.jsonl",
        "sft_diag_plan": "sft_diag_plan.jsonl",
    }.items():
        info[name] = {
            "file_name": filename,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _report(
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    plan_records: Sequence[Dict[str, Any]],
    base_examples: Sequence[BaseExample],
    heldout_ids: set[str],
    out_path: Path,
) -> None:
    def hist(records: Sequence[Dict[str, Any]], key: str) -> Counter:
        return Counter(str(rec.get("meta", {}).get(key, "")) for rec in records)

    base_family = Counter(ex.family.name for ex in base_examples)
    base_f1_action = Counter(ex.action for ex in base_examples if ex.family.name == "F1")
    train_action = hist(train_records, "gt_action")
    val_action = hist(val_records, "gt_action")
    train_aug = hist(train_records, "augmentation")
    confidence = hist(train_records, "evidence_confidence")
    train_boundary = hist(train_records, "boundary_pair")
    val_boundary = hist(val_records, "boundary_pair")
    train_override = hist(train_records, "should_override")
    val_override = hist(val_records, "should_override")
    train_override_action = hist(train_records, "should_override_action")

    lines = [
        "# Diagnostic SFT Report",
        "",
        "## Inputs",
        f"- train examples: {len(base_examples)}",
        f"- heldout IDs loaded: {len(heldout_ids)}",
        f"- family distribution: `{dict(sorted(base_family.items()))}`",
        f"- raw F1 action distribution: `{dict(sorted(base_f1_action.items()))}`",
        "",
        "## Outputs",
        f"- train records: {len(train_records)}",
        f"- val records: {len(val_records)}",
        f"- non-F1 diagnosis_plan records: {len(plan_records)}",
        f"- train action distribution: `{dict(sorted(train_action.items()))}`",
        f"- val action distribution: `{dict(sorted(val_action.items()))}`",
        f"- train augmentation distribution: `{dict(sorted(train_aug.items()))}`",
        f"- train confidence distribution: `{dict(sorted(confidence.items()))}`",
        f"- train boundary-pair records: `{dict(sorted(train_boundary.items()))}`",
        f"- val boundary-pair records: `{dict(sorted(val_boundary.items()))}`",
        f"- train should_override distribution: `{dict(sorted(train_override.items()))}`",
        f"- val should_override distribution: `{dict(sorted(val_override.items()))}`",
        f"- train action-level override distribution: `{dict(sorted(train_override_action.items()))}`",
        "",
        "## Safety Checks",
        "- Assistant outputs diagnostic JSON only; no boxed answers.",
        "- User prompts include compact Track A option codes for prompt parity.",
        "- Assistant outputs are guarded against Track A C-code leakage.",
        "- Heldout F1 IDs are excluded from the training split.",
        "- Non-F1 rows are written as diagnosis_plan data for later optional training, not mixed into first-round F1 SFT.",
        "- Candidate ordering is deployable deterministic ordering in train and validation; no label-aware promotion.",
        "",
        "## Boundary Notes",
        "- lift_tilt/insufficient_data action selection is based on lift_tilt candidate subtype, not MR count alone.",
        "- F1 coverage hypothesis with isolated tertiary target supports lift_tilt.",
        "- Generic high-PRB healthy-radio/primary fallback supports insufficient_data.",
        "- MR count affects confidence calibration only.",
        "- should_override is the intended Phase 3 override signal; confidence is diagnostic metadata.",
        "- Use action-level and should_override smoke tests first; only add submission overrides after the gate passes.",
        "",
        "## Recommended Next Gate",
        "- Train a 100-200 sample smoke adapter before the full rank-32 run.",
        "- Evaluate all heldout lift_tilt and insufficient_data rows for action accuracy and confidence calibration.",
        "- Track should_override precision/recall on deterministic-wrong boundary rows.",
        "- Block integration if boundary action accuracy regresses or should_override precision is unsafe.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build diagnostic JSON SFT data")
    ap.add_argument("--train-json", type=Path, default=_repo_root() / "data/Phase_1/train.json")
    ap.add_argument("--trajectory-jsonl", type=Path, default=_repo_root() / "artifacts/owner_d/data/trajectory_memory_combined.jsonl")
    ap.add_argument("--heldout-ids", type=Path, default=_resolve_default_heldout())
    ap.add_argument("--out-dir", type=Path, default=_repo_root())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--no-resample", action="store_true")
    ap.add_argument("--no-hard-negatives", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw_examples = _load_json(args.train_json)
    heldout_ids = _load_heldout_ids(args.heldout_ids)

    rag = None
    if not args.no_rag:
        rag = StructuredCaseRAG(str(args.train_json), trajectory_jsonl_path=str(args.trajectory_jsonl))
        rag.build_index()

    base_examples = _build_base_examples(raw_examples)
    train_base = [
        ex for ex in base_examples
        if ex.family.name == "F1" and ex.ps.scenario_id not in heldout_ids
    ]
    val_base = [
        ex for ex in base_examples
        if ex.family.name == "F1" and ex.ps.scenario_id in heldout_ids
    ]
    plan_base = [ex for ex in base_examples if ex.family.name != "F1"]

    if args.no_resample:
        train_records = [
            rec for ex in train_base
            if (rec := _make_record(ex.ps, ex.family, ex.sf, rag, split="train"))
        ]
    else:
        train_records, _ = _resampled_records(train_base, rag, rng)

    if not args.no_hard_negatives:
        hard_records, _ = _hard_negative_records(train_base, rag, per_source=1)
        train_records.extend(hard_records)

    val_records = [
        rec for ex in val_base
        if (rec := _make_record(ex.ps, ex.family, ex.sf, rag, split="val"))
    ]
    plan_records = [
        rec for ex in plan_base
        if (rec := _make_record(ex.ps, ex.family, ex.sf, rag, split="plan"))
    ]

    rng.shuffle(train_records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(train_records, args.out_dir / "sft_diag_train.jsonl")
    _write_jsonl(val_records, args.out_dir / "sft_diag_val.jsonl")
    _write_jsonl(plan_records, args.out_dir / "sft_diag_plan.jsonl")
    _update_dataset_info(args.out_dir / "dataset_info.json")
    _report(
        train_records,
        val_records,
        plan_records,
        base_examples,
        heldout_ids,
        args.out_dir / "sft_diag_report.md",
    )

    print(f"Wrote {len(train_records)} train records")
    print(f"Wrote {len(val_records)} val records")
    print(f"Wrote {len(plan_records)} plan records")
    print(f"Report: {args.out_dir / 'sft_diag_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
