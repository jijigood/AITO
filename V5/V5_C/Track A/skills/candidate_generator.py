#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Candidate Generator — V2 Architecture  (includes Skill H: multi-answer composer)

Translates ScenarioFeatures + ParsedScenario into a small ranked list of
concrete C-code candidates for Qwen to choose from.

Key principle (from ARCHITECTURE_V2.md §7.5):
  "Qwen should rarely choose from all 22 options directly. The system should
   first reduce the decision to a small candidate set."

Output:
  - single-answer families (F1, F4): 2–5 ranked C-codes
  - multiple-answer families (F2, F3): 2–4 ranked C-code combinations (Skill H)

IMPORTANT: option slot → action_type is NOT fixed across scenarios.
We parse option labels dynamically (via ParsedScenario.parsed_options) and
match by (action_type, target_role) rather than by slot number.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from skills.scenario_parser import ParsedScenario
from skills.scenario_router import Family
from skills.feature_engine import ScenarioFeatures


# ──────────────────────────────────────────────────────────────────────────────
# Issue-type → action_type priority lists
# ──────────────────────────────────────────────────────────────────────────────

# For each issue type, what actions should be tried (in priority order)?
# Each entry is (action_type, target_role, score_boost)
_ISSUE_ACTION_MAP: Dict[str, List[Tuple[str, Optional[str], float]]] = {
    "late_handover": [
        ("decrease_a3",        "primary",   1.0),   # decrease A3 for serving cell
        ("decrease_a3",        "secondary", 0.6),   # fallback: other cell
        ("decrease_interfreq", "primary",   0.3),
    ],
    "missing_neighbor": [
        ("add_neighbor",       "primary",   1.0),   # missing neighbor is typically the primary cell
        ("add_neighbor",       None,        0.8),   # fallback: any add_neighbor option
        ("decrease_a3",        "primary",   0.4),
    ],
    "inter_freq_load": [
        ("decrease_interfreq", "primary",   1.0),
        ("decrease_interfreq", "secondary", 0.8),
        ("decrease_a3",        "primary",   0.3),
    ],
    "interference": [
        ("decrease_power",     "secondary", 1.0),   # reduce power of interferer
        ("press_down_tilt",    "secondary", 0.8),   # tilt down interferer
        ("decrease_power",     "primary",   0.5),
        ("press_down_tilt",    "primary",   0.4),
    ],
    "coverage_hole": [
        ("increase_power",     "primary",   1.0),
        ("lift_tilt",          "primary",   0.8),
        ("increase_power",     "secondary", 0.4),
    ],
    "overshoot_overlap": [
        ("press_down_tilt",    "secondary", 1.0),
        ("decrease_power",     "secondary", 0.8),
        ("press_down_tilt",    "primary",   0.5),
        ("adjust_azimuth",     "secondary", 0.4),
    ],
    "pdcch_issue": [
        ("pdcch_2sym",         "primary",   1.0),
        ("pdcch_2sym",         "secondary", 0.7),
    ],
    "backhaul_server": [
        ("check_server",       None,        1.0),
        ("insufficient_data",  None,        0.3),
    ],
}

# Complementary action pairs for multi-answer composition (Skill H)
# (action_type_A, role_A, action_type_B, role_B)
_MULTI_PAIRS: List[Tuple[str, str, str, str]] = [
    # Reduce coverage of overpowering cell
    ("decrease_power",  "secondary", "press_down_tilt",  "secondary"),
    ("decrease_power",  "primary",   "press_down_tilt",  "primary"),
    # Coverage-shaping fix for F2
    ("adjust_azimuth",  "secondary", "increase_power",   "secondary"),
    # Improve weak cell
    ("increase_power",  "primary",   "lift_tilt",        "primary"),
    ("increase_power",  "secondary", "lift_tilt",        "secondary"),
    # HO threshold tuning (both cells)
    ("decrease_a3",     "primary",   "increase_a3",      "secondary"),
    ("decrease_a3",     "secondary", "increase_a3",      "primary"),
    # Load + mobility
    ("decrease_interfreq", "primary",  "decrease_a3",    "primary"),
    ("decrease_interfreq", "secondary","decrease_a3",    "secondary"),
    # Missing neighbour + HO threshold
    ("add_neighbor",    None,        "decrease_a3",      "primary"),
]

# 4-action template for F3 (5 BS, always 4 answers)
_F3_4ACTION_TEMPLATES: List[List[Tuple[str, str]]] = [
    [("decrease_power", "primary"), ("press_down_tilt", "primary"),
     ("increase_a3", "primary"),    ("increase_a3", "secondary")],
    [("increase_power", "secondary"), ("lift_tilt", "secondary"),
     ("decrease_a3", "primary"),    ("decrease_interfreq", "primary")],
    [("decrease_power", "secondary"), ("press_down_tilt", "secondary"),
     ("decrease_a3", "primary"),    ("increase_a3", "primary")],
    [("adjust_azimuth", "primary"), ("press_down_tilt", "primary"),
     ("decrease_a3", "primary"),    ("increase_a3", "secondary")],
]

_F1_PREFERRED_ACTIONS = {
    "add_neighbor",
    "adjust_azimuth",
    "check_server",
    "decrease_a3",
    "insufficient_data",
    "lift_tilt",
    "pdcch_2sym",
}

_F1_ARBITRATION_ACTIONS = _F1_PREFERRED_ACTIONS | {"press_down_tilt"}

_F1_SAFE_TOP_ACTIONS = {
    "add_neighbor",
    "check_server",
    "decrease_a3",
    "insufficient_data",
    "lift_tilt",
    "pdcch_2sym",
}

# Submission 2 audit note:
# the single-answer guard is intentionally disabled by default. A 60-row
# Phase2 probe produced a high diff rate, so this remains an audit-only helper
# until it can be gated more narrowly with ground-truth evidence.
_ENABLE_F1_SINGLE_TOP_GUARD = False


# ──────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────────────────────────────────────────────

def _score_option(
    opt: Dict[str, Any],
    action_type: str,
    target_role: Optional[str],
    ps: ParsedScenario,
    boost: float = 1.0,
    serving_cell_id: Optional[str] = None,
) -> float:
    """Score a single parsed option against a desired (action_type, target_role)."""
    if opt["action_type"] != action_type:
        return 0.0
    if action_type in ("check_server", "insufficient_data"):
        return boost  # no cell targeting needed
    tc = opt.get("target_cell")
    if tc is None:
        return boost * 0.5
    actual_role = ps.cell_roles.get(tc)

    # add_neighbor: the option's `cells` list names both endpoints of the new
    # neighbor relation.  The correct option includes the UE's serving cell as
    # one endpoint.  Use this to break ties between options targeting the same
    # target role.
    if action_type == "add_neighbor" and serving_cell_id is not None:
        cells = opt.get("cells", [])
        if serving_cell_id in cells:
            return boost  # serving cell is part of this relation — correct pair
        # Option does not involve the serving cell → heavy penalty
        if target_role is None or actual_role == target_role:
            return boost * 0.20
        return boost * 0.05

    if target_role is None:
        return boost
    if actual_role == target_role:
        return boost
    # Wrong role — partial credit only
    return boost * 0.15


def _find_best_option(
    ps: ParsedScenario,
    action_type: str,
    target_role: Optional[str],
    boost: float = 1.0,
    exclude: Optional[set] = None,
    serving_cell_id: Optional[str] = None,
) -> Optional[Tuple[str, float]]:
    """
    Find the highest-scoring C-code for a given (action_type, target_role).
    Returns (c_code, score) or None.
    """
    best_code, best_score = None, 0.0
    for cid, opt in ps.parsed_options.items():
        if exclude and cid in exclude:
            continue
        s = _score_option(opt, action_type, target_role, ps, boost,
                          serving_cell_id=serving_cell_id)
        if s > best_score:
            best_score = s
            best_code = cid
    return (best_code, best_score) if best_code else None


def _cell_role(ps: ParsedScenario, cell_id: Optional[str]) -> Optional[str]:
    if not cell_id:
        return None
    return ps.cell_roles.get(cell_id)


def _best_neighbor_cell_id(
    ps: ParsedScenario,
    sf: ScenarioFeatures,
    exclude: Optional[set] = None,
) -> Optional[str]:
    pci = sf.mobility.best_neighbor_pci
    if pci is None:
        return None
    cell_id = ps.pci_to_cell_id.get(int(pci))
    if exclude and cell_id in exclude:
        return None
    return cell_id


def _first_distinct_cell_id(
    ps: ParsedScenario,
    exclude: Optional[set] = None,
    preferred_roles: Tuple[str, ...] = ("secondary", "tertiary", "primary"),
) -> Optional[str]:
    excluded = {cell_id for cell_id in (exclude or set()) if cell_id}
    for role in preferred_roles:
        for cell_id in ps.role_to_cells.get(role, []):
            if cell_id not in excluded:
                return cell_id
    for cell_id in ps.cell_roles:
        if cell_id not in excluded:
            return cell_id
    return None


def _score_option_for_cell(
    opt: Dict[str, Any],
    action_type: str,
    target_cell_id: Optional[str],
    ps: ParsedScenario,
    boost: float = 1.0,
    relation_cell_id: Optional[str] = None,
) -> float:
    """Score an option against an exact target cell, falling back to role only if needed."""
    if opt["action_type"] != action_type:
        return 0.0

    if action_type in ("check_server", "insufficient_data"):
        return boost

    cells = opt.get("cells", [])
    tc = opt.get("target_cell")

    if action_type == "add_neighbor":
        if target_cell_id and relation_cell_id:
            if target_cell_id in cells and relation_cell_id in cells:
                return boost
            if relation_cell_id in cells:
                return boost * 0.55
            if target_cell_id in cells:
                return boost * 0.35
            return 0.0
        if relation_cell_id:
            return boost if relation_cell_id in cells else 0.0
        if target_cell_id:
            return boost if target_cell_id in cells else 0.0
        return boost * 0.5

    if target_cell_id:
        if tc == target_cell_id:
            return boost
        target_role = _cell_role(ps, target_cell_id)
        if target_role and ps.cell_roles.get(tc) == target_role:
            return boost * 0.55
        return 0.0

    return boost * 0.5


def _find_best_option_for_cell(
    ps: ParsedScenario,
    action_type: str,
    target_cell_id: Optional[str],
    boost: float = 1.0,
    exclude: Optional[set] = None,
    relation_cell_id: Optional[str] = None,
) -> Optional[Tuple[str, float]]:
    best_code, best_score = None, 0.0
    for cid, opt in ps.parsed_options.items():
        if exclude and cid in exclude:
            continue
        score = _score_option_for_cell(
            opt,
            action_type,
            target_cell_id,
            ps,
            boost=boost,
            relation_cell_id=relation_cell_id,
        )
        if score > best_score:
            best_code, best_score = cid, score
    return (best_code, best_score) if best_code else None


# ──────────────────────────────────────────────────────────────────────────────
# Single-answer candidate generation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RankedCandidate:
    c_code: str
    score: float
    action_type: str
    target_cell: Optional[str]
    reason: str


def _upsert_candidate(
    scored: Dict[str, float],
    reasons: Dict[str, str],
    c_code: str,
    score: float,
    reason: str,
) -> None:
    if c_code not in scored or scored[c_code] < score:
        scored[c_code] = score
        reasons[c_code] = reason


def _add_recommended_single_candidates(
    ps: ParsedScenario,
    sf: ScenarioFeatures,
    scored: Dict[str, float],
    reasons: Dict[str, str],
) -> None:
    """
    Add a small amount of family-agnostic diversification based on each skill's
    recommended action. This is especially useful when composite scores are low
    and the strict issue ranking would otherwise yield only check_server /
    insufficient_data candidates.
    """
    recommendations = [
        (
            sf.mobility.recommended_action_family,
            sf.mobility.recommended_target_role,
            max(sf.score_late_handover, sf.score_missing_neighbor, 0.08),
            "mobility recommendation",
        ),
        (
            sf.inter_freq.recommended_action_family,
            sf.inter_freq.recommended_target_role,
            max(sf.score_inter_freq_load, 0.08),
            "inter-frequency recommendation",
        ),
        (
            sf.geometry.recommended_action_family,
            sf.geometry.recommended_target_role,
            max(sf.score_overshoot_overlap, 0.08),
            "geometry recommendation",
        ),
        (
            sf.power.recommended_action_family,
            sf.power.recommended_target_role,
            max(sf.score_interference, sf.score_coverage_hole, 0.08),
            "power recommendation",
        ),
        (
            sf.capacity.recommended_action_family,
            sf.capacity.recommended_target_role,
            max(sf.score_pdcch_issue, 0.08),
            "capacity recommendation",
        ),
        (
            sf.transport.recommended_action_family,
            None,
            max(sf.score_backhaul_server, 0.08),
            "transport recommendation",
        ),
    ]

    for action_type, target_role, base_score, reason in recommendations:
        if not action_type or action_type == "none":
            continue
        if (
            reason == "geometry recommendation"
            and action_type == "lift_tilt"
            and (sf.score_coverage_hole or 0.0) < 0.30
            and (getattr(sf, "score_insufficient_data", 0.0) or 0.0) < 0.30
        ):
            continue
        hit = _find_best_option(ps, action_type, target_role, boost=base_score)
        if hit:
            c_code, score = hit
            _upsert_candidate(scored, reasons, c_code, score, reason)


def _is_f1_like_single(family: Family, ps: ParsedScenario) -> bool:
    return (
        not family.is_multiple
        and family.name != "F4"
        and ps.num_base_stations <= 5
    )


def _add_f1_high_prb_candidates(
    ps: ParsedScenario,
    sf: ScenarioFeatures,
    scored: Dict[str, float],
    reasons: Dict[str, str],
) -> None:
    """
    F1 high-load, healthy-radio cases are the known hard boundary between
    `insufficient_data` and tertiary `lift_tilt`. Keep both hypotheses visible
    so the tool/LLM layer can arbitrate instead of losing the true option early.
    """
    sinr = sf.mobility.serving_sinr_db
    prb = sf.inter_freq.max_dl_prb_util
    if sinr is None or prb is None:
        return
    if sinr <= 5 or prb <= 70:
        return

    current_insuff = getattr(sf, "score_insufficient_data", 0.0) or 0.0
    # Prefer lift_tilt over insufficient_data in high-PRB healthy-radio group.
    # Phase 2 shows lift_tilt ≈2× more common; keep insuff as fallback only.
    lift_score = max(sf.score_coverage_hole or 0.0, 0.72)
    for target_role in ("primary", "tertiary"):
        hit = _find_best_option(ps, "lift_tilt", target_role, boost=lift_score)
        if hit:
            _upsert_candidate(
                scored,
                reasons,
                hit[0],
                hit[1],
                f"F1 high-PRB healthy-radio → lift_tilt/{target_role} hypothesis",
            )
            break  # prefer primary; fall through to tertiary only if primary absent

    insuff_score = max(current_insuff, 0.45)
    hit = _find_best_option(ps, "insufficient_data", None, boost=insuff_score)
    if hit:
        _upsert_candidate(
            scored,
            reasons,
            hit[0],
            hit[1],
            "F1 high-PRB healthy-radio → insufficient_data hypothesis",
        )


def _add_f1_arbitration_candidates(
    ps: ParsedScenario,
    sf: ScenarioFeatures,
    scored: Dict[str, float],
    reasons: Dict[str, str],
) -> None:
    """
    Keep the full F1 decision surface visible for single-answer Phase 2 cases.

    The 5-BS single-answer family is new in Phase 2 and tends to present a
    strong overlap/geometry signal.  If we only follow that signal, every case
    collapses to press_down_tilt.  Instead, inject the narrow F1 action set
    with evidence-weighted scores and keep press_down_tilt as a lower-priority
    hypothesis for the LLM arbiter to compare.
    """
    serving_cell = sf.mobility.serving_cell_id
    serving_role = _cell_role(ps, serving_cell)
    neighbor_cell = _best_neighbor_cell_id(ps, sf)
    neighbor_role = _cell_role(ps, neighbor_cell)
    five_bs_single = ps.num_base_stations == 5
    base_floor = 0.24 if five_bs_single else 0.08

    def add(
        action_type: str,
        target_cell: Optional[str],
        target_role: Optional[str],
        boost: float,
        reason: str,
        relation_cell: Optional[str] = None,
    ) -> None:
        hit = None
        if target_cell or action_type == "add_neighbor":
            hit = _find_best_option_for_cell(
                ps,
                action_type,
                target_cell,
                boost=boost,
                relation_cell_id=relation_cell,
            )
        if hit is None:
            hit = _find_best_option(ps, action_type, target_role, boost=boost)
        if hit:
            _upsert_candidate(scored, reasons, hit[0], hit[1], reason)

    # Handover and missing-neighbour hypotheses.
    add(
        "decrease_a3",
        serving_cell,
        serving_role or "primary",
        max(sf.score_late_handover, base_floor),
        "F1 arbitration candidate: decrease_a3 on serving cell",
    )
    add(
        "add_neighbor",
        neighbor_cell,
        neighbor_role or "secondary",
        max(sf.score_missing_neighbor, base_floor),
        "F1 arbitration candidate: add_neighbor between serving and best neighbor",
        relation_cell=serving_cell,
    )

    # Healthy-radio ambiguity: capacity/server/insufficient-data.
    add(
        "pdcch_2sym",
        serving_cell,
        serving_role or "primary",
        max(sf.score_pdcch_issue, base_floor * 0.90),
        "F1 arbitration candidate: pdcch_2sym on serving cell",
    )
    add(
        "check_server",
        None,
        None,
        max(sf.score_backhaul_server, base_floor * 0.90),
        "F1 arbitration candidate: check_server",
    )
    add(
        "insufficient_data",
        None,
        None,
        max(getattr(sf, "score_insufficient_data", 0.0) or 0.0, base_floor),
        "F1 arbitration candidate: insufficient_data",
    )

    # Coverage-shaping hypothesis.  Try exact serving and tertiary variants
    # because Phase 1 lift_tilt errors were usually target-cell mistakes.
    lift_boost = max(sf.score_coverage_hole, base_floor)
    add(
        "lift_tilt",
        serving_cell,
        serving_role or "primary",
        lift_boost,
        "F1 arbitration candidate: lift_tilt on serving cell",
    )
    add(
        "lift_tilt",
        None,
        "tertiary",
        max(lift_boost * 0.96, base_floor * 0.95),
        "F1 arbitration candidate: lift_tilt/tertiary",
    )

    # Keep press_down_tilt in the candidate set, but never let it become the
    # deterministic default for F1_5BS.  The LLM can still select it from the
    # all-options list when the evidence truly supports overlap/overshoot.
    press_boost = max(sf.score_overshoot_overlap * 0.20, 0.10)
    if five_bs_single:
        press_boost = min(press_boost, max(base_floor - 0.04, 0.10))
    add(
        "press_down_tilt",
        neighbor_cell,
        neighbor_role or "secondary",
        press_boost,
        "F1 arbitration candidate only: press_down_tilt on best-neighbor/interferer",
    )

    # adjust_azimuth: add as a candidate at moderate priority (does not beat
    # press_down_tilt from issue_map); kept as a candidate for LLM arbitration.
    azimuth_boost = max(
        sf.score_overshoot_overlap * 0.45,   # geometry signal from overlap
        sf.score_coverage_hole * 0.35,       # coverage shape signal
        base_floor,
    )
    add(
        "adjust_azimuth",
        serving_cell,
        serving_role or "primary",
        azimuth_boost,
        "F1 arbitration candidate: adjust_azimuth on serving cell",
    )
    add(
        "adjust_azimuth",
        neighbor_cell,
        neighbor_role or "secondary",
        max(azimuth_boost * 0.85, base_floor),
        "F1 arbitration candidate: adjust_azimuth on best-neighbor",
    )


def _apply_f1_action_prior(
    ps: ParsedScenario,
    scored: Dict[str, float],
    reasons: Dict[str, str],
) -> None:
    """
    Phase-1 F1 single-answer labels use a narrow action family. Demote geometry
    and power-shaping actions unless a later LLM/tool layer deliberately revives
    them from the full option list.
    """
    for c_code, score in list(scored.items()):
        action_type = ps.parsed_options.get(c_code, {}).get("action_type")
        if action_type in _F1_PREFERRED_ACTIONS:
            continue
        if action_type == "press_down_tilt" and ps.num_base_stations == 5:
            scored[c_code] = min(score, 0.20)
            reasons[c_code] = f"F1_5BS press_down_tilt kept as arbiter-only candidate: {reasons.get(c_code, '')}"
            continue
        scored[c_code] = min(score * 0.25, 0.12)
        reasons[c_code] = f"F1 non-primary action demoted: {reasons.get(c_code, '')}"


def _apply_f1_single_top_guard(
    ps: ParsedScenario,
    ranked: List[Tuple[str, float]],
    reasons: Dict[str, str],
) -> List[Tuple[str, float]]:
    """
    Keep F1 single-answer deterministic top1 inside the validated label family.

    Phase-1 F1 single-answer labels never use overlap/power/azimuth actions.
    Keep those options visible for arbitration, but do not let them become the
    deterministic answer when a safe F1 candidate is already present.
    """
    if not ranked:
        return ranked
    top_code, _ = ranked[0]
    top_action = ps.parsed_options.get(top_code, {}).get("action_type")
    if top_action in _F1_SAFE_TOP_ACTIONS:
        return ranked

    for idx, (candidate_code, _) in enumerate(ranked[1:], start=1):
        candidate_action = ps.parsed_options.get(candidate_code, {}).get("action_type")
        if candidate_action in _F1_SAFE_TOP_ACTIONS:
            guarded = ranked[:]
            guarded.insert(0, guarded.pop(idx))
            reasons[candidate_code] = (
                f"F1 single-answer top guard over {top_action}: "
                f"{reasons.get(candidate_code, '')}"
            )
            return guarded
    return ranked


def _generate_f4_candidates(
    ps: ParsedScenario,
    sf: ScenarioFeatures,
    top_n: int,
) -> List[RankedCandidate]:
    """
    Family-specific candidate generation for 13-BS questions.

    Training data for this family is extremely regular:
    - every answer is `decrease_interfreq`
    - there are exactly two such options
    - the correct target role is consistently `secondary`
    """
    scored: Dict[str, float] = {}
    reasons: Dict[str, str] = {}

    base = max(sf.score_inter_freq_load, 0.7)
    preferred_role = sf.inter_freq.recommended_target_role or "secondary"

    for target_role, weight in (
        (preferred_role, 1.0),
        ("secondary", 0.98),
        ("primary", 0.75),
        (None, 0.5),
    ):
        hit = _find_best_option(ps, "decrease_interfreq", target_role, boost=base * weight)
        if hit:
            c_code, score = hit
            _upsert_candidate(
                scored,
                reasons,
                c_code,
                score,
                f"F4 inter-frequency prior → decrease_interfreq/{target_role}",
            )

    for special in ("check_server", "insufficient_data"):
        hit = _find_best_option(ps, special, None, boost=0.04)
        if hit:
            c_code, score = hit
            _upsert_candidate(scored, reasons, c_code, score, f"low-score fallback: {special}")

    ranked = sorted(scored.items(), key=lambda x: -x[1])[:top_n]
    result = []
    for c_code, score in ranked:
        opt = ps.parsed_options.get(c_code, {})
        result.append(RankedCandidate(
            c_code=c_code,
            score=score,
            action_type=opt.get("action_type", "?"),
            target_cell=opt.get("target_cell"),
            reason=reasons.get(c_code, ""),
        ))
    return result


def generate_single_candidates(
    ps: ParsedScenario,
    sf: ScenarioFeatures,
    family: Family,
    top_n: int = 5,
) -> List[RankedCandidate]:
    """
    Generate a ranked list of up to top_n C-code candidates for single-answer scenarios.

    Strategy:
    1. Order issues by their composite score.
    2. For each issue, find the best-matching option(s).
    3. Deduplicate and rank.
    """
    if family.name == "F4":
        return _generate_f4_candidates(ps, sf, top_n)

    scored: Dict[str, float] = {}
    reasons: Dict[str, str] = {}
    f1_like_single = _is_f1_like_single(family, ps)

    issue_scores = [
        ("late_handover",    sf.score_late_handover),
        ("missing_neighbor", sf.score_missing_neighbor),
        ("inter_freq_load",  sf.score_inter_freq_load),
        ("interference",     sf.score_interference),
        ("coverage_hole",    sf.score_coverage_hole),
        ("overshoot_overlap",sf.score_overshoot_overlap),
        ("pdcch_issue",      sf.score_pdcch_issue),
        ("backhaul_server",  sf.score_backhaul_server),
    ]
    # Sort issues by score descending
    issue_scores.sort(key=lambda x: -x[1])

    for issue, issue_score in issue_scores:
        if issue_score < 0.05:
            continue
        action_list = _ISSUE_ACTION_MAP.get(issue, [])
        for action_type, target_role, action_boost in action_list:
            # For add_neighbor, pass serving_cell_id to pick the option whose
            # `cells` list includes the UE's serving cell (the correct relation endpoint).
            scid = sf.mobility.serving_cell_id if action_type == "add_neighbor" else None
            hit = _find_best_option(ps, action_type, target_role,
                                    boost=issue_score * action_boost,
                                    serving_cell_id=scid)
            if hit:
                c_code, score = hit
                _upsert_candidate(
                    scored,
                    reasons,
                    c_code,
                    score,
                    f"{issue} (score={issue_score:.2f}) → {action_type}/{target_role}",
                )

    _add_recommended_single_candidates(ps, sf, scored, reasons)
    if f1_like_single:
        _add_f1_high_prb_candidates(ps, sf, scored, reasons)
        _add_f1_arbitration_candidates(ps, sf, scored, reasons)

    # V3 FIX: inject serving-cell-targeted candidates for handover issues.
    # The serving cell's role is often "secondary" (not "primary"), which
    # the static _ISSUE_ACTION_MAP misses.
    serving_role = None
    if sf.mobility.serving_cell_id:
        serving_role = ps.cell_roles.get(sf.mobility.serving_cell_id)
    if serving_role:
        # V3 FIX: inject serving-cell-targeted candidates with a slight boost
        # (1.05×).  The serving cell's role is often "secondary", which the
        # static _ISSUE_ACTION_MAP (hardcoded to "primary") misses.
        if sf.score_late_handover >= 0.3:
            hit = _find_best_option(ps, "decrease_a3", serving_role,
                                     boost=sf.score_late_handover * 1.05)
            if hit:
                _upsert_candidate(scored, reasons, hit[0], hit[1],
                                   f"late_handover → decrease_a3/{serving_role} (serving cell)")
        if sf.score_missing_neighbor >= 0.3:
            nb_role = sf.mobility.recommended_target_role or "primary"
            hit = _find_best_option(ps, "add_neighbor", nb_role,
                                     boost=sf.score_missing_neighbor * 1.05,
                                     serving_cell_id=sf.mobility.serving_cell_id)
            if hit:
                _upsert_candidate(scored, reasons, hit[0], hit[1],
                                   f"missing_neighbor → add_neighbor/{nb_role} (neighbor cell)")
        if sf.score_pdcch_issue >= 0.3:
            hit = _find_best_option(ps, "pdcch_2sym", serving_role,
                                     boost=sf.score_pdcch_issue * 1.05)
            if hit:
                _upsert_candidate(scored, reasons, hit[0], hit[1],
                                   f"pdcch_issue → pdcch_2sym/{serving_role} (serving cell)")
        if sf.score_coverage_hole >= 0.3:
            # Both lift_tilt and increase_power are valid coverage fixes.
            # Boost the serving-cell-targeted lift_tilt to compete.
            for act in ("lift_tilt", "increase_power"):
                hit = _find_best_option(ps, act, serving_role,
                                         boost=sf.score_coverage_hole * 1.05)
                if hit:
                    _upsert_candidate(scored, reasons, hit[0], hit[1],
                                       f"coverage_hole → {act}/{serving_role} (serving cell)")

    tertiary_lift_hint = (
        (sf.geometry.recommended_action_family == "lift_tilt" and sf.geometry.recommended_target_role == "tertiary")
        or
        (sf.power.recommended_action_family == "lift_tilt" and sf.power.recommended_target_role == "tertiary")
    )
    if tertiary_lift_hint and sf.score_coverage_hole >= 0.3:
        hit = _find_best_option(
            ps,
            "lift_tilt",
            "tertiary",
            boost=sf.score_coverage_hole * 1.15,
        )
        if hit:
            _upsert_candidate(
                scored,
                reasons,
                hit[0],
                hit[1],
                "tool evidence → lift_tilt/tertiary",
            )

    if f1_like_single and sf.score_coverage_hole >= 0.3:
        hit = _find_best_option(
            ps,
            "lift_tilt",
            "tertiary",
            boost=max(sf.score_coverage_hole * 1.10, 0.50),
        )
        if hit:
            _upsert_candidate(
                scored,
                reasons,
                hit[0],
                hit[1],
                "F1 coverage hypothesis → lift_tilt/tertiary",
            )

    # Always include check_server and insufficient_data.
    # V3: boost these when the feature engine signals healthy radio
    # (high PRB or low PRB + healthy SINR → these become primary candidates).
    insufficient_score = getattr(sf, "score_insufficient_data", None) or 0.0
    for special in ("check_server", "insufficient_data"):
        if special == "insufficient_data":
            boost = max(insufficient_score, 0.05)
        else:
            boost = max(sf.score_backhaul_server, 0.05)
        hit = _find_best_option(ps, special, None, boost=boost)
        if hit:
            c_code, score = hit
            _upsert_candidate(scored, reasons, c_code, max(score, 0.05),
                              f"{special} (server_score={sf.score_backhaul_server:.2f}, insufficient_score={insufficient_score:.2f})")

    if f1_like_single:
        _apply_f1_action_prior(ps, scored, reasons)

    # Sort and build result
    ranked = sorted(scored.items(), key=lambda x: -x[1])[:top_n]
    if f1_like_single and _ENABLE_F1_SINGLE_TOP_GUARD:
        ranked = _apply_f1_single_top_guard(ps, ranked, reasons)
    result = []
    for c_code, score in ranked:
        opt = ps.parsed_options.get(c_code, {})
        result.append(RankedCandidate(
            c_code=c_code,
            score=score,
            action_type=opt.get("action_type", "?"),
            target_cell=opt.get("target_cell"),
            reason=reasons.get(c_code, ""),
        ))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Skill H — Multi-answer composer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnswerSet:
    c_codes: List[str]            # ascending order
    score: float
    reason: str


def _resolve_pair(
    ps: ParsedScenario,
    a_type: str, a_role: Optional[str],
    b_type: str, b_role: Optional[str],
    base_score: float,
    exclude: set,
) -> Optional[AnswerSet]:
    """Find concrete C-codes for a (type_A, role_A) + (type_B, role_B) pair."""
    hit_a = _find_best_option(ps, a_type, a_role, boost=base_score, exclude=exclude)
    if not hit_a:
        return None
    c_a, score_a = hit_a

    ex2 = exclude | {c_a}
    hit_b = _find_best_option(ps, b_type, b_role, boost=base_score, exclude=ex2)
    if not hit_b:
        return None
    c_b, score_b = hit_b

    if c_a == c_b:
        return None

    codes = sorted([c_a, c_b], key=lambda x: int(x[1:]))
    return AnswerSet(
        c_codes=codes,
        score=(score_a + score_b) / 2,
        reason=f"{a_type}/{a_role} + {b_type}/{b_role}",
    )


def _resolve_pair_for_cells(
    ps: ParsedScenario,
    a_type: str,
    a_cell: Optional[str],
    b_type: str,
    b_cell: Optional[str],
    base_score: float,
    exclude: set,
    reason: str,
    a_relation_cell: Optional[str] = None,
    b_relation_cell: Optional[str] = None,
    a_role_fallback: Optional[str] = None,
    b_role_fallback: Optional[str] = None,
) -> Optional[AnswerSet]:
    """Find a two-action answer set using exact cell targets before role fallback."""
    hit_a = _find_best_option_for_cell(
        ps,
        a_type,
        a_cell,
        boost=base_score,
        exclude=exclude,
        relation_cell_id=a_relation_cell,
    )
    if hit_a is None and a_role_fallback:
        hit_a = _find_best_option(
            ps,
            a_type,
            a_role_fallback,
            boost=base_score * 0.72,
            exclude=exclude,
        )
    if not hit_a:
        return None

    c_a, score_a = hit_a
    ex2 = exclude | {c_a}
    hit_b = _find_best_option_for_cell(
        ps,
        b_type,
        b_cell,
        boost=base_score,
        exclude=ex2,
        relation_cell_id=b_relation_cell,
    )
    if hit_b is None and b_role_fallback:
        hit_b = _find_best_option(
            ps,
            b_type,
            b_role_fallback,
            boost=base_score * 0.72,
            exclude=ex2,
        )
    if not hit_b:
        return None

    c_b, score_b = hit_b
    if c_a == c_b:
        return None

    codes = sorted([c_a, c_b], key=lambda x: int(x[1:]))
    return AnswerSet(
        c_codes=codes,
        score=(score_a + score_b) / 2,
        reason=reason,
    )


def _resolve_f2_four_action_candidate(
    ps: ParsedScenario,
    serving_cell: Optional[str],
    neighbor_cell: Optional[str],
    base_score: float,
) -> Optional[AnswerSet]:
    if not serving_cell or not neighbor_cell:
        return None

    steps = [
        ("decrease_a3", serving_cell, None),
        ("add_neighbor", neighbor_cell, serving_cell),
        ("decrease_power", serving_cell, None),
        ("press_down_tilt", serving_cell, None),
    ]
    codes: List[str] = []
    scores: List[float] = []
    used: set = set()
    for action_type, target_cell, relation_cell in steps:
        hit = _find_best_option_for_cell(
            ps,
            action_type,
            target_cell,
            boost=base_score,
            exclude=used,
            relation_cell_id=relation_cell,
        )
        if not hit:
            return None
        c_code, score = hit
        codes.append(c_code)
        scores.append(score)
        used.add(c_code)

    if len(set(codes)) != 4:
        return None

    return AnswerSet(
        c_codes=sorted(codes, key=lambda x: int(x[1:])),
        score=sum(scores) / len(scores),
        reason=(
            "F2 comprehensive 4-action fix: decrease_a3 + add_neighbor + "
            "decrease_power + press_down_tilt "
            f"(serving={serving_cell}, best_neighbor={neighbor_cell})"
        ),
    )


def _append_unique_answer_set(result: List[AnswerSet], pair: Optional[AnswerSet]) -> None:
    if pair and pair.c_codes not in [r.c_codes for r in result]:
        result.append(pair)


def generate_multi_candidates(
    ps: ParsedScenario,
    sf: ScenarioFeatures,
    family: Family,
    top_n_sets: int = 4,
) -> List[AnswerSet]:
    """
    Skill H: generate ranked answer SETS for multiple-answer scenarios.

    F3 (5 BS) always needs exactly 4 answers.
    F2 (4 BS) needs 2 answers (76%) or 4 answers (24%).
    """
    result: List[AnswerSet] = []
    used_codes: set = set()

    if ps.num_base_stations == 13:
        # Phase 2 adds 13-BS multiple-answer cases.  The 13-BS family is still
        # dominated by inter-frequency/load fixes, so the safest multi-answer
        # set is the two concrete decrease_interfreq options.
        base = max(sf.score_inter_freq_load, 0.7)
        preferred_role = sf.inter_freq.recommended_target_role or "secondary"
        scored_codes: List[Tuple[str, float]] = []
        for c_code, opt in ps.parsed_options.items():
            if opt.get("action_type") != "decrease_interfreq":
                continue
            role = ps.cell_roles.get(opt.get("target_cell"))
            weight = 1.0
            if role == preferred_role:
                weight = 1.08
            elif role == "secondary":
                weight = 1.04
            elif role == "primary":
                weight = 0.96
            scored_codes.append((c_code, base * weight))

        scored_codes.sort(key=lambda x: (-x[1], int(x[0][1:])))
        if len(scored_codes) >= 2:
            codes = sorted([c for c, _ in scored_codes[:2]], key=lambda x: int(x[1:]))
            score = sum(s for _, s in scored_codes[:2]) / 2.0
            result.append(AnswerSet(
                c_codes=codes,
                score=score,
                reason="Phase2 13BS inter-frequency pair",
            ))

        # Keep one broader fallback candidate for diagnostics/manual review.
        fallback = _resolve_pair(
            ps,
            "decrease_interfreq", preferred_role,
            "decrease_a3", "primary",
            base * 0.75,
            set(c for r in result for c in r.c_codes),
        )
        if fallback:
            result.append(fallback)

        return result[:top_n_sets]

    if family.name == "F3":
        # Try all 4-action templates
        top_issue, top_score = sf.top_issue()
        top_score = max(top_score, 0.3)
        for template in _F3_4ACTION_TEMPLATES:
            codes = []
            total_score = 0.0
            ex = set()
            valid = True
            for a_type, a_role in template:
                hit = _find_best_option(ps, a_type, a_role, boost=top_score, exclude=ex)
                if not hit:
                    valid = False
                    break
                c_code, sc = hit
                codes.append(c_code)
                total_score += sc
                ex.add(c_code)
            if valid and len(set(codes)) == 4:
                sorted_codes = sorted(codes, key=lambda x: int(x[1:]))
                result.append(AnswerSet(
                    c_codes=sorted_codes,
                    score=total_score / 4,
                    reason=f"F3 template: {[t[0] for t in template]}",
                ))
        if not result:
            # Fallback: top 4 single candidates combined
            singles = generate_single_candidates(ps, sf, family, top_n=4)
            if len(singles) >= 4:
                codes = sorted([s.c_code for s in singles], key=lambda x: int(x[1:]))
                result.append(AnswerSet(c_codes=codes, score=0.3, reason="fallback: top-4 singles"))
        return result[:top_n_sets]

    # F2: relationship-aware 2-action pairs.
    #
    # Older code selected pairs mostly by inferred primary/secondary role.
    # That role is derived from option frequency and is brittle in Phase 2.
    # Prefer exact serving / best-neighbour cell relationships, with role-based
    # templates retained only as low-priority fallbacks.
    issue_scores = sorted([
        ("late_handover",    sf.score_late_handover),
        ("missing_neighbor", sf.score_missing_neighbor),
        ("interference",     sf.score_interference),
        ("coverage_hole",    sf.score_coverage_hole),
        ("overshoot_overlap",sf.score_overshoot_overlap),
        ("insufficient_data", getattr(sf, "score_insufficient_data", 0.0) or 0.0),
    ], key=lambda x: -x[1])
    top_issue = issue_scores[0][0] if issue_scores else "uncertain"
    top_score = max(issue_scores[0][1], 0.3) if issue_scores else 0.3
    serving_cell = sf.mobility.serving_cell_id
    neighbor_cell = _best_neighbor_cell_id(ps, sf, exclude={serving_cell})
    serving_role = _cell_role(ps, serving_cell)
    neighbor_role = _cell_role(ps, neighbor_cell)
    tool_evidence = getattr(sf, "tool_evidence", None)
    has_tool_evidence = bool(
        tool_evidence is not None
        and getattr(tool_evidence, "tool_call_count", 0)
    )
    late_missing_score = max(sf.score_late_handover, sf.score_missing_neighbor)
    four_action_score = 0.12
    if (
        has_tool_evidence
        and top_issue in ("late_handover", "missing_neighbor")
        and late_missing_score >= 0.62
    ):
        four_action_score = max(0.70, min(late_missing_score, 0.76))
    elif has_tool_evidence and late_missing_score >= 0.45:
        four_action_score = max(four_action_score, min(late_missing_score * 0.60, 0.36))
    if serving_cell and neighbor_cell:
        _append_unique_answer_set(
            result,
            _resolve_f2_four_action_candidate(
                ps,
                serving_cell,
                neighbor_cell,
                four_action_score,
            ),
        )

    if not has_tool_evidence:
        preferred_pairs: List[Tuple[str, Optional[str], str, Optional[str]]] = []
        if top_issue == "overshoot_overlap":
            preferred_pairs.extend([
                ("adjust_azimuth", "secondary", "increase_power", "secondary"),
                ("decrease_power", "primary", "press_down_tilt", "primary"),
                ("decrease_power", "secondary", "press_down_tilt", "secondary"),
            ])
        else:
            preferred_pairs.extend([
                ("decrease_power", "primary", "press_down_tilt", "primary"),
                ("adjust_azimuth", "secondary", "increase_power", "secondary"),
                ("decrease_power", "secondary", "press_down_tilt", "secondary"),
            ])

        for a_type, a_role, b_type, b_role in preferred_pairs:
            if len(result) >= top_n_sets:
                break
            pair = _resolve_pair(ps, a_type, a_role, b_type, b_role, top_score, set())
            _append_unique_answer_set(result, pair)

        for a_type, a_role, b_type, b_role in _MULTI_PAIRS:
            if len(result) >= top_n_sets:
                break
            pair = _resolve_pair(ps, a_type, a_role, b_type, b_role, top_score, set())
            _append_unique_answer_set(result, pair)

        result.sort(key=lambda x: -x.score)
        return result[:top_n_sets]

    fallback_interferer = _first_distinct_cell_id(ps, exclude={serving_cell})
    interferer_cell = neighbor_cell or fallback_interferer
    interferer_role = _cell_role(ps, interferer_cell) or neighbor_role or "secondary"
    coverage_cell = serving_cell or ps.get_primary_cell_id() or ps.get_secondary_cell_id()
    coverage_role = _cell_role(ps, coverage_cell) or serving_role or "primary"
    overlap_ratio = getattr(tool_evidence, "overlap_ratio", None)
    neighbor_advantage = sf.mobility.neighbor_advantage_db or 0.0
    missing_relation = bool(
        neighbor_cell
        and not sf.mobility.neighbor_in_config
        and neighbor_advantage > 1.0
    )

    if missing_relation or top_issue == "missing_neighbor":
        score = max(sf.score_missing_neighbor, sf.score_late_handover, 0.68)
        _append_unique_answer_set(
            result,
            _resolve_pair_for_cells(
                ps,
                "add_neighbor", neighbor_cell,
                "decrease_a3", serving_cell,
                score,
                set(),
                (
                    "F2 relationship: add neighbor between serving "
                    f"{serving_cell or '?'} and best neighbor {neighbor_cell or '?'} "
                    f"+ decrease_a3 on serving {serving_cell or '?'}"
                ),
                a_relation_cell=serving_cell,
                a_role_fallback=neighbor_role or "secondary",
                b_role_fallback=serving_role or "primary",
            ),
        )

    overlap_high = overlap_ratio is not None and overlap_ratio > 0.30
    interference_like = top_issue in ("interference", "overshoot_overlap")
    has_distinct_interferer = bool(
        interferer_cell and (not serving_cell or interferer_cell != serving_cell)
    )
    if (overlap_high or interference_like) and has_distinct_interferer:
        score = max(sf.score_interference, sf.score_overshoot_overlap, 0.58)
        if overlap_high:
            score = max(score, 0.74)
        elif overlap_ratio is not None and overlap_ratio < 0.20:
            score = min(score, 0.52)
        high_overlap_pair = _resolve_pair_for_cells(
            ps,
            "decrease_power", interferer_cell,
            "press_down_tilt", interferer_cell,
            score,
            set(),
            (
                "F2 relationship: high-overlap/interference fix on "
                f"interferer {interferer_cell or '?'} "
                f"(serving={serving_cell or '?'}, best_neighbor={neighbor_cell or '?'}, "
                f"overlap={overlap_ratio})"
            ),
            a_role_fallback=interferer_role,
            b_role_fallback=interferer_role,
        )
        if high_overlap_pair and serving_cell:
            target_cells = {
                ps.parsed_options.get(code, {}).get("target_cell")
                for code in high_overlap_pair.c_codes
            }
            if serving_cell in target_cells:
                high_overlap_pair = None
        _append_unique_answer_set(result, high_overlap_pair)

    coverage_like = (
        top_issue in ("coverage_hole", "insufficient_data", "uncertain")
        or not overlap_high
    )
    if coverage_like:
        score = max(sf.score_coverage_hole, top_score * 0.92, 0.58)
        if overlap_ratio is not None and overlap_ratio <= 0.30:
            score = max(score, 0.66)
        _append_unique_answer_set(
            result,
            _resolve_pair_for_cells(
                ps,
                "adjust_azimuth", coverage_cell,
                "increase_power", coverage_cell,
                score,
                set(),
                (
                    "F2 relationship: coverage-shaping fix on serving/weak cell "
                    f"{coverage_cell or '?'} (serving={serving_cell or '?'}, "
                    f"overlap={overlap_ratio})"
                ),
                a_role_fallback=coverage_role,
                b_role_fallback=coverage_role,
            ),
        )
        if neighbor_cell and neighbor_cell != coverage_cell:
            _append_unique_answer_set(
                result,
                _resolve_pair_for_cells(
                    ps,
                    "adjust_azimuth", neighbor_cell,
                    "increase_power", neighbor_cell,
                    score * 0.94,
                    set(),
                    (
                        "F2 relationship fallback: coverage-shaping fix on "
                        f"best neighbor {neighbor_cell}"
                    ),
                    a_role_fallback=neighbor_role or "secondary",
                    b_role_fallback=neighbor_role or "secondary",
                ),
            )

    for a_type, a_role, b_type, b_role in _MULTI_PAIRS:
        if len(result) >= top_n_sets:
            break
        pair = _resolve_pair(ps, a_type, a_role, b_type, b_role, top_score * 0.82, set())
        if pair:
            pair.reason = "F2 role fallback: " + pair.reason
        _append_unique_answer_set(result, pair)

    result.sort(key=lambda x: -x.score)
    return result[:top_n_sets]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateOutput:
    family: Family
    is_multiple: bool
    # For single-answer
    single_candidates: List[RankedCandidate] = field(default_factory=list)
    # For multiple-answer
    multi_candidates: List[AnswerSet] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        """Format candidates for injection into the Qwen selection prompt."""
        lines = ["## Candidate Options\n"]
        if not self.is_multiple:
            for i, c in enumerate(self.single_candidates, 1):
                opt_label = ""
                lines.append(
                    f"{i}. **{c.c_code}** (score={c.score:.2f}) "
                    f"— {c.action_type} [{c.target_cell or 'any'}] "
                    f"| {c.reason}"
                )
        else:
            for i, s in enumerate(self.multi_candidates, 1):
                codes_str = "|".join(s.c_codes)
                lines.append(
                    f"{i}. **{codes_str}** (score={s.score:.2f}) — {s.reason}"
                )
        return "\n".join(lines)

    def top_candidate(self) -> Optional[str]:
        """Return the top-ranked answer string (for confidence fallback)."""
        if not self.is_multiple and self.single_candidates:
            return self.single_candidates[0].c_code
        if self.is_multiple and self.multi_candidates:
            return "|".join(self.multi_candidates[0].c_codes)
        return None


def generate_candidates(
    ps: ParsedScenario,
    sf: ScenarioFeatures,
    family: Family,
) -> CandidateOutput:
    """
    Top-level entry point: generate candidates for a scenario.
    """
    out = CandidateOutput(family=family, is_multiple=family.is_multiple)

    if not family.is_multiple:
        top_n = 8 if family.name == "F1_5BS" else 5
        out.single_candidates = generate_single_candidates(ps, sf, family, top_n=top_n)
    else:
        top_n_sets = 6 if family.name == "F2" else 4
        out.multi_candidates = generate_multi_candidates(ps, sf, family, top_n_sets=top_n_sets)

    return out
