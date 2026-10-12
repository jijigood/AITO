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
        ("add_neighbor",       None,        1.0),   # any add_neighbor option
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


# ──────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────────────────────────────────────────────

def _score_option(
    opt: Dict[str, Any],
    action_type: str,
    target_role: Optional[str],
    ps: ParsedScenario,
    boost: float = 1.0,
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
) -> Optional[Tuple[str, float]]:
    """
    Find the highest-scoring C-code for a given (action_type, target_role).
    Returns (c_code, score) or None.
    """
    best_code, best_score = None, 0.0
    for cid, opt in ps.parsed_options.items():
        if exclude and cid in exclude:
            continue
        s = _score_option(opt, action_type, target_role, ps, boost)
        if s > best_score:
            best_score = s
            best_code = cid
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
    Add a small amount of family-agnostic diversification based on each skills's
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
        hit = _find_best_option(ps, action_type, target_role, boost=base_score)
        if hit:
            c_code, score = hit
            _upsert_candidate(scored, reasons, c_code, score, reason)


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
            hit = _find_best_option(ps, action_type, target_role, boost=issue_score * action_boost)
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
            hit = _find_best_option(ps, "add_neighbor", serving_role,
                                     boost=sf.score_missing_neighbor * 1.05)
            if hit:
                _upsert_candidate(scored, reasons, hit[0], hit[1],
                                   f"missing_neighbor → add_neighbor/{serving_role} (serving cell)")
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

    # Always include check_server and insufficient_data.
    # V3: boost these when the feature engine signals healthy radio
    # (high PRB or low PRB + healthy SINR → these become primary candidates).
    server_boost = max(sf.score_backhaul_server, 0.05)
    for special in ("check_server", "insufficient_data"):
        hit = _find_best_option(ps, special, None, boost=server_boost)
        if hit:
            c_code, score = hit
            _upsert_candidate(scored, reasons, c_code, max(score, 0.05),
                              f"{special} (server_score={sf.score_backhaul_server:.2f})")

    # Sort and build result
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

    if family.name == "F3":
        # Try all 4-action templates
        top_issue, top_score = sf.top_issue()
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

    # F2: 2-action pairs (primary) + optional 4-action sets
    issue_scores = sorted([
        ("late_handover",    sf.score_late_handover),
        ("missing_neighbor", sf.score_missing_neighbor),
        ("interference",     sf.score_interference),
        ("coverage_hole",    sf.score_coverage_hole),
        ("overshoot_overlap",sf.score_overshoot_overlap),
    ], key=lambda x: -x[1])
    top_issue = issue_scores[0][0] if issue_scores else "uncertain"
    top_score = max(issue_scores[0][1], 0.3) if issue_scores else 0.3

    # Family-specific priors learned from the training distribution:
    # F2 has only two dominant answer patterns.
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
        if pair and pair.c_codes not in [r.c_codes for r in result]:
            result.append(pair)

    for a_type, a_role, b_type, b_role in _MULTI_PAIRS:
        if len(result) >= top_n_sets:
            break
        pair = _resolve_pair(ps, a_type, a_role, b_type, b_role, top_score, set())
        if pair and pair.c_codes not in [r.c_codes for r in result]:
            result.append(pair)

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
        out.single_candidates = generate_single_candidates(ps, sf, family, top_n=5)
    else:
        out.multi_candidates = generate_multi_candidates(ps, sf, family, top_n_sets=4)

    return out
