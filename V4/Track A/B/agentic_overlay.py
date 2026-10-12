"""
agentic_overlay.py — Controlled Agentic Loop for Phase 2 high-risk scenarios.

Architecture:
  Deterministic backbone runs first and produces a candidate answer.
  This overlay intercepts only HIGH-RISK cases and runs a 2-call LLM loop
  (at most 1 additional tool invocation) to resolve the ambiguity.

MVP covers two branches:
  1. lift_tilt_vs_insufficient_data   (coverage_hole score ≈ insufficient_data score)
  2. add_neighbor_vs_decrease_a3      (missing_neighbor score ≈ late_handover score)

Integration point in phase2_run._solve_scenario():
    After _format_answer(), before _apply_fine_tuned_verifier():

        if agentic_policy == "high_risk" and llm_verifier is not None:
            answer = run_agentic_overlay(
                ps, sf, gatherer, family, tag,
                scores=_feature_scores(sf),
                deterministic_answer=answer,
                task_description=scenario.get("task", {}).get("description", ""),
                llm_verifier=llm_verifier,
                diagnostics=diagnostics,
                logger=logger,
            )
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Branch → allowed tool menus
# Pre-fetched data (mr_data, config_data, kpi_data, user_plane_data) is
# already available; only ADDITIONAL single calls are listed here.
# get_signaling_plane_event_log is intentionally omitted — returns empty live.
# ---------------------------------------------------------------------------

BRANCH_TOOL_MENUS: Dict[str, List[str]] = {
    "lift_tilt_vs_insufficient_data": [
        "judge_mainlobe_or_not",    # Outside_Mainlobe_Flag -> coverage hole vs no signal
        "calculate_tilt_angle",     # vertical angle -> can tilt help?
        "get_user_location",        # confirm user is far from cell
    ],
    "add_neighbor_vs_decrease_a3": [
        "get_neighboring_cells_pci",    # live neighbor list at worst_time
        "get_neighboring_cell_rsrp",    # RSRP of a specific neighbor
        "get_cell_info",                # is that PCI in the config?
    ],
}


# ---------------------------------------------------------------------------
# Risk level computation
# ---------------------------------------------------------------------------

def compute_risk_level(
    sf,
    family,
    scores: Dict[str, float],
) -> Dict[str, Any]:
    """
    Assess whether this scenario needs the agentic overlay.

    Args:
        sf:     ScenarioFeatures (post feature_engine + tool_checks)
        family: Family object (use family.name for comparisons)
        scores: dict from _feature_scores(sf)

    Returns:
        {
            "level":   "low" | "medium" | "high",
            "branch":  str | None,          # which tool menu to use
            "reasons": List[str],           # for Person D diff report
        }
    """
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_name,  top_score  = sorted_items[0] if sorted_items else ("", 0.0)
    sec_name,  sec_score  = sorted_items[1] if len(sorted_items) > 1 else ("", 0.0)
    gap = round(top_score - sec_score, 4)

    reasons: List[str] = []
    branch:  Optional[str] = None
    level = "low"

    # --- Branch 1: lift_tilt vs insufficient_data ---
    cov   = scores.get("coverage_hole",    0.0)
    insuf = scores.get("insufficient_data", 0.0)
    is_single = not bool(getattr(family, "is_multiple", False))
    if cov > 0.20 and insuf > 0.20:
        diff = round(abs(cov - insuf), 4)
        if diff < 0.25 or (is_single and insuf >= 0.65 and cov >= 0.20):
            branch = "lift_tilt_vs_insufficient_data"
            level  = "high"
            reasons.append(
                f"coverage_and_insufficient_close:cov={cov:.3f},insuf={insuf:.3f},diff={diff:.3f}"
            )

    # --- Branch 2: add_neighbor vs decrease_a3 ---
    miss = scores.get("missing_neighbor", 0.0)
    late = scores.get("late_handover",    0.0)
    if miss > 0.20 and late > 0.20 and branch is None:
        diff = round(abs(miss - late), 4)
        if diff < 0.25 or (is_single and miss >= 0.25 and late >= 0.35):
            branch = "add_neighbor_vs_decrease_a3"
            level  = "high"
            reasons.append(
                f"add_neighbor_vs_late_close:miss={miss:.3f},late={late:.3f},diff={diff:.3f}"
            )

    # --- General ambiguity (medium only, no tool menu) ---
    if level == "low":
        if gap < 0.12:
            level = "medium"
            reasons.append(f"candidate_gap={gap:.3f}")
        else:
            reasons.append(f"top={top_name}:{top_score:.3f},gap={gap:.3f}")

    if not reasons:
        reasons.append(f"top={top_name}:{top_score:.3f},gap={gap:.3f}")

    return {"level": level, "branch": branch, "reasons": reasons}


# ---------------------------------------------------------------------------
# Pre-fetched evidence builder (no extra API calls)
# ---------------------------------------------------------------------------

def _build_pre_fetched_evidence(
    ps,
    sf,
    gatherer,
    scores: Dict[str, float],
) -> Dict[str, Any]:
    """
    Assemble evidence already in memory. This is what we hand to the LLM
    without any additional API round-trips.
    """
    mob = sf.mobility  # MobilityFeatures dataclass

    serving_pci     = gatherer.serving_pci
    serving_cell_id = (
        getattr(mob, "serving_cell_id", None)
        or (ps.pci_to_cell_id.get(int(serving_pci)) if serving_pci else None)
        or "?"
    )

    best_nbr_pci   = getattr(mob, "best_neighbor_pci",    None)
    best_nbr_rsrp  = getattr(mob, "best_neighbor_rsrp_dbm", None)
    nbr_advantage  = getattr(mob, "neighbor_advantage_db",  None)
    nbr_in_config  = getattr(mob, "neighbor_in_config",     None)
    best_nbr_cell_id = (
        ps.pci_to_cell_id.get(int(best_nbr_pci)) if best_nbr_pci else None
    )

    mr_count = (
        int(ps.mr.shape[0])
        if (getattr(ps, "mr", None) is not None and not ps.mr.empty)
        else 0
    )

    return {
        "worst_time":              gatherer.worst_time,
        "serving_pci":             serving_pci,
        "serving_cell_id":         serving_cell_id,
        "serving_sinr_dB":         getattr(mob, "serving_sinr_db",  None),
        "serving_rsrp_dBm":        getattr(mob, "serving_rsrp_dbm", None),
        "best_neighbor_pci":       best_nbr_pci,
        "best_neighbor_cell_id":   best_nbr_cell_id,
        "best_neighbor_rsrp_dBm":  best_nbr_rsrp,
        "neighbor_advantage_dB":   nbr_advantage,
        "neighbor_in_config":      nbr_in_config,
        "mr_count":                mr_count,
        "feature_scores":          {k: round(v, 3) for k, v in scores.items()},
    }


# ---------------------------------------------------------------------------
# Constrained tool execution
# ---------------------------------------------------------------------------

def _normalise_tool_args(
    tool_name: str,
    args: Dict[str, Any],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """Fill missing common tool args from already-known evidence."""
    args = dict(args or {})
    if args.get("time") in (None, "", "<worst_time>"):
        args["time"] = evidence.get("worst_time")

    if tool_name in {"judge_mainlobe_or_not", "calculate_tilt_angle"}:
        # These branch checks are defined against the serving cell at worst_time.
        # Override model-supplied PCI to avoid invalid or copied neighbor PCI.
        args["pci"] = evidence.get("serving_pci")
    elif tool_name in {"get_neighboring_cell_rsrp", "get_cell_info"}:
        # These checks inspect the best visible neighbor.
        args["pci"] = evidence.get("best_neighbor_pci") or evidence.get("serving_pci")
    elif tool_name == "get_user_location":
        args.pop("pci", None)
    elif tool_name == "get_neighboring_cells_pci":
        args.pop("pci", None)

    return {k: v for k, v in args.items() if v is not None and v != ""}


def _validate_and_execute_tool(
    tool_request: Dict[str, Any],
    branch: str,
    evidence: Dict[str, Any],
    gatherer,
    logger: logging.Logger,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate that the LLM-requested tool is in BRANCH_TOOL_MENUS[branch],
    then execute via gatherer._call_tool().

    Returns (tool_name, result) or (None, None) on validation/call failure.
    The LLM cannot invoke tools outside the branch menu.
    """
    tool_name = (tool_request.get("tool") or "").strip()
    args      = tool_request.get("args") or {}

    allowed = BRANCH_TOOL_MENUS.get(branch, [])
    if tool_name not in allowed:
        logger.warning(
            "[agentic] rejected tool='%s' (not in menu for branch='%s'); allowed=%s",
            tool_name, branch, allowed,
        )
        return None, None

    if not isinstance(args, dict):
        args = {}
    args = _normalise_tool_args(tool_name, args, evidence)

    try:
        result = gatherer._call_tool(tool_name, **args)
        error  = result.get("error") if isinstance(result, dict) else None
        if error:
            logger.warning("[agentic] tool '%s' returned error: %s", tool_name, error)
            return tool_name, None
        return tool_name, result
    except Exception as exc:
        logger.warning("[agentic] tool '%s' raised: %s", tool_name, exc)
        return tool_name, None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_STEP1_SYSTEM = """\
You are a 5G network diagnosis expert. You are resolving an ambiguous fault scenario.

Pre-fetched evidence is provided. You may request ONE additional tool call to resolve
the ambiguity, or skip directly to diagnosis if the evidence is already sufficient.

Output ONLY valid JSON — one of these two formats (no markdown, no explanation):

Request a tool:
{{"type": "tool_request", "tool": "<tool_name>", "args": {{"time": "<worst_time>", "pci": <int>}}}}

Skip to diagnosis (evidence is sufficient):
{{"type": "skip_to_diagnosis"}}

Available tools for this branch: {available_tools}
"""

_STEP2_SYSTEM = """\
You are a 5G network diagnosis expert. Based on all available evidence, output your
final diagnosis as JSON.

Output ONLY this JSON object (no markdown, no explanation):
{{
  "root_cause": "<missing_neighbor|late_handover|coverage_hole|insufficient_data>",
  "action": "<add_neighbor|decrease_a3|lift_tilt|insufficient_data>",
  "target_pci": <integer or null>,
  "target_cell_id": "<cell_id string or null>",
  "relation_pci": <integer or null>,
  "relation_cell_id": "<cell_id string or null>",
  "evidence_confidence": "<high|medium|low>"
}}

Rules:
- For add_neighbor:  target = the missing neighbor cell; relation = the serving cell.
- For decrease_a3:   target = the serving cell; relation = null.
- For lift_tilt:     target = the serving cell; relation = null.
- For insufficient_data: target = null; relation = null.
- Use cell_id format (e.g. "3279943_2") wherever possible; pci is the integer.
"""


def _evidence_block(evidence: Dict[str, Any]) -> str:
    return json.dumps(evidence, indent=2, ensure_ascii=False)


def _options_text(ps) -> str:
    return "\n".join(
        f"{cid}: {opt.get('raw') or opt.get('label') or '(no label)'}"
        for cid, opt in sorted(
            ps.parsed_options.items(),
            key=lambda x: int(x[0][1:]) if x[0][1:].isdigit() else 99,
        )
    )


def _build_step1_messages(
    branch: str,
    evidence: Dict[str, Any],
    task_description: str,
    options_text: str,
) -> List[Dict[str, str]]:
    system = _STEP1_SYSTEM.format(
        available_tools=", ".join(BRANCH_TOOL_MENUS.get(branch, []))
    )
    user = (
        f"Diagnostic branch: {branch}\n\n"
        f"Task:\n{task_description}\n\n"
        f"Answer options:\n{options_text}\n\n"
        f"Pre-fetched evidence:\n{_evidence_block(evidence)}\n\n"
        "Do you need one additional tool call, or is the evidence sufficient?\n"
        "Output JSON."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_step2_messages(
    branch: str,
    evidence: Dict[str, Any],
    tool_name: Optional[str],
    tool_result: Optional[Dict[str, Any]],
    task_description: str,
    options_text: str,
) -> List[Dict[str, str]]:
    tool_section = ""
    if tool_name and tool_result is not None:
        tool_section = (
            f"\nAdditional tool result [{tool_name}]:\n"
            f"{json.dumps(tool_result, indent=2, ensure_ascii=False)}\n"
        )
    user = (
        f"Diagnostic branch: {branch}\n\n"
        f"Task:\n{task_description}\n\n"
        f"Answer options:\n{options_text}\n\n"
        f"Pre-fetched evidence:\n{_evidence_block(evidence)}"
        f"{tool_section}\n"
        "Output your diagnosis JSON."
    )
    return [{"role": "system", "content": _STEP2_SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# Diagnosis → C-code mapper
# ---------------------------------------------------------------------------

def _map_diagnosis_to_ccode(
    diagnosis: Dict[str, Any],
    ps,
    deterministic_answer: str,
    logger: logging.Logger,
) -> str:
    """
    Map LLM diagnosis {action, target_cell_id, relation_cell_id} to a C-code.
    Matching priority:
      1. action_type match (required)
      2. For add_neighbor: both target and relation appear in option cells (+2 each)
      3. For single-cell actions: target appears in cells (+3)
    Falls back to deterministic_answer if no option scores above 0.
    """
    action           = (diagnosis.get("action")          or "").strip()
    target_cell_id   = (diagnosis.get("target_cell_id")  or "").strip()
    relation_cell_id = (diagnosis.get("relation_cell_id") or "").strip()
    target_pci       = diagnosis.get("target_pci")
    relation_pci     = diagnosis.get("relation_pci")

    # Resolve cell_ids from PCIs if not provided
    if not target_cell_id and target_pci:
        target_cell_id = ps.pci_to_cell_id.get(int(target_pci), "")
    if not relation_cell_id and relation_pci:
        relation_cell_id = ps.pci_to_cell_id.get(int(relation_pci), "")

    action_norm = action.lower().replace("_", "").replace(" ", "")
    best_cid:   Optional[str]   = None
    best_score: float           = -1.0
    matched_action_count = 0

    for cid, opt in ps.parsed_options.items():
        opt_action = (opt.get("action_type") or opt.get("action") or "").strip()
        opt_cells: List[str] = opt.get("cells") or []

        # Action must loosely match (normalise underscores/spaces)
        oa_norm  = opt_action.lower().replace("_", "").replace(" ", "")
        if not (action_norm in oa_norm or oa_norm in action_norm):
            continue
        matched_action_count += 1

        score = 1.0  # base: action matched

        if action_norm == "addneighbor":
            if target_cell_id   and target_cell_id   in opt_cells:
                score += 2.0
            if relation_cell_id and relation_cell_id in opt_cells:
                score += 2.0
        else:
            if target_cell_id and target_cell_id in opt_cells:
                score += 3.0

        if score > best_score:
            best_score = score
            best_cid   = cid

    no_target_actions = {"insufficientdata", "checkserver"}
    if action_norm in no_target_actions:
        min_score = 1.0
    elif action_norm == "addneighbor":
        # Do not let the agent pick an arbitrary add-neighbor option.  It must
        # identify at least one endpoint, and both endpoints are preferred.
        min_score = 3.0
    else:
        # decrease_a3 / lift_tilt are cell-targeted; require a concrete target
        # unless there is only one option of that action type.
        min_score = 1.0 if matched_action_count == 1 and not target_cell_id else 4.0

    if best_cid and best_score >= min_score:
        logger.info(
            "[agentic] mapped diagnosis → %s  (action=%s target=%s relation=%s score=%.1f)",
            best_cid, action, target_cell_id, relation_cell_id, best_score,
        )
        return best_cid

    logger.warning(
        "[agentic] mapping failed/low-confidence (action=%s target=%s relation=%s "
        "best=%s score=%.1f min=%.1f) → falling back to deterministic=%s",
        action, target_cell_id, relation_cell_id, best_cid, best_score, min_score,
        deterministic_answer,
    )
    return deterministic_answer


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_agentic_overlay(
    ps,
    sf,
    gatherer,
    family,
    tag: str,
    scores: Dict[str, float],
    deterministic_answer: str,
    task_description: str,
    llm_verifier,
    diagnostics: Dict[str, Any],
    logger: logging.Logger,
) -> str:
    """
    Run the controlled agentic overlay for one scenario.

    Always returns a non-empty C-code string. Falls back to deterministic_answer
    at any failure point (LLM error, JSON parse failure, mapping failure).

    Writes to diagnostics:
        agentic_path:           "skipped" | "high_risk" | "llm_error_step1" |
                                "llm_error_step2" | "json_parse_failed" | "completed"
        agentic_branch:         str | None
        agentic_risk:           {level, branch, reasons}
        agentic_tools_called:   List[{tool, args, result_keys}]
        agentic_diagnosis:      dict from LLM step 2
        agentic_answer:         str   (LLM-derived, before fallback check)
        deterministic_answer:   str   (what the backbone would have returned)
    """
    risk = compute_risk_level(sf, family, scores)
    diagnostics["agentic_risk"] = risk

    if risk["level"] != "high" or not risk.get("branch"):
        diagnostics["agentic_path"] = "skipped"
        return deterministic_answer

    branch = risk["branch"]
    diagnostics["agentic_path"]          = "high_risk"
    diagnostics["agentic_branch"]        = branch
    diagnostics["agentic_tools_called"]  = []
    diagnostics["deterministic_answer"]  = deterministic_answer

    opts_text = _options_text(ps)
    evidence  = _build_pre_fetched_evidence(ps, sf, gatherer, scores)
    diagnostics["agentic_prefetched_evidence"] = evidence

    # ------------------------------------------------------------------
    # LLM call 1 — tool request or skip?
    # ------------------------------------------------------------------
    step1_msgs = _build_step1_messages(branch, evidence, task_description, opts_text)

    raw1, err1 = llm_verifier.call(step1_msgs, enable_thinking=False)
    diagnostics["agentic_step1_raw_preview"] = raw1[:500] if raw1 else ""
    if err1:
        logger.warning("[agentic] step1 error: %s", err1)
        diagnostics["agentic_path"] = "llm_error_step1"
        return deterministic_answer

    tool_name:   Optional[str]            = None
    tool_result: Optional[Dict[str, Any]] = None

    parsed1 = _extract_json(raw1)
    if parsed1 and parsed1.get("type") == "tool_request":
        tool_name, tool_result = _validate_and_execute_tool(
            parsed1, branch, evidence, gatherer, logger
        )
        if tool_name:
            diagnostics["agentic_tools_called"].append({
                "tool":        tool_name,
                "args":        parsed1.get("args", {}),
                "result_keys": (
                    list(tool_result.keys())
                    if isinstance(tool_result, dict)
                    else str(type(tool_result))
                ) if tool_result is not None else "call_failed",
            })

    # ------------------------------------------------------------------
    # LLM call 2 — final diagnosis
    # ------------------------------------------------------------------
    step2_msgs = _build_step2_messages(
        branch, evidence, tool_name, tool_result, task_description, opts_text
    )

    raw2, err2 = llm_verifier.call(step2_msgs, enable_thinking=False)
    diagnostics["agentic_step2_raw_preview"] = raw2[:500] if raw2 else ""
    if err2:
        logger.warning("[agentic] step2 error: %s", err2)
        diagnostics["agentic_path"] = "llm_error_step2"
        return deterministic_answer

    diagnosis = _extract_json(raw2)
    if not diagnosis:
        logger.warning("[agentic] step2 JSON parse failed; raw=%r", raw2[:300])
        diagnostics["agentic_path"] = "json_parse_failed"
        return deterministic_answer

    diagnostics["agentic_diagnosis"] = diagnosis
    diagnostics["agentic_path"]      = "completed"

    agentic_answer = _map_diagnosis_to_ccode(
        diagnosis, ps, deterministic_answer, logger
    )
    diagnostics["agentic_answer"] = agentic_answer

    logger.info(
        "[agentic] branch=%s det=%s agentic=%s confidence=%s",
        branch, deterministic_answer, agentic_answer,
        diagnosis.get("evidence_confidence", "?"),
    )
    return agentic_answer
