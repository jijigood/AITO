#!/usr/bin/env python3
"""Audit diagnostic SFT ceilings before another LoRA training run.

The two questions this answers:
1. Is the ground-truth answer visible in deterministic top-k candidates?
2. Where do insufficient_data labels fail under the current candidate system?
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from skills.agentic_overlay import compute_risk_level
from skills.candidate_generator import generate_multi_candidates, generate_single_candidates
from skills.feature_engine import run_all_skills
from skills.scenario_parser import parse_scenario
from skills.scenario_router import route


DEFAULT_HELDOUT_CANDIDATES = [
    Path("heldout_f1_val_ids.txt"),
    Path("/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track A/heldout_f1_val_ids.txt"),
    Path("/home/huangxiaolin/AITO/dataset/Track A/v3/heldout_f1_val_ids.txt"),
]


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_heldout_ids(path: Optional[Path]) -> set[str]:
    if path is None:
        for candidate in DEFAULT_HELDOUT_CANDIDATES:
            if candidate.exists():
                path = candidate
                break
    if path is None or not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _answer_codes(answer: Optional[str]) -> List[str]:
    if not answer:
        return []
    text = str(answer).strip()
    text = text.replace("\\boxed{", "").replace("}", "")
    codes = re.findall(r"C\d+", text)
    return sorted(set(codes), key=lambda x: int(x[1:]))


def _feature_scores(sf: Any) -> Dict[str, float]:
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


def _mr_count(ps: Any) -> int:
    mr = getattr(ps, "mr", None)
    if mr is None or mr.empty:
        return 0
    return int(mr.shape[0])


def _candidate_rows(ps: Any, family: Any, sf: Any, top_k: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if getattr(family, "is_multiple", False):
        candidates = generate_multi_candidates(ps, sf, family, top_n_sets=top_k)
        for rank, cand in enumerate(candidates, start=1):
            codes = list(cand.c_codes)
            opts = [ps.parsed_options.get(code, {}) for code in codes]
            rows.append({
                "rank": rank,
                "codes": codes,
                "score": round(float(cand.score or 0.0), 4),
                "reason": cand.reason,
                "actions": [opt.get("action_type", "?") for opt in opts],
                "target_cells": [opt.get("target_cell") for opt in opts],
                "target_roles": [
                    ps.cell_roles.get(opt.get("target_cell"))
                    for opt in opts
                ],
            })
        return rows

    candidates = generate_single_candidates(ps, sf, family, top_n=top_k)
    for rank, cand in enumerate(candidates, start=1):
        rows.append({
            "rank": rank,
            "codes": [cand.c_code],
            "score": round(float(cand.score or 0.0), 4),
            "reason": cand.reason,
            "actions": [cand.action_type],
            "target_cells": [cand.target_cell],
            "target_roles": [ps.cell_roles.get(cand.target_cell)],
        })
    return rows


def _exact_rank(gt_codes: Sequence[str], candidates: Sequence[Dict[str, Any]]) -> Optional[int]:
    gt_set = set(gt_codes)
    if not gt_set:
        return None
    for row in candidates:
        if set(row.get("codes") or []) == gt_set:
            return int(row["rank"])
    return None


def _coverage_bucket(
    gt_codes: Sequence[str],
    parsed_options: Dict[str, Dict[str, Any]],
    exact_rank: Optional[int],
    top_k: int,
) -> str:
    if not gt_codes:
        return "no_gt"
    if any(code not in parsed_options for code in gt_codes):
        return "gt_not_parsed"
    if exact_rank == 1:
        return "top1"
    if exact_rank is not None and exact_rank <= 3:
        return "top3"
    if exact_rank is not None and exact_rank <= top_k:
        return f"top{top_k}"
    return f"options_not_top{top_k}"


def _gt_actions(ps: Any, gt_codes: Sequence[str]) -> List[str]:
    return [
        str(ps.parsed_options.get(code, {}).get("action_type", "?"))
        for code in gt_codes
    ]


def _top(row: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    if not row:
        return default
    values = row.get(key)
    if isinstance(values, list):
        return values[0] if values else default
    return values if values is not None else default


def _is_generic_primary_lift(row: Optional[Dict[str, Any]]) -> bool:
    if not row:
        return False
    action = str(_top(row, "actions", "") or "").lower()
    reason = str(row.get("reason") or "").lower()
    role = str(_top(row, "target_roles", "") or "").lower()
    return (
        action == "lift_tilt"
        and "high-prb healthy-radio" in reason
        and role == "primary"
        and "coverage hypothesis" not in reason
    )


def _insufficient_category(top_row: Optional[Dict[str, Any]]) -> str:
    if not top_row:
        return "no_candidate"
    action = str(_top(top_row, "actions", "") or "").lower()
    reason = str(top_row.get("reason") or "").lower()
    role = str(_top(top_row, "target_roles", "") or "").lower()
    if action == "insufficient_data":
        return "top_already_insufficient_data"
    if action == "lift_tilt":
        if "coverage hypothesis" in reason and role == "tertiary":
            return "top_lift_tilt_coverage_tertiary"
        if "high-prb healthy-radio" in reason and role == "primary":
            return "top_lift_tilt_generic_primary"
        return "top_lift_tilt_other"
    return f"top_{action or 'unknown'}"


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            serialised = {}
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    serialised[key] = json.dumps(value, ensure_ascii=False)
                else:
                    serialised[key] = value
            writer.writerow(serialised)


def _pct(n: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(100.0 * n / total):.1f}%"


def _table(counter: Counter, total: Optional[int] = None, limit: Optional[int] = None) -> List[str]:
    total = total if total is not None else sum(counter.values())
    lines = ["| item | count | pct |", "|---|---:|---:|"]
    items = counter.most_common(limit)
    for key, count in items:
        lines.append(f"| {key} | {count} | {_pct(count, total)} |")
    return lines


def _group_count(rows: Iterable[Dict[str, Any]], *keys: str) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        counter[" / ".join(str(row.get(key, "")) for key in keys)] += 1
    return counter


def _write_report(
    path: Path,
    coverage_rows: Sequence[Dict[str, Any]],
    insufficient_rows: Sequence[Dict[str, Any]],
    top_k: int,
    heldout_count: int,
) -> None:
    total = len(coverage_rows)
    errors = [r for r in coverage_rows if r.get("error")]
    non_error = [r for r in coverage_rows if not r.get("error")]
    exact_override = [r for r in non_error if str(r.get("should_override_exact")) == "True"]
    action_override = [r for r in non_error if str(r.get("should_override_action")) == "True"]

    lines: List[str] = [
        "# Diagnostic Training Ceiling Audit",
        "",
        f"- scenarios: {total}",
        f"- heldout ids loaded: {heldout_count}",
        f"- candidate top_k: {top_k}",
        f"- errors: {len(errors)}",
        "",
        "## GT coverage bucket",
        "",
    ]
    lines.extend(_table(Counter(r.get("coverage_bucket", "") for r in non_error), len(non_error)))

    lines.extend([
        "",
        "## GT coverage by family",
        "",
    ])
    lines.extend(_table(_group_count(non_error, "family", "coverage_bucket"), len(non_error), limit=30))

    lines.extend([
        "",
        "## Single-answer GT action coverage",
        "",
    ])
    single_rows = [r for r in non_error if r.get("is_multiple") is False]
    lines.extend(_table(_group_count(single_rows, "gt_action", "coverage_bucket"), len(single_rows), limit=40))

    lines.extend([
        "",
        "## Override supervision ceiling",
        "",
        f"- exact answer differs from deterministic top1: {len(exact_override)} / {len(non_error)} ({_pct(len(exact_override), len(non_error))})",
        f"- single-answer action differs from deterministic top action: {len(action_override)} / {len(single_rows)} ({_pct(len(action_override), len(single_rows))})",
        "",
        "## Insufficient data failure audit",
        "",
        f"- insufficient_data GT rows: {len(insufficient_rows)}",
        "",
        "### Category",
        "",
    ])
    lines.extend(_table(Counter(r.get("insufficient_category", "") for r in insufficient_rows), len(insufficient_rows)))

    lines.extend([
        "",
        "### GT coverage for insufficient_data",
        "",
    ])
    lines.extend(_table(Counter(r.get("coverage_bucket", "") for r in insufficient_rows), len(insufficient_rows)))

    lines.extend([
        "",
        "### Risk branch for insufficient_data",
        "",
    ])
    lines.extend(_table(_group_count(insufficient_rows, "risk_level", "risk_branch"), len(insufficient_rows), limit=30))

    lines.extend([
        "",
        "### Current subtype gate would invoke overlay",
        "",
    ])
    lines.extend(_table(Counter(str(r.get("current_subtype_gate_invokes")) for r in insufficient_rows), len(insufficient_rows)))

    lines.extend([
        "",
        "### Insufficient data by family/category",
        "",
    ])
    lines.extend(_table(_group_count(insufficient_rows, "family", "insufficient_category"), len(insufficient_rows), limit=40))

    if errors:
        lines.extend([
            "",
            "## Errors",
            "",
        ])
        lines.extend(_table(Counter(str(r.get("error")) for r in errors), len(errors)))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(args: argparse.Namespace) -> None:
    rows = _load_json(args.train_json)
    heldout_ids = _load_heldout_ids(args.heldout_ids)
    coverage_rows: List[Dict[str, Any]] = []
    insufficient_rows: List[Dict[str, Any]] = []

    for idx, raw in enumerate(rows, start=1):
        sid = raw.get("scenario_id", "")
        if args.progress and (idx == 1 or idx % args.progress == 0):
            print(f"[audit] {idx}/{len(rows)} {sid}", flush=True)

        base: Dict[str, Any] = {
            "scenario_id": sid,
            "split": "heldout" if sid in heldout_ids else "train",
            "tag": raw.get("tag", ""),
        }
        try:
            ps = parse_scenario(raw)
            family = route(ps.tag, ps.num_base_stations)
            sf = run_all_skills(ps)
            scores = _feature_scores(sf)
            risk = compute_risk_level(sf, family, scores)
            candidates = _candidate_rows(ps, family, sf, args.top_k)
            gt_codes = _answer_codes(ps.ground_truth)
            rank = _exact_rank(gt_codes, candidates)
            bucket = _coverage_bucket(gt_codes, ps.parsed_options, rank, args.top_k)
            gt_actions = _gt_actions(ps, gt_codes)
            top_row = candidates[0] if candidates else None
            top_codes = top_row.get("codes") if top_row else []
            top_actions = top_row.get("actions") if top_row else []
            exact_top = set(top_codes or []) == set(gt_codes)
            single = not bool(getattr(family, "is_multiple", False))
            gt_action = gt_actions[0] if single and gt_actions else ""
            top_action = (top_actions or [""])[0] if single else "|".join(top_actions or [])
            action_diff = bool(single and gt_action and top_action and gt_action != top_action)

            row = {
                **base,
                "family": family.name,
                "num_base_stations": ps.num_base_stations,
                "is_multiple": bool(family.is_multiple),
                "option_count": len(ps.parsed_options),
                "gt_answer": ps.ground_truth or "",
                "gt_codes": gt_codes,
                "gt_actions": gt_actions,
                "gt_action": gt_action,
                "gt_in_parsed_options": all(code in ps.parsed_options for code in gt_codes),
                "gt_missing_codes": [code for code in gt_codes if code not in ps.parsed_options],
                "gt_exact_rank": rank or "",
                "coverage_bucket": bucket,
                "det_top_codes": top_codes,
                "det_top_action": top_action,
                "det_top_reason": top_row.get("reason", "") if top_row else "",
                "det_top_score": top_row.get("score", "") if top_row else "",
                "det_top_target_cell": _top(top_row, "target_cells", ""),
                "det_top_target_role": _top(top_row, "target_roles", ""),
                "should_override_exact": not exact_top,
                "should_override_action": action_diff,
                "risk_level": risk.get("level"),
                "risk_branch": risk.get("branch") or "",
                "risk_reasons": risk.get("reasons") or [],
                "mr_count": _mr_count(ps),
                "score_coverage_hole": scores.get("coverage_hole", 0.0),
                "score_insufficient_data": scores.get("insufficient_data", 0.0),
                "score_missing_neighbor": scores.get("missing_neighbor", 0.0),
                "score_late_handover": scores.get("late_handover", 0.0),
            }
            coverage_rows.append(row)

            if single and gt_action == "insufficient_data":
                insuf_row = dict(row)
                insuf_row["insufficient_category"] = _insufficient_category(top_row)
                insuf_row["current_subtype_gate_invokes"] = bool(
                    family.name in {"F1", "F1_5BS"}
                    and risk.get("branch") == "lift_tilt_vs_insufficient_data"
                    and _is_generic_primary_lift(top_row)
                )
                insufficient_rows.append(insuf_row)
        except Exception as exc:  # noqa: BLE001 - audit should keep going
            coverage_rows.append({**base, "error": f"{type(exc).__name__}: {exc}"})

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = out_dir / "gt_candidate_coverage.csv"
    insufficient_path = out_dir / "insufficient_data_failure_audit.csv"
    report_path = out_dir / "gt_candidate_coverage_report.md"
    _write_csv(coverage_path, coverage_rows)
    _write_csv(insufficient_path, insufficient_rows)
    _write_report(report_path, coverage_rows, insufficient_rows, args.top_k, len(heldout_ids))
    print(f"[audit] wrote {coverage_path}")
    print(f"[audit] wrote {insufficient_path}")
    print(f"[audit] wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, default=Path("data/Phase_1/train.json"))
    parser.add_argument("--heldout-ids", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results/diagnostic_audits"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--progress", type=int, default=200)
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
