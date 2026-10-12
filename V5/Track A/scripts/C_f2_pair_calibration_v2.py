#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
C_f2_pair_calibration_v2.py — Fixed version of C's F2 pair calibration.

Changes vs V5_C original:
  1. Default base_csv → v5_11 (baseline + F2 structural fix; no bad F1 flips).
  2. --exclude_ids: skip rows already analyzed by D as "keep" with conflicting
     evidence; defaults to the 6 D-keep rows that C's original patch overwrote.
  3. flip_score fixed: original formula (confidence × top_issue_score) collapses
     to 0.0 for "uncertain" cases (top_issue_score == 0 by definition). Fixed to
     use phase1_confidence directly so uncertain candidates rank at 1.0, not 0.0.
  4. Tilt code selection sorted numerically for stable output.
  5. Default max_flips=999 (effectively uncapped; 70 clean candidates after D-keep exclusion).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skills.scenario_parser import parse_scenario, parse_option_label
from skills.feature_engine import run_all_skills


F2_PAIR_AZIMUTH = frozenset({"adjust_azimuth", "increase_power"})
F2_PAIR_TILT    = frozenset({"decrease_power", "press_down_tilt"})
F2_LEGAL_PAIRS  = (F2_PAIR_AZIMUTH, F2_PAIR_TILT)

# D's "keep" rows from spot-check that C's original patch overwrote.
# These had azimuth pair rated KEEP based on API-derived top_issue=late_handover.
# C's offline run computed top_issue=uncertain for the same rows and flipped them
# back to tilt — a direct contradiction. Exclude them by default.
D_KEEP_EXCLUDE = {
    "f428f517-0bcd-4d13-b31a-929eb3584363",
    "732ac9cf-5ad2-4934-a5f6-f41d6edde04e",
    "ddb040fe-708c-472d-8ef6-584b8ec5b7ec",
    "f02d340d-18c0-43c9-ae9c-c6132c5fe719",
    "e83a09dd-795a-4066-9f5c-2de6da46dfea",
    "a4176c2c-458b-4c9c-b89b-c0c873747d0b",
}


def _load_action_map(options: List[Dict[str, str]]) -> Dict[str, str]:
    return {
        str(o.get("id", "")).strip(): parse_option_label(str(o.get("label", ""))).get("action_type", "?")
        for o in options
    }


def _answer_to_actions(action_map: Dict[str, str], answer: str) -> frozenset:
    codes = [c.strip() for c in (answer or "").split("|") if c.strip()]
    return frozenset(action_map.get(c, "?") for c in codes)


def _get_pair_type(actions: frozenset) -> Optional[str]:
    if actions == F2_PAIR_AZIMUTH:
        return "azimuth"
    if actions == F2_PAIR_TILT:
        return "tilt"
    return None


def _sort_codes(codes: List[str]) -> List[str]:
    """Sort C-codes numerically: C3 < C9 < C12."""
    def key(c: str) -> int:
        return int(c[1:]) if c.startswith("C") and c[1:].isdigit() else 9999
    return sorted(codes, key=key)


def analyze_phase1_train(train_json: Path) -> Dict[str, Any]:
    """Extract top_issue → pair distribution from Phase 1 F2 training data."""
    scenarios = json.loads(train_json.read_text(encoding="utf-8"))

    top_issue_to_pairs: Dict[str, List[str]] = defaultdict(list)
    all_f2_rows = 0
    valid_f2_rows = 0

    for raw in scenarios:
        tag = raw.get("tag", "")
        if "multiple" not in tag.lower():
            continue
        wi = raw.get("context", {}).get("wireless_network_information", {})
        num_bs = int(str(wi.get("num_base_stations", "4")).strip())
        if num_bs != 4:
            continue

        all_f2_rows += 1
        answer = raw.get("answer", "")
        if not answer:
            continue

        options = (raw.get("task") or {}).get("options") or []
        action_map = _load_action_map(options)
        actions = _answer_to_actions(action_map, answer)
        pair_type = _get_pair_type(actions)
        if pair_type is None:
            continue

        valid_f2_rows += 1
        try:
            ps = parse_scenario(raw)
            features = run_all_skills(ps)
            top_issue, _ = features.top_issue()
        except Exception:
            top_issue = "unknown"

        top_issue_to_pairs[top_issue].append(pair_type)

    distribution: Dict[str, Any] = {}
    for issue, pairs in top_issue_to_pairs.items():
        count = Counter(pairs)
        total = sum(count.values())
        az, ti = count.get("azimuth", 0), count.get("tilt", 0)
        distribution[issue] = {
            "count": total,
            "azimuth": az,
            "tilt": ti,
            "azimuth_pct": az / total if total else 0.0,
            "tilt_pct": ti / total if total else 0.0,
            "majority": "azimuth" if az >= ti else "tilt",
            "confidence": abs(az - ti) / total if total else 0.0,
        }

    return {
        "total_f2_rows": all_f2_rows,
        "valid_f2_rows": valid_f2_rows,
        "top_issue_distribution": distribution,
    }


def evaluate_phase2_candidates(
    phase1_dist: Dict[str, Any],
    test_json: Path,
    base_csv: Path,
    confidence_threshold: float,
    exclude_ids: Set[str],
) -> Dict[str, Any]:
    """Find Phase 2 F2 rows where current pair should be flipped per Phase 1 prior."""
    scenarios = json.loads(test_json.read_text(encoding="utf-8"))
    rows, base_answers = _load_submission(base_csv)

    candidates = []
    f2_count = 0
    card2_count = 0
    excluded_count = 0

    for raw in scenarios:
        sid = str(raw.get("scenario_id", "")).strip()
        if not sid:
            continue

        tag = raw.get("tag", "")
        if "multiple" not in tag.lower():
            continue
        wi = raw.get("context", {}).get("wireless_network_information", {})
        num_bs = int(str(wi.get("num_base_stations", "4")).strip())
        if num_bs != 4:
            continue

        f2_count += 1

        if sid in exclude_ids:
            excluded_count += 1
            continue

        current_answer = base_answers.get(sid, "")
        if not current_answer:
            continue

        options = (raw.get("task") or {}).get("options") or []
        action_map = _load_action_map(options)
        current_actions = _answer_to_actions(action_map, current_answer)
        current_pair = _get_pair_type(current_actions)
        if current_pair is None:
            continue

        codes = [c.strip() for c in current_answer.split("|") if c.strip()]
        if len(codes) != 2:
            continue

        card2_count += 1

        try:
            ps = parse_scenario(raw)
            features = run_all_skills(ps)
            top_issue, top_score = features.top_issue()
        except Exception:
            top_issue = "unknown"
            top_score = 0.0

        dist = phase1_dist.get("top_issue_distribution", {}).get(top_issue)
        if dist is None:
            continue

        phase1_majority = dist["majority"]
        confidence = dist["confidence"]

        if current_pair == phase1_majority or confidence < confidence_threshold:
            continue

        # FIX: original formula `confidence * top_issue_score` collapses to 0.0
        # for "uncertain" (top_issue_score == 0 by definition). Use phase1
        # confidence directly so these candidates rank at their true signal strength.
        flip_score = round(confidence * max(top_score, 1.0), 4) if top_score == 0.0 else round(confidence * top_score, 4)

        candidates.append({
            "scenario_id": sid,
            "top_issue": top_issue,
            "top_issue_score": round(top_score, 4),
            "current_answer": current_answer,
            "current_pair": current_pair,
            "current_actions": sorted(current_actions),
            "phase1_majority": phase1_majority,
            "phase1_confidence": round(confidence, 4),
            "phase1_bucket_size": dist["count"],
            "phase1_azimuth_pct": round(dist["azimuth_pct"], 4),
            "flip_score": flip_score,
        })

    candidates.sort(key=lambda x: -x["flip_score"])

    return {
        "total_f2_rows": f2_count,
        "card2_f2_rows": card2_count,
        "excluded_count": excluded_count,
        "candidates": candidates,
        "high_confidence_count": len(candidates),
    }


def _load_submission(path: Path) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [{k: (v or "") for k, v in row.items()} for row in csv.DictReader(f)]
    answers = {row["ID"]: row.get("Track A", "") for row in rows}
    return rows, answers


def _write_submission(path: Path, rows: List[Dict[str, str]], answers: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ID", "Track A", "Track B"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fieldnames}
            out["Track A"] = answers.get(row["ID"], row.get("Track A", ""))
            writer.writerow(out)


def generate_patch_csv(
    base_csv: Path,
    candidates: List[Dict[str, Any]],
    test_json: Path,
    out_csv: Path,
    max_flips: int,
) -> Dict[str, Any]:
    scenarios = json.loads(test_json.read_text(encoding="utf-8"))
    scenarios_by_id = {str(s.get("scenario_id", "")).strip(): s for s in scenarios}

    rows, base_answers = _load_submission(base_csv)
    new_answers = dict(base_answers)

    flips_applied = []
    skipped = []

    for cand in candidates[:max_flips]:
        sid = cand["scenario_id"]
        target_pair = cand["phase1_majority"]

        scenario = scenarios_by_id.get(sid)
        if not scenario:
            skipped.append({"sid": sid, "reason": "scenario not found"})
            continue

        options = (scenario.get("task") or {}).get("options") or []
        action_map = {
            str(o.get("id", "")).strip(): parse_option_label(str(o.get("label", "")))
            for o in options
        }

        target_actions = F2_PAIR_AZIMUTH if target_pair == "azimuth" else F2_PAIR_TILT
        matching_codes = [
            cid for cid, parsed in action_map.items()
            if parsed.get("action_type") in target_actions
        ]

        if len(matching_codes) < 2:
            skipped.append({"sid": sid, "reason": f"not enough {target_pair} codes"})
            continue

        # FIX: sort codes numerically for stable output
        sorted_codes = _sort_codes(matching_codes[:2])
        new_answer = "|".join(sorted_codes)
        new_answers[sid] = new_answer
        flips_applied.append({
            "sid": sid,
            "old": cand["current_answer"],
            "old_pair": cand["current_pair"],
            "new": new_answer,
            "new_pair": target_pair,
            "top_issue": cand["top_issue"],
            "confidence": cand["phase1_confidence"],
            "flip_score": cand["flip_score"],
        })

    _write_submission(out_csv, rows, new_answers)
    return {
        "flips_applied": flips_applied,
        "skipped": skipped,
        "total_flips": len(flips_applied),
        "out_csv": str(out_csv),
    }


def _generate_report(result: Dict[str, Any], out_path: Path) -> None:
    p1 = result["phase1_analysis"]
    p2 = result["phase2_evaluation"]
    patch = result["patch_result"]
    params = result["params"]

    lines = [
        "# C — F2 Pair Calibration Report (v2)",
        "",
        "## Changes vs V5_C original",
        "",
        "- Base CSV: v5_11 (baseline + F2 structural fixes only; no v5_07 F1 flips)",
        "- Excluded: 6 D-keep rows where D's API-based analysis contradicts C's offline top_issue",
        "- flip_score fixed: uncertain cases now score at phase1_confidence (1.0), not 0.0",
        "- Tilt code selection sorted numerically for deterministic output",
        "",
        "## Parameters",
        "",
        f"- Confidence threshold: {params['confidence_threshold']}",
        f"- Max flips: {params['max_flips']}",
        f"- Base CSV: {Path(params['base_csv']).name}",
        f"- Excluded IDs: {len(params['exclude_ids'])}",
        "",
        "## Phase 1 Analysis",
        "",
        f"Total F2 rows: {p1['total_f2_rows']}",
        f"Valid F2 rows with legal pairs: {p1['valid_f2_rows']}",
        "",
        "### top_issue -> pair distribution",
        "",
        "| top_issue | total | azimuth | tilt | azimuth % | tilt % | majority | confidence |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for issue, d in sorted(p1["top_issue_distribution"].items(), key=lambda x: -x[1]["count"]):
        lines.append(
            f"| {issue} | {d['count']} | {d['azimuth']} | {d['tilt']} | "
            f"{d['azimuth_pct']:.1%} | {d['tilt_pct']:.1%} | {d['majority']} | {d['confidence']:.2f} |"
        )

    lines.extend([
        "",
        "## Phase 2 Evaluation",
        "",
        f"Total F2 rows: {p2['total_f2_rows']}",
        f"Cardinality-2 F2 rows: {p2['card2_f2_rows']}",
        f"Excluded (D-keep conflicts): {p2['excluded_count']}",
        f"Flip candidates: {p2['high_confidence_count']}",
        "",
        "### Candidates (sorted by flip_score)",
        "",
        "| scenario_id | top_issue | current_pair | target_pair | confidence | flip_score |",
        "|---|---|---|---|---|---|",
    ])
    for c in p2["candidates"][:params["max_flips"]]:
        lines.append(
            f"| {c['scenario_id']} | {c['top_issue']} | {c['current_pair']} | "
            f"{c['phase1_majority']} | {c['phase1_confidence']:.2f} | {c['flip_score']:.4f} |"
        )

    lines.extend([
        "",
        "## Patch Result",
        "",
        f"Flips applied: {patch['total_flips']}",
        f"Skipped: {len(patch.get('skipped', []))}",
        "",
        "### Applied Flips",
        "",
        "| scenario_id | top_issue | old | old_pair | new | new_pair | confidence |",
        "|---|---|---|---|---|---|---|",
    ])
    for flip in patch.get("flips_applied", []):
        lines.append(
            f"| {flip['sid']} | {flip['top_issue']} | {flip['old']} | {flip['old_pair']} | "
            f"{flip['new']} | {flip['new_pair']} | {flip['confidence']:.2f} |"
        )

    if patch.get("skipped"):
        lines.extend(["", "### Skipped", ""])
        for s in patch["skipped"]:
            lines.append(f"- {s['sid']}: {s['reason']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="F2 pair calibration v2 (fixed flip_score + D-keep exclusion)")
    ap.add_argument("--train_json", type=Path,
                    default=ROOT / "data" / "Phase_1" / "train.json")
    ap.add_argument("--test_json", type=Path,
                    default=ROOT / "data" / "Phase_2" / "test.json")
    ap.add_argument("--base_csv", type=Path,
                    default=ROOT / "submissions" / "submission_600_v5_11_baseline_plus_f2_pair_fix.csv")
    ap.add_argument("--confidence_threshold", type=float, default=0.70)
    ap.add_argument("--max_flips", type=int, default=999,
                    help="Cap on flips applied (999 = effectively uncapped; 70 clean candidates)")
    ap.add_argument("--exclude_ids", type=str, nargs="*", default=None,
                    help="Scenario IDs to skip (default: D's 6 keep-conflict rows)")
    ap.add_argument("--out_csv", type=Path,
                    default=ROOT / "submissions" / "submission_600_C_f2_calibrated_v2.csv")
    ap.add_argument("--report_json", type=Path,
                    default=ROOT / "reports" / "C_f2_pair_calibration_v2.json")
    ap.add_argument("--report_md", type=Path,
                    default=ROOT / "reports" / "C_f2_pair_calibration_v2.md")
    args = ap.parse_args()

    exclude_ids: Set[str] = set(args.exclude_ids) if args.exclude_ids is not None else D_KEEP_EXCLUDE

    print("Analyzing Phase 1 F2 training data...")
    p1 = analyze_phase1_train(args.train_json)
    print(f"  F2 rows: {p1['total_f2_rows']} total, {p1['valid_f2_rows']} valid")
    for issue, d in sorted(p1["top_issue_distribution"].items(), key=lambda x: -x[1]["count"]):
        print(f"  {issue:22s}  n={d['count']:3d}  az={d['azimuth']:3d}({d['azimuth_pct']:.0%})  "
              f"tilt={d['tilt']:3d}({d['tilt_pct']:.0%})  majority={d['majority']:8s}  conf={d['confidence']:.2f}")

    print("\nEvaluating Phase 2 F2 candidates...")
    p2 = evaluate_phase2_candidates(
        p1, args.test_json, args.base_csv, args.confidence_threshold, exclude_ids
    )
    print(f"  F2 rows: {p2['total_f2_rows']} total, {p2['card2_f2_rows']} card-2")
    print(f"  Excluded (D-keep conflicts): {p2['excluded_count']}")
    print(f"  Flip candidates: {p2['high_confidence_count']}")

    print("\nGenerating patch CSV...")
    patch = generate_patch_csv(args.base_csv, p2["candidates"], args.test_json, args.out_csv, args.max_flips)
    print(f"  Flips applied: {patch['total_flips']}, skipped: {len(patch.get('skipped', []))}")
    for flip in patch["flips_applied"]:
        print(f"    {flip['sid'][:8]}...  {flip['old_pair']} → {flip['new_pair']}  "
              f"(top_issue={flip['top_issue']} conf={flip['confidence']:.2f})")

    full = {
        "phase1_analysis": p1,
        "phase2_evaluation": p2,
        "patch_result": patch,
        "params": {
            "confidence_threshold": args.confidence_threshold,
            "max_flips": args.max_flips,
            "base_csv": str(args.base_csv),
            "train_json": str(args.train_json),
            "test_json": str(args.test_json),
            "exclude_ids": sorted(exclude_ids),
        },
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _generate_report(full, args.report_md)
    print(f"\nReports: {args.report_json.name}, {args.report_md.name}")
    print(f"Patch CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
