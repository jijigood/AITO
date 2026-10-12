#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
F2 pair calibration using Phase 1 training data.

This script:
1. Runs feature pipeline on Phase 1 F2 training data to extract top_issue
2. Builds conditional distribution: top_issue -> legal pair
3. Applies this distribution to Phase 2 F2 rows to find high-confidence flips
4. Outputs ranked flip list and patch CSV

Legal F2 pairs:
- azimuth pair: {adjust_azimuth, increase_power}
- tilt pair: {decrease_power, press_down_tilt}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skills.scenario_parser import parse_scenario, parse_option_label
from skills.feature_engine import run_all_skills


F2_PAIR_AZIMUTH = frozenset({"adjust_azimuth", "increase_power"})
F2_PAIR_TILT = frozenset({"decrease_power", "press_down_tilt"})
F2_LEGAL_PAIRS = (F2_PAIR_AZIMUTH, F2_PAIR_TILT)


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
    elif actions == F2_PAIR_TILT:
        return "tilt"
    return None


def analyze_phase1_train(train_json: Path) -> Dict[str, Any]:
    """Extract top_issue -> pair distribution from Phase 1 F2 training data."""
    scenarios = json.loads(train_json.read_text(encoding="utf-8"))
    
    top_issue_to_pairs: Dict[str, List[str]] = defaultdict(list)
    all_f2_rows = 0
    valid_f2_rows = 0
    
    for raw_scenario in scenarios:
        tag = raw_scenario.get("tag", "")
        if "multiple" not in tag.lower():
            continue
        
        num_bs = int(str(raw_scenario.get("context", {}).get("wireless_network_information", {}).get("num_base_stations", "4")).strip())
        if num_bs != 4:
            continue
        
        all_f2_rows += 1
        answer = raw_scenario.get("answer", "")
        if not answer:
            continue
        
        options = (raw_scenario.get("task") or {}).get("options") or []
        action_map = _load_action_map(options)
        actions = _answer_to_actions(action_map, answer)
        pair_type = _get_pair_type(actions)
        
        if pair_type is None:
            continue
        
        valid_f2_rows += 1
        
        try:
            ps = parse_scenario(raw_scenario)
            features = run_all_skills(ps)
            top_issue, _ = features.top_issue()
        except Exception:
            top_issue = "unknown"
        
        top_issue_to_pairs[top_issue].append(pair_type)
    
    distribution = {}
    for top_issue, pairs in top_issue_to_pairs.items():
        count = Counter(pairs)
        total = sum(count.values())
        dist = {
            "count": total,
            "azimuth": count.get("azimuth", 0),
            "tilt": count.get("tilt", 0),
            "azimuth_pct": count.get("azimuth", 0) / total if total > 0 else 0.0,
            "tilt_pct": count.get("tilt", 0) / total if total > 0 else 0.0,
            "majority": "azimuth" if count.get("azimuth", 0) >= count.get("tilt", 0) else "tilt",
            "confidence": abs(count.get("azimuth", 0) - count.get("tilt", 0)) / total if total > 0 else 0.0,
        }
        distribution[top_issue] = dist
    
    return {
        "total_f2_rows": all_f2_rows,
        "valid_f2_rows": valid_f2_rows,
        "top_issue_distribution": distribution,
    }


def evaluate_phase2_candidates(
    phase1_dist: Dict[str, Any],
    test_json: Path,
    base_csv: Path,
    confidence_threshold: float = 0.70,
) -> List[Dict[str, Any]]:
    """Find Phase 2 F2 rows where current pair differs from Phase 1 majority with high confidence."""
    
    scenarios = json.loads(test_json.read_text(encoding="utf-8"))
    rows, base_answers = _load_submission(base_csv)
    
    candidates = []
    f2_count = 0
    card2_count = 0
    
    for raw_scenario in scenarios:
        sid = str(raw_scenario.get("scenario_id", "")).strip()
        if not sid:
            continue
        
        tag = raw_scenario.get("tag", "")
        if "multiple" not in tag.lower():
            continue
        
        num_bs = int(str(raw_scenario.get("context", {}).get("wireless_network_information", {}).get("num_base_stations", "4")).strip())
        if num_bs != 4:
            continue
        
        f2_count += 1
        current_answer = base_answers.get(sid, "")
        if not current_answer:
            continue
        
        options = (raw_scenario.get("task") or {}).get("options") or []
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
            ps = parse_scenario(raw_scenario)
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
        
        if current_pair != phase1_majority and confidence >= confidence_threshold:
            candidates.append({
                "scenario_id": sid,
                "top_issue": top_issue,
                "top_issue_score": round(top_score, 4),
                "current_answer": current_answer,
                "current_pair": current_pair,
                "current_actions": list(current_actions),
                "phase1_majority": phase1_majority,
                "phase1_confidence": round(confidence, 4),
                "phase1_bucket_size": dist["count"],
                "phase1_azimuth_pct": round(dist["azimuth_pct"], 4),
                "flip_score": round(confidence * top_score, 4),
            })
    
    candidates.sort(key=lambda x: -x["flip_score"])
    
    return {
        "total_f2_rows": f2_count,
        "card2_f2_rows": card2_count,
        "candidates": candidates,
        "high_confidence_count": len([c for c in candidates if c["phase1_confidence"] >= confidence_threshold]),
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
    max_flips: int = 25,
) -> Dict[str, Any]:
    """Generate a patch CSV with the highest-confidence flips."""
    
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
            skipped.append({"sid": sid, "reason": "scenario not found in test.json"})
            continue
        
        options = (scenario.get("task") or {}).get("options") or []
        action_map = {
            str(o.get("id", "")).strip(): parse_option_label(str(o.get("label", "")))
            for o in options
        }
        
        if target_pair == "azimuth":
            target_actions = F2_PAIR_AZIMUTH
        else:
            target_actions = F2_PAIR_TILT
        
        matching_codes = [
            cid for cid, parsed in action_map.items()
            if parsed.get("action_type") in target_actions
        ]
        
        if len(matching_codes) >= 2:
            new_answer = "|".join(matching_codes[:2])
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
        else:
            skipped.append({"sid": sid, "reason": f"not enough {target_pair} action codes found"})
    
    _write_submission(out_csv, rows, new_answers)
    
    return {
        "flips_applied": flips_applied,
        "skipped": skipped,
        "total_flips": len(flips_applied),
        "out_csv": str(out_csv),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="F2 pair calibration using Phase 1 training data")
    ap.add_argument("--train_json", type=Path,
                    default=ROOT / "data" / "Phase_1" / "train.json",
                    help="Path to Phase 1 training JSON")
    ap.add_argument("--test_json", type=Path,
                    default=ROOT / "data" / "Phase_2" / "test.json",
                    help="Path to Phase 2 test JSON")
    ap.add_argument("--base_csv", type=Path,
                    default=ROOT / "submissions" / "submission_600_v5_10_adj_only_plus_f2_pair_fix.csv",
                    help="Base submission CSV to apply patches to")
    ap.add_argument("--confidence_threshold", type=float, default=0.70,
                    help="Minimum confidence threshold for flips")
    ap.add_argument("--max_flips", type=int, default=25,
                    help="Maximum number of flips to apply")
    ap.add_argument("--out_csv", type=Path,
                    default=ROOT / "submissions" / "submission_600_C_f2_calibrated.csv")
    ap.add_argument("--report_json", type=Path,
                    default=ROOT / "reports" / "C_f2_pair_calibration.json")
    ap.add_argument("--report_md", type=Path,
                    default=ROOT / "reports" / "C_f2_pair_calibration.md")
    args = ap.parse_args()
    
    print("Analyzing Phase 1 F2 training data...")
    phase1_result = analyze_phase1_train(args.train_json)
    print(f"Phase 1 F2 rows: {phase1_result['total_f2_rows']} total, {phase1_result['valid_f2_rows']} valid")
    
    print("\nPhase 1 top_issue -> pair distribution:")
    for top_issue, dist in sorted(phase1_result["top_issue_distribution"].items(), key=lambda x: -x[1]["count"]):
        print(f"  {top_issue:20s}: n={dist['count']:3d}  azimuth={dist['azimuth']:3d}({dist['azimuth_pct']:.1%})  tilt={dist['tilt']:3d}({dist['tilt_pct']:.1%})  majority={dist['majority']:8s}  conf={dist['confidence']:.2f}")
    
    print("\nEvaluating Phase 2 F2 candidates...")
    phase2_result = evaluate_phase2_candidates(
        phase1_result,
        args.test_json,
        args.base_csv,
        args.confidence_threshold,
    )
    print(f"Phase 2 F2 rows: {phase2_result['total_f2_rows']} total, {phase2_result['card2_f2_rows']} card-2")
    print(f"High-confidence flip candidates: {phase2_result['high_confidence_count']}")
    
    print("\nGenerating patch CSV...")
    patch_result = generate_patch_csv(
        args.base_csv,
        phase2_result["candidates"],
        args.test_json,
        args.out_csv,
        args.max_flips,
    )
    print(f"Flips applied: {patch_result['total_flips']}")
    
    full_result = {
        "phase1_analysis": phase1_result,
        "phase2_evaluation": phase2_result,
        "patch_result": patch_result,
        "params": {
            "confidence_threshold": args.confidence_threshold,
            "max_flips": args.max_flips,
            "base_csv": str(args.base_csv),
            "train_json": str(args.train_json),
            "test_json": str(args.test_json),
        },
    }
    
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(full_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nJSON report written to: {args.report_json}")
    
    _generate_markdown_report(full_result, args.report_md)
    print(f"Markdown report written to: {args.report_md}")
    
    print(f"\nPatch CSV written to: {patch_result['out_csv']}")


def _generate_markdown_report(result: Dict[str, Any], out_path: Path) -> None:
    phase1 = result["phase1_analysis"]
    phase2 = result["phase2_evaluation"]
    patch = result["patch_result"]
    params = result["params"]
    
    lines = [
        "# C — F2 Pair Calibration Report",
        "",
        "Generated: 2026-05-16",
        "",
        "## Parameters",
        "",
        f"- Confidence threshold: {params['confidence_threshold']}",
        f"- Max flips: {params['max_flips']}",
        f"- Base CSV: {params['base_csv'].split('/')[-1]}",
        "",
        "## Phase 1 Analysis",
        "",
        f"Total F2 rows: {phase1['total_f2_rows']}",
        f"Valid F2 rows with legal pairs: {phase1['valid_f2_rows']}",
        "",
        "### top_issue -> pair distribution",
        "",
        "| top_issue | total | azimuth | tilt | azimuth % | tilt % | majority | confidence |",
        "|---|---|---|---|---|---|---|---|",
    ]
    
    for top_issue, dist in sorted(phase1["top_issue_distribution"].items(), key=lambda x: -x[1]["count"]):
        lines.append(
            f"| {top_issue} | {dist['count']} | {dist['azimuth']} | {dist['tilt']} | "
            f"{dist['azimuth_pct']:.1%} | {dist['tilt_pct']:.1%} | {dist['majority']} | {dist['confidence']:.2f} |"
        )
    
    lines.extend([
        "",
        "## Phase 2 Evaluation",
        "",
        f"Total F2 rows: {phase2['total_f2_rows']}",
        f"Cardinality-2 F2 rows: {phase2['card2_f2_rows']}",
        f"High-confidence flip candidates: {phase2['high_confidence_count']}",
        "",
        "### Recommended Flips (sorted by flip_score)",
        "",
        "| scenario_id | top_issue | current_pair | target_pair | confidence | flip_score |",
        "|---|---|---|---|---|---|",
    ])
    
    for cand in phase2["candidates"][:25]:
        lines.append(
            f"| {cand['scenario_id']} | {cand['top_issue']} | {cand['current_pair']} | "
            f"{cand['phase1_majority']} | {cand['phase1_confidence']:.2f} | {cand['flip_score']:.4f} |"
        )
    
    if not phase2["candidates"]:
        lines.append("| (no candidates found) | | | | | |")
    
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
    
    if not patch.get("flips_applied"):
        lines.append("| (no flips applied) | | | | | | |")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()