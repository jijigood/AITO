#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
V3 Offline Evaluation Framework

Evaluates the deterministic pipeline (no Qwen, no server) against training data.
Use this to rapidly iterate on feature_engine / candidate_generator changes.

Usage:
    # Quick 200-sample eval:
    python eval_v3.py

    # Full 2000-sample eval with per-action breakdown:
    python eval_v3.py --full --per-action

    # Evaluate specific family only:
    python eval_v3.py --family F1 --full

    # Write error analysis CSV:
    python eval_v3.py --full --errors errors_v3.csv
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skills.scenario_parser import parse_scenario
from skills.scenario_router import route
from skills.feature_engine import run_all_skills
from skills.candidate_generator import generate_candidates
from skills.answer_parser import iou_score


def count_bs_from_config(ex: dict) -> int:
    """Count base stations from network_configuration_data."""
    config = ex.get("data", {}).get("network_configuration_data", "")
    if not config:
        return 0
    rows = [r.strip() for r in config.strip().split("\n") if r.strip()]
    return max(0, len(rows) - 1)


def evaluate(
    train_path: str,
    max_samples: Optional[int] = None,
    target_family: Optional[str] = None,
    per_action: bool = False,
    error_csv: Optional[str] = None,
    seed: int = 42,
) -> Dict:
    """
    Run offline evaluation.

    Returns dict with metrics.
    """
    with open(train_path, "r", encoding="utf-8") as f:
        train = json.load(f)

    if max_samples and max_samples < len(train):
        random.seed(seed)
        train = random.sample(train, max_samples)

    total_iou = 0.0
    family_iou: Dict[str, float] = {}
    family_count: Dict[str, int] = {}
    family_exact: Dict[str, int] = {}

    action_correct: Counter = Counter()
    action_total: Counter = Counter()
    action_in_top2: Counter = Counter()
    action_in_top3: Counter = Counter()

    errors: List[Dict] = []

    t0 = time.time()

    for i, ex in enumerate(train):
        ps = parse_scenario(ex)
        family = route(ps.tag, ps.num_base_stations)
        fam = family.name

        if target_family and fam != target_family:
            continue

        sf = run_all_skills(ps)
        cands = generate_candidates(ps, sf, family)
        pred = cands.top_candidate() if cands else ""
        gt = ex["answer"]
        iou = iou_score(pred, gt)
        total_iou += iou

        family_iou[fam] = family_iou.get(fam, 0) + iou
        family_count[fam] = family_count.get(fam, 0) + 1
        family_exact[fam] = family_exact.get(fam, 0) + (1 if iou == 1.0 else 0)

        # Per-action tracking (F1 only)
        if fam == "F1":
            gt_opt = ps.parsed_options.get(gt, {})
            action = gt_opt.get("action_type", "unknown")
            action_total[action] += 1
            if iou == 1.0:
                action_correct[action] += 1
            if cands and cands.single_candidates:
                codes = [c.c_code for c in cands.single_candidates]
                if gt in codes[:2]:
                    action_in_top2[action] += 1
                if gt in codes[:3]:
                    action_in_top3[action] += 1

        # Error collection
        if iou < 1.0:
            top_issue, top_score = sf.top_issue()
            error = {
                "scenario_id": ex.get("scenario_id", ""),
                "family": fam,
                "gt": gt,
                "pred": pred,
                "iou": round(iou, 3),
                "top_issue": top_issue,
                "top_issue_score": round(top_score, 3),
            }
            if not family.is_multiple:
                gt_opt = ps.parsed_options.get(gt, {})
                error["gt_action"] = gt_opt.get("action_type", "?")
                error["gt_target_cell"] = gt_opt.get("target_cell", "?")
                error["gt_target_role"] = ps.cell_roles.get(
                    gt_opt.get("target_cell", ""), "?"
                )
                # Was GT in candidate list at all?
                if cands and cands.single_candidates:
                    cand_codes = [c.c_code for c in cands.single_candidates]
                    error["gt_rank"] = (
                        cand_codes.index(gt) + 1 if gt in cand_codes else -1
                    )
                    error["candidates"] = "|".join(
                        f"{c.c_code}({c.score:.2f})" for c in cands.single_candidates
                    )
                else:
                    error["gt_rank"] = -1
                    error["candidates"] = ""
            errors.append(error)

    elapsed = time.time() - t0
    n_total = sum(family_count.values())

    # Print results
    print(f"\n{'='*60}")
    print(f"V3 Offline Evaluation")
    print(f"{'='*60}")
    print(f"Samples: {n_total}  Time: {elapsed:.1f}s")
    print(f"Overall mean IOU: {total_iou/max(n_total,1):.4f}")
    print()

    print("Per-family breakdown:")
    print(f"  {'Family':<6} {'IOU':>8} {'Exact':>8} {'N':>6}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*6}")
    for fam in sorted(family_iou.keys()):
        avg = family_iou[fam] / family_count[fam]
        exact = family_exact[fam] / family_count[fam]
        print(f"  {fam:<6} {avg:>8.4f} {exact:>7.1%} {family_count[fam]:>6}")

    if per_action and action_total:
        print()
        print("F1 per-action breakdown:")
        print(f"  {'Action':<20} {'Top1':>8} {'Top2':>8} {'Top3':>8} {'Total':>6}")
        print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
        for action in sorted(action_total.keys()):
            t = action_total[action]
            c1 = action_correct[action]
            c2 = action_in_top2[action]
            c3 = action_in_top3[action]
            print(
                f"  {action:<20} {c1/t:>7.0%} {c2/t:>7.0%} {c3/t:>7.0%} {t:>6}"
            )

    if error_csv and errors:
        with open(error_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=errors[0].keys())
            writer.writeheader()
            writer.writerows(errors)
        print(f"\nError analysis written to: {error_csv} ({len(errors)} errors)")

    print(f"{'='*60}\n")

    return {
        "mean_iou": total_iou / max(n_total, 1),
        "family_iou": {
            k: v / family_count[k] for k, v in family_iou.items()
        },
        "n_total": n_total,
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3 Offline Evaluation")
    parser.add_argument(
        "--train-json",
        default="data/Phase_1/train.json",
        help="Path to training data",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate all 2000 examples (default: 200)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override sample count",
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        choices=["F1", "F2", "F3", "F4"],
        help="Evaluate only one family",
    )
    parser.add_argument(
        "--per-action",
        action="store_true",
        help="Show per-action breakdown for F1",
    )
    parser.add_argument(
        "--errors",
        type=str,
        default=None,
        help="Write error analysis to CSV",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    max_samples = args.max_samples
    if max_samples is None and not args.full:
        max_samples = 200

    evaluate(
        train_path=args.train_json,
        max_samples=max_samples,
        target_family=args.family,
        per_action=args.per_action,
        error_csv=args.errors,
        seed=args.seed,
    )
