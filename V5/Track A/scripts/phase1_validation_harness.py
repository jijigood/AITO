#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase-1 validation harness for Track A V6 work.

Purpose:
  - Build a frozen Phase-2-weighted Phase-1 holdout.
  - Evaluate deterministic/current mechanisms offline using embedded Phase-1 data.
  - Report per-family IOU, F1 action precision/recall, and structural legality.

This script intentionally does not call the official API and does not mutate the
main Phase 2 runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from phase2_run import _format_answer, _phase2_family
from skills.answer_parser import iou_score
from skills.candidate_generator import generate_candidates
from skills.feature_engine import run_all_skills
from skills.scenario_parser import infer_scenario_tag, parse_scenario


F1_ALLOWED_ACTIONS = {
    "add_neighbor",
    "check_server",
    "decrease_a3",
    "insufficient_data",
    "lift_tilt",
    "pdcch_2sym",
}

F2_LEGAL_ACTION_SETS = {
    frozenset({"adjust_azimuth", "increase_power"}),
    frozenset({"decrease_power", "press_down_tilt"}),
}


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON at {path}")
    return data


def family_name(scenario: Dict[str, Any]) -> str:
    tag = infer_scenario_tag(scenario)
    wi = scenario.get("context", {}).get("wireless_network_information", {})
    num_bs = int(str(wi.get("num_base_stations", "4")).strip())
    _, family, _ = _phase2_family(tag, num_bs)

    if tag == "single-answer" and num_bs == 5:
        return "F1_5BS"
    if tag == "multiple-answer" and num_bs == 5:
        return "F3_5BS_MULTI"
    if tag == "multiple-answer" and num_bs == 13:
        return "F4M_13BS_MULTI"
    return family.name


def option_actions(scenario: Dict[str, Any], answer: str) -> List[str]:
    if not answer:
        return []
    ps = parse_scenario(scenario)
    actions: List[str] = []
    for code in [c for c in str(answer).split("|") if c]:
        actions.append(str((ps.parsed_options.get(code) or {}).get("action_type", "?")))
    return actions


def normalise_answer(answer: Any) -> str:
    if answer is None:
        return ""
    if isinstance(answer, list):
        return "|".join(str(x).strip() for x in answer if str(x).strip())
    return str(answer).strip()


def phase2_weights(phase2_scenarios: List[Dict[str, Any]]) -> Dict[str, float]:
    counts = Counter(family_name(s) for s in phase2_scenarios)
    total = sum(counts.values()) or 1
    return {family: count / total for family, count in sorted(counts.items())}


def build_holdout(
    train_scenarios: List[Dict[str, Any]],
    phase2_scenarios: List[Dict[str, Any]],
    sample_size: int,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    by_family: Dict[str, List[str]] = defaultdict(list)
    by_id = {str(s.get("scenario_id")): s for s in train_scenarios}
    for scenario in train_scenarios:
        sid = str(scenario.get("scenario_id"))
        if sid and normalise_answer(scenario.get("answer")):
            by_family[family_name(scenario)].append(sid)

    for ids in by_family.values():
        rng.shuffle(ids)

    weights = phase2_weights(phase2_scenarios)
    selected: Dict[str, List[str]] = {family: [] for family in weights}
    target_counts = {
        family: int(round(sample_size * weight))
        for family, weight in weights.items()
    }

    # First pass: fill each Phase2 family target from matching Phase1 data.
    for family in weights:
        available = by_family.get(family, [])
        target = min(target_counts.get(family, 0), len(available))
        selected[family] = available[:target]

    # If Phase2 has families scarce/absent in Phase1, redistribute the remainder
    # to available families in Phase2-weight order.
    total_selected = sum(len(ids) for ids in selected.values())
    ordered_families = sorted(weights, key=lambda f: -weights[f])
    while total_selected < sample_size:
        progressed = False
        for family in ordered_families:
            available = by_family.get(family, [])
            if len(selected.get(family, [])) < len(available):
                selected.setdefault(family, []).append(available[len(selected[family])])
                total_selected += 1
                progressed = True
                if total_selected >= sample_size:
                    break
        if not progressed:
            break

    selected_ids = [sid for family in sorted(selected) for sid in selected[family]]
    selected_set = set(selected_ids)
    dev_ids = [
        sid for sid in sorted(by_id)
        if sid not in selected_set and normalise_answer(by_id[sid].get("answer"))
    ]

    availability = {
        family: {
            "phase1_available": len(by_family.get(family, [])),
            "phase2_weight": weights.get(family, 0.0),
            "target": target_counts.get(family, 0),
            "selected": len(selected.get(family, [])),
            "low_confidence": len(by_family.get(family, [])) < 20,
        }
        for family in sorted(weights)
    }
    # Include Phase1-only families for visibility.
    for family, ids in sorted(by_family.items()):
        availability.setdefault(
            family,
            {
                "phase1_available": len(ids),
                "phase2_weight": 0.0,
                "target": 0,
                "selected": 0,
                "low_confidence": len(ids) < 20,
            },
        )

    return {
        "seed": seed,
        "sample_size_requested": sample_size,
        "sample_size_actual": len(selected_ids),
        "ids": selected_ids,
        "dev_ids": dev_ids,
        "availability": availability,
        "phase2_weights": weights,
        "selected_by_family": selected,
    }


def deterministic_answer(scenario: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    tag = infer_scenario_tag(scenario)
    wi = scenario.get("context", {}).get("wireless_network_information", {})
    num_bs = int(str(wi.get("num_base_stations", "4")).strip())
    _, family, family_override = _phase2_family(tag, num_bs)

    ps = parse_scenario(scenario)
    sf = run_all_skills(ps)
    candidates = generate_candidates(ps, sf, family)
    answer = _format_answer(candidates, tag)
    top_issue, top_score = sf.top_issue()
    candidate_rows: List[Dict[str, Any]] = []
    if not candidates.is_multiple:
        for c in candidates.single_candidates:
            candidate_rows.append({
                "answer": c.c_code,
                "score": float(c.score),
                "action_type": c.action_type,
                "target_cell": c.target_cell,
                "reason": c.reason,
            })
    else:
        for c in candidates.multi_candidates:
            candidate_rows.append({
                "answer": "|".join(c.c_codes),
                "score": float(c.score),
                "reason": c.reason,
            })
    return answer, {
        "tag": tag,
        "num_bs": num_bs,
        "family": family.name,
        "family_override": family_override,
        "top_issue": top_issue,
        "top_issue_score": float(top_score),
        "candidate_rows": candidate_rows,
        "candidate_output": candidates,
        "parsed_scenario": ps,
    }


def strict_f1_allowed_answer(scenario: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    answer, info = deterministic_answer(scenario)
    family = family_name(scenario)
    if family not in {"F1", "F1_5BS"}:
        return answer, info

    candidates = info["candidate_output"]
    ps = info["parsed_scenario"]
    for c in candidates.single_candidates:
        action = str((ps.parsed_options.get(c.c_code) or {}).get("action_type", ""))
        if action in F1_ALLOWED_ACTIONS:
            info = dict(info)
            info["strict_guard_replaced"] = c.c_code != answer
            info["strict_guard_original"] = answer
            return c.c_code, info
    return answer, info


MECHANISMS = {
    "deterministic_current": deterministic_answer,
    "strict_f1_allowed_guard": strict_f1_allowed_answer,
}


def structural_flags(family: str, actions: List[str], answer: str) -> Dict[str, bool]:
    flags = {
        "f1_illegal": False,
        "f2_bad": False,
        "f3_bad_cardinality": False,
        "f4_bad": False,
        "f4m_bad": False,
    }
    if family in {"F1", "F1_5BS"}:
        flags["f1_illegal"] = len(actions) != 1 or actions[0] not in F1_ALLOWED_ACTIONS
    elif family == "F2":
        flags["f2_bad"] = (
            len(actions) != 2
            or len(set(actions)) != 2
            or frozenset(actions) not in F2_LEGAL_ACTION_SETS
        )
    elif family == "F3_5BS_MULTI":
        flags["f3_bad_cardinality"] = len([c for c in answer.split("|") if c]) != 4
    elif family == "F4":
        flags["f4_bad"] = len(actions) != 1 or actions[0] != "decrease_interfreq"
    elif family == "F4M_13BS_MULTI":
        flags["f4m_bad"] = len(actions) != 2 or any(a != "decrease_interfreq" for a in actions)
    return flags


def evaluate(
    scenarios: Iterable[Dict[str, Any]],
    mechanism_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if mechanism_name not in MECHANISMS:
        raise ValueError(f"Unknown mechanism: {mechanism_name}")
    solver = MECHANISMS[mechanism_name]

    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        sid = str(scenario.get("scenario_id"))
        family = family_name(scenario)
        truth = normalise_answer(scenario.get("answer"))
        pred, info = solver(scenario)
        gt_actions = option_actions(scenario, truth)
        pred_actions = option_actions(scenario, pred)
        score = iou_score(pred, truth)
        flags = structural_flags(family, pred_actions, pred)
        rows.append({
            "ID": sid,
            "family": family,
            "ground_truth": truth,
            "prediction": pred,
            "iou": score,
            "exact": float(score == 1.0),
            "gt_action": "|".join(gt_actions),
            "pred_action": "|".join(pred_actions),
            "top_issue": info.get("top_issue", ""),
            "top_issue_score": info.get("top_issue_score", ""),
            **{k: int(v) for k, v in flags.items()},
        })

    summary = summarize(rows)
    summary["mechanism"] = mechanism_name
    return rows, summary


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)

    family_summary: Dict[str, Dict[str, Any]] = {}
    for family, group in sorted(by_family.items()):
        n = len(group)
        family_summary[family] = {
            "n": n,
            "mean_iou": sum(float(r["iou"]) for r in group) / n if n else 0.0,
            "exact_match": sum(float(r["exact"]) for r in group) / n if n else 0.0,
            "f1_illegal": sum(int(r["f1_illegal"]) for r in group),
            "f2_bad": sum(int(r["f2_bad"]) for r in group),
            "f3_bad_cardinality": sum(int(r["f3_bad_cardinality"]) for r in group),
            "f4_bad": sum(int(r["f4_bad"]) for r in group),
            "f4m_bad": sum(int(r["f4m_bad"]) for r in group),
        }

    single_rows = [
        r for r in rows
        if r["family"] in {"F1", "F1_5BS", "F4"}
        and "|" not in r["ground_truth"]
    ]
    action_metrics = action_precision_recall(single_rows)
    n_all = len(rows)
    return {
        "n": n_all,
        "mean_iou": sum(float(r["iou"]) for r in rows) / n_all if n_all else 0.0,
        "exact_match": sum(float(r["exact"]) for r in rows) / n_all if n_all else 0.0,
        "family": family_summary,
        "single_action_metrics": action_metrics,
        "structural_totals": {
            "f1_illegal": sum(int(r["f1_illegal"]) for r in rows),
            "f2_bad": sum(int(r["f2_bad"]) for r in rows),
            "f3_bad_cardinality": sum(int(r["f3_bad_cardinality"]) for r in rows),
            "f4_bad": sum(int(r["f4_bad"]) for r in rows),
            "f4m_bad": sum(int(r["f4m_bad"]) for r in rows),
        },
    }


def action_precision_recall(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    labels = sorted(set(r["gt_action"] for r in rows) | set(r["pred_action"] for r in rows))
    metrics: Dict[str, Dict[str, float]] = {}
    for label in labels:
        if not label:
            continue
        tp = sum(1 for r in rows if r["gt_action"] == label and r["pred_action"] == label)
        pred = sum(1 for r in rows if r["pred_action"] == label)
        gt = sum(1 for r in rows if r["gt_action"] == label)
        metrics[label] = {
            "tp": tp,
            "pred": pred,
            "gt": gt,
            "precision": tp / pred if pred else 0.0,
            "recall": tp / gt if gt else 0.0,
        }
    return metrics


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(
    path: Path,
    holdout: Dict[str, Any],
    summaries: Dict[str, Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# Phase 0.5 Phase1 Validation Harness Report")
    lines.append("")
    lines.append("## Holdout")
    lines.append("")
    lines.append(f"- Seed: `{holdout['seed']}`")
    lines.append(f"- Requested sample size: `{holdout['sample_size_requested']}`")
    lines.append(f"- Actual sample size: `{holdout['sample_size_actual']}`")
    lines.append("")
    lines.append("| Family | Phase1 Available | Phase2 Weight | Target | Selected | Low Confidence |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for family, row in sorted(holdout["availability"].items()):
        lines.append(
            f"| {family} | {row['phase1_available']} | "
            f"{row['phase2_weight']:.3f} | {row['target']} | "
            f"{row['selected']} | {row['low_confidence']} |"
        )

    lines.append("")
    lines.append("## Mechanism Summary")
    lines.append("")
    lines.append("| Mechanism | N | Mean IOU | Exact | F1 Illegal | F2 Bad | F3 Bad Card | F4 Bad | F4M Bad |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, summary in summaries.items():
        st = summary["structural_totals"]
        lines.append(
            f"| {name} | {summary['n']} | {summary['mean_iou']:.4f} | "
            f"{summary['exact_match']:.4f} | {st['f1_illegal']} | {st['f2_bad']} | "
            f"{st['f3_bad_cardinality']} | {st['f4_bad']} | {st['f4m_bad']} |"
        )

    for name, summary in summaries.items():
        lines.append("")
        lines.append(f"## {name} Per-Family")
        lines.append("")
        lines.append("| Family | N | Mean IOU | Exact | F1 Illegal | F2 Bad |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for family, row in sorted(summary["family"].items()):
            lines.append(
                f"| {family} | {row['n']} | {row['mean_iou']:.4f} | "
                f"{row['exact_match']:.4f} | {row['f1_illegal']} | {row['f2_bad']} |"
            )

        lines.append("")
        lines.append(f"## {name} Single-Action Precision/Recall")
        lines.append("")
        lines.append("| Action | GT | Pred | TP | Precision | Recall |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for action, row in sorted(summary["single_action_metrics"].items()):
            lines.append(
                f"| {action} | {row['gt']} | {row['pred']} | {row['tp']} | "
                f"{row['precision']:.4f} | {row['recall']:.4f} |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase1 stratified validation harness")
    parser.add_argument("--train_json", type=Path, default=ROOT / "data/Phase_1/train.json")
    parser.add_argument("--phase2_json", type=Path, default=ROOT / "data/Phase_2/test.json")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "results/phase1_validation")
    parser.add_argument("--sample_size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        default=["deterministic_current", "strict_f1_allowed_guard"],
        choices=sorted(MECHANISMS),
    )
    parser.add_argument("--eval_all", action="store_true", help="Evaluate all Phase1 train rows in addition to holdout")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = load_json(args.train_json)
    phase2 = load_json(args.phase2_json)
    by_id = {str(s.get("scenario_id")): s for s in train}

    holdout = build_holdout(train, phase2, args.sample_size, args.seed)
    holdout_path = args.out_dir / "holdout_stratified.json"
    holdout_path.write_text(json.dumps(holdout, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    holdout_scenarios = [by_id[sid] for sid in holdout["ids"] if sid in by_id]
    summaries: Dict[str, Dict[str, Any]] = {}
    for mechanism in args.mechanisms:
        rows, summary = evaluate(holdout_scenarios, mechanism)
        summaries[mechanism] = summary
        write_csv(args.out_dir / f"holdout_{mechanism}_predictions.csv", rows)

    if args.eval_all:
        all_summaries: Dict[str, Dict[str, Any]] = {}
        for mechanism in args.mechanisms:
            rows, summary = evaluate(train, mechanism)
            all_summaries[mechanism] = summary
            write_csv(args.out_dir / f"all_{mechanism}_predictions.csv", rows)
        (args.out_dir / "all_eval_summary.json").write_text(
            json.dumps(all_summaries, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    summary_path = args.out_dir / "holdout_eval_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    markdown_report(ROOT / "reports/phase0_5_validation_harness.md", holdout, summaries)

    print(f"holdout: {holdout_path}")
    print(f"summary: {summary_path}")
    print(f"report: {ROOT / 'reports/phase0_5_validation_harness.md'}")
    for mechanism, summary in summaries.items():
        print(
            f"{mechanism}: n={summary['n']} mean_iou={summary['mean_iou']:.4f} "
            f"exact={summary['exact_match']:.4f} structural={summary['structural_totals']}"
        )


if __name__ == "__main__":
    main()
