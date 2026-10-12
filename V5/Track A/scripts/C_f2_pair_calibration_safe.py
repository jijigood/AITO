#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Conservative F2 pair calibration.

This fixes the unsafe C calibration prototype by:
  1. using real Phase 2 diagnostics instead of placeholder test.json data,
  2. choosing replacements only from scored diagnostics candidates,
  3. enforcing exact legal F2 pairs before writing any CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skills.candidate_generator import generate_candidates
from skills.feature_engine import run_all_skills
from skills.scenario_parser import parse_option_label, parse_scenario
from skills.scenario_router import route


F2_PAIR_AZIMUTH = frozenset({"adjust_azimuth", "increase_power"})
F2_PAIR_TILT = frozenset({"decrease_power", "press_down_tilt"})
F2_LEGAL_PAIRS = {F2_PAIR_AZIMUTH, F2_PAIR_TILT}
PAIR_NAMES = {
    F2_PAIR_AZIMUTH: "azimuth",
    F2_PAIR_TILT: "tilt",
}


def _split_answer(answer: str) -> List[str]:
    return [part.strip() for part in str(answer or "").split("|") if part.strip()]


def _stable_fold(sid: str, fold_mod: int) -> int:
    digest = hashlib.md5(sid.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % fold_mod


def _num_bs(raw: Dict[str, Any]) -> int:
    wi = (raw.get("context") or {}).get("wireless_network_information") or {}
    try:
        return int(str(wi.get("num_base_stations", "4")).strip())
    except ValueError:
        return 4


def _is_f2_raw(raw: Dict[str, Any]) -> bool:
    return "multiple" in str(raw.get("tag", "")).lower() and _num_bs(raw) == 4


def _action_map_from_options(options: Iterable[Dict[str, str]]) -> Dict[str, str]:
    return {
        str(opt.get("id", "")).strip(): str(parse_option_label(str(opt.get("label", ""))).get("action_type", "?"))
        for opt in options
    }


def _load_phase2_action_maps(test_json: Path) -> Dict[str, Dict[str, str]]:
    scenarios = json.loads(test_json.read_text(encoding="utf-8"))
    return {
        str(s.get("scenario_id", "")).strip(): _action_map_from_options((s.get("task") or {}).get("options") or [])
        for s in scenarios
    }


def _pair_kind_from_actions(actions: Iterable[str]) -> Optional[str]:
    action_set = frozenset(actions)
    return PAIR_NAMES.get(action_set)


def _answer_actions(action_map: Dict[str, str], answer: str) -> List[str]:
    return [action_map.get(code, "?") for code in _split_answer(answer)]


def _answer_pair_kind(action_map: Dict[str, str], answer: str) -> Optional[str]:
    codes = _split_answer(answer)
    if len(codes) != 2:
        return None
    return _pair_kind_from_actions(_answer_actions(action_map, answer))


def _is_legal_pair_answer(action_map: Dict[str, str], answer: str) -> bool:
    return _answer_pair_kind(action_map, answer) is not None


def _load_submission(path: Path) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [{k: (v or "") for k, v in row.items()} for row in csv.DictReader(f)]
    return rows, {row["ID"]: row.get("Track A", "") for row in rows}


def _write_submission(path: Path, rows: List[Dict[str, str]], answers: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Track A", "Track B"])
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in ("ID", "Track A", "Track B")}
            out["Track A"] = answers.get(row["ID"], row.get("Track A", ""))
            writer.writerow(out)


def _load_diagnostics(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("scenario_id") or "").strip()
            if sid:
                out[sid] = row
    return out


def _phase1_f2_records(train_json: Path) -> List[Dict[str, Any]]:
    rows = json.loads(train_json.read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = []
    for raw in rows:
        if not _is_f2_raw(raw):
            continue
        sid = str(raw.get("scenario_id", "")).strip()
        action_map = _action_map_from_options((raw.get("task") or {}).get("options") or [])
        gt_pair = _answer_pair_kind(action_map, str(raw.get("answer", "")))
        if gt_pair is None:
            continue
        ps = parse_scenario(raw)
        sf = run_all_skills(ps)
        top_issue, top_score = sf.top_issue()
        routed_family = route(ps.tag, ps.num_base_stations)
        candidate_output = generate_candidates(ps, sf, routed_family)
        deterministic_answer = candidate_output.top_candidate() or ""
        deterministic_pair = _answer_pair_kind(action_map, deterministic_answer)
        records.append({
            "sid": sid,
            "top_issue": top_issue,
            "top_score": float(top_score or 0.0),
            "gt_pair": gt_pair,
            "deterministic_answer": deterministic_answer,
            "deterministic_pair": deterministic_pair,
        })
    return records


def _build_distribution(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, Counter] = defaultdict(Counter)
    for row in records:
        buckets[row["top_issue"]][row["gt_pair"]] += 1
    out: Dict[str, Dict[str, Any]] = {}
    for issue, counts in buckets.items():
        total = sum(counts.values())
        majority, majority_count = counts.most_common(1)[0]
        out[issue] = {
            "count": total,
            "azimuth": counts.get("azimuth", 0),
            "tilt": counts.get("tilt", 0),
            "majority": majority,
            "majority_rate": round(majority_count / total, 6) if total else 0.0,
            "azimuth_pct": round(counts.get("azimuth", 0) / total, 6) if total else 0.0,
            "tilt_pct": round(counts.get("tilt", 0) / total, 6) if total else 0.0,
        }
    return out


def _crossval_smoke(records: List[Dict[str, Any]], fold_mod: int, min_bucket_size: int, min_majority_rate: float) -> Dict[str, Any]:
    stats = Counter()
    by_transition: Dict[str, Counter] = defaultdict(Counter)
    for fold in range(fold_mod):
        train = [r for r in records if _stable_fold(r["sid"], fold_mod) != fold]
        holdout = [r for r in records if _stable_fold(r["sid"], fold_mod) == fold]
        dist = _build_distribution(train)
        for row in holdout:
            old_pair = row.get("deterministic_pair")
            gt_pair = row["gt_pair"]
            old_correct = old_pair == gt_pair
            stats["rows"] += 1
            stats["old_correct"] += int(old_correct)
            bucket = dist.get(row["top_issue"])
            if not bucket:
                continue
            target_pair = bucket["majority"]
            if bucket["count"] < min_bucket_size or bucket["majority_rate"] < min_majority_rate:
                continue
            if old_pair is None or old_pair == target_pair:
                continue
            new_correct = target_pair == gt_pair
            transition = f"{old_pair}->{target_pair}"
            stats["candidate_flips"] += 1
            stats["new_correct"] += int(new_correct)
            stats["flip_improves"] += int((not old_correct) and new_correct)
            stats["flip_hurts"] += int(old_correct and (not new_correct))
            by_transition[transition]["n"] += 1
            by_transition[transition]["old_correct"] += int(old_correct)
            by_transition[transition]["new_correct"] += int(new_correct)
    rows = max(stats["rows"], 1)
    flips = max(stats["candidate_flips"], 1)
    return {
        "fold_mod": fold_mod,
        "rows": stats["rows"],
        "old_accuracy": round(stats["old_correct"] / rows, 6),
        "candidate_flips": stats["candidate_flips"],
        "flip_new_precision": round(stats["new_correct"] / flips, 6),
        "flip_improves": stats["flip_improves"],
        "flip_hurts": stats["flip_hurts"],
        "net_gain_rows": stats["flip_improves"] - stats["flip_hurts"],
        "by_transition": {
            key: {
                "n": val["n"],
                "old_precision": round(val["old_correct"] / max(val["n"], 1), 6),
                "new_precision": round(val["new_correct"] / max(val["n"], 1), 6),
            }
            for key, val in sorted(by_transition.items())
        },
    }


def _candidate_pair_kind(action_map: Dict[str, str], candidate_answer: str) -> Optional[str]:
    return _answer_pair_kind(action_map, candidate_answer)


def _best_scored_candidate_for_pair(
    diag: Dict[str, Any],
    action_map: Dict[str, str],
    target_pair: str,
) -> Optional[Tuple[str, float, str]]:
    best: Optional[Tuple[str, float, str]] = None
    for cand in diag.get("candidates") or []:
        answer = str(cand.get("answer") or "")
        if _candidate_pair_kind(action_map, answer) != target_pair:
            continue
        score = float(cand.get("score") or 0.0)
        reason = str(cand.get("reason") or "")
        if best is None or score > best[1]:
            best = (answer, score, reason)
    return best


def evaluate_phase2(
    *,
    test_json: Path,
    diagnostics_jsonl: Path,
    base_csv: Path,
    distribution: Dict[str, Dict[str, Any]],
    min_bucket_size: int,
    min_majority_rate: float,
    min_top_score: float,
    max_score_drop: float,
    max_flips: int,
) -> Dict[str, Any]:
    rows, answers = _load_submission(base_csv)
    action_maps = _load_phase2_action_maps(test_json)
    diagnostics = _load_diagnostics(diagnostics_jsonl)
    new_answers = dict(answers)
    candidates: List[Dict[str, Any]] = []
    skipped = Counter()

    for sid, diag in diagnostics.items():
        if diag.get("family") != "F2":
            continue
        current = answers.get(sid, "")
        action_map = action_maps.get(sid, {})
        current_pair = _answer_pair_kind(action_map, current)
        if current_pair is None:
            skipped["current_not_legal_pair"] += 1
            continue
        top_issue = str((diag.get("top_issue") or {}).get("name") or "")
        top_score = float((diag.get("top_issue") or {}).get("score") or 0.0)
        bucket = distribution.get(top_issue)
        if not bucket:
            skipped["no_phase1_bucket"] += 1
            continue
        target_pair = str(bucket["majority"])
        if target_pair == current_pair:
            skipped["already_majority_pair"] += 1
            continue
        if bucket["count"] < min_bucket_size:
            skipped["bucket_too_small"] += 1
            continue
        if float(bucket["majority_rate"]) < min_majority_rate:
            skipped["majority_rate_too_low"] += 1
            continue
        if top_score < min_top_score:
            skipped["top_score_too_low"] += 1
            continue
        current_score = None
        for cand in diag.get("candidates") or []:
            if str(cand.get("answer") or "") == current:
                current_score = float(cand.get("score") or 0.0)
                break
        target = _best_scored_candidate_for_pair(diag, action_map, target_pair)
        if target is None:
            skipped["no_scored_target_pair_candidate"] += 1
            continue
        target_answer, target_score, target_reason = target
        if current_score is not None and target_score + max_score_drop < current_score:
            skipped["target_score_drop_too_large"] += 1
            continue
        candidates.append({
            "scenario_id": sid,
            "top_issue": top_issue,
            "top_score": round(top_score, 6),
            "current_answer": current,
            "current_pair": current_pair,
            "current_score": current_score,
            "target_answer": target_answer,
            "target_pair": target_pair,
            "target_score": round(target_score, 6),
            "target_reason": target_reason,
            "bucket_count": bucket["count"],
            "bucket_majority_rate": bucket["majority_rate"],
            "priority": round(float(bucket["majority_rate"]) * top_score * target_score, 6),
        })

    candidates.sort(
        key=lambda row: (
            -float(row["priority"]),
            -float(row["bucket_majority_rate"]),
            -float(row["target_score"]),
            row["scenario_id"],
        )
    )
    applied = []
    for row in candidates[:max_flips]:
        new_answers[row["scenario_id"]] = row["target_answer"]
        applied.append(row)
    return {
        "base_rows": rows,
        "answers": new_answers,
        "candidates": candidates,
        "applied": applied,
        "skipped": dict(skipped),
        "diagnostic_f2_rows": sum(1 for row in diagnostics.values() if row.get("family") == "F2"),
    }


def audit_f2_legality(test_json: Path, csv_path: Path) -> Dict[str, Any]:
    action_maps = _load_phase2_action_maps(test_json)
    scenarios = json.loads(test_json.read_text(encoding="utf-8"))
    f2_ids = {
        str(s.get("scenario_id", "")).strip()
        for s in scenarios
        if _is_f2_raw(s)
    }
    rows, answers = _load_submission(csv_path)
    bad = []
    pair_counts = Counter()
    for sid in f2_ids:
        answer = answers.get(sid, "")
        pair = _answer_pair_kind(action_maps.get(sid, {}), answer)
        if pair is None:
            bad.append({
                "scenario_id": sid,
                "answer": answer,
                "actions": _answer_actions(action_maps.get(sid, {}), answer),
                "cardinality": len(_split_answer(answer)),
            })
        else:
            pair_counts[pair] += 1
    return {
        "f2_rows": len(f2_ids),
        "bad_rows": bad,
        "bad_count": len(bad),
        "pair_counts": dict(pair_counts),
    }


def _write_reports(result: Dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# C F2 Pair Calibration Safe Report",
        "",
        "## Phase1 Distribution",
        "",
        "| top_issue | count | azimuth | tilt | majority | majority_rate |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for issue, bucket in sorted(result["phase1_distribution"].items(), key=lambda item: (-item[1]["count"], item[0])):
        lines.append(
            f"| {issue} | {bucket['count']} | {bucket['azimuth']} | {bucket['tilt']} | "
            f"{bucket['majority']} | {bucket['majority_rate']:.3f} |"
        )
    smoke = result["phase1_crossval_smoke"]
    lines.extend([
        "",
        "## Phase1 Crossval Smoke",
        "",
        f"- rows: {smoke['rows']}",
        f"- old_accuracy: {smoke['old_accuracy']:.4f}",
        f"- candidate_flips: {smoke['candidate_flips']}",
        f"- flip_new_precision: {smoke['flip_new_precision']:.4f}",
        f"- flip_improves: {smoke['flip_improves']}",
        f"- flip_hurts: {smoke['flip_hurts']}",
        f"- net_gain_rows: {smoke['net_gain_rows']}",
        "",
        "## Phase2 Candidates",
        "",
        f"- diagnostic_f2_rows: {result['phase2']['diagnostic_f2_rows']}",
        f"- candidates: {len(result['phase2']['candidates'])}",
        f"- applied: {len(result['phase2']['applied'])}",
        f"- skipped: {result['phase2']['skipped']}",
        "",
        "| scenario_id | top_issue | current_pair | target_pair | current_score | target_score | majority_rate | priority |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ])
    for row in result["phase2"]["applied"]:
        current_score = row["current_score"]
        current_score_text = "" if current_score is None else f"{current_score:.4f}"
        lines.append(
            f"| {row['scenario_id']} | {row['top_issue']} | {row['current_pair']} | {row['target_pair']} | "
            f"{current_score_text} | {row['target_score']:.4f} | {row['bucket_majority_rate']:.3f} | {row['priority']:.4f} |"
        )
    audit = result["audit"]
    lines.extend([
        "",
        "## F2 Legality Audit",
        "",
        f"- bad_count: {audit['bad_count']}",
        f"- pair_counts: {audit['pair_counts']}",
    ])
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Conservative F2 pair calibration using real diagnostics")
    ap.add_argument("--train_json", type=Path, default=ROOT / "data" / "Phase_1" / "train.json")
    ap.add_argument("--test_json", type=Path, default=ROOT / "data" / "Phase_2" / "test.json")
    ap.add_argument("--diagnostics_jsonl", type=Path,
                    default=ROOT / "results" / "phase2" / "sub2_lora_boundary" / "result_diag_lora_boundary_only.diagnostics.jsonl")
    ap.add_argument("--base_csv", type=Path,
                    default=ROOT / "submissions" / "submission_600_v5_10_adj_only_plus_f2_pair_fix.csv")
    ap.add_argument("--out_csv", type=Path,
                    default=ROOT / "submissions" / "submission_600_C_f2_calibrated_safe.csv")
    ap.add_argument("--report_json", type=Path,
                    default=ROOT / "reports" / "C_f2_pair_calibration_safe.json")
    ap.add_argument("--report_md", type=Path,
                    default=ROOT / "reports" / "C_f2_pair_calibration_safe.md")
    ap.add_argument("--min_bucket_size", type=int, default=20)
    ap.add_argument("--min_majority_rate", type=float, default=0.70)
    ap.add_argument("--min_top_score", type=float, default=0.50)
    ap.add_argument("--max_score_drop", type=float, default=0.08)
    ap.add_argument("--max_flips", type=int, default=25)
    ap.add_argument("--fold_mod", type=int, default=5)
    args = ap.parse_args()

    records = _phase1_f2_records(args.train_json)
    distribution = _build_distribution(records)
    smoke = _crossval_smoke(
        records,
        fold_mod=args.fold_mod,
        min_bucket_size=args.min_bucket_size,
        min_majority_rate=args.min_majority_rate,
    )
    phase2 = evaluate_phase2(
        test_json=args.test_json,
        diagnostics_jsonl=args.diagnostics_jsonl,
        base_csv=args.base_csv,
        distribution=distribution,
        min_bucket_size=args.min_bucket_size,
        min_majority_rate=args.min_majority_rate,
        min_top_score=args.min_top_score,
        max_score_drop=args.max_score_drop,
        max_flips=args.max_flips,
    )
    _write_submission(args.out_csv, phase2["base_rows"], phase2["answers"])
    audit = audit_f2_legality(args.test_json, args.out_csv)
    result = {
        "params": {
            "train_json": str(args.train_json),
            "test_json": str(args.test_json),
            "diagnostics_jsonl": str(args.diagnostics_jsonl),
            "base_csv": str(args.base_csv),
            "out_csv": str(args.out_csv),
            "min_bucket_size": args.min_bucket_size,
            "min_majority_rate": args.min_majority_rate,
            "min_top_score": args.min_top_score,
            "max_score_drop": args.max_score_drop,
            "max_flips": args.max_flips,
        },
        "phase1_distribution": distribution,
        "phase1_crossval_smoke": smoke,
        "phase2": {
            "diagnostic_f2_rows": phase2["diagnostic_f2_rows"],
            "candidates": phase2["candidates"],
            "applied": phase2["applied"],
            "skipped": phase2["skipped"],
        },
        "audit": audit,
    }
    _write_reports(result, args.report_json, args.report_md)
    print(json.dumps({
        "records": len(records),
        "crossval": smoke,
        "phase2_candidates": len(phase2["candidates"]),
        "applied": len(phase2["applied"]),
        "audit_bad_count": audit["bad_count"],
        "out_csv": str(args.out_csv),
        "report_json": str(args.report_json),
        "report_md": str(args.report_md),
    }, indent=2, ensure_ascii=False))
    if audit["bad_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
