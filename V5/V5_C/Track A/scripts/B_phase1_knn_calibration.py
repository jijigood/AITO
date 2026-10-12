#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Calibrate structured F1 KNN votes on Phase 1 train/holdout.

This is an offline diagnostic script.  It does not call Track A APIs and does
not create a submission.  It estimates whether KNN vote share is meaningful by
running train -> holdout retrieval where Phase 1 ground truth is known.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skills.candidate_generator import generate_candidates
from skills.feature_engine import run_all_skills
from skills.scenario_parser import parse_option_label, parse_scenario
from skills.scenario_router import route


SAFE_F1_ACTIONS = {
    "add_neighbor",
    "check_server",
    "decrease_a3",
    "insufficient_data",
    "lift_tilt",
    "pdcch_2sym",
}


def _answer_action(raw: Dict[str, Any], answer: str) -> str:
    options = (raw.get("task") or {}).get("options") or []
    action_map = {
        str(opt.get("id", "")).strip(): parse_option_label(str(opt.get("label", ""))).get("action_type", "unknown")
        for opt in options
    }
    codes = [part.strip() for part in str(answer or "").split("|") if part.strip()]
    if len(codes) != 1:
        return "multi"
    return action_map.get(codes[0], "unknown")


def _num_bs(raw: Dict[str, Any]) -> int:
    wi = (raw.get("context") or {}).get("wireless_network_information") or {}
    try:
        return int(str(wi.get("num_base_stations", "4")).strip())
    except ValueError:
        return 4


def _is_holdout(sid: str, fold_mod: int, holdout_fold: int) -> bool:
    digest = hashlib.md5(sid.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % fold_mod == holdout_fold


def _vec(sf) -> np.ndarray:
    tp = sf.throughput
    mob = sf.mobility
    ifreq = sf.inter_freq
    arr = np.array([
        (tp.worst_throughput_mbps or 0.0) / 500.0,
        (tp.throughput_drop_magnitude or 0.0) / 500.0,
        ((mob.serving_rsrp_dbm or -140.0) + 140.0) / 60.0,
        ((mob.serving_sinr_db or -20.0) + 20.0) / 40.0,
        (mob.neighbor_advantage_db or 0.0) / 20.0,
        ((mob.a3_margin_db or -10.0) + 10.0) / 20.0,
        (ifreq.max_dl_prb_util or 0.0) / 100.0,
        float(len(getattr(mob, "visible_neighbor_pcis", []) or [])) / 5.0,
        sf.score_late_handover,
        sf.score_missing_neighbor,
        sf.score_interference,
        sf.score_coverage_hole,
        sf.score_overshoot_overlap,
        sf.score_pdcch_issue,
        sf.score_backhaul_server,
        getattr(sf, "score_insufficient_data", 0.0) or 0.0,
    ], dtype=np.float32)
    return np.clip(arr, 0.0, 1.0)


def _old_action(raw: Dict[str, Any], sf) -> str:
    family = route(str(raw.get("tag", "")), _num_bs(raw))
    ps = parse_scenario(raw)
    output = generate_candidates(ps, sf, family)
    candidates = getattr(output, "single_candidates", None) or []
    if not candidates:
        return "none"
    return candidates[0].action_type


def _load_f1_records(train_json: Path, fold_mod: int, holdout_fold: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = json.loads(train_json.read_text(encoding="utf-8"))
    train: List[Dict[str, Any]] = []
    holdout: List[Dict[str, Any]] = []

    for raw in rows:
        sid = str(raw.get("scenario_id", "")).strip()
        tag = str(raw.get("tag", ""))
        family = route(tag, _num_bs(raw))
        if family.name != "F1":
            continue
        gt_action = _answer_action(raw, str(raw.get("answer", "")))
        if gt_action not in SAFE_F1_ACTIONS:
            continue
        ps = parse_scenario(raw)
        sf = run_all_skills(ps)
        record = {
            "sid": sid,
            "gt_action": gt_action,
            "old_action": _old_action(raw, sf),
            "vec": _vec(sf),
        }
        if _is_holdout(sid, fold_mod, holdout_fold):
            holdout.append(record)
        else:
            train.append(record)
    return train, holdout


def _cosine_topk(query: np.ndarray, matrix: np.ndarray, k: int) -> np.ndarray:
    q = query / (np.linalg.norm(query) + 1e-8)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    sims = m @ q
    k = min(k, len(sims))
    return np.argsort(-sims)[:k]


def _bucket(share: float) -> str:
    if share >= 0.80:
        return ">=0.80"
    if share >= 0.70:
        return "0.70-0.79"
    if share >= 0.60:
        return "0.60-0.69"
    if share >= 0.50:
        return "0.50-0.59"
    if share >= 0.40:
        return "0.40-0.49"
    return "<0.40"


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate F1 KNN action votes on Phase 1 holdout")
    ap.add_argument("--train_json", type=Path, default=Path("data/Phase_1/train.json"))
    ap.add_argument("--out_csv", type=Path, default=Path("reports/B_phase1_knn_calibration.csv"))
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--fold_mod", type=int, default=5)
    ap.add_argument("--holdout_fold", type=int, default=0)
    args = ap.parse_args()

    train_json = args.train_json
    if not train_json.exists():
        fallback = Path("/home/huangxiaolin/AITO/5.4/Track A/data/Phase_1/train.json")
        if fallback.exists():
            train_json = fallback
        else:
            raise SystemExit(f"missing train_json: {args.train_json}")

    train, holdout = _load_f1_records(train_json, args.fold_mod, args.holdout_fold)
    matrix = np.stack([row["vec"] for row in train], axis=0)

    bucket_stats: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)
    detail_rows: List[Dict[str, Any]] = []
    train_actions = [row["gt_action"] for row in train]

    for row in holdout:
        idxs = _cosine_topk(row["vec"], matrix, args.k)
        votes = Counter(train_actions[int(idx)] for idx in idxs)
        vote_action, vote_count = votes.most_common(1)[0]
        share = vote_count / max(len(idxs), 1)
        transition = f"{row['old_action']}->{vote_action}"
        changed = vote_action != row["old_action"]
        correct = vote_action == row["gt_action"]
        old_correct = row["old_action"] == row["gt_action"]

        key = (_bucket(share), transition if changed else "no_change", vote_action)
        bucket_stats[key]["n"] += 1
        bucket_stats[key]["vote_correct"] += int(correct)
        bucket_stats[key]["old_correct"] += int(old_correct)
        bucket_stats[key]["changed"] += int(changed)

        detail_rows.append({
            "scenario_id": row["sid"],
            "gt_action": row["gt_action"],
            "old_action": row["old_action"],
            "vote_action": vote_action,
            "share": round(share, 4),
            "share_bucket": _bucket(share),
            "transition": transition,
            "vote_correct": int(correct),
            "old_correct": int(old_correct),
            "votes": dict(votes),
        })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "share_bucket",
                "transition",
                "vote_action",
                "n",
                "changed",
                "vote_precision",
                "old_precision",
            ],
        )
        writer.writeheader()
        for (share_bucket, transition, vote_action), stats in sorted(bucket_stats.items()):
            n = stats["n"]
            writer.writerow({
                "share_bucket": share_bucket,
                "transition": transition,
                "vote_action": vote_action,
                "n": n,
                "changed": stats["changed"],
                "vote_precision": round(stats["vote_correct"] / n, 4) if n else 0.0,
                "old_precision": round(stats["old_correct"] / n, 4) if n else 0.0,
            })

    detail_path = args.out_csv.with_name(args.out_csv.stem + "_details.jsonl")
    with detail_path.open("w", encoding="utf-8") as f:
        for row in detail_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    changed_rows = [row for row in detail_rows if row["old_action"] != row["vote_action"]]
    print(json.dumps({
        "train_json": str(train_json),
        "train_records": len(train),
        "holdout_records": len(holdout),
        "k": args.k,
        "changed_vote_rows": len(changed_rows),
        "changed_vote_precision": round(
            sum(row["vote_correct"] for row in changed_rows) / max(len(changed_rows), 1),
            4,
        ),
        "out_csv": str(args.out_csv),
        "detail_jsonl": str(detail_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
