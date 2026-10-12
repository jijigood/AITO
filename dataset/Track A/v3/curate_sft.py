#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
curate_sft.py — Owner A's SFT-data curation step.

Consumes Owner D's scoring library (skills/data_tools.py) and decides which
training scenarios go into sft_v3.jsonl.

Deliverable: A1a in TEAM_DIVISION.md §2.3.

Output:
  - curated_train_ids.txt    (one scenario_id per line; consumed by build_sft_v3.py)
  - curation_report.md       (human-readable counts: dropped per reason, kept per class)

Usage:
  python curate_sft.py \\
      --train_dir train_data \\
      --out_ids curated_train_ids.txt \\
      --report  curation_report.md
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from main_v3 import _sf_to_feature_vec
from skills.answer_parser import validate_and_normalise
from skills.candidate_generator import generate_candidates
from skills.data_tools import score_all_scenarios
from skills.feature_engine import run_all_skills
from skills.scenario_parser import parse_scenario
from skills.scenario_router import route


# Inclusion thresholds — A owns these numbers.
# Re-tune on val IOU; document every change in curation_report.md.
THRESHOLDS = {
    "duplicate_drop":   0.98,   # drop if cosine sim > this
    "confidence_drop":  0.30,   # drop if label confidence < this
    "outlier_flag":     3.00,   # flag for manual review (do not auto-drop)
    "ambiguity_weight": 0.70,   # down-weight (not drop) training loss for these
}


def _resolve_train_json(train_dir: Path) -> Path:
    if train_dir.is_file():
        return train_dir
    candidate = train_dir / "train.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Unable to resolve train.json from {train_dir}")


def _load_f1_examples(train_json: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    with open(train_json, encoding="utf-8") as f:
        raw_examples = json.load(f)

    f1_examples: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for raw in raw_examples:
        ps = parse_scenario(raw)
        family = route(ps.tag, ps.num_base_stations)
        if family.name != "F1":
            continue
        if not validate_and_normalise(ps.ground_truth or ""):
            continue
        f1_examples.append(raw)
        by_id[ps.scenario_id] = raw
    return f1_examples, by_id


def _build_fallback_scores(
    examples: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    scenario_rows: List[Dict[str, Any]] = []
    for raw in examples:
        ps = parse_scenario(raw)
        family = route(ps.tag, ps.num_base_stations)
        sf = run_all_skills(ps)
        candidates = generate_candidates(ps, sf, family)
        gt = validate_and_normalise(ps.ground_truth or "")
        gt_opt = ps.parsed_options.get(gt, {})
        candidate_scores: Dict[str, float] = {}
        ordered_codes: List[str] = []
        if candidates and candidates.single_candidates:
            for cand in candidates.single_candidates:
                candidate_scores[cand.c_code] = cand.score
                ordered_codes.append(cand.c_code)
        feature_vec = _sf_to_feature_vec(sf)
        scenario_rows.append(
            {
                "scenario_id": ps.scenario_id,
                "feature_vec": feature_vec,
                "action_type": gt_opt.get("action_type", "unknown"),
                "gt": gt,
                "candidate_scores": candidate_scores,
                "ordered_codes": ordered_codes,
                "gt_present": gt in ps.parsed_options,
            }
        )

    vecs = np.stack([row["feature_vec"] for row in scenario_rows], axis=0).astype(np.float32)
    dist_matrix = np.sqrt(((vecs[:, None, :] - vecs[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dist_matrix, np.inf)

    # Action-wise centroid distance as a light-weight outlier proxy.
    action_to_indices: Dict[str, List[int]] = {}
    for idx, row in enumerate(scenario_rows):
        action_to_indices.setdefault(row["action_type"], []).append(idx)

    centroids: Dict[str, np.ndarray] = {}
    action_dists: Dict[str, List[float]] = {}
    for action, indices in action_to_indices.items():
        subset = vecs[indices]
        centroid = subset.mean(axis=0)
        centroids[action] = centroid
        action_dists[action] = [
            float(np.linalg.norm(vecs[i] - centroid)) for i in indices
        ]

    score_map: Dict[str, Dict[str, float]] = {}
    for idx, row in enumerate(scenario_rows):
        action = row["action_type"]
        min_dist = float(dist_matrix[idx].min())
        duplicate = max(0.0, 1.0 - min(min_dist, 1.0))

        dists = action_dists[action]
        own_dist = float(np.linalg.norm(vecs[idx] - centroids[action]))
        mean_dist = float(np.mean(dists)) if dists else 0.0
        std_dist = float(np.std(dists)) if dists else 0.0
        outlier = 0.0 if std_dist < 1e-8 else max(0.0, (own_dist - mean_dist) / std_dist)

        if not row["gt_present"]:
            confidence = 0.0
        elif row["gt"] not in row["candidate_scores"]:
            confidence = 0.40
        else:
            gt_score = float(row["candidate_scores"][row["gt"]])
            other_scores = [
                score for code, score in row["candidate_scores"].items() if code != row["gt"]
            ]
            second = max(other_scores) if other_scores else 0.0
            margin = max(0.0, gt_score - second)
            confidence = 0.45 + 0.40 * min(gt_score, 1.0) + 0.15 * min(margin, 1.0)
            if row["ordered_codes"] and row["ordered_codes"][0] == row["gt"]:
                confidence += 0.10
            confidence = min(confidence, 1.0)

        ordered_scores = [
            row["candidate_scores"][code]
            for code in row["ordered_codes"][:2]
            if code in row["candidate_scores"]
        ]
        if len(ordered_scores) < 2 or ordered_scores[0] <= 1e-8:
            ambiguity = 0.0
        else:
            ambiguity = min(1.0, ordered_scores[1] / ordered_scores[0])

        score_map[row["scenario_id"]] = {
            "duplicate": duplicate,
            "outlier": outlier,
            "confidence": confidence,
            "ambiguity": ambiguity,
        }

    return score_map


def _compute_scores(
    train_ids: List[str],
    fallback_examples: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    try:
        return "data_tools", score_all_scenarios(train_ids)
    except NotImplementedError:
        pass
    except Exception:
        pass
    return "fallback", _build_fallback_scores(fallback_examples)


def curate(
    train_dir: Path,
    out_ids: Path,
    report: Path,
    val_size: int = 200,
    val_ids_path: Path | None = None,
    weights_path: Path | None = None,
) -> None:
    train_json = _resolve_train_json(train_dir)
    f1_examples, by_id = _load_f1_examples(train_json)
    sorted_ids = sorted(by_id.keys())
    heldout_val_ids = sorted_ids[-min(val_size, len(sorted_ids)):] if val_size > 0 else []
    heldout_val_set = set(heldout_val_ids)
    train_ids = [sid for sid in sorted_ids if sid not in heldout_val_set]

    if val_ids_path is None:
        val_ids_path = out_ids.with_name("heldout_f1_val_ids.txt")
    if weights_path is None:
        weights_path = out_ids.with_name("curation_weights.json")

    mode, score_map = _compute_scores(sorted_ids, f1_examples)

    buckets: Dict[str, str] = {}
    weights: Dict[str, float] = {}
    action_counts: Counter = Counter()

    for sid in train_ids:
        raw = by_id[sid]
        ps = parse_scenario(raw)
        gt = validate_and_normalise(ps.ground_truth or "")
        action = ps.parsed_options.get(gt, {}).get("action_type", "unknown")
        action_counts[action] += 1

        scores = score_map[sid]
        if scores["duplicate"] > THRESHOLDS["duplicate_drop"]:
            buckets[sid] = "DROP_DUPLICATE"
            continue
        if scores["confidence"] < THRESHOLDS["confidence_drop"]:
            buckets[sid] = "DROP_LOW_CONFIDENCE"
            continue

        buckets[sid] = "KEEP_FLAGGED" if scores["outlier"] > THRESHOLDS["outlier_flag"] else "KEEP"
        weights[sid] = 0.5 if scores["ambiguity"] > THRESHOLDS["ambiguity_weight"] else 1.0

    kept_ids = [sid for sid in train_ids if buckets.get(sid, "").startswith("KEEP")]
    dropped_ids = [sid for sid in train_ids if sid not in kept_ids]

    out_ids.parent.mkdir(parents=True, exist_ok=True)
    out_ids.write_text("\n".join(kept_ids) + ("\n" if kept_ids else ""), encoding="utf-8")
    val_ids_path.write_text(
        "\n".join(heldout_val_ids) + ("\n" if heldout_val_ids else ""), encoding="utf-8"
    )
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)

    bucket_counts = Counter(buckets.values())
    weight_counts = Counter(weights.values())
    report_lines = [
        "# Curation Report",
        "",
        f"- Mode: `{mode}`",
        f"- Train source: `{train_json}`",
        f"- Total F1 examples: `{len(sorted_ids)}`",
        f"- Held-out validation size: `{len(heldout_val_ids)}`",
        f"- Training candidates before curation: `{len(train_ids)}`",
        f"- Kept training IDs: `{len(kept_ids)}`",
        f"- Dropped training IDs: `{len(dropped_ids)}`",
        f"- Curated IDs: `{out_ids}`",
        f"- Held-out val IDs: `{val_ids_path}`",
        f"- Weight metadata: `{weights_path}`",
        "",
        "## Drop Counts",
        "",
    ]
    for name in ("DROP_DUPLICATE", "DROP_LOW_CONFIDENCE", "KEEP", "KEEP_FLAGGED"):
        report_lines.append(f"- `{name}`: {bucket_counts.get(name, 0)}")
    report_lines.extend(
        [
            "",
            "## Weight Counts",
            "",
        ]
    )
    for weight_value, count in sorted(weight_counts.items()):
        report_lines.append(f"- `{weight_value}`: {count}")

    report_lines.extend(["", "## F1 Action Distribution (pre-curation train split)", ""])
    for action, count in sorted(action_counts.items()):
        report_lines.append(f"- `{action}`: {count}")

    report_lines.extend(["", "## Thresholds", ""])
    for key, value in THRESHOLDS.items():
        report_lines.append(f"- `{key}`: {value}")

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=Path, required=True)
    ap.add_argument("--out_ids",   type=Path, default=Path("curated_train_ids.txt"))
    ap.add_argument("--report",    type=Path, default=Path("curation_report.md"))
    ap.add_argument("--val_size",  type=int, default=200)
    ap.add_argument("--val_ids",   type=Path, default=None)
    ap.add_argument("--weights_out", type=Path, default=None)
    args = ap.parse_args()

    curate(
        args.train_dir,
        args.out_ids,
        args.report,
        val_size=args.val_size,
        val_ids_path=args.val_ids,
        weights_path=args.weights_out,
    )


if __name__ == "__main__":
    main()
