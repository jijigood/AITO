#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
build_sft_v3.py — Owner A's SFT dataset builder.

Reads the curated id list from curate_sft.py, runs the V3 pipeline on each
scenario, and writes inference-parity (system, user, assistant) triples to
sft_v3.jsonl.

Deliverable: A1b in TEAM_DIVISION.md §2.3.

Inference parity rule (SCHEMA.md §4):
  build_sft_v3.py and main_v3.py MUST import the same _build_f1_prompt symbol.
  Do not copy-paste the prompt; import it.

Usage:
  python build_sft_v3.py \\
      --ids    curated_train_ids.txt \\
      --train  train_data \\
      --out    sft_v3.jsonl \\
      --upsample                     # apply class-balanced sampling (§2.5)
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from main_v3 import (
    QWEN_V3_SYSTEM_PROMPT,
    _build_f1_prompt,
    _retrieve_rag_hits,
    _sf_to_feature_vec,
)
from skills.scenario_router import route, Family
from skills.feature_engine import run_all_skills
from skills.candidate_generator import generate_single_candidates
from skills.scenario_parser import parse_scenario
from skills.tool_caller import ToolCaller
from rag_retriever_v2 import StructuredCaseRAG


# Class-balanced sampling weights (§2.5 of TEAM_DIVISION.md).
# V4 update: lift_tilt and insufficient_data increased to improve disambiguation.
UPSAMPLE_WEIGHTS = {
    "lift_tilt":          35,
    "insufficient_data":  30,
    "add_neighbor":       15,
    "pdcch_2sym":         10,
    "check_server":        7,
    "decrease_a3":         3,
}


def _target_action_counts(total: int) -> Dict[str, int]:
    """Convert target shares into exact per-action counts that sum to `total`."""
    if total <= 0:
        return {action: 0 for action in UPSAMPLE_WEIGHTS}

    total_weight = sum(UPSAMPLE_WEIGHTS.values())
    exact = {
        action: (total * weight / total_weight)
        for action, weight in UPSAMPLE_WEIGHTS.items()
    }
    counts = {
        action: int(value)
        for action, value in exact.items()
    }
    remaining = total - sum(counts.values())
    ranked = sorted(
        UPSAMPLE_WEIGHTS,
        key=lambda action: (
            exact[action] - counts[action],
            UPSAMPLE_WEIGHTS[action],
            action,
        ),
        reverse=True,
    )
    for action in ranked[:remaining]:
        counts[action] += 1
    return counts


def _rebalance_records(records: List[dict], final_total: int) -> List[dict]:
    """
    Rebalance to exact target counts while preserving total size.

    Preference order inside each class:
      1. original records
      2. existing augmentation records
      3. cloned filler from the same class if still below quota
    """
    by_action: Dict[str, List[dict]] = {}
    for record in records:
        action = record["meta"].get("action_type", "unknown")
        by_action.setdefault(action, []).append(record)

    targets = _target_action_counts(final_total)
    rebalanced: List[dict] = []

    for action, target in targets.items():
        pool = list(by_action.get(action, []))
        if target <= 0:
            continue
        if not pool:
            raise ValueError(f"No records available for action={action!r}, cannot rebalance.")

        originals = [rec for rec in pool if "augmentation" not in rec.get("meta", {})]
        augmented = [rec for rec in pool if "augmentation" in rec.get("meta", {})]
        random.shuffle(originals)
        random.shuffle(augmented)

        chosen: List[dict] = []
        if len(originals) >= target:
            chosen.extend(originals[:target])
        else:
            chosen.extend(originals)
            deficit = target - len(chosen)

            if len(augmented) >= deficit:
                chosen.extend(augmented[:deficit])
            else:
                chosen.extend(augmented)
                filler_pool = augmented or originals
                if not filler_pool:
                    raise ValueError(f"Empty filler pool for action={action!r}.")
                for idx in range(target - len(chosen)):
                    clone = json.loads(json.dumps(filler_pool[idx % len(filler_pool)]))
                    clone["meta"]["augmentation"] = "rebalance_fill"
                    chosen.append(clone)

        rebalanced.extend(chosen)

    if len(rebalanced) != final_total:
        raise ValueError(
            f"Rebalanced dataset size mismatch: expected {final_total}, got {len(rebalanced)}."
        )
    return rebalanced


def _resolve_train_json(train_path: Path) -> Path:
    if train_path.is_file():
        return train_path
    candidate = train_path / "train.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Unable to resolve train.json from {train_path}")


def build_rationale(ps, sf, gt_c_code: str) -> str:
    """
    One-line CoT tag appended to the assistant turn to teach the decision rules.
    Stripped at inference for latency.

    Example output:
        "SINR=6.2, PRB=82% -> coverage_hole -> lift_tilt/secondary"
    """
    opt = ps.parsed_options.get(gt_c_code, {})
    action = opt.get("action_type", "?")
    target_cell = opt.get("target_cell")
    role = ps.cell_roles.get(target_cell, "") if target_cell else ""
    sinr = sf.mobility.serving_sinr_db if sf.mobility.serving_sinr_db is not None else 0.0
    rsrp = sf.mobility.serving_rsrp_dbm if sf.mobility.serving_rsrp_dbm is not None else -140.0
    prb = sf.inter_freq.max_dl_prb_util or 0.0
    role_suffix = f"/{role}" if role else ""
    return (
        f"SINR={sinr:.1f}dB "
        f"RSRP={rsrp:.0f}dBm "
        f"PRB={prb:.0f}% "
        f"→ {sf.top_issue()[0]} → {action}{role_suffix}"
    )


def _is_boundary_case(sf) -> bool:
    """True when lift_tilt and insufficient_data evidence are within 0.10 of each other."""
    cov = sf.score_coverage_hole
    insuff = getattr(sf, "score_insufficient_data", None) or 0.0
    return abs(cov - insuff) < 0.10 and max(cov, insuff) >= 0.25


def build_record(
    raw: Dict[str, Any],
    rag: Optional[StructuredCaseRAG],
    tool_caller: Optional[ToolCaller] = None,
) -> Optional[dict]:
    """
    Build one (system, user, assistant) record for F1 scenarios.
    Returns None for non-F1 (those bypass the model).

    When `tool_caller` is provided, boundary lift_tilt/insufficient_data cases
    receive live tool calls so the ``## Server Tool Evidence`` prompt section
    is non-empty in training data, reducing the train/inference distribution gap.
    """
    ps = parse_scenario(raw)
    family = route(ps.tag, ps.num_base_stations)
    if family.name != "F1":
        return None
    sf = run_all_skills(ps)

    # Boundary tool call injection (S1-2): only runs when server is available.
    if tool_caller is not None and _is_boundary_case(sf):
        worst_time = sf.throughput.worst_time or ""
        try:
            tool_evidence = tool_caller.call_targeted(ps.scenario_id, sf, worst_time)
            if tool_evidence.tool_call_count > 0:
                sf.tool_evidence = tool_evidence
                sf = tool_caller.augment_features(sf, tool_evidence)
                sf.tool_evidence = tool_evidence
        except Exception:
            pass  # server errors must not abort SFT generation

    candidates = generate_single_candidates(ps, sf, family, top_n=6)
    candidate_tuples = [
        (cand.c_code, cand.score, cand.reason)
        for cand in candidates
    ]
    rag_hits = []
    if rag is not None:
        rag_hits = _retrieve_rag_hits(
            rag,
            sf,
            family.name,
            top_k=6,
            exclude_scenario_id=ps.scenario_id,
        )

    user = _build_f1_prompt(ps, sf, candidate_tuples, rag_hits)
    gt_c_code = ps.ground_truth       # one C-code for F1
    rationale = build_rationale(ps, sf, gt_c_code)
    gt_opt = ps.parsed_options.get(gt_c_code, {})
    target_cell = gt_opt.get("target_cell")
    target_role = ps.cell_roles.get(target_cell, "") if target_cell else ""
    feature_vec = _sf_to_feature_vec(sf).tolist()   # now 16-dim

    return {
        "system":    QWEN_V3_SYSTEM_PROMPT,
        "user":      user,
        "assistant": f"\\boxed{{{gt_c_code}}}\n# {rationale}",
        "messages": [
            {"role": "system", "content": QWEN_V3_SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": f"\\boxed{{{gt_c_code}}}\n# {rationale}"},
        ],
        "meta": {
            "scenario_id": ps.scenario_id,
            "action_type": gt_opt.get("action_type", "unknown"),
            "target_role": target_role,
            "feature_vec": feature_vec,
        },
    }


def _upsample(records: List[dict]) -> List[dict]:
    by_action: Dict[str, List[dict]] = {}
    for record in records:
        action = record["meta"].get("action_type", "unknown")
        by_action.setdefault(action, []).append(record)

    total = len(records)
    augmented = list(records)
    for action, target_share in UPSAMPLE_WEIGHTS.items():
        current = by_action.get(action, [])
        if not current:
            continue
        desired = int(round(total * (target_share / 100.0)))
        if desired <= len(current):
            continue
        deficit = desired - len(current)
        for idx in range(deficit):
            src = current[idx % len(current)]
            clone = json.loads(json.dumps(src))
            clone["meta"]["augmentation"] = "upsample"
            augmented.append(clone)
    return augmented


def _inject_boundary_cases(records: List[dict], final_total: int = 0) -> List[dict]:
    by_action: Dict[str, List[dict]] = {}
    for record in records:
        by_action.setdefault(record["meta"].get("action_type", "unknown"), []).append(record)

    left = by_action.get("lift_tilt", [])
    right = by_action.get("insufficient_data", [])
    if not left or not right:
        return records

    left_vecs = np.array([rec["meta"]["feature_vec"] for rec in left], dtype=float)
    right_vecs = np.array([rec["meta"]["feature_vec"] for rec in right], dtype=float)
    left_norm = left_vecs / (np.linalg.norm(left_vecs, axis=1, keepdims=True) + 1e-8)
    right_norm = right_vecs / (np.linalg.norm(right_vecs, axis=1, keepdims=True) + 1e-8)

    augmented = list(records)
    sims_lr = left_norm @ right_norm.T
    sims_rl = right_norm @ left_norm.T

    for idx, rec in enumerate(left):
        nearest = np.argsort(-sims_lr[idx])[:2]
        for j in nearest:
            clone = json.loads(json.dumps(right[int(j)]))
            clone["meta"]["augmentation"] = "hard_negative_boundary"
            augmented.append(clone)

    for idx, rec in enumerate(right):
        nearest = np.argsort(-sims_rl[idx])[:2]
        for j in nearest:
            clone = json.loads(json.dumps(left[int(j)]))
            clone["meta"]["augmentation"] = "hard_negative_boundary"
            augmented.append(clone)

    if final_total <= 0:
        return augmented

    # Re-balance to exact target counts so boundary negatives do not swamp the
    # final class mix while still remaining available as training examples.
    return _rebalance_records(augmented, final_total)


def _strip_internal_fields(record: dict) -> dict:
    clean = json.loads(json.dumps(record))
    clean["meta"].pop("feature_vec", None)
    return clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids",   type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--out",   type=Path, default=Path("sft_v3.jsonl"))
    ap.add_argument("--upsample", action="store_true")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="If set, use this competition server URL to inject tool evidence into "
             "ambiguous lift_tilt/insufficient_data boundary records (S1-2).",
    )
    args = ap.parse_args()

    random.seed(args.seed)

    ids = [line.strip() for line in args.ids.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_json = _resolve_train_json(args.train)
    with open(train_json, encoding="utf-8") as f:
        raw_examples = json.load(f)
    raw_by_id = {raw.get("scenario_id", ""): raw for raw in raw_examples}

    weights: Dict[str, float] = {}
    if args.weights and args.weights.exists():
        with open(args.weights, encoding="utf-8") as f:
            weights = json.load(f)

    rag = StructuredCaseRAG(str(train_json))
    rag.build_index()

    tool_caller: Optional[ToolCaller] = None
    if args.server_url:
        tool_caller = ToolCaller(args.server_url)
        print(f"[SFT] Tool caller enabled for boundary cases: {args.server_url}")

    records: List[dict] = []
    for sid in ids:
        raw = raw_by_id.get(sid)
        if raw is None:
            continue
        record = build_record(raw, rag, tool_caller=tool_caller)
        if record is None:
            continue
        record["meta"]["sample_weight"] = float(weights.get(sid, 1.0))
        records.append(record)

    if args.upsample:
        records = _upsample(records)
        records = _inject_boundary_cases(records, final_total=len(records))

    random.shuffle(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_strip_internal_fields(record), ensure_ascii=False) + "\n")

    hist = Counter(record["meta"].get("action_type", "unknown") for record in records)
    print(f"Wrote {len(records)} records to {args.out}")
    print("Action histogram:")
    for action, count in sorted(hist.items()):
        print(f"  {action:<20} {count}")


if __name__ == "__main__":
    main()
