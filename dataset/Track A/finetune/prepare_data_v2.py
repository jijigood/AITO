#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
V2 Fine-tuning Dataset Generator — Track A

Generates SFT training data that exactly matches the main_v2.py inference format:
  [system]   QWEN_V2_SYSTEM_PROMPT  (short selection instruction)
  [user]     structured evidence + ranked candidates + all options
  [assistant] \\boxed{answer}

This replaces prepare_data.py (V1) which produced 9–15 turn tool-calling
conversations that do not match the V2 single-turn inference format.

Key design decisions:
  - RAG context is OMITTED from training data to avoid data leakage
    (each example would retrieve its own neighbours). The model learns
    to select from evidence + candidates; RAG is additive at inference time.
  - If the V2 pipeline fails for an example (parse/skills error), the
    example is skipped with a warning and counted.
  - Label: always the ground-truth answer from train.json (pipe-delimited
    C-codes in ascending numeric order, already normalised in training data).

Output:
    finetune/sft_v2_train.jsonl  (1800 examples)
    finetune/sft_v2_val.jsonl    (200 examples)

Usage:
    cd "Track A"
    python finetune/prepare_data_v2.py [--train_json data/Phase_1/train.json]
"""

import argparse
import json
import os
import random
import sys
import traceback
from typing import Any, Dict, List, Optional

# Ensure the Track A root is on the path so local modules resolve correctly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TRACK_A_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _TRACK_A_DIR)

from main_v2 import QWEN_V2_SYSTEM_PROMPT, _build_qwen_prompt, _sf_to_feature_vec
from skills.scenario_parser import parse_scenario
from skills.scenario_router import route
from skills.feature_engine import run_all_skills
from skills.candidate_generator import generate_candidates
from skills.answer_parser import validate_and_normalise


# ──────────────────────────────────────────────────────────────────────────────
# Conversation builder
# ──────────────────────────────────────────────────────────────────────────────

def build_v2_conversation(
    raw: Dict[str, Any],
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Build a 3-message SFT conversation for one training example.

    Returns a dict:
        {
            "messages":     [system, user, assistant],
            "scenario_id":  str,
            "answer":       str,   # ground truth
            "family":       str,   # F1/F2/F3/F4
            "top_issue":    str,
        }
    or None if the example cannot be processed.
    """
    scenario_id = raw.get("scenario_id", "?")
    answer = raw.get("answer", "")

    # Ground truth must exist and be valid
    normalised_answer = validate_and_normalise(answer)
    if not normalised_answer:
        if verbose:
            print(f"  [skip] {scenario_id}: invalid/missing answer '{answer}'")
        return None

    # ── V2 pipeline (local — no HTTP) ─────────────────────────────────────────
    try:
        ps = parse_scenario(raw)
    except Exception as exc:
        if verbose:
            print(f"  [skip] {scenario_id}: parse_scenario failed — {exc}")
        return None

    try:
        family = route(ps.tag, ps.num_base_stations)
    except Exception as exc:
        if verbose:
            print(f"  [skip] {scenario_id}: route() failed — {exc}")
        return None

    try:
        sf = run_all_skills(ps)
    except Exception as exc:
        if verbose:
            print(f"  [skip] {scenario_id}: run_all_skills() failed — {exc}")
        return None

    try:
        candidates = generate_candidates(ps, sf, family)
        candidates_text = candidates.format_for_prompt()
    except Exception as exc:
        if verbose:
            print(f"  [warn] {scenario_id}: generate_candidates() failed — {exc}; continuing without candidates")
        candidates_text = ""

    # ── Build prompt (no RAG — avoids data leakage) ───────────────────────────
    try:
        user_msg = _build_qwen_prompt(
            scenario=raw,
            family_name=family.name,
            sf=sf,
            candidates_text=candidates_text,
            rag_ctx="",          # intentionally omitted during training
        )
    except Exception as exc:
        if verbose:
            print(f"  [skip] {scenario_id}: _build_qwen_prompt() failed — {exc}")
        return None

    # ── Assistant turn: just the boxed answer ─────────────────────────────────
    assistant_content = f"\\boxed{{{normalised_answer}}}"

    top_issue, _ = sf.top_issue()

    return {
        "messages": [
            {"role": "system",    "content": QWEN_V2_SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_content},
        ],
        "scenario_id": scenario_id,
        "answer":      normalised_answer,
        "family":      family.name,
        "top_issue":   top_issue,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate V2 SFT dataset for Track A fine-tuning")
    parser.add_argument(
        "--train_json",
        type=str,
        default=os.path.join(_TRACK_A_DIR, "data", "Phase_1", "train.json"),
        help="Path to train.json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=_SCRIPT_DIR,
        help="Output directory for JSONL files (default: finetune/)",
    )
    parser.add_argument("--val_size",  type=int, default=200, help="Validation split size")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    print(f"Loading training data from {args.train_json}...")
    with open(args.train_json, encoding="utf-8") as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples.")

    records: List[Dict[str, Any]] = []
    skipped = 0
    family_counts: Dict[str, int] = {}

    for i, raw in enumerate(examples):
        result = build_v2_conversation(raw, verbose=args.verbose)
        if result is None:
            skipped += 1
            continue
        records.append(result)
        family_counts[result["family"]] = family_counts.get(result["family"], 0) + 1

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(examples)} | usable={len(records)} | skipped={skipped}")

    print(f"\nTotal usable: {len(records)} | Skipped: {skipped}")
    print(f"Family distribution: {family_counts}")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(records)
    val_set   = records[: args.val_size]
    train_set = records[args.val_size :]

    # Report answer-type distribution in train set
    issue_counts: Dict[str, int] = {}
    for r in train_set:
        k = r["top_issue"]
        issue_counts[k] = issue_counts.get(k, 0) + 1
    print(f"Train top-issue distribution: {dict(sorted(issue_counts.items(), key=lambda x: -x[1]))}")

    def write_jsonl(path: str, data: List[Dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} records → {path}")

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "sft_v2_train.jsonl"), train_set)
    write_jsonl(os.path.join(args.out_dir, "sft_v2_val.jsonl"),   val_set)

    # ── Estimate token budget ──────────────────────────────────────────────────
    # Rough heuristic: 1 token ≈ 3.5 chars for mixed English/numbers
    sample = records[:50]
    avg_chars = sum(
        len(r["messages"][1]["content"]) + len(r["messages"][2]["content"])
        for r in sample
    ) / max(len(sample), 1)
    avg_tokens_est = int(avg_chars / 3.5)
    print(f"\nEstimated avg tokens per example (user+assistant): ~{avg_tokens_est}")
    print(f"Recommended cutoff_len in llamafactory_config_v2.yaml: {min(2048, avg_tokens_est * 2)}")
    print("\nDone. Next: update llamafactory_config_v2.yaml and run train.py --config finetune/llamafactory_config_v2.yaml")


if __name__ == "__main__":
    main()
