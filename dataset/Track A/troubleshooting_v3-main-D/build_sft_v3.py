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
from pathlib import Path

from main_v3 import _build_f1_prompt, QWEN_V3_SYSTEM_PROMPT   # shared symbol
from skills.scenario_parser import parse_scenario as parse
from skills.scenario_router import route, Family
from skills.feature_engine import run_all_skills
from skills.candidate_generator import generate_single_candidates
from rag_retriever_v2 import StructuredCaseRAG


# Class-balanced sampling weights (§2.5 of TEAM_DIVISION.md).
UPSAMPLE_WEIGHTS = {
    "lift_tilt":          30,
    "insufficient_data":  25,
    "add_neighbor":       18,
    "pdcch_2sym":         12,
    "check_server":       10,
    "decrease_a3":         5,
}


def build_rationale(sf, gt_c_code: str) -> str:
    """
    One-line CoT tag appended to the assistant turn to teach the decision rules.
    Stripped at inference for latency.

    Example output:
        "SINR=6.2, PRB=82% -> coverage_hole -> lift_tilt/secondary"
    """
    # TODO(Owner A): implement in A1b.
    raise NotImplementedError("Owner A: A1b deliverable")


def build_record(scenario_id: str, train_dir: Path, rag: StructuredCaseRAG) -> dict | None:
    """
    Build one (system, user, assistant) record for F1 scenarios.
    Returns None for non-F1 (those bypass the model).
    """
    ps = parse(scenario_id, train_dir)
    if route(ps) != Family.F1:
        return None
    sf = run_all_skills(ps)
    candidates = generate_single_candidates(ps, sf)[:6]
    rag_hits = rag.retrieve(sf, k=6)

    user = _build_f1_prompt(ps, sf, candidates, rag_hits)
    gt_c_code = ps.ground_truth       # one C-code for F1
    rationale = build_rationale(sf, gt_c_code)

    return {
        "system":    QWEN_V3_SYSTEM_PROMPT,
        "user":      user,
        "assistant": f"\\boxed{{{gt_c_code}}}\n# {rationale}",
        "meta":      {"scenario_id": scenario_id, "action_type": ps.gt_action_type},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids",   type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--out",   type=Path, default=Path("sft_v3.jsonl"))
    ap.add_argument("--upsample", action="store_true")
    args = ap.parse_args()

    # TODO(Owner A):
    # 1. Load curated ids
    # 2. Instantiate StructuredCaseRAG
    # 3. For each id, build_record(); skip None
    # 4. If --upsample, replicate records per UPSAMPLE_WEIGHTS
    # 5. Write jsonl; print histogram of action types
    raise NotImplementedError("Owner A: A1b deliverable")


if __name__ == "__main__":
    main()
