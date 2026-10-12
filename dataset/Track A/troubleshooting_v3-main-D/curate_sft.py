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
from collections import Counter
from pathlib import Path

from skills.data_tools import score_all_scenarios


# Inclusion thresholds — A owns these numbers.
# Re-tune on val IOU; document every change in curation_report.md.
THRESHOLDS = {
    "duplicate_drop":   0.98,   # drop if cosine sim > this
    "confidence_drop":  0.30,   # drop if label confidence < this
    "outlier_flag":     3.00,   # flag for manual review (do not auto-drop)
    "ambiguity_weight": 0.70,   # down-weight (not drop) training loss for these
}


def curate(train_dir: Path, out_ids: Path, report: Path) -> None:
    # TODO(Owner A): implement in A1a. Skeleton:
    #
    # 1. Enumerate scenario_ids from train_dir
    # 2. Call data_tools.score_all_scenarios(scenario_ids)
    # 3. Apply THRESHOLDS → bucket each scenario: KEEP / DROP / WEIGHT
    # 4. Write curated_train_ids.txt
    # 5. Write curation_report.md with per-reason counts
    raise NotImplementedError("Owner A: A1a deliverable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=Path, required=True)
    ap.add_argument("--out_ids",   type=Path, default=Path("curated_train_ids.txt"))
    ap.add_argument("--report",    type=Path, default=Path("curation_report.md"))
    args = ap.parse_args()

    curate(args.train_dir, args.out_ids, args.report)


if __name__ == "__main__":
    main()
