#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
submit.py — Owner A's final submission pipeline.

Runs main_v3.py against the competition eval server, writes result_final.csv,
and uploads via the portal.

Deliverable: A6 in TEAM_DIVISION.md §2.3.

Safety:
  --dry-run mode runs everything locally but does not upload. MUST pass a
  full dry-run before any real submission.

Usage:
  python submit.py \\
      --server_url http://eval.server:7860 \\
      --model_url  http://localhost:8001/v1 \\
      --lora       checkpoints/v3/ckpt_latest \\
      --output_dir submissions/run_$(date +%Y%m%d_%H%M%S) \\
      --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_url", required=True)
    ap.add_argument("--model_url",  required=True)
    ap.add_argument("--lora",       required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # TODO(Owner A) A6:
    # 1. Run main_v3.py with given args
    # 2. Parse result.csv; sanity-check C-codes are all in C1..C22
    # 3. Compute mean IOU on held-out (if labels available)
    # 4. If not --dry-run, upload result_final.csv to leaderboard
    # 5. Write submission_receipt.json with timestamp + server response
    raise NotImplementedError("Owner A: A6 deliverable")


if __name__ == "__main__":
    main()
