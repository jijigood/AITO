#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Check the combined 600-row Phase 2 submission format."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


VALID_CODE_RE = re.compile(r"^C(?:[1-9]|1[0-9]|2[0-4])(?:\|C(?:[1-9]|1[0-9]|2[0-4]))*$")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate 600-row Track A/B submission template")
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("--expected_rows", type=int, default=600)
    ap.add_argument("--expected_track_a_answers", type=int, default=500)
    args = ap.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as f:
        rows = [{k: (v or "") for k, v in row.items()} for row in csv.DictReader(f)]

    errors = []
    if len(rows) != args.expected_rows:
        errors.append(f"row count {len(rows)} != {args.expected_rows}")

    if rows:
        missing_cols = [col for col in ("ID", "Track A", "Track B") if col not in rows[0]]
        if missing_cols:
            errors.append(f"missing columns: {missing_cols}")

    ids = [row.get("ID", "").strip() for row in rows]
    if any(not sid for sid in ids):
        errors.append("blank ID exists")
    if len(ids) != len(set(ids)):
        errors.append("duplicate ID exists")

    track_a_answers = 0
    track_b_nonempty = 0
    bad_answers = []
    unsorted_answers = []
    for idx, row in enumerate(rows, 2):
        answer = row.get("Track A", "").strip()
        if row.get("Track B", "").strip():
            track_b_nonempty += 1
        if not answer:
            continue
        track_a_answers += 1
        if not VALID_CODE_RE.fullmatch(answer):
            bad_answers.append((idx, row.get("ID", ""), answer))
            continue
        codes = answer.split("|")
        nums = [int(code[1:]) for code in codes]
        if nums != sorted(nums):
            unsorted_answers.append((idx, row.get("ID", ""), answer))

    if track_a_answers != args.expected_track_a_answers:
        errors.append(f"Track A answered {track_a_answers} != {args.expected_track_a_answers}")
    if track_b_nonempty:
        errors.append(f"Track B non-empty rows: {track_b_nonempty}")
    if bad_answers:
        errors.append(f"invalid Track A answers: {bad_answers[:5]}")
    if unsorted_answers:
        print(f"warning: unsorted answers first={unsorted_answers[:5]}")

    if errors:
        print("NOT READY")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(2)

    print("READY")
    print(f"- rows: {len(rows)}")
    print(f"- Track A answered: {track_a_answers}")
    print(f"- Track B non-empty: {track_b_nonempty}")


if __name__ == "__main__":
    main()
