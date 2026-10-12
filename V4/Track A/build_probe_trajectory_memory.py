#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Build probe-derived trajectory memory from Phase 2 diagnostics JSONL files.

This script is intentionally tolerant of missing files. When no probe
diagnostics are present in the workspace, it writes an empty JSONL and reports
that no probe memory could be extracted yet.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from rag_retriever_v2 import load_probe_trajectory_records


def main() -> int:
    ap = argparse.ArgumentParser(description="Build probe trajectory memory from diagnostics JSONL files")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "artifacts/owner_d/data/trajectory_probe_memory.jsonl",
        help="Output JSONL path",
    )
    ap.add_argument(
        "--diagnostics",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit diagnostics JSONL paths",
    )
    args = ap.parse_args()

    records = load_probe_trajectory_records(
        [str(p) for p in args.diagnostics] if args.diagnostics else None
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    branch_counts = Counter(record["branch"] for record in records)
    print("probe_records", len(records))
    print("branch_counts", json.dumps(dict(branch_counts.most_common()), ensure_ascii=False))
    if records:
        print("sample", json.dumps(records[0], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
