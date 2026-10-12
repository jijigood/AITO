#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Merge Phase 1 trajectory memory and optional Phase 2 probe memory.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge train/probe trajectory memory")
    ap.add_argument("--train", type=Path, default=Path(__file__).parent / "artifacts/owner_d/data/trajectory_examples.jsonl")
    ap.add_argument("--probe", type=Path, default=Path(__file__).parent / "artifacts/owner_d/data/trajectory_probe_memory.jsonl")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "artifacts/owner_d/data/trajectory_memory_combined.jsonl")
    args = ap.parse_args()

    train_rows = _load_jsonl(args.train)
    probe_rows = _load_jsonl(args.probe)

    merged = {row["scenario_id"]: row for row in train_rows}
    for row in probe_rows:
        merged[row["scenario_id"]] = row

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in sorted(merged.values(), key=lambda r: (str(r.get("source_origin", "")), str(r.get("scenario_id", "")))):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter((row.get("source_origin", "unknown")) for row in merged.values())
    print("combined_records", len(merged))
    print("source_counts", json.dumps(dict(sorted(counts.items())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
