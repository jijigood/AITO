#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
build_trajectory_examples.py — Owner D Task 02 dataset builder.

Reads Phase 1 train.json and writes prompt-safe trajectory records to JSONL.
The conversion logic lives in rag_retriever_v2.py so Task 03 can reuse the
same schema-aware record builder for indexing and retrieval.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional

from rag_retriever_v2 import build_trajectory_records_from_train


def _resolve_train_json(train_path: Path) -> Path:
    if train_path.is_file():
        return train_path
    candidate = train_path / "train.json"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Unable to resolve train.json from {train_path}")


def _write_jsonl(records, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _guard_smoke_output(out_path: Path, limit: Optional[int]) -> None:
    default_out = (Path(__file__).parent / "artifacts/owner_d/data/trajectory_examples.jsonl").resolve()
    if limit is None:
        return
    if out_path.resolve() == default_out:
        raise SystemExit(
            "Refusing to overwrite trajectory_examples.jsonl with a limited sample. "
            "Pass --out to a different path for smoke tests."
        )


def _summarize(records) -> str:
    families = Counter(record["family"] for record in records)
    branches = Counter(record["branch"] for record in records)
    actions = Counter(record["action"] for record in records)
    confidences = Counter(record["confidence"] for record in records)
    lines = [f"records={len(records)}"]
    lines.append("families=" + json.dumps(dict(sorted(families.items())), ensure_ascii=False))
    lines.append("top_branches=" + json.dumps(dict(branches.most_common(8)), ensure_ascii=False))
    lines.append("top_actions=" + json.dumps(dict(actions.most_common(8)), ensure_ascii=False))
    lines.append("confidences=" + json.dumps(dict(sorted(confidences.items())), ensure_ascii=False))
    return "\n".join(lines)


def main(limit: Optional[int] = None) -> int:
    ap = argparse.ArgumentParser(description="Build trajectory_examples.jsonl from Phase 1 train.json")
    ap.add_argument(
        "--train",
        type=Path,
        default=Path(__file__).parent / "data/Phase_1/train.json",
        help="Path to train.json or its parent directory",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "artifacts/owner_d/data/trajectory_examples.jsonl",
        help="Output JSONL path",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=limit,
        help="Optional record limit for smoke tests",
    )
    args = ap.parse_args()

    train_json = _resolve_train_json(args.train)
    _guard_smoke_output(args.out, args.limit)
    records = build_trajectory_records_from_train(str(train_json), limit=args.limit)
    _write_jsonl(records, args.out)
    print(_summarize(records))
    if records:
        print("sample=" + json.dumps(records[0], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
