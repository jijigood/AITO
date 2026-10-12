#!/usr/bin/env python3
"""Validate Track B result.csv formatting against the Phase test questions."""

import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from answer_validation import validate_answer
from prompts import detect_family


def load_questions(path: str) -> dict[int, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {int(item["task"]["id"]): item["task"]["question"] for item in data}


def load_results(path: str) -> tuple[dict[int, str], list[int]]:
    results: dict[int, str] = {}
    duplicates: list[int] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            qid = int(row["id"])
            if qid in results:
                duplicates.append(qid)
            results[qid] = row.get("prediction", "")
    return results, duplicates


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Track B result.csv format")
    parser.add_argument("--input", default="data/Phase_1/test.json")
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    questions = load_questions(args.input)
    results, duplicates = load_results(args.result)

    missing = sorted(set(questions) - set(results))
    extra = sorted(set(results) - set(questions))
    risks = []
    family_counts = Counter()
    clean_counts = Counter()

    for qid in sorted(questions):
        question = questions[qid]
        family = detect_family(question)
        family_counts[family] += 1
        answer = results.get(qid, "")
        validation = validate_answer(answer, family, question)
        if validation.ok:
            clean_counts[family] += 1
        else:
            preview = answer.replace("\n", " | ")[:140]
            risks.append((qid, family, validation.reasons, preview))

    print(
        f"questions={len(questions)} results={len(results)} missing={len(missing)} "
        f"extra={len(extra)} duplicates={len(duplicates)}"
    )
    if missing:
        print("missing ids:", ",".join(map(str, missing)))
    if extra:
        print("extra ids:", ",".join(map(str, extra)))
    if duplicates:
        print("duplicate ids:", ",".join(map(str, sorted(set(duplicates)))))
    print("family_counts:", dict(family_counts))
    print("format_clean:", sum(clean_counts.values()), "/", len(questions), dict(clean_counts))
    print("format_risks:", len(risks))
    for qid, family, reasons, preview in risks:
        print(f"Q{qid:02d} {family:8s} reasons={list(reasons)} preview={preview!r}")

    return 1 if missing or extra or duplicates or risks else 0


if __name__ == "__main__":
    raise SystemExit(main())
