#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Build a small branch coverage report for trajectory memory and snippets.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


HIGH_RISK_BRANCHES = [
    "lift_tilt_vs_insufficient_data",
    "add_neighbor_vs_decrease_a3",
    "inter_freq_load",
    "pdcch_issue",
    "backhaul_server",
    "overshoot_overlap",
]


def _load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Build branch coverage report for trajectory memory")
    ap.add_argument("--memory", type=Path, default=Path(__file__).parent / "artifacts/owner_d/data/trajectory_memory_combined.jsonl")
    ap.add_argument("--snippets", type=Path, default=Path(__file__).parent / "artifacts/owner_d/docs/trajectory_prompt_snippets.md")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "artifacts/owner_d/docs/trajectory_coverage_report.md")
    args = ap.parse_args()

    rows = _load_jsonl(args.memory)
    by_branch = Counter()
    by_branch_source = defaultdict(Counter)
    for row in rows:
        branch = row.get("branch", "unknown")
        src = row.get("source_origin", "unknown")
        by_branch[branch] += 1
        by_branch_source[branch][src] += 1

    snippet_text = args.snippets.read_text(encoding="utf-8") if args.snippets.exists() else ""
    lines = ["# Trajectory Coverage Report", ""]
    lines.append(f"- Total memory records: `{len(rows)}`")
    probe_count = sum(1 for row in rows if row.get("source_origin") == "phase2_probe")
    lines.append(f"- Probe-derived records: `{probe_count}`")
    lines.append("")
    lines.append("| Branch | Memory Count | Source Split | Snippet Present |")
    lines.append("|---|---:|---|---|")
    for branch in HIGH_RISK_BRANCHES:
        split = ", ".join(
            f"{src}:{count}"
            for src, count in sorted(by_branch_source.get(branch, {}).items())
        ) or "none"
        present = "yes" if branch in snippet_text else "no"
        lines.append(f"| `{branch}` | {by_branch.get(branch, 0)} | {split} | {present} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
