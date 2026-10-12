#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Summarize Phase 2 LLM arbitration behavior from diagnostics JSONL."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    arbitration = [
        row for row in rows
        if (row.get("llm_arbitrator") or {}).get("called")
    ]
    overrides = [
        row for row in arbitration
        if (row.get("llm_arbitrator") or {}).get("used_for_final")
    ]
    retries = [
        row for row in arbitration
        if (row.get("llm_arbitrator") or {}).get("retry_used")
    ]
    reasons = Counter(
        reason
        for row in arbitration
        for reason in (row.get("llm_arbitrator") or {}).get("reasons", [])
    )
    validation = Counter(
        (row.get("llm_arbitrator") or {}).get("validation", "")
        for row in arbitration
    )
    errors = Counter(
        (row.get("llm_arbitrator") or {}).get("error", "")
        for row in rows
        if (row.get("llm_arbitrator") or {}).get("error")
    )
    changed_by_family = Counter(row.get("family", "") for row in overrides)
    called_by_family = Counter(row.get("family", "") for row in arbitration)

    return {
        "rows": len(rows),
        "arbitration_called": len(arbitration),
        "arbitration_rate": len(arbitration) / max(len(rows), 1),
        "llm_overrides": len(overrides),
        "llm_override_rate": len(overrides) / max(len(arbitration), 1),
        "retries": len(retries),
        "called_by_family": dict(called_by_family),
        "overrides_by_family": dict(changed_by_family),
        "top_reasons": reasons.most_common(12),
        "validation": dict(validation),
        "errors": dict(errors),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("diagnostics_jsonl", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    report = build_report(_load_jsonl(args.diagnostics_jsonl))
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
