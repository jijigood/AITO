#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Select representative Phase 2 scenario IDs for the first live-server probe.

The goal is calibration, not scoring.  The selector covers each observed
(tag, num_base_stations) group and spreads IDs by option layout so the first
token-backed run is likely to expose response-key/parser/tooling mismatches.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from skills.scenario_parser import parse_scenario


def _group_key(scenario: Dict[str, Any]) -> Tuple[str, int]:
    tag = str(scenario.get("tag", "single-answer"))
    wi = scenario.get("context", {}).get("wireless_network_information", {})
    try:
        bs = int(str(wi.get("num_base_stations", "4")).strip())
    except ValueError:
        bs = 4
    return tag, bs


def _stable_rank(text: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{text}".encode("utf-8")).hexdigest()


def _action_signature(scenario: Dict[str, Any]) -> str:
    ps = parse_scenario(scenario)
    actions = []
    for cid in sorted(ps.parsed_options, key=lambda x: int(x[1:])):
        actions.append(ps.parsed_options[cid].get("action_type", "unknown"))
    return "|".join(actions)


def _action_counts(scenario: Dict[str, Any]) -> Dict[str, int]:
    ps = parse_scenario(scenario)
    return dict(Counter(opt.get("action_type", "unknown") for opt in ps.parsed_options.values()))


def select_probe(
    scenarios: List[Dict[str, Any]],
    per_group: int,
    seed: int,
    total: int = 0,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        grouped[_group_key(scenario)].append(scenario)

    selected: List[Dict[str, Any]] = []
    for key in sorted(grouped, key=lambda x: (x[0], x[1])):
        rows = grouped[key]
        rows = sorted(
            rows,
            key=lambda s: (
                _action_signature(s),
                _stable_rank(str(s.get("scenario_id", "")), seed),
            ),
        )

        seen_signatures = set()
        picked: List[Dict[str, Any]] = []
        for scenario in rows:
            sig = _action_signature(scenario)
            if sig in seen_signatures:
                continue
            picked.append(scenario)
            seen_signatures.add(sig)
            if len(picked) >= per_group:
                break

        if len(picked) < per_group:
            picked_ids = {s.get("scenario_id") for s in picked}
            fill = sorted(rows, key=lambda s: _stable_rank(str(s.get("scenario_id", "")), seed))
            for scenario in fill:
                if scenario.get("scenario_id") in picked_ids:
                    continue
                picked.append(scenario)
                if len(picked) >= per_group:
                    break

        selected.extend(picked)

    if total and len(selected) > total:
        selected = sorted(selected, key=lambda s: _stable_rank(str(s.get("scenario_id", "")), seed))[:total]
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description="Select representative Phase 2 probe IDs")
    ap.add_argument("--test_json", type=Path, default=Path("data/Phase_2/test.json"))
    ap.add_argument("--per_group", type=int, default=3)
    ap.add_argument("--total", type=int, default=0, help="Optional hard cap across all groups")
    ap.add_argument("--seed", type=int, default=20260505)
    ap.add_argument("--out_ids", type=Path, default=Path("results/phase2/probe_ids.txt"))
    ap.add_argument("--out_summary", type=Path, default=Path("results/phase2/probe_summary.json"))
    args = ap.parse_args()

    scenarios = json.loads(args.test_json.read_text(encoding="utf-8"))
    selected = select_probe(scenarios, per_group=args.per_group, seed=args.seed, total=args.total)

    args.out_ids.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)

    ids = [str(s.get("scenario_id", "")).strip() for s in selected]
    args.out_ids.write_text("\n".join(ids) + "\n", encoding="utf-8")

    group_counts = Counter(f"{tag}:{bs}" for tag, bs in map(_group_key, selected))
    summary = {
        "test_json": str(args.test_json),
        "selected": len(selected),
        "out_ids": str(args.out_ids),
        "groups": dict(sorted(group_counts.items())),
        "items": [
            {
                "scenario_id": str(s.get("scenario_id", "")),
                "tag": _group_key(s)[0],
                "num_base_stations": _group_key(s)[1],
                "action_counts": _action_counts(s),
            }
            for s in selected
        ],
    }
    args.out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "selected": len(selected),
        "groups": summary["groups"],
        "out_ids": str(args.out_ids),
        "out_summary": str(args.out_summary),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
