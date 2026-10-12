#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
F2 multi-answer pair repair for Phase 2 Track A submissions.

Phase 1 F2 answers are always cardinality 2 with one of two legal action pairs:
  - {adjust_azimuth, increase_power}   ("azimuth pair", 62% of Phase 1 F2)
  - {decrease_power, press_down_tilt}  ("tilt pair", 38% of Phase 1 F2)

Phase 2 diagnostics show 10 F2 rows that violate this:
  - 8 rows with cardinality 4 containing F1-style actions
    {add_neighbor, decrease_a3, decrease_power, press_down_tilt}
  - 2 rows with cardinality 2 but actions {add_neighbor, decrease_a3} (no F2-legal codes)

For each violator, replace with the highest-scoring 2-code candidate whose
action set is one of the two legal F2 pairs.  No API, no LLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skills.scenario_parser import parse_option_label


F2_PAIR_AZIMUTH = frozenset({"adjust_azimuth", "increase_power"})
F2_PAIR_TILT = frozenset({"decrease_power", "press_down_tilt"})
F2_LEGAL_PAIRS = (F2_PAIR_AZIMUTH, F2_PAIR_TILT)


def _load_action_map(test_json: Path) -> Dict[str, Dict[str, str]]:
    scenarios = json.loads(test_json.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, str]] = {}
    for s in scenarios:
        sid = str(s.get("scenario_id", "")).strip()
        opts = (s.get("task") or {}).get("options") or []
        out[sid] = {
            str(o.get("id", "")).strip(): parse_option_label(str(o.get("label", ""))).get("action_type", "?")
            for o in opts
        }
    return out


def _row_actions(action_map: Dict[str, Dict[str, str]], sid: str, codes: List[str]) -> frozenset:
    am = action_map.get(sid, {})
    return frozenset(am.get(c, "?") for c in codes)


def _is_f2_violation(action_map: Dict[str, Dict[str, str]], sid: str, answer: str) -> bool:
    codes = [c.strip() for c in (answer or "").split("|") if c.strip()]
    if len(codes) != 2:
        return True
    return _row_actions(action_map, sid, codes) not in F2_LEGAL_PAIRS


def _find_best_legal_pair(
    diag: Dict[str, Any],
    action_map: Dict[str, Dict[str, str]],
    sid: str,
) -> Optional[Tuple[str, frozenset, float]]:
    """Return (answer, action_set, score) of the highest-scoring legal F2 pair candidate."""
    best: Optional[Tuple[str, frozenset, float]] = None
    for cand in diag.get("candidates") or []:
        code = str(cand.get("answer") or "")
        score = float(cand.get("score") or 0.0)
        parts = [p.strip() for p in code.split("|") if p.strip()]
        if len(parts) != 2:
            continue
        actions = _row_actions(action_map, sid, parts)
        if actions not in F2_LEGAL_PAIRS:
            continue
        if best is None or score > best[2]:
            best = (code, actions, score)
    return best


def _load_diagnostics(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("scenario_id") or "").strip()
            if sid:
                out[sid] = row
    return out


def _load_submission(path: Path) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [{k: (v or "") for k, v in row.items()} for row in csv.DictReader(f)]
    answers = {row["ID"]: row.get("Track A", "") for row in rows}
    return rows, answers


def _write_submission(path: Path, rows: List[Dict[str, str]], answers: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ID", "Track A", "Track B"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fieldnames}
            out["Track A"] = answers.get(row["ID"], row.get("Track A", ""))
            writer.writerow(out)


def repair(
    base_csv: Path,
    test_json: Path,
    diagnostics_jsonl: Path,
    out_csv: Path,
) -> Dict[str, Any]:
    rows, base_answers = _load_submission(base_csv)
    action_map = _load_action_map(test_json)
    diagnostics = _load_diagnostics(diagnostics_jsonl)

    new_answers = dict(base_answers)
    changes: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for sid, diag in diagnostics.items():
        if diag.get("family") != "F2":
            continue
        current = base_answers.get(sid, "")
        if not _is_f2_violation(action_map, sid, current):
            continue
        best = _find_best_legal_pair(diag, action_map, sid)
        if best is None:
            skipped.append({"sid": sid, "current": current, "reason": "no legal F2 pair in candidates"})
            continue
        new_code, new_actions, new_score = best
        old_codes = [c.strip() for c in (current or "").split("|") if c.strip()]
        new_answers[sid] = new_code
        changes.append({
            "sid": sid,
            "old": current,
            "old_actions": list(_row_actions(action_map, sid, old_codes)),
            "old_cardinality": len(old_codes),
            "new": new_code,
            "new_actions": list(new_actions),
            "new_score": round(new_score, 4),
            "pair_kind": "azimuth" if new_actions == F2_PAIR_AZIMUTH else "tilt",
            "top_issue": (diag.get("top_issue") or {}).get("name"),
        })

    _write_submission(out_csv, rows, new_answers)

    pair_split = Counter(c["pair_kind"] for c in changes)
    return {
        "base_csv": str(base_csv),
        "out_csv": str(out_csv),
        "changed_rows": len(changes),
        "skipped_rows": len(skipped),
        "pair_split_in_fix": dict(pair_split),
        "changes": changes,
        "skipped": skipped,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair F2 cardinality / action-type violations")
    ap.add_argument("--base_csv", type=Path,
                    default=ROOT / "submissions" / "submission_600_v5_07_adj_only.csv")
    ap.add_argument("--test_json", type=Path,
                    default=ROOT / "data" / "Phase_2" / "test.json")
    ap.add_argument("--diagnostics_jsonl", type=Path,
                    default=ROOT / "results" / "phase2" / "sub2_lora_boundary"
                            / "result_diag_lora_boundary_only.diagnostics.jsonl")
    ap.add_argument("--out_csv", type=Path,
                    default=ROOT / "submissions"
                            / "submission_600_v5_10_adj_only_plus_f2_pair_fix.csv")
    ap.add_argument("--report_json", type=Path,
                    default=ROOT / "reports" / "C_f2_pair_repair.json")
    args = ap.parse_args()

    summary = repair(args.base_csv, args.test_json, args.diagnostics_jsonl, args.out_csv)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")

    print(json.dumps({
        "base_csv": summary["base_csv"],
        "out_csv": summary["out_csv"],
        "changed_rows": summary["changed_rows"],
        "skipped_rows": summary["skipped_rows"],
        "pair_split_in_fix": summary["pair_split_in_fix"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
