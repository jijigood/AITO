#!/usr/bin/env python
"""Build V6.1 Phase-2 submission variants from audited F1 fixes."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase1_validation_harness import family_name, option_actions


BASE = ROOT / "submission_600_diag_lora_boundary_resubmit.csv"
V5_11 = ROOT / "submissions/submission_600_v5_11_baseline_plus_f2_pair_fix.csv"
V5_21 = ROOT / "submissions/submission_600_v5_21_safe_f1_5bs_decrease_a3.csv"
QWEN_RAW54 = ROOT / "results/phase2/f1_v5_20_audit/qwen_raw54_plus_6_summary.csv"
KNN_VOTES = ROOT / "results/phase2/knn_patch/knn_f1_votes.csv"
PHASE2_JSON = ROOT / "data/Phase_2/test.json"
OUT_DIR = ROOT / "submissions"
REPORT = ROOT / "reports/v6_1_build_report.md"

LORA_OVERRIDES = {
    "6060dca7": "C22",
    "1784b64c": "C20",
}

F1_ALLOWED_ACTIONS = {
    "add_neighbor",
    "check_server",
    "decrease_a3",
    "insufficient_data",
    "lift_tilt",
    "pdcch_2sym",
}

F2_LEGAL_ACTION_SETS = {
    frozenset({"adjust_azimuth", "increase_power"}),
    frozenset({"decrease_power", "press_down_tilt"}),
}


def read_submission(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_submission(path: Path, headers: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def by_id(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {r["ID"]: r for r in rows}


def load_phase2() -> Dict[str, Dict[str, Any]]:
    with PHASE2_JSON.open(encoding="utf-8") as f:
        scenarios = json.load(f)
    return {str(s["scenario_id"]): s for s in scenarios}


def action_of(scenarios: Dict[str, Dict[str, Any]], sid: str, answer: str) -> str:
    actions = option_actions(scenarios[sid], answer)
    return actions[0] if len(actions) == 1 else "|".join(actions)


def load_qwen_raw54() -> Dict[str, str]:
    raw: Dict[str, str] = {}
    with QWEN_RAW54.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            raw[r["ID"]] = r["qwen_raw"].strip()
    if len(raw) != 54:
        raise RuntimeError(f"Expected 54 qwen raw rows, got {len(raw)}")
    return raw


def load_knn_fallbacks() -> Dict[str, str]:
    if not KNN_VOTES.exists():
        return {}
    fallbacks: Dict[str, str] = {}
    with KNN_VOTES.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vote_answer = (r.get("vote_answer") or r.get("new") or "").strip()
            if vote_answer:
                fallbacks[r["ID"]] = vote_answer
    return fallbacks


def diff_answers(a: Dict[str, Dict[str, str]], b: Dict[str, Dict[str, str]]) -> List[str]:
    return [
        sid for sid, row in a.items()
        if row.get("Track A", "").strip() != b.get(sid, {}).get("Track A", "").strip()
    ]


def f2_fix_map(base: Dict[str, Dict[str, str]], v5_11: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    return {sid: v5_11[sid]["Track A"].strip() for sid in diff_answers(base, v5_11)}


def f1_5bs_fix_map(
    base: Dict[str, Dict[str, str]],
    v5_21: Dict[str, Dict[str, str]],
    scenarios: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    fixes: Dict[str, str] = {}
    for sid in diff_answers(base, v5_21):
        if sid not in scenarios:
            continue
        if family_name(scenarios[sid]) == "F1_5BS":
            fixes[sid] = v5_21[sid]["Track A"].strip()
    return fixes


def apply_map(
    base_rows: List[Dict[str, str]],
    answer_map: Dict[str, str],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in base_rows:
        new_row = dict(row)
        sid = new_row["ID"]
        if sid in answer_map:
            new_row["Track A"] = answer_map[sid]
        rows.append(new_row)

    for prefix, answer in LORA_OVERRIDES.items():
        for row in rows:
            if row["ID"].startswith(prefix):
                row["Track A"] = answer
                break
    return rows


def strict_raw54_map(
    raw54: Dict[str, str],
    fallbacks: Dict[str, str],
    scenarios: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, str], List[str]]:
    strict: Dict[str, str] = {}
    notes: List[str] = []
    for sid, answer in raw54.items():
        action = action_of(scenarios, sid, answer)
        if action in F1_ALLOWED_ACTIONS:
            strict[sid] = answer
            continue

        fallback = fallbacks.get(sid, "")
        fallback_action = action_of(scenarios, sid, fallback) if fallback else ""
        if fallback_action in F1_ALLOWED_ACTIONS:
            strict[sid] = fallback
            notes.append(
                f"{sid}: raw {answer}/{action} -> fallback {fallback}/{fallback_action}"
            )
            continue

        raise RuntimeError(
            f"No strict F1 fallback for {sid}: raw {answer}/{action}, "
            f"fallback {fallback}/{fallback_action}"
        )
    return strict, notes


def audit_submission(
    rows: List[Dict[str, str]],
    scenarios: Dict[str, Dict[str, Any]],
    base: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    row_map = by_id(rows)
    track_a = [r for r in rows if r.get("Track A", "").strip()]
    track_b = [r for r in rows if r.get("Track B", "").strip()]
    diffs = diff_answers(base, row_map)

    family_counts = Counter()
    action_counts = Counter()
    illegal_f1: List[Tuple[str, str, str, str]] = []
    f2_bad: List[Tuple[str, str, List[str]]] = []

    for row in track_a:
        sid = row["ID"]
        if sid not in scenarios:
            continue
        answer = row["Track A"].strip()
        family = family_name(scenarios[sid])
        family_counts[family] += 1
        actions = option_actions(scenarios[sid], answer)
        for action in actions:
            action_counts[(family, action)] += 1

        if family in {"F1", "F1_5BS"}:
            action = actions[0] if len(actions) == 1 else "|".join(actions)
            if action not in F1_ALLOWED_ACTIONS:
                illegal_f1.append((sid, answer, action, family))

        if family == "F2":
            if frozenset(actions) not in F2_LEGAL_ACTION_SETS:
                f2_bad.append((sid, answer, actions))

    lora_status = {}
    for prefix, expected in LORA_OVERRIDES.items():
        match = [r for r in rows if r["ID"].startswith(prefix)]
        actual = match[0]["Track A"].strip() if match else ""
        lora_status[prefix] = {
            "actual": actual,
            "expected": expected,
            "ok": actual == expected,
        }

    return {
        "rows": len(rows),
        "track_a": len(track_a),
        "track_b": len(track_b),
        "diff_vs_base": len(diffs),
        "family_counts": dict(sorted(family_counts.items())),
        "action_counts": {f"{k[0]}:{k[1]}": v for k, v in sorted(action_counts.items())},
        "illegal_f1": illegal_f1,
        "f2_bad": f2_bad,
        "lora_status": lora_status,
        "diff_ids": sorted(diffs),
    }


def md_table(rows: List[List[Any]]) -> List[str]:
    if not rows:
        return []
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for idx, row in enumerate(rows):
        lines.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))) + " |")
        if idx == 0:
            lines.append("| " + " | ".join("-" * widths[i] for i in range(len(row))) + " |")
    return lines


def write_report(
    audits: Dict[str, Dict[str, Any]],
    strict_notes: List[str],
    maps: Dict[str, Dict[str, str]],
) -> None:
    lines: List[str] = []
    lines.append("# V6.1 Build Report")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Base: `{BASE.name}`")
    lines.append(f"- Raw-54 source: `{QWEN_RAW54.relative_to(ROOT)}`")
    lines.append(f"- F1_5BS six-fix source: `{V5_21.relative_to(ROOT)}`")
    lines.append(f"- F2 structural-fix source: `{V5_11.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## Fix Maps")
    lines.append("")
    lines.append(f"- Raw-54 F1 fixes: `{len(maps['raw54'])}`")
    lines.append(f"- Strict Raw-54 F1 fixes: `{len(maps['strict54'])}`")
    lines.append(f"- F1_5BS validated fixes: `{len(maps['f1_5bs'])}`")
    lines.append(f"- F2 structural fixes: `{len(maps['f2'])}`")
    if strict_notes:
        lines.append("")
        lines.append("## Strict F1 Fallbacks")
        lines.append("")
        for note in strict_notes:
            lines.append(f"- {note}")

    lines.append("")
    lines.append("## Output Audits")
    rows = [["Variant", "Rows", "Track A", "Track B", "Diff vs Base", "F1 Illegal", "F2 Bad", "LoRA OK"]]
    for name, audit in audits.items():
        lora_ok = all(v["ok"] for v in audit["lora_status"].values())
        rows.append([
            name,
            audit["rows"],
            audit["track_a"],
            audit["track_b"],
            audit["diff_vs_base"],
            len(audit["illegal_f1"]),
            len(audit["f2_bad"]),
            lora_ok,
        ])
    lines.extend(md_table(rows))

    for name, audit in audits.items():
        lines.append("")
        lines.append(f"## {name}")
        lines.append("")
        lines.append("Family counts:")
        for family, count in audit["family_counts"].items():
            lines.append(f"- `{family}`: {count}")
        lines.append("")
        lines.append("F1/F1_5BS action distribution:")
        for key, count in audit["action_counts"].items():
            if key.startswith("F1:") or key.startswith("F1_5BS:"):
                lines.append(f"- `{key}`: {count}")
        if audit["illegal_f1"]:
            lines.append("")
            lines.append("F1 illegal rows:")
            for sid, answer, action, family in audit["illegal_f1"]:
                lines.append(f"- `{sid}` `{family}` `{answer}` `{action}`")
        if audit["f2_bad"]:
            lines.append("")
            lines.append("F2 bad rows:")
            for sid, answer, actions in audit["f2_bad"]:
                lines.append(f"- `{sid}` `{answer}` `{actions}`")
        lines.append("")
        lines.append("LoRA:")
        for prefix, status in audit["lora_status"].items():
            lines.append(
                f"- `{prefix}`: {status['actual']} expected {status['expected']} "
                f"({'OK' if status['ok'] else 'FAIL'})"
            )

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    headers, base_rows = read_submission(BASE)
    _, v5_11_rows = read_submission(V5_11)
    _, v5_21_rows = read_submission(V5_21)

    base_map = by_id(base_rows)
    v5_11_map = by_id(v5_11_rows)
    v5_21_map = by_id(v5_21_rows)
    scenarios = load_phase2()

    raw54 = load_qwen_raw54()
    fallbacks = load_knn_fallbacks()
    strict54, strict_notes = strict_raw54_map(raw54, fallbacks, scenarios)
    f1_5bs = f1_5bs_fix_map(base_map, v5_21_map, scenarios)
    f2 = f2_fix_map(base_map, v5_11_map)

    variants = {
        "v6_1_raw54_base": raw54,
        "v6_1_strict60_base": {**strict54, **f1_5bs},
        "v6_1_strict60_plus_f2fix": {**strict54, **f1_5bs, **f2},
    }

    outputs = {
        "v6_1_raw54_base": OUT_DIR / "submission_600_v6_1_raw54_base.csv",
        "v6_1_strict60_base": OUT_DIR / "submission_600_v6_1_strict60_base.csv",
        "v6_1_strict60_plus_f2fix": OUT_DIR / "submission_600_v6_1_strict60_plus_f2fix.csv",
    }

    audits: Dict[str, Dict[str, Any]] = {}
    for name, fixes in variants.items():
        rows = apply_map(base_rows, fixes)
        write_submission(outputs[name], headers, rows)
        audits[name] = audit_submission(rows, scenarios, base_map)
        print(
            f"{name}: wrote {outputs[name]} "
            f"diff={audits[name]['diff_vs_base']} "
            f"f1_illegal={len(audits[name]['illegal_f1'])} "
            f"f2_bad={len(audits[name]['f2_bad'])}"
        )

    maps = {
        "raw54": raw54,
        "strict54": strict54,
        "f1_5bs": f1_5bs,
        "f2": f2,
    }
    write_report(audits, strict_notes, maps)
    print(f"report: {REPORT}")


if __name__ == "__main__":
    main()
