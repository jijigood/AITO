#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Offline F1 distribution repair for Phase 2 Track A submissions.

This script does not call the competition server.  It starts from a known
600-row submission, applies reproducible F1/F1_5BS post-processing variants
from structured KNN votes and diagnostics, and writes new 600-row CSVs plus a
compact report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skills.scenario_parser import parse_option_label


SAFE_F1_ACTIONS = {
    "add_neighbor",
    "check_server",
    "decrease_a3",
    "insufficient_data",
    "lift_tilt",
    "pdcch_2sym",
}

DEFAULT_BASE = Path("submission_600_diag_lora_boundary_resubmit.csv")
DEFAULT_TEST = Path("data/Phase_2/test.json")
DEFAULT_DIAG = Path("results/phase2/sub2_lora_boundary/result_diag_lora_boundary_only.diagnostics.jsonl")
DEFAULT_VOTES = Path("results/phase2/knn_patch/knn_f1_votes.csv")
DEFAULT_OUT = Path("results/phase2/f1_distribution_repair")


def _split_answer(answer: str) -> List[str]:
    return [part.strip() for part in str(answer or "").split("|") if part.strip()]


def _load_submission(path: Path) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [{k: (v or "") for k, v in row.items()} for row in csv.DictReader(f)]
    return rows, {row["ID"]: row for row in rows}


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


def _load_action_map(test_json: Path) -> Dict[str, Dict[str, str]]:
    scenarios = json.loads(test_json.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, str]] = {}
    for scenario in scenarios:
        sid = str(scenario.get("scenario_id", "")).strip()
        options = (scenario.get("task") or {}).get("options") or []
        out[sid] = {
            str(opt.get("id", "")).strip(): parse_option_label(str(opt.get("label", ""))).get("action_type", "unknown")
            for opt in options
        }
    return out


def _answer_action(action_map: Dict[str, Dict[str, str]], sid: str, answer: str) -> str:
    codes = _split_answer(answer)
    if not codes:
        return ""
    if len(codes) > 1:
        return "multi"
    return action_map.get(sid, {}).get(codes[0], "?")


def _load_diagnostics(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("scenario_id") or row.get("id") or "").strip()
            if sid:
                out[sid] = row
    return out


def _is_f1_single(diag: Optional[Dict[str, Any]]) -> bool:
    if not diag:
        return False
    return diag.get("family") in {"F1", "F1_5BS"} and not bool(diag.get("is_multiple"))


def _load_votes(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = dict(row)
            try:
                item["share"] = float(item.get("share") or 0.0)
            except ValueError:
                item["share"] = 0.0
            rows.append(item)
    return rows


def _best_candidate_for_action(diag: Dict[str, Any], action: str) -> Optional[str]:
    candidates = [
        cand
        for cand in (diag.get("candidates") or [])
        if cand.get("action_type") == action and cand.get("answer")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda c: float(c.get("score") or 0.0), reverse=True)
    return str(candidates[0]["answer"])


def _best_safe_non_adjust_candidate(diag: Dict[str, Any]) -> Optional[str]:
    candidates = []
    for cand in diag.get("candidates") or []:
        action = cand.get("action_type")
        if action in SAFE_F1_ACTIONS and action != "adjust_azimuth" and cand.get("answer"):
            candidates.append(cand)
    if not candidates:
        return None
    candidates.sort(key=lambda c: float(c.get("score") or 0.0), reverse=True)
    return str(candidates[0]["answer"])


def _apply_knn_votes(
    base_answers: Dict[str, str],
    diagnostics: Dict[str, Dict[str, Any]],
    votes: Iterable[Dict[str, Any]],
    min_share: float,
    force_actions: Optional[set] = None,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    answers = dict(base_answers)
    changes: List[Dict[str, Any]] = []
    for vote in votes:
        sid = str(vote.get("ID") or "").strip()
        if not _is_f1_single(diagnostics.get(sid)):
            continue
        vote_action = str(vote.get("vote") or "").strip()
        if force_actions is None and float(vote.get("share") or 0.0) < min_share:
            continue
        if force_actions is not None and vote_action not in force_actions and float(vote.get("share") or 0.0) < min_share:
            continue
        old = str(vote.get("old") or "").strip()
        new = str(vote.get("new") or "").strip()
        if not new:
            continue
        current = answers.get(sid, "")
        if old and current != old:
            continue
        if current == new:
            continue
        answers[sid] = new
        changes.append(
            {
                "id": sid,
                "old": current,
                "new": new,
                "rule": f"knn_vote_share>={min_share:.2f}",
                "vote": vote_action,
                "share": float(vote.get("share") or 0.0),
            }
        )
    return answers, changes


def _apply_hardblock_adjust(
    base_answers: Dict[str, str],
    action_map: Dict[str, Dict[str, str]],
    diagnostics: Dict[str, Dict[str, Any]],
    votes: Iterable[Dict[str, Any]],
    min_share: float,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    answers, changes = _apply_knn_votes(
        base_answers,
        diagnostics,
        votes,
        min_share=min_share,
        force_actions=SAFE_F1_ACTIONS,
    )
    changed_ids = {row["id"] for row in changes}
    for sid, answer in list(answers.items()):
        diag = diagnostics.get(sid)
        if not _is_f1_single(diag):
            continue
        if _answer_action(action_map, sid, answer) != "adjust_azimuth":
            continue
        new = _best_safe_non_adjust_candidate(diag)
        if not new or new == answer:
            continue
        answers[sid] = new
        changes.append(
            {
                "id": sid,
                "old": answer,
                "new": new,
                "rule": "hardblock_adjust_best_safe_candidate",
                "vote": "",
                "share": 0.0,
                "already_knn": sid in changed_ids,
            }
        )
    return answers, changes


def _distribution(action_map: Dict[str, Dict[str, str]], answers: Dict[str, str], diagnostics: Dict[str, Dict[str, Any]]) -> Counter:
    counter: Counter = Counter()
    for sid, answer in answers.items():
        if not answer:
            counter["blank"] += 1
            continue
        diag = diagnostics.get(sid)
        if _is_f1_single(diag):
            counter[f"F1_like:{_answer_action(action_map, sid, answer)}"] += 1
        else:
            counter[_answer_action(action_map, sid, answer)] += 1
    return counter


def _f1_distribution(action_map: Dict[str, Dict[str, str]], answers: Dict[str, str], diagnostics: Dict[str, Dict[str, Any]]) -> Counter:
    counter: Counter = Counter()
    for sid, diag in diagnostics.items():
        if not _is_f1_single(diag):
            continue
        counter[_answer_action(action_map, sid, answers.get(sid, ""))] += 1
    return counter


def _transitions(
    action_map: Dict[str, Dict[str, str]],
    base_answers: Dict[str, str],
    new_answers: Dict[str, str],
) -> Counter:
    counter: Counter = Counter()
    for sid, old in base_answers.items():
        new = new_answers.get(sid, "")
        if old == new:
            continue
        counter[(
            _answer_action(action_map, sid, old),
            _answer_action(action_map, sid, new),
        )] += 1
    return counter


def _phase1_scaled_targets() -> Dict[str, float]:
    phase1_counts = {
        "check_server": 239,
        "decrease_a3": 231,
        "lift_tilt": 254,
        "insufficient_data": 252,
        "pdcch_2sym": 240,
        "add_neighbor": 240,
    }
    total = sum(phase1_counts.values())
    return {k: v * 269.0 / total for k, v in phase1_counts.items()}


def _check_server_probe(diagnostics: Dict[str, Dict[str, Any]], answers: Dict[str, str], action_map: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    has_check = 0
    high_conf = 0
    by_current: Counter = Counter()
    for sid, diag in diagnostics.items():
        if not _is_f1_single(diag):
            continue
        check = _best_candidate_for_action(diag, "check_server")
        if not check:
            continue
        has_check += 1
        current_action = _answer_action(action_map, sid, answers.get(sid, ""))
        by_current[current_action] += 1
        fs = diag.get("feature_scores") or {}
        ev = diag.get("raw_evidence_summary") or {}
        traffic = ev.get("traffic") or {}
        serving = ev.get("serving") or {}
        prb = traffic.get("max_dl_prb_util")
        sinr = serving.get("sinr_db")
        try:
            phase1_like_server = (
                float(prb) < 30
                and float(sinr) > 8
                and float(fs.get("backhaul_server") or 0.0) > float(fs.get("pdcch_issue") or 0.0)
            )
        except (TypeError, ValueError):
            phase1_like_server = False
        if phase1_like_server:
            high_conf += 1
    return {
        "check_server_candidate_present": has_check,
        "phase1_like_high_conf_check_server": high_conf,
        "candidate_by_current_action": dict(by_current),
    }


def _write_report(
    path: Path,
    base_name: str,
    variants: List[Dict[str, Any]],
    check_probe: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# F1 Distribution Repair Report")
    lines.append("")
    lines.append(f"Base: `{base_name}`")
    lines.append("")
    lines.append("## Phase1-scaled F1 Targets")
    for action, target in sorted(_phase1_scaled_targets().items()):
        lines.append(f"- {action}: {target:.1f}")
    lines.append("")
    lines.append("## Check Server Probe")
    lines.append(f"- check_server candidate present: {check_probe['check_server_candidate_present']}")
    lines.append(f"- phase1-like high-confidence check_server cases: {check_probe['phase1_like_high_conf_check_server']}")
    lines.append(f"- candidate by current action: {check_probe['candidate_by_current_action']}")
    lines.append("")
    lines.append("## Variants")
    for variant in variants:
        lines.append(f"### {variant['name']}")
        lines.append(f"- file: `{variant['path']}`")
        lines.append(f"- changed rows vs base: {variant['changed_rows']}")
        lines.append(f"- F1/F1_5BS distribution: {variant['f1_distribution']}")
        lines.append(f"- top transitions: {variant['top_transitions']}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build offline F1 distribution repair submission variants")
    ap.add_argument("--base_csv", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--test_json", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--diagnostics_jsonl", type=Path, default=DEFAULT_DIAG)
    ap.add_argument("--votes_csv", type=Path, default=DEFAULT_VOTES)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    base_rows, base_by_id = _load_submission(args.base_csv)
    base_answers = {sid: row.get("Track A", "") for sid, row in base_by_id.items()}
    action_map = _load_action_map(args.test_json)
    diagnostics = _load_diagnostics(args.diagnostics_jsonl)
    votes = _load_votes(args.votes_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    variant_specs: List[Tuple[str, Dict[str, str], List[Dict[str, Any]]]] = []
    for threshold in (0.0, 0.40, 0.50, 0.60, 0.70):
        answers, changes = _apply_knn_votes(base_answers, diagnostics, votes, min_share=threshold)
        label = "all" if threshold == 0.0 else f"conf{int(threshold * 100):02d}"
        variant_specs.append((f"knn_{label}", answers, changes))

    answers, changes = _apply_hardblock_adjust(
        base_answers,
        action_map,
        diagnostics,
        votes,
        min_share=0.50,
    )
    variant_specs.append(("hardblock_adjust_knn50_else_candidate", answers, changes))

    answers, changes = _apply_knn_votes(base_answers, diagnostics, votes, min_share=0.50, force_actions={"check_server"})
    variant_specs.append(("knn_conf50_plus_all_check_votes", answers, changes))

    variants: List[Dict[str, Any]] = []
    for name, answers, changes in variant_specs:
        out_csv = args.out_dir / f"submission_600_f1repair_{name}.csv"
        _write_submission(out_csv, base_rows, answers)
        changed_rows = sum(1 for sid, old in base_answers.items() if answers.get(sid, "") != old)
        trans = _transitions(action_map, base_answers, answers)
        variant = {
            "name": name,
            "path": str(out_csv),
            "changed_rows": changed_rows,
            "f1_distribution": dict(_f1_distribution(action_map, answers, diagnostics)),
            "full_distribution": dict(_distribution(action_map, answers, diagnostics)),
            "top_transitions": [f"{old}->{new}:{count}" for (old, new), count in trans.most_common(12)],
            "rules_applied": len(changes),
        }
        variants.append(variant)

    summary_json = args.out_dir / "repair_summary.json"
    summary = {
        "base_csv": str(args.base_csv),
        "diagnostics_jsonl": str(args.diagnostics_jsonl),
        "votes_csv": str(args.votes_csv),
        "base_f1_distribution": dict(_f1_distribution(action_map, base_answers, diagnostics)),
        "phase1_scaled_targets": _phase1_scaled_targets(),
        "check_server_probe": _check_server_probe(diagnostics, base_answers, action_map),
        "variants": variants,
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(args.out_dir / "repair_report.md", str(args.base_csv), variants, summary["check_server_probe"])

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
