#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
phase2_audit.py -- pre-submission checks for Track A Phase 2.

This script does not call the competition server and does not submit anything.
It validates a generated results.csv against Phase 2 test.json, and optionally
summarizes phase2_run.py diagnostics JSONL.
"""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


VALID_CODE_RE = re.compile(r"^C(?:[1-9]|1[0-9]|2[0-4])(?:\|C(?:[1-9]|1[0-9]|2[0-4]))*$")


def _split_answer(answer: str) -> List[str]:
    return [part.strip() for part in str(answer or "").split("|") if part.strip()]


def _code_num(code: str) -> int:
    return int(code[1:])


def _expected_answer_range(scenario: Dict[str, Any]) -> Tuple[int, int]:
    tag = str(scenario.get("tag", "single-answer")).lower()
    wi = scenario.get("context", {}).get("wireless_network_information", {})
    try:
        num_bs = int(str(wi.get("num_base_stations", "4")).strip())
    except ValueError:
        num_bs = 4

    if "multiple" not in tag:
        return (1, 1)
    if num_bs == 5:
        return (4, 4)
    return (2, 4)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[idx]


def _load_test(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_id = {str(row.get("scenario_id", "")).strip(): row for row in rows}
    return rows, by_id


def _load_result(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: (v or "") for k, v in row.items()} for row in csv.DictReader(f)]


def _load_diagnostics(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path or not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append({"scenario_id": f"__bad_json_line_{line_no}", "error": str(exc)})
    return rows


def audit(
    test_json: Path,
    result_csv: Path,
    diagnostics_jsonl: Path = None,
    allow_partial: bool = False,
    max_api_error_rate: float = 0.02,
    min_answer_rate: float = 1.0,
    max_low_confidence_rate: float = 0.15,
    max_missing_critical_data_rate: float = 0.02,
) -> Dict[str, Any]:
    scenarios, scenario_by_id = _load_test(test_json)
    expected_ids = [str(row.get("scenario_id", "")).strip() for row in scenarios]
    expected_set = set(expected_ids)

    result_rows = _load_result(result_csv)
    diagnostics = _load_diagnostics(diagnostics_jsonl) if diagnostics_jsonl else []

    hard_errors: List[str] = []
    warnings: List[str] = []

    required_cols = ["ID", "Track A", "Track B"]
    if result_rows:
        missing_cols = [col for col in required_cols if col not in result_rows[0]]
        if missing_cols:
            hard_errors.append(f"missing required columns: {missing_cols}")
    else:
        hard_errors.append("result CSV has no data rows")

    ids = [str(row.get("ID", "")).strip() for row in result_rows]
    id_counts = Counter(ids)
    duplicate_ids = sorted([sid for sid, count in id_counts.items() if sid and count > 1])
    if duplicate_ids:
        hard_errors.append(f"duplicate IDs: {duplicate_ids[:10]}")

    result_set = {sid for sid in ids if sid}
    missing_ids = sorted(expected_set - result_set)
    extra_ids = sorted(result_set - expected_set)
    if missing_ids and not allow_partial:
        hard_errors.append(f"missing {len(missing_ids)} expected IDs, first={missing_ids[:5]}")
    if extra_ids:
        hard_errors.append(f"extra {len(extra_ids)} unknown IDs, first={extra_ids[:5]}")

    invalid_answers = []
    cardinality_errors = []
    empty_answers = []
    code_counter: Counter[str] = Counter()
    answer_len_counter: Counter[int] = Counter()
    tag_counter: Counter[str] = Counter()
    tag_answer_len_counter: Counter[str] = Counter()

    for idx, row in enumerate(result_rows, 2):
        sid = str(row.get("ID", "")).strip()
        answer = str(row.get("Track A", "")).strip()
        track_b = str(row.get("Track B", "")).strip()

        if not sid:
            hard_errors.append(f"blank ID at CSV line {idx}")
            continue
        if sid not in scenario_by_id:
            continue

        scenario = scenario_by_id[sid]
        tag = str(scenario.get("tag", "single-answer"))
        tag_counter[tag] += 1

        if not answer:
            empty_answers.append((idx, sid))
            continue
        if track_b:
            warnings.append(f"Track B is non-empty for Track A row {sid}: {track_b[:80]!r}")

        if not VALID_CODE_RE.fullmatch(answer):
            invalid_answers.append((idx, sid, answer))
            continue

        codes = _split_answer(answer)
        if len(codes) != len(set(codes)):
            cardinality_errors.append((idx, sid, answer, "duplicate codes in answer"))
        if codes != sorted(codes, key=_code_num):
            warnings.append(f"codes not sorted at line {idx}: {sid} -> {answer}")

        min_n, max_n = _expected_answer_range(scenario)
        if not (min_n <= len(codes) <= max_n):
            cardinality_errors.append((idx, sid, answer, f"expected {min_n}-{max_n} codes"))

        for code in codes:
            code_counter[code] += 1
        answer_len_counter[len(codes)] += 1
        tag_answer_len_counter[f"{tag}:{len(codes)}"] += 1

    if empty_answers:
        hard_errors.append(f"empty Track A answers: {len(empty_answers)}, first={empty_answers[:5]}")
    if invalid_answers:
        hard_errors.append(f"invalid Track A answers: {len(invalid_answers)}, first={invalid_answers[:5]}")
    if cardinality_errors:
        hard_errors.append(f"answer cardinality errors: {len(cardinality_errors)}, first={cardinality_errors[:5]}")

    answered = sum(1 for row in result_rows if str(row.get("Track A", "")).strip())
    answer_rate = answered / max(len(result_rows), 1)
    coverage_rate = len(result_set & expected_set) / max(len(expected_ids), 1)
    if answer_rate < min_answer_rate:
        hard_errors.append(f"answer rate {answer_rate:.2%} below threshold {min_answer_rate:.2%}")

    diag_by_id = {str(row.get("scenario_id", "")).strip(): row for row in diagnostics}
    api_error_counter: Counter[str] = Counter()
    api_call_counter: Counter[str] = Counter()
    diag_error_counter: Counter[str] = Counter()
    elapsed_values: List[float] = []
    family_counter: Counter[str] = Counter()
    family_override_counter: Counter[str] = Counter()
    missing_data_counter: Counter[str] = Counter()
    candidate_empty = 0
    low_confidence = 0
    top_candidate_scores: List[float] = []
    missing_critical_data = 0

    for row in diagnostics:
        if row.get("elapsed_seconds") is not None:
            try:
                elapsed_values.append(float(row["elapsed_seconds"]))
            except (TypeError, ValueError):
                pass
        if row.get("family"):
            family_counter[str(row["family"])] += 1
        if row.get("family_override"):
            family_override_counter[str(row["family_override"])] += 1
        if row.get("error"):
            diag_error_counter[str(row["error"]).splitlines()[0][:120]] += 1
        if row.get("candidates") == []:
            candidate_empty += 1
        candidates = row.get("candidates") or []
        if candidates:
            try:
                score = float(candidates[0].get("score", 0.0))
                top_candidate_scores.append(score)
                if score <= 0.06:
                    low_confidence += 1
            except (TypeError, ValueError, AttributeError):
                pass
        for name, available in (row.get("data_available") or {}).items():
            if available is False:
                missing_data_counter[str(name)] += 1
        shapes = row.get("data_shapes") or {}
        critical_missing = False
        for name in ("user_plane", "net_config"):
            info = shapes.get(name) or {}
            try:
                if int(info.get("rows", 0)) <= 0 or int(info.get("cols", 0)) <= 1:
                    critical_missing = True
            except (TypeError, ValueError):
                critical_missing = True
        if critical_missing:
            missing_critical_data += 1
        for call in row.get("api_calls", []) or []:
            tool = str(call.get("tool", "unknown"))
            api_call_counter[tool] += 1
            if not call.get("ok", False):
                api_error_counter[tool] += 1

    api_calls_total = sum(api_call_counter.values())
    api_errors_total = sum(api_error_counter.values())
    api_error_rate = api_errors_total / max(api_calls_total, 1)
    if diagnostics and api_error_rate > max_api_error_rate:
        hard_errors.append(
            f"API error rate {api_error_rate:.2%} exceeds threshold {max_api_error_rate:.2%}"
        )
    low_confidence_rate = low_confidence / max(len(top_candidate_scores), 1)
    if diagnostics and top_candidate_scores and low_confidence_rate > max_low_confidence_rate:
        hard_errors.append(
            f"low-confidence top candidate rate {low_confidence_rate:.2%} exceeds threshold "
            f"{max_low_confidence_rate:.2%}"
        )
    missing_critical_data_rate = missing_critical_data / max(len(diagnostics), 1)
    if diagnostics and missing_critical_data_rate > max_missing_critical_data_rate:
        hard_errors.append(
            f"critical parsed data missing/invalid for {missing_critical_data_rate:.2%} of diagnostics rows"
        )
    if diagnostics and len(diag_by_id) < len(result_set):
        warnings.append(f"diagnostics cover {len(diag_by_id)} IDs but result has {len(result_set)} IDs")
    if not diagnostics:
        warnings.append("no diagnostics JSONL provided; API/runtime audit skipped")

    report = {
        "submit_ready": not hard_errors,
        "hard_errors": hard_errors,
        "warnings": warnings[:50],
        "paths": {
            "test_json": str(test_json),
            "result_csv": str(result_csv),
            "diagnostics_jsonl": str(diagnostics_jsonl) if diagnostics_jsonl else None,
        },
        "rows": {
            "expected": len(expected_ids),
            "result": len(result_rows),
            "answered": answered,
            "answer_rate": answer_rate,
            "coverage_rate": coverage_rate,
            "missing_ids": len(missing_ids),
            "extra_ids": len(extra_ids),
            "duplicate_ids": len(duplicate_ids),
        },
        "answer_distribution": {
            "by_code": dict(sorted(code_counter.items(), key=lambda kv: _code_num(kv[0]))),
            "by_answer_len": dict(sorted(answer_len_counter.items())),
            "by_tag": dict(tag_counter),
            "by_tag_and_answer_len": dict(sorted(tag_answer_len_counter.items())),
        },
        "diagnostics": {
            "rows": len(diagnostics),
            "family_counts": dict(family_counter),
            "family_override_counts": dict(family_override_counter),
            "scenario_errors": dict(diag_error_counter),
            "candidate_empty": candidate_empty,
            "low_confidence_top_candidates": low_confidence,
            "low_confidence_rate": low_confidence_rate if top_candidate_scores else 0.0,
            "top_candidate_score": {
                "mean": mean(top_candidate_scores) if top_candidate_scores else 0.0,
                "p50": _percentile(top_candidate_scores, 50),
                "p95": _percentile(top_candidate_scores, 95),
                "min": min(top_candidate_scores) if top_candidate_scores else 0.0,
                "max": max(top_candidate_scores) if top_candidate_scores else 0.0,
            },
            "missing_data_by_source": dict(missing_data_counter),
            "missing_critical_data": missing_critical_data,
            "missing_critical_data_rate": missing_critical_data_rate if diagnostics else 0.0,
            "api_calls_total": api_calls_total,
            "api_errors_total": api_errors_total,
            "api_error_rate": api_error_rate,
            "api_calls_by_tool": dict(api_call_counter),
            "api_errors_by_tool": dict(api_error_counter),
            "elapsed_seconds": {
                "mean": mean(elapsed_values) if elapsed_values else 0.0,
                "p50": _percentile(elapsed_values, 50),
                "p95": _percentile(elapsed_values, 95),
                "max": max(elapsed_values) if elapsed_values else 0.0,
            },
        },
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Track A Phase 2 result CSV before submission")
    ap.add_argument("--test_json", type=Path, default=Path("data/Phase_2/test.json"))
    ap.add_argument("--result_csv", type=Path, required=True)
    ap.add_argument("--diagnostics_jsonl", type=Path, default=None)
    ap.add_argument("--out_report", type=Path, default=None)
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--max_api_error_rate", type=float, default=0.02)
    ap.add_argument("--min_answer_rate", type=float, default=1.0)
    ap.add_argument("--max_low_confidence_rate", type=float, default=0.15)
    ap.add_argument("--max_missing_critical_data_rate", type=float, default=0.02)
    args = ap.parse_args()

    report = audit(
        test_json=args.test_json,
        result_csv=args.result_csv,
        diagnostics_jsonl=args.diagnostics_jsonl,
        allow_partial=args.allow_partial,
        max_api_error_rate=args.max_api_error_rate,
        min_answer_rate=args.min_answer_rate,
        max_low_confidence_rate=args.max_low_confidence_rate,
        max_missing_critical_data_rate=args.max_missing_critical_data_rate,
    )

    out_report = args.out_report or args.result_csv.with_name(args.result_csv.stem + ".audit.json")
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "submit_ready": report["submit_ready"],
        "rows": report["rows"],
        "hard_error_count": len(report["hard_errors"]),
        "warning_count": len(report["warnings"]),
        "diagnostics": report["diagnostics"],
        "out_report": str(out_report),
    }, indent=2, ensure_ascii=False))

    if not report["submit_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
