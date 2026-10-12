#!/usr/bin/env python
"""Measure whether backhaul_server score can support a check_server promotion."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase1_validation_harness import family_name, normalise_answer, option_actions
from phase2_run import _format_answer, _phase2_family
from skills.answer_parser import iou_score
from skills.candidate_generator import generate_candidates
from skills.feature_engine import run_all_skills
from skills.scenario_parser import infer_scenario_tag, parse_scenario


TRAIN_JSON = ROOT / "data/Phase_1/train.json"
PHASE2_JSON = ROOT / "data/Phase_2/test.json"
PHASE2_DIAG = ROOT / "results/phase2/sub2_lora_boundary/result_diag_lora_boundary_only.diagnostics.jsonl"
OUT_JSON = ROOT / "results/phase1_validation/backhaul_checkserver_gate.json"
OUT_MD = ROOT / "reports/v6_2_backhaul_gate_measurement.md"

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.45, 0.48, 0.49, 0.5]


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def check_server_code(scenario: Dict[str, Any]) -> str:
    ps = parse_scenario(scenario)
    for code, option in ps.parsed_options.items():
        if option.get("action_type") == "check_server":
            return code
    return ""


def deterministic_prediction(scenario: Dict[str, Any]) -> Tuple[str, float]:
    tag = infer_scenario_tag(scenario)
    wi = scenario.get("context", {}).get("wireless_network_information", {})
    num_bs = int(str(wi.get("num_base_stations", "4")).strip())
    _, family, _ = _phase2_family(tag, num_bs)

    ps = parse_scenario(scenario)
    sf = run_all_skills(ps)
    candidates = generate_candidates(ps, sf, family)
    return _format_answer(candidates, tag), float(sf.score_backhaul_server)


def phase1_rows(scenarios: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        family = family_name(scenario)
        if family not in {"F1", "F1_5BS"}:
            continue
        truth = normalise_answer(scenario.get("answer"))
        if not truth or "|" in truth:
            continue
        pred, backhaul = deterministic_prediction(scenario)
        gt_action = "|".join(option_actions(scenario, truth))
        pred_action = "|".join(option_actions(scenario, pred))
        check_code = check_server_code(scenario)
        rows.append({
            "id": str(scenario.get("scenario_id")),
            "family": family,
            "truth": truth,
            "pred": pred,
            "gt_action": gt_action,
            "pred_action": pred_action,
            "backhaul": backhaul,
            "check_code": check_code,
            "iou": iou_score(pred, truth),
        })
    return rows


def threshold_summary(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        selected = [r for r in rows if r["backhaul"] >= threshold]
        impacted = [
            r for r in selected
            if r["check_code"] and r["pred_action"] != "check_server"
        ]
        gt_check = sum(1 for r in selected if r["gt_action"] == "check_server")
        impacted_gt_check = sum(1 for r in impacted if r["gt_action"] == "check_server")

        before = sum(float(r["iou"]) for r in rows) / len(rows) if rows else 0.0
        after_scores = []
        changed = 0
        harmed = 0
        helped = 0
        same_wrong = 0
        for r in rows:
            new_pred = r["pred"]
            if r["backhaul"] >= threshold and r["check_code"] and r["pred_action"] != "check_server":
                new_pred = r["check_code"]
                changed += 1
            before_iou = float(r["iou"])
            after_iou = iou_score(new_pred, r["truth"])
            if after_iou > before_iou:
                helped += 1
            elif after_iou < before_iou:
                harmed += 1
            elif changed and new_pred == r["check_code"] and after_iou == 0:
                same_wrong += 1
            after_scores.append(after_iou)

        after = sum(after_scores) / len(after_scores) if after_scores else 0.0
        summaries[key] = {
            "selected": len(selected),
            "gt_check_server": gt_check,
            "signal_precision": gt_check / len(selected) if selected else None,
            "already_pred_check_server": sum(1 for r in selected if r["pred_action"] == "check_server"),
            "impacted": len(impacted),
            "impacted_gt_check_server": impacted_gt_check,
            "impacted_precision": impacted_gt_check / len(impacted) if impacted else None,
            "before_iou": before,
            "after_iou": after,
            "delta_iou": after - before,
            "changed": changed,
            "helped": helped,
            "harmed": harmed,
            "same_wrong": same_wrong,
            "selected_gt_actions": dict(Counter(r["gt_action"] for r in selected)),
            "impacted_gt_actions": dict(Counter(r["gt_action"] for r in impacted)),
        }
    return summaries


def phase2_histogram(scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    if not PHASE2_DIAG.exists():
        return {"error": f"missing {PHASE2_DIAG}"}
    with PHASE2_DIAG.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            sid = str(data.get("scenario_id", ""))
            scenario = scenarios.get(sid)
            if scenario is None:
                continue
            family = data.get("routed_family") or data.get("family") or family_name(scenario)
            if family not in {"F1", "F1_5BS"}:
                continue
            score = float((data.get("feature_scores") or {}).get("backhaul_server", 0.0))
            answer = str(data.get("answer", "")).strip()
            actions = option_actions(scenario, answer)
            rows.append({
                "id": sid,
                "family": family,
                "score": score,
                "answer": answer,
                "action": actions[0] if len(actions) == 1 else "|".join(actions),
            })

    bins = Counter(f"{r['score']:.1f}" for r in rows)
    by_threshold: Dict[str, Any] = {}
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        selected = [r for r in rows if r["score"] >= threshold]
        by_threshold[key] = {
            "selected": len(selected),
            "families": dict(Counter(r["family"] for r in selected)),
            "current_actions": dict(Counter(r["action"] for r in selected)),
        }
    return {
        "n": len(rows),
        "histogram": dict(sorted(bins.items())),
        "thresholds": by_threshold,
    }


def pct(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.4f}"


def write_report(data: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# V6.2 Backhaul Check-Server Gate Measurement")
    lines.append("")
    lines.append("## Phase1 F1 Signal")
    lines.append("")
    lines.append(f"- Rows measured: `{data['phase1']['n']}`")
    lines.append(f"- Current mean IOU: `{data['phase1']['current_mean_iou']:.4f}`")
    lines.append("")
    lines.append("| Threshold | Selected | GT check_server | Signal precision | Already pred check | Impacted | Impact precision | Delta IOU | Helped | Harmed |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for threshold, row in data["phase1"]["thresholds"].items():
        lines.append(
            f"| {threshold} | {row['selected']} | {row['gt_check_server']} | "
            f"{pct(row['signal_precision'])} | {row['already_pred_check_server']} | "
            f"{row['impacted']} | {pct(row['impacted_precision'])} | "
            f"{row['delta_iou']:.4f} | {row['helped']} | {row['harmed']} |"
        )

    lines.append("")
    lines.append("## Phase2 Diagnostic Distribution")
    lines.append("")
    phase2 = data["phase2"]
    lines.append(f"- Rows measured: `{phase2.get('n', 0)}`")
    lines.append(f"- Backhaul histogram: `{phase2.get('histogram', {})}`")
    lines.append("")
    lines.append("| Threshold | Selected | Families | Current actions |")
    lines.append("|---:|---:|---|---|")
    for threshold, row in phase2.get("thresholds", {}).items():
        lines.append(
            f"| {threshold} | {row['selected']} | `{row['families']}` | `{row['current_actions']}` |"
        )

    lines.append("")
    lines.append("## Decision")
    lines.append("")
    t04 = data["phase1"]["thresholds"]["0.40"]
    t05 = data["phase1"]["thresholds"]["0.50"]
    if t04["signal_precision"] is not None and t04["signal_precision"] >= 0.9:
        lines.append("- Phase1 says `backhaul_server >= 0.4` is a clean check_server signal.")
    else:
        lines.append("- Phase1 does not support a broad `backhaul_server >= 0.4` promotion.")
    lines.append(
        f"- At `>=0.4`, Phase1 signal precision is `{pct(t04['signal_precision'])}`, "
        f"while Phase2 would promote/check `{phase2.get('thresholds', {}).get('0.40', {}).get('selected', 0)}` rows."
    )
    lines.append(
        f"- At `>=0.5`, Phase1 signal precision is `{pct(t05['signal_precision'])}`, "
        f"but Phase2 only has `{phase2.get('thresholds', {}).get('0.50', {}).get('selected', 0)}` rows."
    )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    train = load_json(TRAIN_JSON)
    phase2 = {str(s["scenario_id"]): s for s in load_json(PHASE2_JSON)}

    rows = phase1_rows(train)
    phase1 = {
        "n": len(rows),
        "current_mean_iou": sum(float(r["iou"]) for r in rows) / len(rows),
        "score_histogram": dict(Counter(f"{r['backhaul']:.1f}" for r in rows)),
        "thresholds": threshold_summary(rows),
    }
    data = {
        "phase1": phase1,
        "phase2": phase2_histogram(phase2),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_report(data)
    print(f"phase1 rows: {phase1['n']}")
    for threshold, row in phase1["thresholds"].items():
        print(
            f"t>={threshold}: selected={row['selected']} "
            f"gt_check={row['gt_check_server']} "
            f"precision={pct(row['signal_precision'])} "
            f"impacted={row['impacted']} delta={row['delta_iou']:.4f}"
        )
    print(f"phase2 histogram: {data['phase2'].get('histogram', {})}")
    print(f"report: {OUT_MD}")


if __name__ == "__main__":
    main()
