#!/usr/bin/env python
"""Gate a tiny add_neighbor/decrease_a3 disambiguation before touching V6.1."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from itertools import product
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
SPLIT_JSON = ROOT / "results/phase1_harness_split.json"
HOLDOUT_JSON = ROOT / "results/phase1_validation/holdout_stratified.json"
BASE_SUBMISSION = ROOT / "submission_600_diag_lora_boundary_resubmit.csv"
V6_1_SUBMISSION = ROOT / "submissions/submission_600_v6_1_strict60_plus_f2fix.csv"
PHASE2_DIAG = ROOT / "results/phase2/sub2_lora_boundary/result_diag_lora_boundary_only.diagnostics.jsonl"
OUT_SUBMISSION = ROOT / "submissions/submission_600_v6_2_addneighbor_gate.csv"
OUT_JSON = ROOT / "results/phase1_validation/add_neighbor_gate.json"
OUT_MD = ROOT / "reports/v6_2_add_neighbor_gate.md"

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


@dataclass(frozen=True)
class RuleParams:
    add_to_dec_enabled: bool
    dec_to_add_enabled: bool
    add_to_dec_missing_min: float
    add_to_dec_late_max: float
    add_to_dec_margin_max: float
    dec_to_add_missing_max: float
    dec_to_add_late_min: float
    dec_to_add_margin_min: float

    def name(self) -> str:
        return (
            f"a2d={int(self.add_to_dec_enabled)}"
            f":m>={self.add_to_dec_missing_min:.2f}"
            f":l<={self.add_to_dec_late_max:.2f}"
            f":a3<={self.add_to_dec_margin_max:.2f}|"
            f"d2a={int(self.dec_to_add_enabled)}"
            f":m<={self.dec_to_add_missing_max:.2f}"
            f":l>={self.dec_to_add_late_min:.2f}"
            f":a3>={self.dec_to_add_margin_min:.2f}"
        )


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_submission(path: Path) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = {r["ID"]: r for r in reader}
        return list(reader.fieldnames or []), rows


def write_submission(path: Path, headers: List[str], rows: Dict[str, Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for sid in rows:
            writer.writerow(rows[sid])


def best_code_by_action_from_candidates(candidates: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for cand in getattr(candidates, "single_candidates", []) or []:
        out.setdefault(cand.action_type, cand.c_code)
    return out


def best_code_by_action_from_diag(candidates: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for cand in candidates:
        action = str(cand.get("action_type", ""))
        answer = str(cand.get("answer", "")).strip()
        if action and answer:
            out.setdefault(action, answer)
    return out


def phase1_records() -> Dict[str, Dict[str, Any]]:
    rows = load_json(TRAIN_JSON)
    records: Dict[str, Dict[str, Any]] = {}
    for scenario in rows:
        sid = str(scenario.get("scenario_id"))
        if family_name(scenario) != "F1":
            continue
        truth = normalise_answer(scenario.get("answer"))
        if not truth or "|" in truth:
            continue

        tag = infer_scenario_tag(scenario)
        wi = scenario.get("context", {}).get("wireless_network_information", {})
        num_bs = int(str(wi.get("num_base_stations", "4")).strip())
        _, family, _ = _phase2_family(tag, num_bs)
        ps = parse_scenario(scenario)
        sf = run_all_skills(ps)
        candidates = generate_candidates(ps, sf, family)
        pred = _format_answer(candidates, tag)
        best_by_action = best_code_by_action_from_candidates(candidates)
        records[sid] = {
            "id": sid,
            "truth": truth,
            "pred": pred,
            "gt_action": "|".join(option_actions(scenario, truth)),
            "pred_action": "|".join(option_actions(scenario, pred)),
            "best_by_action": best_by_action,
            "missing": float(sf.score_missing_neighbor),
            "late": float(sf.score_late_handover),
            "a3_margin": sf.mobility.a3_margin_db,
            "neighbor_advantage": sf.mobility.neighbor_advantage_db,
            "neighbor_in_config": bool(sf.mobility.neighbor_in_config),
            "mobility_issue_type": sf.mobility.mobility_issue_type,
            "baseline_iou": iou_score(pred, truth),
        }
    return records


def _leq(value: Optional[float], threshold: float) -> bool:
    return value is not None and value <= threshold


def _geq(value: Optional[float], threshold: float) -> bool:
    return value is not None and value >= threshold


def apply_rule_to_record(record: Dict[str, Any], params: RuleParams) -> Tuple[str, str]:
    pred = str(record["pred"])
    action = str(record["pred_action"])
    by_action = record.get("best_by_action") or {}

    if (
        params.add_to_dec_enabled
        and action == "add_neighbor"
        and by_action.get("decrease_a3")
        and record["missing"] >= params.add_to_dec_missing_min
        and record["late"] <= params.add_to_dec_late_max
        and _leq(record.get("a3_margin"), params.add_to_dec_margin_max)
    ):
        return by_action["decrease_a3"], "add_neighbor_to_decrease_a3"

    if (
        params.dec_to_add_enabled
        and action == "decrease_a3"
        and by_action.get("add_neighbor")
        and record["missing"] <= params.dec_to_add_missing_max
        and record["late"] >= params.dec_to_add_late_min
        and _geq(record.get("a3_margin"), params.dec_to_add_margin_min)
    ):
        return by_action["add_neighbor"], "decrease_a3_to_add_neighbor"

    return pred, "unchanged"


def action_metrics(rows: List[Dict[str, Any]], pred_key: str) -> Dict[str, Dict[str, float]]:
    labels = sorted(set(r["gt_action"] for r in rows) | set(r[f"{pred_key}_action"] for r in rows))
    out: Dict[str, Dict[str, float]] = {}
    for label in labels:
        if not label:
            continue
        gt = sum(1 for r in rows if r["gt_action"] == label)
        pred = sum(1 for r in rows if r[f"{pred_key}_action"] == label)
        tp = sum(1 for r in rows if r["gt_action"] == label and r[f"{pred_key}_action"] == label)
        out[label] = {
            "gt": gt,
            "pred": pred,
            "tp": tp,
            "precision": tp / pred if pred else 0.0,
            "recall": tp / gt if gt else 0.0,
        }
    return out


def evaluate(records: Dict[str, Dict[str, Any]], ids: Iterable[str], params: RuleParams) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for sid in ids:
        if sid not in records:
            continue
        record = dict(records[sid])
        new_pred, reason = apply_rule_to_record(record, params)
        record["new_pred"] = new_pred
        record["reason"] = reason
        record["new_action"] = record["pred_action"]
        if new_pred != record["pred"]:
            scenario = _PHASE1_BY_ID[sid]
            record["new_action"] = "|".join(option_actions(scenario, new_pred))
        record["new_iou"] = iou_score(new_pred, record["truth"])
        rows.append(record)

    base = sum(float(r["baseline_iou"]) for r in rows) / len(rows) if rows else 0.0
    new = sum(float(r["new_iou"]) for r in rows) / len(rows) if rows else 0.0
    changed = [r for r in rows if r["reason"] != "unchanged"]
    helped = [r for r in changed if r["new_iou"] > r["baseline_iou"]]
    harmed = [r for r in changed if r["new_iou"] < r["baseline_iou"]]
    same = [r for r in changed if r["new_iou"] == r["baseline_iou"]]
    base_metrics = action_metrics(
        [{**r, "baseline_action": r["pred_action"]} for r in rows],
        "baseline",
    )
    new_metrics = action_metrics(rows, "new")
    return {
        "n": len(rows),
        "baseline_iou": base,
        "new_iou": new,
        "delta_iou": new - base,
        "changed": len(changed),
        "helped": len(helped),
        "harmed": len(harmed),
        "same": len(same),
        "reason_counts": dict(Counter(r["reason"] for r in changed)),
        "gt_actions_changed": dict(Counter(r["gt_action"] for r in changed)),
        "base_action_metrics": base_metrics,
        "new_action_metrics": new_metrics,
        "changed_rows": [
            {
                "id": r["id"],
                "truth": r["truth"],
                "old": r["pred"],
                "new": r["new_pred"],
                "gt_action": r["gt_action"],
                "old_action": r["pred_action"],
                "new_action": r["new_action"],
                "reason": r["reason"],
                "old_iou": r["baseline_iou"],
                "new_iou": r["new_iou"],
                "missing": r["missing"],
                "late": r["late"],
                "a3_margin": r["a3_margin"],
                "mobility_issue_type": r["mobility_issue_type"],
            }
            for r in changed
        ],
    }


def grid_params() -> List[RuleParams]:
    params: List[RuleParams] = []
    for a2d_on, d2a_on in product([False, True], [False, True]):
        for a2d_missing, a2d_late, a2d_margin in product(
            [0.70, 0.74, 0.75],
            [0.35, 0.36, 0.40],
            [2.50, 3.00, 3.50, 99.0],
        ):
            for d2a_missing, d2a_late, d2a_margin in product(
                [0.30, 0.31, 0.35],
                [0.65, 0.69, 0.70],
                [2.50, 3.00, 3.50],
            ):
                params.append(
                    RuleParams(
                        add_to_dec_enabled=a2d_on,
                        dec_to_add_enabled=d2a_on,
                        add_to_dec_missing_min=a2d_missing,
                        add_to_dec_late_max=a2d_late,
                        add_to_dec_margin_max=a2d_margin,
                        dec_to_add_missing_max=d2a_missing,
                        dec_to_add_late_min=d2a_late,
                        dec_to_add_margin_min=d2a_margin,
                    )
                )
    return params


def choose_rule(records: Dict[str, Dict[str, Any]], dev_ids: List[str]) -> Tuple[RuleParams, Dict[str, Any]]:
    baseline = RuleParams(False, False, 0.75, 0.35, 3.0, 0.30, 0.70, 3.0)
    baseline_eval = evaluate(records, dev_ids, baseline)
    baseline_decrease_precision = baseline_eval["base_action_metrics"].get("decrease_a3", {}).get("precision", 0.0)
    candidates: List[Tuple[float, int, int, RuleParams, Dict[str, Any]]] = []
    for params in grid_params():
        result = evaluate(records, dev_ids, params)
        decrease_precision = result["new_action_metrics"].get("decrease_a3", {}).get("precision", 0.0)
        if result["delta_iou"] <= 0:
            continue
        if decrease_precision + 1e-12 < max(0.80, baseline_decrease_precision - 1e-12):
            continue
        if result["harmed"] != 0:
            continue
        candidates.append((result["delta_iou"], result["helped"], result["changed"], params, result))

    if not candidates:
        return baseline, baseline_eval
    candidates.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)
    return candidates[0][3], candidates[0][4]


def load_phase2_diag() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with PHASE2_DIAG.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                out[str(data.get("scenario_id"))] = data
    return out


def phase2_record_from_diag(sid: str, scenario: Dict[str, Any], answer: str, diag: Dict[str, Any]) -> Dict[str, Any]:
    features = diag.get("feature_scores") or {}
    raw = diag.get("raw_evidence_summary") or {}
    best_neighbor = raw.get("best_neighbor") or {}
    action = "|".join(option_actions(scenario, answer))
    return {
        "id": sid,
        "pred": answer,
        "pred_action": action,
        "best_by_action": best_code_by_action_from_diag(diag.get("candidates") or []),
        "missing": float(features.get("missing_neighbor", 0.0)),
        "late": float(features.get("late_handover", 0.0)),
        "a3_margin": best_neighbor.get("a3_margin_db"),
    }


def apply_phase2(params: RuleParams) -> Dict[str, Any]:
    headers, base_rows = read_submission(BASE_SUBMISSION)
    _, v6_rows = read_submission(V6_1_SUBMISSION)
    scenarios = {str(s["scenario_id"]): s for s in load_json(PHASE2_JSON)}
    diagnostics = load_phase2_diag()

    protected_ids = {
        sid for sid, row in v6_rows.items()
        if row.get("Track A", "").strip() != base_rows.get(sid, {}).get("Track A", "").strip()
        and family_name(scenarios[sid]) in {"F1", "F1_5BS"}
    }
    out_rows = {sid: dict(row) for sid, row in v6_rows.items()}
    changes: List[Dict[str, Any]] = []
    for sid, row in list(out_rows.items()):
        if sid in protected_ids or sid not in scenarios or sid not in diagnostics:
            continue
        family = family_name(scenarios[sid])
        if family not in {"F1", "F1_5BS"}:
            continue
        answer = row.get("Track A", "").strip()
        if not answer or "|" in answer:
            continue
        record = phase2_record_from_diag(sid, scenarios[sid], answer, diagnostics[sid])
        new_answer, reason = apply_rule_to_record(record, params)
        if reason != "unchanged" and new_answer != answer:
            old_action = record["pred_action"]
            new_action = "|".join(option_actions(scenarios[sid], new_answer))
            out_rows[sid]["Track A"] = new_answer
            changes.append({
                "id": sid,
                "old": answer,
                "new": new_answer,
                "old_action": old_action,
                "new_action": new_action,
                "reason": reason,
                "missing": record["missing"],
                "late": record["late"],
                "a3_margin": record["a3_margin"],
            })

    for prefix, expected in LORA_OVERRIDES.items():
        for sid, row in out_rows.items():
            if sid.startswith(prefix):
                row["Track A"] = expected

    write_submission(OUT_SUBMISSION, headers, out_rows)
    audit = audit_submission(out_rows, scenarios, base_rows)
    return {
        "submission": str(OUT_SUBMISSION.relative_to(ROOT)),
        "changes": changes,
        "audit": audit,
        "protected_f1_fix_rows": len(protected_ids),
    }


def audit_submission(
    rows: Dict[str, Dict[str, str]],
    scenarios: Dict[str, Dict[str, Any]],
    base_rows: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    track_a = [r for r in rows.values() if r.get("Track A", "").strip()]
    track_b = [r for r in rows.values() if r.get("Track B", "").strip()]
    diff = [
        sid for sid, row in rows.items()
        if row.get("Track A", "").strip() != base_rows.get(sid, {}).get("Track A", "").strip()
    ]
    illegal_f1 = []
    f2_bad = []
    for sid, row in rows.items():
        if sid not in scenarios:
            continue
        answer = row.get("Track A", "").strip()
        if not answer:
            continue
        family = family_name(scenarios[sid])
        actions = option_actions(scenarios[sid], answer)
        if family in {"F1", "F1_5BS"} and (len(actions) != 1 or actions[0] not in F1_ALLOWED_ACTIONS):
            illegal_f1.append((sid, answer, actions))
        if family == "F2" and frozenset(actions) not in F2_LEGAL_ACTION_SETS:
            f2_bad.append((sid, answer, actions))
    lora = {}
    for prefix, expected in LORA_OVERRIDES.items():
        match = [row for sid, row in rows.items() if sid.startswith(prefix)]
        actual = match[0].get("Track A", "").strip() if match else ""
        lora[prefix] = actual == expected
    return {
        "rows": len(rows),
        "track_a": len(track_a),
        "track_b": len(track_b),
        "diff_vs_baseline": len(diff),
        "illegal_f1": len(illegal_f1),
        "f2_bad": len(f2_bad),
        "lora_ok": all(lora.values()),
    }


def compact_eval(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n": result["n"],
        "baseline_iou": result["baseline_iou"],
        "new_iou": result["new_iou"],
        "delta_iou": result["delta_iou"],
        "changed": result["changed"],
        "helped": result["helped"],
        "harmed": result["harmed"],
        "reason_counts": result["reason_counts"],
        "base_decrease_precision": result["base_action_metrics"].get("decrease_a3", {}).get("precision", 0.0),
        "new_decrease_precision": result["new_action_metrics"].get("decrease_a3", {}).get("precision", 0.0),
        "base_add_recall": result["base_action_metrics"].get("add_neighbor", {}).get("recall", 0.0),
        "new_add_recall": result["new_action_metrics"].get("add_neighbor", {}).get("recall", 0.0),
    }


def write_report(data: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# V6.2 Add-Neighbor Gate")
    lines.append("")
    lines.append(f"- Selected rule: `{data['selected_rule']}`")
    lines.append(f"- Gate passed: `{data['gate_passed']}`")
    lines.append("")
    lines.append("| Split | N | Baseline IOU | New IOU | Delta | Changed | Helped | Harmed | decrease_a3 P old->new | add_neighbor R old->new |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split in ["dev", "test", "holdout", "all"]:
        row = data["eval"][split]
        lines.append(
            f"| {split} | {row['n']} | {row['baseline_iou']:.4f} | {row['new_iou']:.4f} | "
            f"{row['delta_iou']:.4f} | {row['changed']} | {row['helped']} | {row['harmed']} | "
            f"{row['base_decrease_precision']:.4f}->{row['new_decrease_precision']:.4f} | "
            f"{row['base_add_recall']:.4f}->{row['new_add_recall']:.4f} |"
        )
    lines.append("")
    lines.append("## Rule Actions")
    lines.append("")
    for split in ["dev", "test", "holdout", "all"]:
        lines.append(f"- `{split}`: `{data['eval'][split]['reason_counts']}`")
    if data.get("phase2"):
        lines.append("")
        lines.append("## Phase2 Candidate")
        lines.append("")
        p2 = data["phase2"]
        lines.append(f"- Submission: `{p2['submission']}`")
        lines.append(f"- Protected F1/F1_5BS v6_1 fix rows: `{p2['protected_f1_fix_rows']}`")
        lines.append(f"- Phase2 rule changes: `{len(p2['changes'])}`")
        lines.append(f"- Audit: `{p2['audit']}`")
        for change in p2["changes"]:
            lines.append(
                f"- `{change['id']}` {change['old']}/{change['old_action']} -> "
                f"{change['new']}/{change['new_action']} ({change['reason']}, "
                f"a3={change['a3_margin']})"
            )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


_PHASE1_BY_ID: Dict[str, Dict[str, Any]] = {}


def main() -> None:
    global _PHASE1_BY_ID
    phase1 = load_json(TRAIN_JSON)
    _PHASE1_BY_ID = {str(s.get("scenario_id")): s for s in phase1}
    split = load_json(SPLIT_JSON)
    holdout = load_json(HOLDOUT_JSON)

    records = phase1_records()
    dev_ids = [sid for sid in split["dev"] if sid in records]
    test_ids = [sid for sid in split["test"] if sid in records]
    holdout_ids = [sid for sid in holdout["ids"] if sid in records]
    all_ids = sorted(records)

    selected, dev_eval = choose_rule(records, dev_ids)
    test_eval = evaluate(records, test_ids, selected)
    holdout_eval = evaluate(records, holdout_ids, selected)
    all_eval = evaluate(records, all_ids, selected)
    base_test_dec_p = test_eval["base_action_metrics"].get("decrease_a3", {}).get("precision", 0.0)
    new_test_dec_p = test_eval["new_action_metrics"].get("decrease_a3", {}).get("precision", 0.0)
    all_new_dec_p = all_eval["new_action_metrics"].get("decrease_a3", {}).get("precision", 0.0)
    gate_passed = (
        test_eval["delta_iou"] > 0
        and test_eval["harmed"] == 0
        and new_test_dec_p + 1e-12 >= base_test_dec_p
        and all_new_dec_p + 1e-12 >= 0.80
    )

    data: Dict[str, Any] = {
        "selected_rule": selected.name(),
        "gate_passed": gate_passed,
        "eval": {
            "dev": compact_eval(dev_eval),
            "test": compact_eval(test_eval),
            "holdout": compact_eval(holdout_eval),
            "all": compact_eval(all_eval),
        },
        "changed_rows": {
            "test": test_eval["changed_rows"],
            "all": all_eval["changed_rows"],
        },
    }

    if gate_passed:
        data["phase2"] = apply_phase2(selected)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_report(data)

    print(f"selected: {selected.name()}")
    for split_name in ["dev", "test", "holdout", "all"]:
        row = data["eval"][split_name]
        print(
            f"{split_name}: n={row['n']} base={row['baseline_iou']:.4f} "
            f"new={row['new_iou']:.4f} delta={row['delta_iou']:.4f} "
            f"changed={row['changed']} helped={row['helped']} harmed={row['harmed']} "
            f"decP={row['base_decrease_precision']:.4f}->{row['new_decrease_precision']:.4f}"
        )
    print(f"gate_passed: {gate_passed}")
    if data.get("phase2"):
        print(f"phase2 submission: {data['phase2']['submission']}")
        print(f"phase2 changes: {len(data['phase2']['changes'])}")
    print(f"report: {OUT_MD}")


if __name__ == "__main__":
    main()
