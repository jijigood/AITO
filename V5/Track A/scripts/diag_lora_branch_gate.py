#!/usr/bin/env python
"""Gate diag-LoRA only on the lift_tilt vs insufficient_data branch.

This script deliberately keeps the experiment narrow:
  - Phase-1 F1 split only.
  - Only scenarios whose deterministic answer is lift_tilt are sent to LoRA.
  - LoRA outputs may only replace lift_tilt with lift_tilt/insufficient_data.
  - Everything outside that branch is scored as unchanged deterministic output.

The prompt and diagnosis mapper are reused from the V4 diagnostic SFT/eval
scripts so the fine-tuned adapter sees the same contract it was trained on.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
V4_ROOT = ROOT.parents[1] / "V4" / "Track A"

sys.path.insert(0, str(ROOT))
sys.path.append(str(V4_ROOT))

from phase2_run import _format_answer, _phase2_family
from skills.answer_parser import iou_score
from skills.candidate_generator import generate_candidates
from skills.feature_engine import run_all_skills
from skills.scenario_parser import infer_scenario_tag, parse_scenario


TRAIN_JSON = ROOT / "data/Phase_1/train.json"
PHASE2_JSON = ROOT / "data/Phase_2/test.json"
SPLIT_JSON = ROOT / "results/phase1_harness_split.json"
OUT_DIR = ROOT / "results/diag_lora_branch_v6_3"
OUT_MD = ROOT / "reports/v6_3_diag_lora_branch_gate.md"
PHASE2_DIAG = ROOT / "results/phase2/sub2_lora_boundary/result_diag_lora_boundary_only.diagnostics.jsonl"
V6_2_SUBMISSION = ROOT / "submissions/submission_600_v6_2_addneighbor_gate.csv"
V6_3_SUBTYPE_SUBMISSION = ROOT / "submissions/submission_600_v6_3_subtype_lift_insuff_gate.csv"
V6_3_SUBTYPE_MD = ROOT / "reports/v6_3_subtype_lift_insuff_gate.md"

PAIR_ACTIONS = {"lift_tilt", "insufficient_data"}
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
LORA_OVERRIDES = {
    "6060dca7": "C22",
    "1784b64c": "C20",
}
DEFAULT_MODEL_URL = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL_NAME = "diag"


def _load_v4_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DIAG_SFT = _load_v4_module("diag_sft_v4_reuse", V4_ROOT / "build_diagnostic_sft.py")
DIAG_EVAL = _load_v4_module("diag_eval_v4_reuse", V4_ROOT / "eval_diag_lora.py")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_submission(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    import csv

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_submission(path: Path, headers: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def code_action(scenario: Dict[str, Any], answer: str) -> str:
    if not answer or "|" in answer:
        return ""
    ps = parse_scenario(scenario)
    return str((ps.parsed_options.get(answer) or {}).get("action_type", ""))


def family_name(scenario: Dict[str, Any]) -> str:
    tag = infer_scenario_tag(scenario)
    wi = scenario.get("context", {}).get("wireless_network_information", {})
    try:
        num_bs = int(str(wi.get("num_base_stations", "4")).strip())
    except (TypeError, ValueError):
        num_bs = 4
    _, family, _ = _phase2_family(tag, num_bs)
    return family.name


def answer_actions(scenario: Dict[str, Any], answer: str) -> List[str]:
    ps = parse_scenario(scenario)
    actions: List[str] = []
    for code in [part.strip() for part in str(answer or "").split("|") if part.strip()]:
        actions.append(str((ps.parsed_options.get(code) or {}).get("action_type", "?")))
    return actions


def deterministic_context(scenario: Dict[str, Any]) -> Tuple[str, str, Any, Any, Any, Any]:
    tag = infer_scenario_tag(scenario)
    ps = parse_scenario(scenario)
    sf = run_all_skills(ps)
    _, family, _ = _phase2_family(tag, ps.num_base_stations)
    output = generate_candidates(ps, sf, family)
    answer = _format_answer(output, tag)
    action = str((ps.parsed_options.get(answer) or {}).get("action_type", ""))
    return answer, action, ps, sf, family, output


def classify_lift_subtype(ps: Any, candidates: Sequence[Dict[str, Any]]) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    first_lift: Optional[Dict[str, Any]] = None
    insufficient_code: Optional[str] = None
    for cand in candidates:
        action = str(cand.get("action_type") or cand.get("action") or "")
        if action == "insufficient_data" and insufficient_code is None:
            insufficient_code = str(cand.get("answer") or cand.get("option_code") or "") or None
        if action == "lift_tilt" and first_lift is None:
            first_lift = cand
    if not first_lift:
        return "unknown", None, insufficient_code

    target = first_lift.get("target_cell") or first_lift.get("target_cell_id")
    role = ps.cell_roles.get(target) if target else None
    reason = str(first_lift.get("reason") or "")
    if "F1 coverage hypothesis" in reason and role == "tertiary":
        return "coverage_hypothesis_tertiary", first_lift, insufficient_code
    if "high-PRB healthy-radio" in reason and role == "primary":
        return "generic_high_prb_primary_fallback", first_lift, insufficient_code
    return f"other:{role}:{reason[:60]}", first_lift, insufficient_code


def build_prompt_record(
    scenario: Dict[str, Any],
    split: str,
    det_answer: str,
    det_action: str,
    ps: Any,
    sf: Any,
    family: Any,
    output: Any,
) -> Dict[str, Any]:
    gt = str(scenario.get("answer") or "").strip()
    gt_action = str((ps.parsed_options.get(gt) or {}).get("action_type", ""))
    evidence = DIAG_SFT._evidence_json(ps, sf, family)
    options = DIAG_SFT._available_option_rows(ps, sf)
    candidates = DIAG_SFT._candidate_rows(
        ps,
        sf,
        output,
        limit=8,
        force_actions=sorted(PAIR_ACTIONS),
    )
    user = DIAG_SFT._user_prompt(
        ps,
        family,
        evidence,
        options,
        candidates,
        DIAG_SFT._tool_evidence_json(),
        [],
        candidate_actions=sorted(PAIR_ACTIONS),
    )
    return {
        "messages": [
            {"role": "system", "content": DIAG_SFT.SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "meta": {
            "scenario_id": ps.scenario_id,
            "split": split,
            "family": family.name,
            "gt": gt,
            "gt_action": gt_action,
            "deterministic_answer": det_answer,
            "deterministic_action": det_action,
            "branch": "lift_tilt_vs_insufficient_data",
            "boundary_pair": DIAG_SFT._is_lift_insufficient_boundary(candidates, sorted(PAIR_ACTIONS)),
        },
    }


def prepare_records(limit: Optional[int] = None) -> Dict[str, Any]:
    scenarios = {str(s.get("scenario_id")): s for s in load_json(TRAIN_JSON)}
    split = load_json(SPLIT_JSON)
    split_ids = {
        "dev": list(split.get("dev", [])),
        "test": list(split.get("test", [])),
    }
    branch_records: List[Dict[str, Any]] = []
    all_f1_rows: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for split_name, ids in split_ids.items():
        for sid in ids:
            scenario = scenarios.get(str(sid))
            if not scenario or family_name(scenario) != "F1":
                continue
            gt = str(scenario.get("answer") or "").strip()
            if not gt or "|" in gt:
                continue
            det_answer, det_action, ps, sf, family, output = deterministic_context(scenario)
            gt_action = str((ps.parsed_options.get(gt) or {}).get("action_type", ""))
            base_iou = iou_score(det_answer, gt)
            row = {
                "scenario_id": sid,
                "split": split_name,
                "gt": gt,
                "gt_action": gt_action,
                "deterministic_answer": det_answer,
                "deterministic_action": det_action,
                "baseline_iou": base_iou,
            }
            all_f1_rows.append(row)
            counts[f"{split_name}:all"] += 1
            counts[f"{split_name}:det:{det_action}"] += 1
            counts[f"{split_name}:gt:{gt_action}"] += 1
            if det_action != "lift_tilt":
                continue
            record = build_prompt_record(
                scenario,
                split_name,
                det_answer,
                det_action,
                ps,
                sf,
                family,
                output,
            )
            branch_records.append(record)
            counts[f"{split_name}:branch"] += 1
            counts[f"{split_name}:branch_gt:{gt_action}"] += 1
            if limit is not None and len(branch_records) >= limit:
                break
        if limit is not None and len(branch_records) >= limit:
            break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "branch_prompts.jsonl", branch_records)
    write_jsonl(OUT_DIR / "all_f1_baseline.jsonl", all_f1_rows)
    summary = {
        "branch_records": len(branch_records),
        "all_f1_rows": len(all_f1_rows),
        "counts": dict(sorted(counts.items())),
        "outputs": {
            "branch_prompts": str(OUT_DIR / "branch_prompts.jsonl"),
            "all_f1_baseline": str(OUT_DIR / "all_f1_baseline.jsonl"),
        },
    }
    write_json(OUT_DIR / "prepare_summary.json", summary)
    return summary


def _prompt_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": msg.get("role", ""), "content": msg.get("content", "")}
        for msg in record.get("messages", [])
        if msg.get("role") in {"system", "user"}
    ]


def call_model(
    prompts_path: Path,
    out_path: Path,
    model_url: str,
    model_name: str,
    max_tokens: int,
    timeout: float,
    limit: Optional[int],
    resume: bool,
    split_filter: str,
) -> Dict[str, Any]:
    # Keep localhost requests away from any 10080 proxy environment.
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)

    prompts = read_jsonl(prompts_path)
    if split_filter != "all":
        prompts = [
            record for record in prompts
            if str((record.get("meta") or {}).get("split") or "") == split_filter
        ]
    if limit is not None:
        prompts = prompts[:limit]

    done: Dict[str, Dict[str, Any]] = {}
    if resume and out_path.exists():
        for row in read_jsonl(out_path):
            sid = str(row.get("scenario_id") or "")
            if sid:
                done[sid] = row

    client = DIAG_EVAL.OpenAIDiagClient(model_url, model_name, timeout)
    rows: List[Dict[str, Any]] = [done[sid] for sid in done]

    for idx, record in enumerate(prompts, 1):
        meta = record.get("meta") or {}
        sid = str(meta.get("scenario_id") or "")
        if sid in done:
            print(f"[{idx:04d}/{len(prompts):04d}] skip {sid[:8]} cached", flush=True)
            continue
        raw = ""
        error = None
        try:
            raw = client.call(_prompt_messages(record), max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "idx": idx,
            "scenario_id": sid,
            "split": meta.get("split"),
            "gt": meta.get("gt"),
            "gt_action": meta.get("gt_action"),
            "deterministic_answer": meta.get("deterministic_answer"),
            "deterministic_action": meta.get("deterministic_action"),
            "raw": raw,
            "api_error": error,
        }
        rows.append(row)
        write_jsonl(out_path, rows)
        status = "ERR" if error else "OK"
        print(f"[{idx:04d}/{len(prompts):04d}] {status} {sid[:8]}", flush=True)

    summary = {
        "requested": len(prompts),
        "written": len(rows),
        "api_errors": sum(1 for r in rows if r.get("api_error")),
        "out_path": str(out_path),
        "split_filter": split_filter,
    }
    write_json(OUT_DIR / "model_call_summary.json", summary)
    return summary


def _scenario_map() -> Dict[str, Dict[str, Any]]:
    return {str(s.get("scenario_id")): s for s in load_json(TRAIN_JSON)}


def _as_bool(value: Any) -> bool:
    return DIAG_EVAL._as_bool(value)


def _diagnosis_to_pair_answer(
    row: Dict[str, Any],
    scenarios: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    sid = str(row.get("scenario_id") or "")
    scenario = scenarios[sid]
    ps = parse_scenario(scenario)
    det_answer = str(row.get("deterministic_answer") or "")
    gt = str(row.get("gt") or scenario.get("answer") or "").strip()
    raw = str(row.get("raw") or "")
    diagnosis, json_error = DIAG_EVAL._extract_json_object(raw)
    json_valid = diagnosis is not None and json_error is None
    if not json_valid:
        diagnosis = {}
    mapped_answer, mapped_ok, map_status, map_score = DIAG_EVAL._map_diagnosis_to_ccode(
        diagnosis,
        ps,
        det_answer,
    )
    mapped_action = str((ps.parsed_options.get(mapped_answer) or {}).get("action_type", ""))
    pair_ok = bool(json_valid and mapped_ok and mapped_action in PAIR_ACTIONS)
    pair_answer = mapped_answer if pair_ok else det_answer
    confidence = str(diagnosis.get("evidence_confidence") or "").strip().lower()
    should_override = _as_bool(diagnosis.get("should_override"))
    return {
        "scenario_id": sid,
        "split": row.get("split"),
        "gt": gt,
        "gt_action": row.get("gt_action"),
        "deterministic_answer": det_answer,
        "deterministic_action": row.get("deterministic_action"),
        "raw": raw,
        "api_error": row.get("api_error"),
        "json_valid": json_valid,
        "json_error": json_error,
        "pred_action": diagnosis.get("action"),
        "pred_confidence": confidence,
        "pred_should_override": should_override,
        "mapped_answer": mapped_answer,
        "mapped_action": mapped_action,
        "mapped_ok": mapped_ok,
        "map_status": map_status,
        "map_score": map_score,
        "pair_ok": pair_ok,
        "pair_answer": pair_answer,
        "det_iou": iou_score(det_answer, gt),
        "mapped_iou": iou_score(pair_answer, gt),
    }


def _answer_for_config(parsed: Dict[str, Any], config: str) -> str:
    det = str(parsed.get("deterministic_answer") or "")
    pair = str(parsed.get("pair_answer") or det)
    if not parsed.get("pair_ok"):
        return det
    if config == "mapped":
        return pair
    if config == "conf_high":
        return pair if parsed.get("pred_confidence") == "high" else det
    if config == "conf_medium_high":
        return pair if parsed.get("pred_confidence") in {"medium", "high"} else det
    if config == "should_override":
        return pair if parsed.get("pred_should_override") else det
    if config == "confidence_or_should_override":
        if parsed.get("pred_confidence") in {"medium", "high"} or parsed.get("pred_should_override"):
            return pair
        return det
    raise ValueError(f"Unknown config: {config}")


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _action_metrics(rows: Sequence[Dict[str, Any]], answer_key: str, scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    labels = sorted({str(r.get("gt_action") or "") for r in rows})
    out: Dict[str, Dict[str, float]] = {}
    for label in labels:
        if not label:
            continue
        gt_n = 0
        pred_n = 0
        tp = 0
        for row in rows:
            scenario = scenarios[str(row["scenario_id"])]
            pred_action = code_action(scenario, str(row.get(answer_key) or ""))
            gt_action = str(row.get("gt_action") or "")
            gt_n += int(gt_action == label)
            pred_n += int(pred_action == label)
            tp += int(gt_action == label and pred_action == label)
        out[label] = {
            "gt": gt_n,
            "pred": pred_n,
            "tp": tp,
            "precision": tp / pred_n if pred_n else 0.0,
            "recall": tp / gt_n if gt_n else 0.0,
        }
    return out


def evaluate_results(results_path: Path) -> Dict[str, Any]:
    scenarios = _scenario_map()
    all_f1 = read_jsonl(OUT_DIR / "all_f1_baseline.jsonl")
    raw_rows = read_jsonl(results_path)
    parsed = [_diagnosis_to_pair_answer(row, scenarios) for row in raw_rows if row.get("raw") or row.get("api_error")]
    parsed_by_id = {str(r["scenario_id"]): r for r in parsed}
    configs = [
        "mapped",
        "conf_high",
        "conf_medium_high",
        "should_override",
        "confidence_or_should_override",
    ]

    eval_rows: List[Dict[str, Any]] = []
    for row in all_f1:
        sid = str(row["scenario_id"])
        merged = dict(row)
        branch = parsed_by_id.get(sid)
        for config in configs:
            if branch:
                answer = _answer_for_config(branch, config)
            else:
                answer = str(row["deterministic_answer"])
            merged[f"{config}_answer"] = answer
            merged[f"{config}_iou"] = iou_score(answer, str(row["gt"]))
        eval_rows.append(merged)

    branch_eval = [r for r in parsed if r.get("split") == "test"]
    branch_baseline = _mean(float(r["det_iou"]) for r in branch_eval)
    config_summaries: Dict[str, Any] = {}
    test_all = [r for r in eval_rows if r.get("split") == "test"]
    test_baseline = _mean(float(r["baseline_iou"]) for r in test_all)
    for config in configs:
        branch_values = []
        changed = []
        helped = []
        harmed = []
        same = []
        for branch in branch_eval:
            answer = _answer_for_config(branch, config)
            new_iou = iou_score(answer, str(branch["gt"]))
            branch_values.append(new_iou)
            if answer != branch["deterministic_answer"]:
                changed.append(branch)
                if new_iou > float(branch["det_iou"]):
                    helped.append(branch)
                elif new_iou < float(branch["det_iou"]):
                    harmed.append(branch)
                else:
                    same.append(branch)
        full_iou = _mean(float(r[f"{config}_iou"]) for r in test_all)
        answer_key = f"{config}_answer"
        config_summaries[config] = {
            "branch_test_n": len(branch_eval),
            "branch_baseline_iou": branch_baseline,
            "branch_iou": _mean(branch_values),
            "branch_delta": _mean(branch_values) - branch_baseline,
            "full_f1_test_baseline_iou": test_baseline,
            "full_f1_test_iou": full_iou,
            "full_f1_test_delta": full_iou - test_baseline,
            "changed": len(changed),
            "helped": len(helped),
            "harmed": len(harmed),
            "same": len(same),
            "changed_gt_actions": dict(Counter(str(r.get("gt_action") or "") for r in changed)),
            "json_valid": sum(1 for r in branch_eval if r.get("json_valid")),
            "pair_ok": sum(1 for r in branch_eval if r.get("pair_ok")),
            "pred_actions": dict(Counter(str(r.get("pred_action") or "") for r in branch_eval)),
            "pred_confidence": dict(Counter(str(r.get("pred_confidence") or "") for r in branch_eval)),
            "action_metrics": _action_metrics(test_all, answer_key, scenarios),
        }

    summary = {
        "results_path": str(results_path),
        "raw_rows": len(raw_rows),
        "parsed_rows": len(parsed),
        "branch_gt_actions_all": dict(Counter(str(r.get("gt_action") or "") for r in parsed)),
        "branch_gt_actions_test": dict(Counter(str(r.get("gt_action") or "") for r in branch_eval)),
        "test_baseline_iou": test_baseline,
        "configs": config_summaries,
    }
    write_json(OUT_DIR / "gate_eval_summary.json", summary)
    write_jsonl(OUT_DIR / "gate_eval_rows.jsonl", eval_rows)
    write_report(summary)
    return summary


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(summary: Dict[str, Any]) -> None:
    lines = [
        "# V6.3 diag-LoRA Branch Gate",
        "",
        "## Scope",
        "- Only Phase-1 F1 TEST rows whose deterministic answer is lift_tilt are evaluated for LoRA replacement.",
        "- LoRA is allowed to choose only lift_tilt/insufficient_data; all other actions fallback to deterministic.",
        "- Non-branch rows remain byte-equivalent at the action level.",
        "",
        "## Input",
        f"- results: `{summary.get('results_path')}`",
        f"- raw rows: `{summary.get('raw_rows')}`",
        f"- parsed rows: `{summary.get('parsed_rows')}`",
        f"- TEST baseline F1 IOU: `{_fmt(summary.get('test_baseline_iou', 0.0))}`",
        f"- branch GT actions TEST: `{summary.get('branch_gt_actions_test')}`",
        "",
        "## Gate Results",
        "",
        "| config | branch IOU | branch delta | full TEST IOU | full delta | changed | helped | harmed | pair_ok |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, cfg in summary.get("configs", {}).items():
        lines.append(
            "| "
            + " | ".join([
                name,
                _fmt(cfg.get("branch_iou", 0.0)),
                _fmt(cfg.get("branch_delta", 0.0)),
                _fmt(cfg.get("full_f1_test_iou", 0.0)),
                _fmt(cfg.get("full_f1_test_delta", 0.0)),
                str(cfg.get("changed", 0)),
                str(cfg.get("helped", 0)),
                str(cfg.get("harmed", 0)),
                str(cfg.get("pair_ok", 0)),
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Pass Rule",
        "- PASS requires branch delta >= 0.05 and full F1 TEST delta > 0.",
        "- Prefer the narrowest passing gate; otherwise keep v6_2 unchanged.",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def phase1_subtype_rule_summary() -> Dict[str, Any]:
    prompts = read_jsonl(OUT_DIR / "branch_prompts.jsonl")
    decoder = json.JSONDecoder()
    by_split: Dict[str, Counter[str]] = defaultdict(Counter)
    by_subtype: Dict[str, Counter[str]] = defaultdict(Counter)
    acc: Dict[str, Dict[str, Any]] = {}
    for record in prompts:
        meta = record.get("meta") or {}
        split = str(meta.get("split") or "")
        gt_action = str(meta.get("gt_action") or "")
        user = next(
            (msg.get("content", "") for msg in record.get("messages", []) if msg.get("role") == "user"),
            "",
        )
        left = user.find("## Boundary Arbitration")
        right = user.find("## Similar Case Evidence Patterns")
        block = user[left:right] if left >= 0 and right > left else ""
        subtype = "missing"
        if "{" in block:
            obj, _ = decoder.raw_decode(block[block.find("{"):])
            subtype = str(((obj.get("top_lift_tilt_candidate") or {}).get("subtype")) or "missing")
        pred = "insufficient_data" if subtype == "generic_high_prb_primary_fallback" else "lift_tilt"
        by_split[split][f"n"] += 1
        by_split[split][f"hit"] += int(pred == gt_action)
        by_split[split][f"gt:{gt_action}"] += 1
        by_subtype[f"{split}:{subtype}"][gt_action] += 1
    for split, counts in by_split.items():
        n = counts["n"]
        hit = counts["hit"]
        acc[split] = {
            "n": n,
            "hits": hit,
            "accuracy": hit / n if n else 0.0,
            "gt_counts": {k[3:]: v for k, v in counts.items() if k.startswith("gt:")},
        }
    return {
        "accuracy": acc,
        "by_subtype": {k: dict(v) for k, v in sorted(by_subtype.items())},
    }


def structural_audit(rows: Sequence[Dict[str, str]], phase2: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    track_a = [row for row in rows if row.get("Track A")]
    track_b = [row for row in rows if row.get("Track B")]
    f1_illegal: List[Dict[str, Any]] = []
    f2_bad: List[Dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    for row in track_a:
        sid = row.get("ID", "")
        scenario = phase2.get(sid)
        if not scenario:
            continue
        answer = row.get("Track A", "")
        actions = answer_actions(scenario, answer)
        action_counts["|".join(actions)] += 1
        fam = family_name(scenario)
        if fam in {"F1", "F1_5BS"}:
            if len(actions) != 1 or actions[0] not in F1_ALLOWED_ACTIONS:
                f1_illegal.append({"id": sid, "answer": answer, "actions": actions, "family": fam})
        if fam == "F2":
            if len(actions) != 2 or frozenset(actions) not in F2_LEGAL_ACTION_SETS:
                f2_bad.append({"id": sid, "answer": answer, "actions": actions})
    override_status: Dict[str, str] = {}
    rows_by_id = {row.get("ID", ""): row for row in rows}
    for prefix, expected in LORA_OVERRIDES.items():
        match = next((sid for sid in rows_by_id if sid.startswith(prefix)), "")
        if not match:
            override_status[prefix] = "NOT_FOUND"
        else:
            actual = rows_by_id[match].get("Track A", "")
            override_status[prefix] = "OK" if actual == expected else f"MISMATCH:{actual}->{expected}"
    return {
        "rows": len(rows),
        "track_a_rows": len(track_a),
        "track_b_nonblank": len(track_b),
        "f1_illegal": len(f1_illegal),
        "f1_illegal_examples": f1_illegal[:10],
        "f2_bad": len(f2_bad),
        "f2_bad_examples": f2_bad[:10],
        "lora_overrides": override_status,
        "action_counts": dict(action_counts),
    }


def build_phase2_subtype_rule() -> Dict[str, Any]:
    if not (OUT_DIR / "branch_prompts.jsonl").exists():
        prepare_records()

    phase2 = {str(s.get("scenario_id")): s for s in load_json(PHASE2_JSON)}
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for line in PHASE2_DIAG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            diagnostics[str(row.get("scenario_id"))] = row

    headers, rows = read_submission(V6_2_SUBMISSION)
    changed: List[Dict[str, Any]] = []
    subtype_counts: Counter[str] = Counter()
    for row in rows:
        sid = str(row.get("ID") or "")
        answer = str(row.get("Track A") or "")
        scenario = phase2.get(sid)
        if not scenario or not answer or "|" in answer:
            continue
        if family_name(scenario) not in {"F1", "F1_5BS"}:
            continue
        ps = parse_scenario(scenario)
        action = str((ps.parsed_options.get(answer) or {}).get("action_type", ""))
        if action != "lift_tilt":
            continue
        diag = diagnostics.get(sid, {})
        subtype, first_lift, insuff_code = classify_lift_subtype(ps, diag.get("candidates") or [])
        subtype_counts[subtype] += 1
        if subtype != "generic_high_prb_primary_fallback" or not insuff_code:
            continue
        insuff_action = str((ps.parsed_options.get(insuff_code) or {}).get("action_type", ""))
        if insuff_action != "insufficient_data":
            continue
        row["Track A"] = insuff_code
        changed.append({
            "id": sid,
            "old": answer,
            "new": insuff_code,
            "family": family_name(scenario),
            "subtype": subtype,
            "reason": (first_lift or {}).get("reason"),
        })

    write_submission(V6_3_SUBTYPE_SUBMISSION, headers, rows)
    audit = structural_audit(rows, phase2)
    phase1_rule = phase1_subtype_rule_summary()
    raw54_path = ROOT / "results/phase2/f1_v5_20_audit/f1_54_ids.txt"
    raw54 = set(raw54_path.read_text(encoding="utf-8").split()) if raw54_path.exists() else set()
    changed_ids = {item["id"] for item in changed}
    summary = {
        "base": str(V6_2_SUBMISSION),
        "out": str(V6_3_SUBTYPE_SUBMISSION),
        "changed": len(changed),
        "changed_rows": changed,
        "phase2_lift_subtypes": dict(subtype_counts),
        "phase1_subtype_rule": phase1_rule,
        "audit": audit,
        "protected_raw54_intersection": sorted(changed_ids & raw54),
    }
    write_json(OUT_DIR / "phase2_subtype_rule_summary.json", summary)
    write_jsonl(OUT_DIR / "phase2_subtype_rule_changed.jsonl", changed)
    write_subtype_report(summary)
    return summary


def write_subtype_report(summary: Dict[str, Any]) -> None:
    phase1 = summary.get("phase1_subtype_rule", {})
    audit = summary.get("audit", {})
    lines = [
        "# V6.3 Subtype lift_tilt/insufficient_data Gate",
        "",
        "## Decision",
        "- Use the diagnostic SFT boundary subtype rule directly; no model call required.",
        "- Rule: generic high-PRB primary fallback -> insufficient_data; coverage tertiary hypothesis -> keep lift_tilt.",
        "- Scope: Phase2 F1/F1_5BS single-answer rows where v6_2 currently answers lift_tilt.",
        "",
        "## Phase1 Gate",
        f"- DEV: `{phase1.get('accuracy', {}).get('dev')}`",
        f"- TEST: `{phase1.get('accuracy', {}).get('test')}`",
        f"- subtype table: `{phase1.get('by_subtype')}`",
        "",
        "## Phase2 Application",
        f"- base: `{summary.get('base')}`",
        f"- output: `{summary.get('out')}`",
        f"- changed rows: `{summary.get('changed')}`",
        f"- Phase2 lift subtypes: `{summary.get('phase2_lift_subtypes')}`",
        f"- protected raw54 intersection: `{summary.get('protected_raw54_intersection')}`",
        "",
        "## Structural Audit",
        f"- rows: `{audit.get('rows')}`",
        f"- Track A rows: `{audit.get('track_a_rows')}`",
        f"- Track B nonblank: `{audit.get('track_b_nonblank')}`",
        f"- F1 illegal: `{audit.get('f1_illegal')}`",
        f"- F2 bad: `{audit.get('f2_bad')}`",
        f"- LoRA overrides: `{audit.get('lora_overrides')}`",
        "",
        "## Changed Rows",
        "",
        "| id | old | new | reason |",
        "|---|---:|---:|---|",
    ]
    for row in summary.get("changed_rows", []):
        lines.append(
            f"| {row.get('id')} | {row.get('old')} | {row.get('new')} | {row.get('reason')} |"
        )
    V6_3_SUBTYPE_MD.parent.mkdir(parents=True, exist_ok=True)
    V6_3_SUBTYPE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare", action="store_true", help="Build branch prompt JSONL and baseline rows.")
    ap.add_argument("--run-model", action="store_true", help="Call local diag model for branch prompts.")
    ap.add_argument("--evaluate", action="store_true", help="Evaluate cached model results.")
    ap.add_argument("--build-phase2-subtype-rule", action="store_true", help="Build no-model v6_3 using the validated boundary subtype rule.")
    ap.add_argument("--model-url", default=DEFAULT_MODEL_URL)
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--split-filter", choices=["all", "dev", "test"], default="all")
    ap.add_argument("--prompts", type=Path, default=OUT_DIR / "branch_prompts.jsonl")
    ap.add_argument("--results", type=Path, default=OUT_DIR / "diag_lora_branch_results.jsonl")
    args = ap.parse_args()

    if not any([args.prepare, args.run_model, args.evaluate, args.build_phase2_subtype_rule]):
        args.prepare = True

    if args.prepare:
        summary = prepare_records(limit=args.limit)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.run_model:
        summary = call_model(
            prompts_path=args.prompts,
            out_path=args.results,
            model_url=args.model_url,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            limit=args.limit,
            resume=args.resume,
            split_filter=args.split_filter,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.evaluate:
        summary = evaluate_results(args.results)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.build_phase2_subtype_rule:
        summary = build_phase2_subtype_rule()
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
