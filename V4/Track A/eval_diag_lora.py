#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Evaluate a diagnostic-JSON LoRA checkpoint served by vLLM/OpenAI API.

This is deliberately different from the old C-code LoRA evaluator.  The model
is expected to output diagnosis JSON, which is then mapped to scenario-local
C-codes only for measuring heldout IOU and gated arbiter behavior.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import httpx
from openai import OpenAI

from skills.answer_parser import iou_score
from skills.candidate_generator import generate_candidates
from skills.feature_engine import run_all_skills
from skills.scenario_parser import parse_scenario
from skills.scenario_router import route


GT_ACTION_GATES: Dict[str, float] = {
    "lift_tilt": 0.70,
    "insufficient_data": 0.70,
    "add_neighbor": 0.65,
}
OVERALL_GATE = 0.75
JSON_VALID_GATE = 0.98
C_CODE_RE = re.compile(r"\bC(?:[1-9]|1\d|2[0-4])\b")


def _is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_train_json(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "train.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot resolve train.json from {path}")


def _scenario_id(record: Dict[str, Any]) -> str:
    meta = record.get("meta") or {}
    if meta.get("scenario_id"):
        return str(meta["scenario_id"])
    for msg in record.get("messages") or []:
        if msg.get("role") == "user":
            match = re.search(r'"scenario_id"\s*:\s*"([^"]+)"', msg.get("content", ""))
            if match:
                return match.group(1)
    return ""


def _assistant_label(record: Dict[str, Any]) -> Dict[str, Any]:
    for msg in record.get("messages") or []:
        if msg.get("role") == "assistant":
            try:
                obj = json.loads(msg.get("content") or "{}")
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def _prompt_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for msg in record.get("messages") or []:
        role = msg.get("role")
        if role in {"system", "user"}:
            messages.append({"role": role, "content": msg.get("content", "")})
    return messages


def _extract_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    text = (text or "").strip()
    if not text:
        return None, "empty"
    if C_CODE_RE.search(text):
        # Not automatically invalid, but it is a prompt-contract violation.
        pass
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj, None
            return None, "json_not_object"
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
    return None, "no_json_object"


def _normalize_action(action: Any) -> str:
    return str(action or "").strip().lower().replace("_", "").replace(" ", "")


def _map_diagnosis_to_ccode(
    diagnosis: Dict[str, Any],
    ps: Any,
    deterministic_answer: str,
) -> Tuple[str, bool, str, float]:
    action = str(diagnosis.get("action") or "").strip()
    target_cell_id = str(diagnosis.get("target_cell_id") or "").strip()
    relation_cell_id = str(diagnosis.get("relation_cell_id") or "").strip()
    target_pci = diagnosis.get("target_pci")
    relation_pci = diagnosis.get("relation_pci")

    if not target_cell_id and target_pci:
        try:
            target_cell_id = ps.pci_to_cell_id.get(int(target_pci), "")
        except (TypeError, ValueError):
            pass
    if not relation_cell_id and relation_pci:
        try:
            relation_cell_id = ps.pci_to_cell_id.get(int(relation_pci), "")
        except (TypeError, ValueError):
            pass

    action_norm = _normalize_action(action)
    best_cid: Optional[str] = None
    best_score = -1.0
    matched_action_count = 0

    for cid, opt in ps.parsed_options.items():
        opt_action = str(opt.get("action_type") or opt.get("action") or "").strip()
        opt_cells: List[str] = opt.get("cells") or []
        opt_norm = _normalize_action(opt_action)
        if not action_norm or not (action_norm in opt_norm or opt_norm in action_norm):
            continue
        matched_action_count += 1
        score = 1.0
        if action_norm == "addneighbor":
            if target_cell_id and target_cell_id in opt_cells:
                score += 2.0
            if relation_cell_id and relation_cell_id in opt_cells:
                score += 2.0
        elif target_cell_id and target_cell_id in opt_cells:
            score += 3.0
        if score > best_score:
            best_score = score
            best_cid = cid

    no_target_actions = {"insufficientdata", "checkserver"}
    if action_norm in no_target_actions:
        min_score = 1.0
    elif action_norm == "addneighbor":
        min_score = 3.0
    else:
        min_score = 1.0 if matched_action_count == 1 and not target_cell_id else 4.0

    if best_cid and best_score >= min_score:
        return best_cid, True, "mapped", best_score
    return deterministic_answer, False, "fallback", best_score


class OpenAIDiagClient:
    def __init__(self, model_url: str, model_name: str, timeout: float) -> None:
        self.model_name = model_name
        self.client = OpenAI(
            base_url=model_url,
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
            timeout=timeout,
            http_client=httpx.Client(
                verify=False,
                timeout=timeout,
                trust_env=not _is_local_url(model_url),
            ),
        )

    def call(self, messages: Sequence[Dict[str, str]], max_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=list(messages),
            temperature=0,
            max_tokens=max_tokens,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
        )
        msg = resp.choices[0].message
        return (
            msg.content
            or getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None)
            or ""
        ).strip()


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / max(len(vals), 1)


def _gt_action(ps: Any, gt: str) -> str:
    return str(ps.parsed_options.get(gt, {}).get("action_type") or "unknown")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate diagnostic JSON LoRA heldout gate")
    ap.add_argument("--val-jsonl", type=Path, default=Path("sft_diag_val.jsonl"))
    ap.add_argument("--train", type=Path, default=Path("/home/huangxiaolin/AITO/dataset/Track A/data/Phase_1/train.json"))
    ap.add_argument("--model-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model-name", default="diag")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--action-filter", nargs="+", default=None)
    ap.add_argument("--gate-confidence", nargs="+", default=["high"])
    ap.add_argument(
        "--gate-mode",
        choices=["confidence", "should_override", "confidence_or_should_override"],
        default="confidence",
        help="How to decide whether mapped LoRA output replaces deterministic top1.",
    )
    ap.add_argument("--out-jsonl", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    records = _load_jsonl(args.val_jsonl)
    train_json = _resolve_train_json(args.train)
    scenarios = {
        str(item.get("scenario_id")): item
        for item in json.loads(train_json.read_text(encoding="utf-8"))
    }

    selected: List[Tuple[Dict[str, Any], Dict[str, Any], Any, Any, str]] = []
    wanted_actions = set(args.action_filter or [])
    for record in records:
        sid = _scenario_id(record)
        scenario = scenarios.get(sid)
        if not scenario:
            continue
        ps = parse_scenario(scenario)
        gt = str(scenario.get("answer") or "").strip()
        action = _gt_action(ps, gt)
        if wanted_actions and action not in wanted_actions:
            continue
        family = route(ps.tag, ps.num_base_stations)
        selected.append((record, scenario, ps, family, action))

    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("No heldout records selected.")

    client = OpenAIDiagClient(args.model_url, args.model_name, args.timeout)
    gate_conf = {str(item).strip().lower() for item in args.gate_confidence}

    rows: List[Dict[str, Any]] = []
    counters: Counter[str] = Counter()
    by_action: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for idx, (record, scenario, ps, family, gt_action) in enumerate(selected, 1):
        sid = _scenario_id(record)
        gt = str(scenario.get("answer") or "").strip()
        sf = run_all_skills(ps)
        candidates = generate_candidates(ps, sf, family)
        deterministic_answer = candidates.top_candidate() if candidates else ""
        det_iou = iou_score(deterministic_answer, gt)

        label = _assistant_label(record)
        raw = ""
        error = None
        try:
            raw = client.call(_prompt_messages(record), max_tokens=args.max_tokens)
        except Exception as exc:
            error = str(exc)

        diagnosis, json_error = _extract_json_object(raw)
        json_valid = diagnosis is not None and json_error is None
        if not json_valid:
            diagnosis = {}

        mapped_answer, mapped_ok, map_status, map_score = _map_diagnosis_to_ccode(
            diagnosis,
            ps,
            deterministic_answer,
        )
        confidence = str(diagnosis.get("evidence_confidence") or "").strip().lower()
        pred_should_override = _as_bool(diagnosis.get("should_override"))
        label_should_override = _as_bool(label.get("should_override"))
        accept_by_confidence = bool(confidence in gate_conf)
        accept_by_should_override = bool(pred_should_override)
        if args.gate_mode == "should_override":
            gate_accept = accept_by_should_override
        elif args.gate_mode == "confidence_or_should_override":
            gate_accept = accept_by_confidence or accept_by_should_override
        else:
            gate_accept = accept_by_confidence
        accept = bool(json_valid and mapped_ok and gate_accept)
        gated_answer = mapped_answer if accept else deterministic_answer

        row = {
            "idx": idx,
            "scenario_id": sid,
            "gt": gt,
            "gt_action": gt_action,
            "label_action": label.get("action"),
            "pred_action": diagnosis.get("action"),
            "pred_confidence": confidence,
            "label_should_override": label_should_override,
            "pred_should_override": pred_should_override,
            "should_override_match_label": pred_should_override == label_should_override,
            "json_valid": json_valid,
            "json_error": json_error,
            "c_code_leak": bool(C_CODE_RE.search(raw or "")),
            "api_error": error,
            "deterministic_answer": deterministic_answer,
            "mapped_answer": mapped_answer,
            "gated_answer": gated_answer,
            "mapped_ok": mapped_ok,
            "map_status": map_status,
            "map_score": map_score,
            "det_iou": det_iou,
            "mapped_iou": iou_score(mapped_answer, gt),
            "gated_iou": iou_score(gated_answer, gt),
            "action_match_label": diagnosis.get("action") == label.get("action"),
            "raw": raw,
        }
        rows.append(row)
        by_action[gt_action].append(row)
        counters["json_valid"] += int(row["json_valid"])
        counters["mapped_ok"] += int(row["mapped_ok"])
        counters["accepted"] += int(accept)
        counters["label_should_override"] += int(label_should_override)
        counters["pred_should_override"] += int(pred_should_override)
        counters["should_override_tp"] += int(pred_should_override and label_should_override)
        counters["should_override_fp"] += int(pred_should_override and not label_should_override)
        counters["should_override_fn"] += int((not pred_should_override) and label_should_override)
        counters["c_code_leak"] += int(row["c_code_leak"])
        counters["api_error"] += int(bool(error))

        print(
            f"[{idx:03d}/{len(selected):03d}] {gt_action:<18} "
            f"det={row['det_iou']:.0f} map={row['mapped_iou']:.0f} gate={row['gated_iou']:.0f} "
            f"json={int(json_valid)} conf={confidence or '-':<6} "
            f"ovr={int(pred_should_override)} pred={str(diagnosis.get('action') or '-'):<18} "
            f"accept={int(accept)}",
            flush=True,
        )

    if args.out_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    n = len(rows)
    det_overall = _mean(row["det_iou"] for row in rows)
    mapped_overall = _mean(row["mapped_iou"] for row in rows)
    gated_overall = _mean(row["gated_iou"] for row in rows)
    json_valid_rate = counters["json_valid"] / max(n, 1)
    override_precision = counters["should_override_tp"] / max(
        counters["should_override_tp"] + counters["should_override_fp"], 1
    )
    override_recall = counters["should_override_tp"] / max(
        counters["should_override_tp"] + counters["should_override_fn"], 1
    )

    print("\n" + "=" * 72)
    print(f"Evaluated                 : {n}")
    print(f"JSON valid                : {json_valid_rate:.3f}  gate>={JSON_VALID_GATE}")
    print(f"Mapped OK                 : {counters['mapped_ok']}/{n}")
    print(f"Accepted by gate          : {counters['accepted']}/{n}  mode={args.gate_mode} conf={sorted(gate_conf)}")
    print(f"Label should_override     : {counters['label_should_override']}/{n}")
    print(f"Pred should_override      : {counters['pred_should_override']}/{n}")
    print(f"should_override precision : {override_precision:.4f}")
    print(f"should_override recall    : {override_recall:.4f}")
    print(f"C-code leak               : {counters['c_code_leak']}/{n}")
    print(f"API errors                : {counters['api_error']}/{n}")
    print(f"Deterministic IOU         : {det_overall:.4f}")
    print(f"Mapped LoRA IOU           : {mapped_overall:.4f}")
    print(f"Gated arbiter IOU         : {gated_overall:.4f}  gate>={OVERALL_GATE}")
    print("\nPer GT action:")
    for action in sorted(by_action):
        vals = by_action[action]
        det = _mean(row["det_iou"] for row in vals)
        mapped = _mean(row["mapped_iou"] for row in vals)
        gated = _mean(row["gated_iou"] for row in vals)
        action_acc = _mean(float(row["pred_action"] == row["label_action"]) for row in vals)
        override_acc = _mean(float(row["should_override_match_label"]) for row in vals)
        override_recall_action = (
            sum(1 for row in vals if row["pred_should_override"] and row["label_should_override"])
            / max(sum(1 for row in vals if row["label_should_override"]), 1)
        )
        gate = GT_ACTION_GATES.get(action)
        gate_str = f" gate>={gate:.2f}" if gate is not None else ""
        print(
            f"  {action:<20} n={len(vals):3d} det={det:.4f} "
            f"map={mapped:.4f} gated={gated:.4f} action_acc={action_acc:.4f} "
            f"override_acc={override_acc:.4f} override_recall={override_recall_action:.4f}{gate_str}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
