#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
eval_ckpt.py — Held-out F1 evaluation for A3/A4/A5 checkpoints.

Runs the 200 heldout F1 scenarios through the actual LoRA model via
vLLM and reports per-action IOU against the promotion gates defined
in TEAM_DIVISION_CN.md §2.6.

NOTE: The RAG index is built from the full train.json, but `main_v3.py`
now excludes the current `scenario_id` during retrieval, so heldout
scenarios do not retrieve themselves.

Usage:
  # Deterministic baseline (no model needed):
  python eval_ckpt.py \\
      --val-ids  heldout_f1_val_ids.txt \\
      --train    train_data \\
      --deterministic

  # With vLLM serving the LoRA adapter:
  python eval_ckpt.py \\
      --val-ids   heldout_f1_val_ids.txt \\
      --train     train_data \\
      --model-url http://localhost:8001/v1 \\
      --model-name qwen

  # Focused tool-caller eval on the two stuck F1 actions:
  python eval_ckpt.py \\
      --val-ids heldout_f1_val_ids.txt \\
      --train   train_data \\
      --model-url  http://localhost:8001/v1 \\
      --server-url http://localhost:7861 \\
      --use-tools \\
      --action-filter lift_tilt insufficient_data

  # Compare three checkpoints sequentially:
  for ckpt in checkpoint-300 checkpoint-400 checkpoint-477; do
    echo "=== $ckpt ===" && python eval_ckpt.py \\
      --val-ids heldout_f1_val_ids.txt --train train_data \\
      --model-url http://localhost:8001/v1 --model-name qwen
  done
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

from skills.answer_parser import iou_score
from skills.scenario_parser import parse_scenario
from skills.tool_caller import ToolCaller
from main_v3 import V3Runner, Environment

# Promotion gates from TEAM_DIVISION_CN.md §2.6
PROMOTION_GATES: Dict[str, float] = {
    "lift_tilt":         0.60,
    "insufficient_data": 0.60,
    "add_neighbor":      0.65,
}
OVERALL_F1_GATE = 0.75
LATENCY_GATE_S  = 90.0


def _resolve_train_json(train_path: Path) -> Path:
    if train_path.is_file():
        return train_path
    candidate = train_path / "train.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot resolve train.json from {train_path}")


def _get_action_type(scenario: dict) -> Optional[str]:
    """Return the action_type of the ground-truth answer option."""
    try:
        ps = parse_scenario(scenario)
        gt = scenario.get("answer", "")
        opt = ps.parsed_options.get(gt, {})
        return opt.get("action_type")
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on heldout F1 val set")
    ap.add_argument("--val-ids",      type=Path,  required=True,
                    help="Path to heldout_f1_val_ids.txt")
    ap.add_argument("--train",        type=Path,  required=True,
                    help="Path to train.json or directory containing it")
    ap.add_argument("--model-url",    type=str,   default="http://localhost:8001/v1",
                    help="vLLM OpenAI-compatible endpoint")
    ap.add_argument("--model-name",   type=str,   default="qwen")
    ap.add_argument("--server-url",   type=str,   default="http://localhost:7860",
                    help="FastAPI tool server endpoint")
    ap.add_argument("--rag-top-k",    type=int,   default=3)
    ap.add_argument("--no-rag",       action="store_true")
    ap.add_argument("--use-tools",    action="store_true",
                    help="Enable server tool calls during F1 solve()")
    ap.add_argument("--mr-sample-threshold", type=int, default=7,
                    help="Threshold for tool-driven insufficient_data boost")
    ap.add_argument("--action-filter", nargs="+", default=None,
                    help="Restrict evaluation to heldout examples whose GT action_type matches one of these values")
    ap.add_argument("--deterministic", action="store_true",
                    help="Skip Qwen, use top candidate only (fast baseline)")
    # A3 checkpoints often spend >256 tokens on rationale before emitting the
    # final boxed answer; 768 avoids systematic fallback-to-deterministic.
    ap.add_argument("--max-tokens",   type=int,   default=768)
    args = ap.parse_args()

    val_ids: Set[str] = set(
        line.strip()
        for line in args.val_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    train_json = _resolve_train_json(args.train)
    with open(train_json, encoding="utf-8") as f:
        all_scenarios = json.load(f)

    val_scenarios = [s for s in all_scenarios if s.get("scenario_id") in val_ids]
    if args.action_filter:
        wanted_actions = {a.strip() for a in args.action_filter if a.strip()}
        val_scenarios = [
            s for s in val_scenarios
            if _get_action_type(s) in wanted_actions
        ]
        print(
            "Filtered actions: "
            + ", ".join(sorted(wanted_actions))
            + f"  -> {len(val_scenarios)} scenarios"
        )
    if not val_scenarios:
        print(f"[ERROR] No heldout scenarios found. Check --val-ids path.")
        return
    print(f"Heldout val scenarios: {len(val_scenarios)}")

    rag = None
    if not args.no_rag and not args.deterministic:
        from rag_retriever_v2 import StructuredCaseRAG
        rag = StructuredCaseRAG(str(train_json))
        rag.build_index()
        print("RAG index built.")

    tool_caller = None
    if args.use_tools:
        tool_caller = ToolCaller(
            args.server_url,
            mr_sample_threshold=args.mr_sample_threshold,
        )
        print(
            f"Tools enabled via {args.server_url} "
            f"(mr_sample_threshold={args.mr_sample_threshold})"
        )

    env = Environment(server_url=args.server_url)  # unused by solve()
    runner = V3Runner(
        environment=env,
        model_url="" if args.deterministic else args.model_url,
        model_name=args.model_name,
        rag_v2=rag,
        rag_top_k=args.rag_top_k,
        tool_caller=tool_caller,
        max_tokens=args.max_tokens,
        deterministic=args.deterministic,
    )

    per_action_ious: Dict[str, List[float]] = defaultdict(list)
    all_f1_ious: List[float] = []
    total_latency = 0.0
    n_skipped = 0

    for scenario in val_scenarios:
        gt = (scenario.get("answer") or "").strip()
        if not gt or gt == "To be determined":
            n_skipped += 1
            continue

        result = runner.solve(scenario)
        pred    = result["answer"]
        iou     = iou_score(pred, gt)
        latency = result.get("latency_s", 0.0)
        total_latency += latency

        all_f1_ious.append(iou)
        action = _get_action_type(scenario)
        if action:
            per_action_ious[action].append(iou)

    n_eval = len(all_f1_ious)
    if n_eval == 0:
        print("[ERROR] No evaluable scenarios.")
        return

    overall_f1  = sum(all_f1_ious) / n_eval
    avg_latency = total_latency / n_eval

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Overall F1 IOU : {overall_f1:.4f}  "
          f"(gate ≥{OVERALL_F1_GATE}  "
          f"{'✓ PASS' if overall_f1 >= OVERALL_F1_GATE else '✗ FAIL'})")
    print(f"  Avg latency    : {avg_latency:.1f}s  "
          f"(gate ≤{LATENCY_GATE_S:.0f}s  "
          f"{'✓ PASS' if avg_latency <= LATENCY_GATE_S else '✗ FAIL'})")
    print(f"  Evaluated      : {n_eval}  (skipped: {n_skipped})")
    print(f"\n  Per-action IOU:")
    for action in sorted(per_action_ious.keys()):
        ious  = per_action_ious[action]
        mean  = sum(ious) / len(ious)
        gate  = PROMOTION_GATES.get(action)
        if gate is not None:
            gstr = f"  gate ≥{gate}  {'✓ PASS' if mean >= gate else '✗ FAIL'}"
        else:
            gstr = ""
        print(f"    {action:<22}  n={len(ious):3d}  iou={mean:.4f}{gstr}")
    print(f"{'='*62}\n")

    # ── Gate summary ──────────────────────────────────────────────────────────
    gates: List[bool] = [
        overall_f1  >= OVERALL_F1_GATE,
        avg_latency <= LATENCY_GATE_S,
    ]
    for action, threshold in PROMOTION_GATES.items():
        ious = per_action_ious.get(action, [])
        mean = sum(ious) / max(len(ious), 1)
        gates.append(mean >= threshold)

    if all(gates):
        print("ALL PROMOTION GATES PASSED — checkpoint qualifies for A3 promotion")
    else:
        n_failed = sum(1 for g in gates if not g)
        print(f"{n_failed} PROMOTION GATE(S) FAILED — review per-action breakdown above")


if __name__ == "__main__":
    main()
