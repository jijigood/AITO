#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Evaluate a local PEFT LoRA checkpoint without vLLM.

This is for Qwen3.5 MoE checkpoints that current vLLM cannot serve. It mirrors
the F1 path in main_v3.py, but calls Transformers + PEFT directly.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from main_v3 import (
    Environment,
    QWEN_V3_SYSTEM_PROMPT,
    _build_f1_prompt,
    _retrieve_rag_hits,
)
from skills.answer_parser import iou_score, parse_answer
from skills.candidate_generator import generate_candidates
from skills.feature_engine import run_all_skills
from skills.scenario_parser import parse_scenario
from skills.scenario_router import route
from skills.tool_caller import ToolCaller, ToolEvidence


def _resolve_train_json(train_path: Path) -> Path:
    if train_path.is_file():
        return train_path
    candidate = train_path / "train.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot resolve train.json from {train_path}")


def _get_action_type(scenario: dict) -> Optional[str]:
    try:
        ps = parse_scenario(scenario)
        gt = scenario.get("answer", "")
        opt = ps.parsed_options.get(gt, {})
        return opt.get("action_type")
    except Exception:
        return None


class LocalLoraGenerator:
    def __init__(
        self,
        base_model: str,
        adapter: str,
        max_memory: Optional[Dict[int, str]] = None,
    ) -> None:
        tokenizer_path = adapter if Path(adapter).exists() else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        first_device = self.model.get_input_embeddings().weight.device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _structured_candidates(candidates: Any) -> List[Any]:
    if not candidates or not candidates.single_candidates:
        return []
    return [
        (cand.c_code, cand.score, cand.reason)
        for cand in candidates.single_candidates
    ]


def _solve_one(
    scenario: Dict[str, Any],
    generator: LocalLoraGenerator,
    rag: Any,
    tool_caller: Optional[ToolCaller],
    max_tokens: int,
    rag_top_k: int,
) -> Dict[str, Any]:
    t0 = time.time()
    scenario_id = scenario.get("scenario_id", "?")
    ps = parse_scenario(scenario)
    family = route(ps.tag, ps.num_base_stations)
    sf = run_all_skills(ps)
    candidates = generate_candidates(ps, sf, family)

    if family.name != "F1":
        answer = candidates.top_candidate() if candidates else ""
        return {
            "scenario_id": scenario_id,
            "answer": answer,
            "family": family.name,
            "raw": "",
            "latency_s": round(time.time() - t0, 2),
            "mode": "deterministic_non_f1",
        }

    if tool_caller is not None:
        evidence = ToolEvidence()
        try:
            evidence = tool_caller.call_targeted(
                scenario_id,
                sf,
                sf.throughput.worst_time,
            )
            if evidence.tool_call_count > 0:
                sf.tool_evidence = evidence
                sf = tool_caller.augment_features(sf, evidence)
                sf.tool_evidence = evidence
                candidates = generate_candidates(ps, sf, family)
        except Exception:
            pass

    top_answer = candidates.top_candidate() if candidates else ""
    rag_hits = []
    if rag is not None:
        rag_hits = _retrieve_rag_hits(
            rag,
            sf,
            family.name,
            top_k=rag_top_k,
            exclude_scenario_id=scenario_id,
        )
    user_msg = _build_f1_prompt(
        ps,
        sf,
        _structured_candidates(candidates),
        rag_hits,
    )
    raw = generator.generate(
        [
            {"role": "system", "content": QWEN_V3_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_new_tokens=max_tokens,
    )
    answer = parse_answer(raw) or top_answer
    return {
        "scenario_id": scenario_id,
        "answer": answer,
        "family": family.name,
        "raw": raw,
        "latency_s": round(time.time() - t0, 2),
        "mode": "local_lora",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="/home/huangxiaolin/AITO/model")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--val-ids", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--action-filter", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--rag-top-k", type=int, default=3)
    ap.add_argument("--use-tools", action="store_true")
    ap.add_argument("--server-url", default="http://localhost:7861")
    ap.add_argument("--mr-sample-threshold", type=int, default=7)
    ap.add_argument("--max-memory", nargs="*", default=["0:76GiB", "1:76GiB"])
    ap.add_argument("--out-jsonl", type=Path, default=None)
    args = ap.parse_args()

    max_memory: Dict[int, str] = {}
    for item in args.max_memory:
        idx, value = item.split(":", 1)
        max_memory[int(idx)] = value

    val_ids: Set[str] = {
        line.strip()
        for line in args.val_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    train_json = _resolve_train_json(args.train)
    scenarios = json.loads(train_json.read_text(encoding="utf-8"))
    scenarios = [s for s in scenarios if s.get("scenario_id") in val_ids]
    if args.action_filter:
        wanted = set(args.action_filter)
        scenarios = [s for s in scenarios if _get_action_type(s) in wanted]
        print(f"Filtered actions: {sorted(wanted)} -> {len(scenarios)} scenarios")
    if args.limit is not None:
        scenarios = scenarios[:args.limit]
    print(f"Evaluating {len(scenarios)} scenarios")

    rag = None
    if not args.no_rag:
        from rag_retriever_v2 import StructuredCaseRAG

        rag = StructuredCaseRAG(str(train_json))
        rag.build_index()
        print("RAG index built")

    tool_caller = None
    if args.use_tools:
        tool_caller = ToolCaller(
            args.server_url,
            mr_sample_threshold=args.mr_sample_threshold,
        )
        print(f"Tools enabled: {args.server_url}")

    print(f"Loading base: {args.base_model}")
    print(f"Loading adapter: {args.adapter}")
    generator = LocalLoraGenerator(
        args.base_model,
        args.adapter,
        max_memory=max_memory or None,
    )

    per_action: Dict[str, List[float]] = defaultdict(list)
    rows: List[Dict[str, Any]] = []
    for idx, scenario in enumerate(scenarios, 1):
        gt = (scenario.get("answer") or "").strip()
        if not gt or gt == "To be determined":
            continue
        result = _solve_one(
            scenario,
            generator,
            rag,
            tool_caller,
            max_tokens=args.max_tokens,
            rag_top_k=args.rag_top_k,
        )
        pred = result["answer"]
        iou = iou_score(pred, gt)
        action = _get_action_type(scenario) or "unknown"
        per_action[action].append(iou)
        row = {
            "scenario_id": scenario.get("scenario_id"),
            "gt": gt,
            "pred": pred,
            "iou": iou,
            "action": action,
            **result,
        }
        rows.append(row)
        print(
            f"[{idx:03d}/{len(scenarios):03d}] {action:<20} "
            f"iou={iou:.3f} pred={pred} gt={gt} raw={result['raw'][:80]!r}"
        )

    if args.out_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    all_ious = [r["iou"] for r in rows]
    mean = sum(all_ious) / max(len(all_ious), 1)
    print("\n" + "=" * 62)
    print(f"Overall IOU: {mean:.4f}  n={len(all_ious)}")
    for action, vals in sorted(per_action.items()):
        print(f"{action:<22} n={len(vals):3d} iou={sum(vals) / len(vals):.4f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
