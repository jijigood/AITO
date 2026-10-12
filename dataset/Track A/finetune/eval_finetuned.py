#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Accuracy check: run the fine-tuned V2 model on N training examples
and compare against ground-truth answers.

Loads the base model + LoRA adapter directly (no vLLM needed).

Usage:
    cd "Track A"
    CUDA_VISIBLE_DEVICES=0 python finetune/eval_finetuned.py \
        --n_samples 20 --seed 0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np

TRACK_A_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(TRACK_A_DIR))

from main_v2 import (
    QWEN_V2_SYSTEM_PROMPT,
    _build_qwen_prompt,
    _sf_to_feature_vec,
)
from skills.answer_parser import iou_score, parse_answer, validate_and_normalise
from skills.candidate_generator import generate_candidates
from skills.feature_engine import run_all_skills
from skills.scenario_parser import parse_scenario
from skills.scenario_router import route
from rag_retriever_v2 import StructuredCaseRAG

ADAPTER_PATH = str(TRACK_A_DIR / "finetune" / "lora_output_v2")
BASE_MODEL   = "/home/huangxiaolin/AITO/model"
TRAIN_JSON   = str(TRACK_A_DIR / "data" / "Phase_1" / "train.json")


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(base_model: str, adapter_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading tokenizer from {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    print(f"Loading base model from {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print("Model ready.")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def call_model(
    model,
    tokenizer,
    messages: List[Dict[str, Any]],
    max_new_tokens: int = 64,
) -> Optional[str]:
    """Apply chat template and run greedy decoding with thinking disabled."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # suppress chain-of-thought preamble
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Strip residual <think>...</think> block if present
    import re
    decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()
    return decoded


# ──────────────────────────────────────────────────────────────────────────────
# V2 pipeline (offline, no HTTP)
# ──────────────────────────────────────────────────────────────────────────────

def run_v2_pipeline(
    scenario: Dict[str, Any],
    model,
    tokenizer,
    rag_v2: Optional[StructuredCaseRAG],
    verbose: bool = False,
) -> Dict[str, Any]:
    sid = scenario.get("scenario_id", "?")
    t0 = time.time()

    # Parse → Route → Skills → Candidates
    try:
        ps = parse_scenario(scenario)
        family = route(ps.tag, ps.num_base_stations)
        sf = run_all_skills(ps)
    except Exception as e:
        return {"scenario_id": sid, "answer": None, "status": f"pipeline_err: {e}", "latency_s": 0}

    try:
        candidates = generate_candidates(ps, sf, family)
        candidates_text = candidates.format_for_prompt()
        top_cand = candidates.top_candidate()
        top_score = (
            candidates.single_candidates[0].score if (not family.is_multiple and candidates.single_candidates)
            else (candidates.multi_candidates[0].score if (family.is_multiple and candidates.multi_candidates) else 0.0)
        )
    except Exception:
        candidates_text, top_cand, top_score = "", None, 0.0

    # RAG retrieval
    rag_ctx = ""
    if rag_v2 is not None:
        try:
            fvec = _sf_to_feature_vec(sf)
            rag_ctx = rag_v2.retrieve_and_format(fvec, family.name, top_k=3)
        except Exception:
            pass

    # Build prompt
    user_msg = _build_qwen_prompt(scenario, family.name, sf, candidates_text, rag_ctx)
    messages = [
        {"role": "system", "content": QWEN_V2_SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    # Model call
    raw = call_model(model, tokenizer, messages)
    answer = parse_answer(raw) if raw else None

    if verbose:
        print(f"  [{sid}] raw='{raw[:60]}' → parsed='{answer}'")

    # Fallback to top candidate if no parseable answer
    status = "ok"
    if not answer:
        answer = top_cand
        status = "fallback_to_candidate"

    latency = time.time() - t0
    return {
        "scenario_id": sid,
        "answer": answer,
        "raw_response": raw,
        "top_candidate": top_cand,
        "top_cand_score": top_score,
        "status": status,
        "latency_s": round(latency, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of training examples to evaluate")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--no_rag",    action="store_true")
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--train_json", default=TRAIN_JSON)
    parser.add_argument("--adapter",    default=ADAPTER_PATH)
    parser.add_argument("--base_model", default=BASE_MODEL)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Fine-tuned V2 Model Accuracy Check")
    print(f"  base:    {args.base_model}")
    print(f"  adapter: {args.adapter}")
    print(f"  samples: {args.n_samples}  seed: {args.seed}")
    print(f"{'='*60}\n")

    # Load training data
    with open(args.train_json, encoding="utf-8") as f:
        all_examples = json.load(f)

    import random
    random.seed(args.seed)
    # Sample uniformly — avoid the examples used for validation (last 200 after shuffle)
    # Since prepare_data_v2 shuffled with seed=42, just pick random training examples
    sample = random.sample(all_examples, min(args.n_samples, len(all_examples)))
    print(f"Sampled {len(sample)} examples from {len(all_examples)} total.\n")

    # Build RAG index
    rag_v2 = None
    if not args.no_rag:
        print("Building RAG index from training data...")
        try:
            rag_v2 = StructuredCaseRAG(args.train_json)
            rag_v2.build_index()
            print("RAG index ready.\n")
        except Exception as e:
            print(f"RAG init failed ({e}), running without RAG.\n")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter)
    print()

    # Run evaluation
    results = []
    iou_scores = []
    fallback_count = 0

    for i, scenario in enumerate(sample):
        gt = validate_and_normalise(scenario.get("answer", ""))
        sid = scenario.get("scenario_id", "?")

        print(f"[{i+1:2d}/{len(sample)}] {sid}  gt={gt}", end="  ", flush=True)

        result = run_v2_pipeline(scenario, model, tokenizer, rag_v2, verbose=args.verbose)
        pred = result["answer"]
        iou = iou_score(gt, pred) if (gt and pred) else 0.0

        if result["status"] != "ok":
            fallback_count += 1

        iou_scores.append(iou)
        result["ground_truth"] = gt
        result["iou"] = iou
        results.append(result)

        status_str = f"✓" if iou == 1.0 else (f"~{iou:.2f}" if iou > 0 else "✗")
        print(f"pred={pred}  iou={iou:.3f} {status_str}  ({result['latency_s']:.1f}s)")

    # Summary
    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
    exact_match = sum(1 for s in iou_scores if s == 1.0)
    partial = sum(1 for s in iou_scores if 0 < s < 1.0)
    zero = sum(1 for s in iou_scores if s == 0.0)

    print(f"\n{'='*60}")
    print(f"Results over {len(results)} examples:")
    print(f"  Mean IOU:      {mean_iou:.4f}")
    print(f"  Exact match:   {exact_match}/{len(results)} ({100*exact_match/len(results):.1f}%)")
    print(f"  Partial IOU:   {partial}/{len(results)}")
    print(f"  Zero IOU:      {zero}/{len(results)}")
    print(f"  Fallback used: {fallback_count}/{len(results)}")
    avg_lat = sum(r["latency_s"] for r in results) / len(results)
    print(f"  Avg latency:   {avg_lat:.1f}s/scenario")
    print(f"{'='*60}")

    if mean_iou >= 0.7:
        print("\n✓ STRONG — proceed to full test run")
    elif mean_iou >= 0.5:
        print("\n~ MODERATE — acceptable, submit and iterate")
    else:
        print("\n✗ LOW — investigate before submitting")


if __name__ == "__main__":
    main()
