#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Direct PEFT/QLoRA fine-tuning script for Qwen3.5-35B-A3B (qwen3_5_moe).

Replaces LLaMA-Factory which has version conflicts with transformers 5.x.
Uses: transformers 5.5.4 + peft 0.15 + bitsandbytes 0.49 + accelerate 1.7

Training data format (sft_v2_train.jsonl):
  Each line = {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

The script:
  1. Loads the tokenizer + model with 4-bit NF4 QLoRA
  2. Applies LoRA via PEFT (rank=16, alpha=32)
  3. Tokenizes messages into input_ids with labels masked on non-assistant tokens
  4. Trains with HuggingFace Trainer (cosine LR schedule, bf16)
  5. Saves LoRA adapter to lora_output_v2/

Usage (from the Track A directory):
    CUDA_VISIBLE_DEVICES=0 python finetune/train_peft.py
    # or with args:
    CUDA_VISIBLE_DEVICES=0 python finetune/train_peft.py \
        --model_path /home/huangxiaolin/AITO/model \
        --data_path finetune/sft_v2_train.jsonl \
        --output_dir finetune/lora_output_v2 \
        --epochs 3

After training, merge with:
    python finetune/merge_lora.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
TRACK_A_DIR = SCRIPT_DIR.parent

DEFAULTS = dict(
    model_path    = "/home/huangxiaolin/AITO/model",
    data_path     = str(SCRIPT_DIR / "sft_v2_train.jsonl"),
    output_dir    = str(SCRIPT_DIR / "lora_output_v2"),
    logging_dir   = str(SCRIPT_DIR / "logs_v2"),
    epochs        = 3,
    batch_size    = 1,
    grad_accum    = 16,
    lr            = 2e-4,
    cutoff_len    = 2048,
    max_samples   = 1800,
    lora_rank     = 16,
    lora_alpha    = 32,
    lora_dropout  = 0.05,
    lora_targets  = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    warmup_ratio  = 0.05,
    weight_decay  = 0.01,
    save_strategy = "epoch",
    save_total_limit = 3,
    seed          = 42,
)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading and tokenisation
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str, max_samples: int) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= max_samples:
                break
    print(f"Loaded {len(records)} records from {path}")
    return records


def tokenize_example(
    example: Dict[str, Any],
    tokenizer,
    cutoff_len: int,
) -> Optional[Dict[str, Any]]:
    """
    Convert a 3-message conversation to input_ids + labels.
    Labels are -100 for system+user tokens (no loss) and actual token ids
    for assistant tokens.
    """
    messages = example["messages"]
    if len(messages) != 3:
        return None

    # Build the full text using the tokenizer's chat template
    # The Qwen chat template handles system/user/assistant formatting
    try:
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return None

    # Also build prefix (system + user without assistant) to find where
    # assistant response begins in the token sequence
    prefix_messages = messages[:2]
    try:
        prefix_text = tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=False,
            add_generation_prompt=True,  # adds <|im_start|>assistant\n
        )
    except Exception:
        return None

    full_ids = tokenizer(
        full_text,
        truncation=True,
        max_length=cutoff_len,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]

    prefix_ids = tokenizer(
        prefix_text,
        truncation=True,
        max_length=cutoff_len,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]

    if len(full_ids) <= len(prefix_ids):
        return None  # assistant response was truncated away

    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    # Truncate both to cutoff_len
    input_ids = full_ids[:cutoff_len]
    labels    = labels[:cutoff_len]

    if len(input_ids) < 4:  # degenerate
        return None

    return {"input_ids": input_ids, "labels": labels}


def build_dataset(records: List[Dict], tokenizer, cutoff_len: int) -> Dataset:
    tokenized = []
    skipped = 0
    for r in records:
        tok = tokenize_example(r, tokenizer, cutoff_len)
        if tok is None:
            skipped += 1
            continue
        tokenized.append(tok)

    print(f"Tokenised {len(tokenized)} examples, skipped {skipped}")
    if tokenized:
        lens = [len(t["input_ids"]) for t in tokenized]
        print(f"  avg tokens: {sum(lens)/len(lens):.0f}  max: {max(lens)}")

    return Dataset.from_list(tokenized)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Qwen3.5-35B-A3B")
    for k, v in DEFAULTS.items():
        t = type(v)
        if isinstance(v, bool):
            parser.add_argument(f"--{k}", action="store_true", default=v)
        else:
            parser.add_argument(f"--{k}", type=t, default=v)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("QLoRA Fine-Tuning — Qwen3.5-35B-A3B")
    print(f"{'='*60}")
    print(f"  model:      {args.model_path}")
    print(f"  data:       {args.data_path}")
    print(f"  output:     {args.output_dir}")
    print(f"  epochs:     {args.epochs}")
    print(f"  batch:      {args.batch_size} × grad_accum {args.grad_accum} = {args.batch_size*args.grad_accum}")
    print(f"  lora_rank:  {args.lora_rank}  lora_alpha: {args.lora_alpha}")
    print(f"  max_len:    {args.cutoff_len}")
    print()

    # ── 1. Load tokenizer ─────────────────────────────────────────────────────
    print("[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 2. Load model with 4-bit QLoRA ────────────────────────────────────────
    print("[2/5] Loading model with 4-bit QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={"": 0},            # force all layers to GPU 0
        # NOTE: do NOT set dtype= here; bnb_config sets compute dtype internally
        attn_implementation="eager",   # flash_attn not installed
    )
    # Enable input grad for kbit models so LoRA gradients flow through frozen base
    model.enable_input_require_grads()

    # ── 3. Apply LoRA ─────────────────────────────────────────────────────────
    print("[3/5] Applying LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets.split(","),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 4. Build dataset ──────────────────────────────────────────────────────
    print("[4/5] Building dataset...")
    records = load_jsonl(args.data_path, args.max_samples)
    train_dataset = build_dataset(records, tokenizer, args.cutoff_len)

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print("[5/5] Starting training...")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)

    # Set logging dir via env var (transformers 5.x deprecates logging_dir argument)
    os.environ["TENSORBOARD_LOGGING_DIR"] = args.logging_dir

    # Compute warmup_steps (replaces deprecated warmup_ratio in transformers 5.x)
    steps_per_epoch = len(train_dataset) // (args.batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        seed=args.seed,
        label_smoothing_factor=0.0,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,   # transformers 5.x: tokenizer renamed to processing_class
    )

    trainer.train()

    print(f"\nTraining complete. Saving LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done. Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
