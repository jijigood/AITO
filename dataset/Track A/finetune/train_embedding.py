#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Contrastive fine-tuning of the RAG embedding model for Track A.

Adapts OTel's approach: fine-tune all-MiniLM-L6-v2 with CoSENTLoss using
anchor/positive/negative triplets built from the 2000 training scenarios.
The fine-tuned model is saved to finetune/embedding_model/ and is
automatically picked up by rag_retriever.py at inference time.

Prerequisites:
    pip install sentence-transformers>=3.0

Usage:
    cd "Track A"
    python finetune/train_embedding.py [--epochs 5] [--batch_size 32] [--no_prepare]

Output:
    finetune/embedding_model/   (fine-tuned SentenceTransformer checkpoint)
"""

import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACK_A_DIR = os.path.dirname(SCRIPT_DIR)
TRIPLETS_PATH = os.path.join(SCRIPT_DIR, "embedding_train.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "embedding_model")
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def ensure_triplets(no_prepare: bool):
    if os.path.exists(TRIPLETS_PATH):
        print(f"[Embedding] Triplet data found at {TRIPLETS_PATH}. Skipping preparation.")
        return
    if no_prepare:
        print(f"[Embedding] ERROR: {TRIPLETS_PATH} not found. "
              "Run prepare_embedding_data.py first or omit --no_prepare.")
        sys.exit(1)
    print("[Embedding] Generating contrastive triplets...")
    prepare_script = os.path.join(SCRIPT_DIR, "prepare_embedding_data.py")
    subprocess.run([sys.executable, prepare_script], cwd=TRACK_A_DIR, check=True)


def load_triplets(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train(epochs: int, batch_size: int, warmup_steps: int):
    # Defer imports so the script is importable without sentence-transformers installed
    try:
        from sentence_transformers import InputExample, SentenceTransformer
        from sentence_transformers.losses import CoSENTLoss
        from torch.utils.data import DataLoader
    except ImportError as e:
        print(f"[Embedding] Import error: {e}")
        print("Install with:  pip install sentence-transformers>=3.0")
        sys.exit(1)

    print(f"[Embedding] Loading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)

    print(f"[Embedding] Loading triplets from {TRIPLETS_PATH}...")
    triplets = load_triplets(TRIPLETS_PATH)
    print(f"[Embedding] {len(triplets)} triplets loaded.")

    # Convert each triplet to InputExample pairs:
    #   anchor + positive  → label 1.0 (high similarity)
    #   anchor + negative  → label 0.0 (low similarity)
    # CoSENTLoss learns a similarity ordering from these scored pairs.
    train_examples = []
    neg_keys = ["negative_1", "negative_2", "negative_3"]
    for record in triplets:
        anchor = record.get("anchor", "")
        positive = record.get("positive", "")
        if not anchor or not positive:
            continue
        train_examples.append(InputExample(texts=[anchor, positive], label=1.0))
        for key in neg_keys:
            neg = record.get(key, "")
            if neg:
                train_examples.append(InputExample(texts=[anchor, neg], label=0.0))

    print(f"[Embedding] {len(train_examples)} training pairs (positives + negatives).")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = CoSENTLoss(model)

    print(f"[Embedding] Fine-tuning for {epochs} epoch(s), batch_size={batch_size}, "
          f"warmup_steps={warmup_steps}...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=OUTPUT_PATH,
        show_progress_bar=True,
    )

    print(f"\n[Embedding] Fine-tuned model saved to: {OUTPUT_PATH}")
    print("[Embedding] rag_retriever.py will automatically use this model on next run.")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune RAG embedding model with CoSENTLoss")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of fine-tuning epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size (default: 32)")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="Linear warmup steps (default: 50)")
    parser.add_argument("--no_prepare", action="store_true",
                        help="Skip triplet generation (requires embedding_train.json to exist)")
    args = parser.parse_args()

    ensure_triplets(no_prepare=args.no_prepare)
    train(epochs=args.epochs, batch_size=args.batch_size, warmup_steps=args.warmup_steps)


if __name__ == "__main__":
    main()
