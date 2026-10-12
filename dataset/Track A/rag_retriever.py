#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
RAG Module for Track A — Few-shot retrieval from the 2000 training examples.

At startup, embeds all training questions with sentence-transformers and builds
a FAISS index. At inference time, retrieves the top-k most similar training
examples and formats them as few-shot demonstrations for the system prompt.

Dependencies (all open-source):
    pip install sentence-transformers faiss-cpu

Usage:
    retriever = RAGRetriever("data/Phase_1/train.json")
    retriever.build_index()
    few_shot_text = retriever.retrieve_and_format(test_question, top_k=3)
"""

import json
import os
import re
import pickle
from typing import List, Dict, Any, Optional

import numpy as np


class RAGRetriever:
    """
    Embedding-based retriever over the Track A training set.

    Each training example is embedded using the concatenation of:
      - task description (problem statement)
      - option labels (C1..C22 labels)

    This gives a fingerprint that captures both the problem type and the
    specific cell IDs / parameter names involved.
    """

    def __init__(
        self,
        train_json_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = ".rag_cache",
    ):
        self.train_json_path = train_json_path
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.index = None
        self.examples: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self._embed_model = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            # Prefer the domain-fine-tuned model when available (produced by
            # finetune/train_embedding.py); fall back to the base model name.
            ft_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "finetune", "embedding_model",
            )
            model_to_load = ft_path if os.path.isdir(ft_path) else self.model_name
            if model_to_load != self.model_name:
                print(f"[RAG] Using fine-tuned embedding model: {ft_path}")
            self._embed_model = SentenceTransformer(model_to_load)
        return self._embed_model

    def _make_query_text(self, scenario: Dict[str, Any]) -> str:
        """
        Build the text to embed for a given scenario dict.
        Works for both training examples (have 'answer') and test questions.
        """
        task = scenario.get("task", {})
        description = task.get("description", "")
        options = task.get("options", [])
        option_text = " | ".join(
            f"{opt['id']}: {opt['label']}" for opt in options
        )
        return f"{description} OPTIONS: {option_text}"

    def _format_tool_calls(self, example: Dict[str, Any]) -> str:
        """
        Build a compact representation of the data available in a training example.
        Since we don't have recorded tool calls in train.json, we summarise the
        key RF facts directly from the embedded CSV data.
        """
        data = example.get("data", {})
        lines = []

        # Parse throughput CSV to find worst timestamp
        user_plane = data.get("user_plane_data", "")
        if user_plane:
            rows = [r for r in user_plane.strip().split("\n") if r]
            if len(rows) > 1:
                # Find min throughput row (columns: Timestamp|UE|Lon|Lat|PCI|ARFCN|RSRP|SINR|DL_TP|...)
                worst_tp = None
                worst_row = None
                for row in rows[1:]:
                    cols = row.split("|")
                    try:
                        tp = float(cols[8]) if len(cols) > 8 else float("inf")
                        if worst_tp is None or tp < worst_tp:
                            worst_tp = tp
                            worst_row = cols
                    except (ValueError, IndexError):
                        continue
                if worst_row:
                    ts = worst_row[0] if worst_row else "?"
                    try:
                        rsrp = worst_row[6]
                        sinr = worst_row[7]
                        pci = worst_row[4]
                        lines.append(f"Worst timestamp: {ts.strip()}")
                        lines.append(f"Serving PCI: {pci.strip()}, RSRP: {rsrp.strip()} dBm, SINR: {sinr.strip()} dB")
                        lines.append(f"Throughput at fault: {worst_tp:.1f} Mbps")
                    except IndexError:
                        pass

        # Notes (usually empty but include if present)
        notes = data.get("notes", "")
        if notes:
            lines.append(f"Notes: {notes}")

        return "\n".join(lines) if lines else "No additional RF data."

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_training_data(self) -> None:
        """Load the training JSON into self.examples."""
        with open(self.train_json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.examples = raw
        print(f"[RAG] Loaded {len(self.examples)} training examples.")

    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Build (or load from cache) the FAISS embedding index.

        The index and embeddings are cached to disk so that subsequent runs
        don't need to re-embed 2000 examples.
        """
        import faiss

        os.makedirs(self.cache_dir, exist_ok=True)
        index_path = os.path.join(self.cache_dir, "rag_index.faiss")
        emb_path = os.path.join(self.cache_dir, "rag_embeddings.npy")
        meta_path = os.path.join(self.cache_dir, "rag_meta.pkl")

        if not force_rebuild and os.path.exists(index_path) and os.path.exists(meta_path):
            print("[RAG] Loading cached index...")
            self.index = faiss.read_index(index_path)
            self.embeddings = np.load(emb_path)
            with open(meta_path, "rb") as f:
                self.examples = pickle.load(f)
            print(f"[RAG] Index loaded ({len(self.examples)} examples).")
            return

        if not self.examples:
            self.load_training_data()

        print("[RAG] Building embedding index (this may take ~30-60s)...")
        model = self._get_embed_model()

        texts = [self._make_query_text(ex) for ex in self.examples]
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        self.embeddings = embeddings.astype(np.float32)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity (normalized vectors)
        self.index.add(self.embeddings)

        # Cache to disk
        faiss.write_index(self.index, index_path)
        np.save(emb_path, self.embeddings)
        with open(meta_path, "wb") as f:
            pickle.dump(self.examples, f)

        print(f"[RAG] Index built and cached ({len(self.examples)} examples, dim={dim}).")

    def retrieve(self, query_scenario: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most similar training examples for a given test scenario.

        Args:
            query_scenario: the test scenario dict (same structure as training, but no 'answer')
            top_k:          number of examples to retrieve

        Returns:
            List of training example dicts, ordered by similarity (most similar first).
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        model = self._get_embed_model()
        query_text = self._make_query_text(query_scenario)
        query_emb = model.encode([query_text], normalize_embeddings=True).astype(np.float32)

        scores, indices = self.index.search(query_emb, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.examples):
                ex = dict(self.examples[idx])
                ex["_similarity"] = float(score)
                results.append(ex)
        return results

    def format_few_shots(self, retrieved: List[Dict[str, Any]]) -> str:
        """
        Format retrieved training examples into a few-shot demonstration string
        suitable for injection into the system prompt.

        Each example shows:
          - The question (options)
          - Key RF observations from the data
          - The correct answer
        """
        if not retrieved:
            return ""

        parts = ["## Few-Shot Examples (similar past cases)\n"]
        for i, ex in enumerate(retrieved, 1):
            task = ex.get("task", {})
            answer = ex.get("answer", "?")
            tag = ex.get("tag", "")

            # Build option list
            options = task.get("options", [])
            option_lines = "\n".join(f"  {opt['id']}: {opt['label']}" for opt in options)

            # Key RF observations
            rf_summary = self._format_tool_calls(ex)

            # Answer with label
            answer_labels = []
            for code in answer.split("|"):
                label = next((o["label"] for o in options if o["id"] == code), code)
                answer_labels.append(f"{code} ({label})")
            answer_str = " | ".join(answer_labels)

            parts.append(
                f"### Example {i} [{tag}] (similarity={ex.get('_similarity', 0):.3f})\n"
                f"**Options:**\n{option_lines}\n\n"
                f"**Key RF observations:**\n{rf_summary}\n\n"
                f"**Correct answer:** {answer_str}\n"
            )

        return "\n".join(parts)

    def retrieve_and_format(self, query_scenario: Dict[str, Any], top_k: int = 3) -> str:
        """
        Convenience method: retrieve + format in one call.
        """
        retrieved = self.retrieve(query_scenario, top_k=top_k)
        return self.format_few_shots(retrieved)
