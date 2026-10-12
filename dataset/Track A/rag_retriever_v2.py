#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Structured Case RAG — V2 Architecture

Unlike V1 (raw text embedding of question + option labels), V2 retrieval
operates on structured case summaries derived from RF features, enabling
family-filtered nearest-neighbour search.

Two retrieval stages (per ARCHITECTURE_V2.md §7.6):
  1. Hard filter by family name (F1/F2/F3/F4) and answer type
  2. Nearest-neighbour by structured feature vector

Each training example is summarised into a numeric feature vector:
  [worst_tp, drop_magnitude, serving_rsrp, serving_sinr,
   neighbor_advantage, a3_margin, max_prb_util, num_neighbors,
   late_handover_score, missing_neighbor_score, interference_score,
   coverage_hole_score, inter_freq_score]

Dependencies: sentence-transformers, faiss-cpu, numpy
"""

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class StructuredCaseRAG:
    """
    Builds and queries a structured case index over the 2000 training examples.

    Each entry stores:
      - family_name: F1/F2/F3/F4
      - answer: ground-truth C-code string
      - feature_vector: 13-dim numeric summary
      - text_summary: compact human-readable evidence string
      - option_labels: {c_code: label} for the correct answer codes
    """

    FEATURE_DIM = 13
    CACHE_DIR = ".rag_cache_v2"

    def __init__(self, train_json_path: str):
        self.train_json_path = train_json_path
        self.cases: List[Dict[str, Any]] = []
        self.family_indices: Dict[str, List[int]] = {}  # family_name → list of case indices
        self.feature_matrix: Optional[np.ndarray] = None  # shape (N, FEATURE_DIM)

    # ──────────────────────────────────────────────────────────────────────────
    # Feature extraction (works on raw training JSON — no HTTP)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_feature_vector(raw: Dict[str, Any]) -> np.ndarray:
        """
        Extract a 13-dim numeric feature vector from a raw training example.
        Applies safe defaults (0.0) for missing values.
        """
        import io
        import pandas as pd

        data = raw.get("data", {})

        def _csv(key: str) -> pd.DataFrame:
            s = data.get(key, "")
            if not s:
                return pd.DataFrame()
            try:
                return pd.read_csv(io.StringIO(s), sep="|")
            except Exception:
                return pd.DataFrame()

        up = _csv("user_plane_data")
        nc = _csv("network_configuration_data")
        tr = _csv("traffic_data")

        # ── Throughput features ──
        worst_tp = 0.0
        drop_magnitude = 0.0
        if not up.empty:
            tp_col = next((c for c in up.columns if "DL Throughput" in c), None)
            if tp_col:
                tp = pd.to_numeric(up[tp_col], errors="coerce").dropna()
                if not tp.empty:
                    worst_tp = float(tp.min())
                    drop_magnitude = float(tp.max() - tp.min())

        # ── RF features at worst timestamp ──
        serving_rsrp = 0.0
        serving_sinr = 0.0
        neighbor_advantage = 0.0
        a3_margin = 0.0
        num_neighbors = 0

        if not up.empty:
            tp_col2 = next((c for c in up.columns if "DL Throughput" in c), None)
            if tp_col2:
                tp_vals = pd.to_numeric(up[tp_col2], errors="coerce")
                worst_idx = int(tp_vals.idxmin()) if not tp_vals.isna().all() else 0
                worst_row = up.iloc[worst_idx]

                rsrp_col = next((c for c in up.columns if "SS-RSRP" in c and "Serving" in c), None)
                sinr_col = next((c for c in up.columns if "SS-SINR" in c and "Serving" in c), None)
                if rsrp_col:
                    try:
                        serving_rsrp = float(worst_row[rsrp_col])
                    except (ValueError, TypeError):
                        pass
                if sinr_col:
                    try:
                        serving_sinr = float(worst_row[sinr_col])
                    except (ValueError, TypeError):
                        pass

                # Neighbour RSRP columns
                n_pci_cols = [c for c in up.columns if "Neighbor" in c and "PCI" in c and "Serving" not in c]
                n_rsrp_cols = [c for c in up.columns if "Neighbor" in c and "RSRP" in c and "Serving" not in c]
                num_neighbors = len(n_pci_cols)
                best_n_rsrp = -999.0
                for i, rc in enumerate(n_rsrp_cols):
                    try:
                        v = float(worst_row[rc])
                        if v > best_n_rsrp:
                            best_n_rsrp = v
                    except (ValueError, TypeError):
                        pass
                if best_n_rsrp > -999:
                    neighbor_advantage = best_n_rsrp - serving_rsrp

        # ── A3 margin from net_config ──
        if not nc.empty:
            pci_col = next((c for c in nc.columns if c.strip() == "PCI"), None)
            a3_col = next((c for c in nc.columns if "A3Offset" in c), None)
            hyst_col = next((c for c in nc.columns if "A3Hyst" in c), None)
            if a3_col and hyst_col and not nc.empty:
                try:
                    a3_offset = float(nc[a3_col].iloc[0]) * 0.5
                    a3_hyst = float(nc[hyst_col].iloc[0]) * 0.5
                    a3_margin = neighbor_advantage - (a3_offset + a3_hyst)
                except (ValueError, TypeError, IndexError):
                    pass

        # ── Traffic features ──
        max_prb_util = 0.0
        if not tr.empty:
            prb_col = next((c for c in tr.columns if "Downlink PRB" in c), None)
            if prb_col:
                vals = pd.to_numeric(tr[prb_col], errors="coerce").dropna()
                max_prb_util = float(vals.max()) if not vals.empty else 0.0

        # ── Issue scores (simplified, fast version of feature_engine) ──
        late_ho_score = 0.0
        missing_nb_score = 0.0
        interference_score = 0.0
        coverage_hole_score = 0.0
        inter_freq_score = 0.0

        if neighbor_advantage > 2 and a3_margin <= 0:
            late_ho_score = min(1.0, abs(a3_margin) / 5.0 + 0.3)
        if serving_sinr < 0 and serving_rsrp > -110:
            interference_score = min(1.0, abs(serving_sinr) / 10.0 + 0.3)
        if serving_rsrp < -110:
            coverage_hole_score = min(1.0, (-110 - serving_rsrp) / 20.0 + 0.5)
        if max_prb_util > 70:
            inter_freq_score = min(1.0, (max_prb_util - 70) / 30.0)

        vec = np.array([
            worst_tp / 500.0,            # normalised throughput
            drop_magnitude / 500.0,      # normalised drop magnitude
            (serving_rsrp + 140) / 60.0, # normalised RSRP [-140, -80] → [0,1]
            (serving_sinr + 20) / 40.0,  # normalised SINR [-20, 20] → [0,1]
            neighbor_advantage / 20.0,   # normalised advantage
            (a3_margin + 10) / 20.0,     # normalised A3 margin [-10, 10] → [0,1]
            max_prb_util / 100.0,        # normalised PRB util
            num_neighbors / 5.0,         # normalised neighbor count
            late_ho_score,
            missing_nb_score,
            interference_score,
            coverage_hole_score,
            inter_freq_score,
        ], dtype=np.float32)

        # Clip to [0, 1]
        return np.clip(vec, 0.0, 1.0)

    @staticmethod
    def _get_family_name(raw: Dict[str, Any]) -> str:
        tag = raw.get("tag", "single-answer")
        ctx = raw.get("context", {})
        num_bs = int(str(ctx.get("wireless_network_information", {}).get("num_base_stations", "4")))
        is_multi = "multiple" in tag.lower()
        if num_bs == 5:
            return "F3"
        if num_bs == 13:
            return "F4"
        return "F2" if is_multi else "F1"

    # ──────────────────────────────────────────────────────────────────────────
    # Index building
    # ──────────────────────────────────────────────────────────────────────────

    def build_index(self, force_rebuild: bool = False) -> None:
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(self.CACHE_DIR, "rag_v2_cases.pkl")
        matrix_path = os.path.join(self.CACHE_DIR, "rag_v2_matrix.npy")

        if not force_rebuild and os.path.exists(cache_path) and os.path.exists(matrix_path):
            print("[RAG V2] Loading cached structured index...")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self.cases = data["cases"]
            self.family_indices = data["family_indices"]
            self.feature_matrix = np.load(matrix_path)
            print(f"[RAG V2] Loaded {len(self.cases)} cases.")
            return

        print("[RAG V2] Building structured feature index...")
        with open(self.train_json_path, encoding="utf-8") as f:
            raw_examples = json.load(f)

        self.cases = []
        self.family_indices = {}
        vectors = []

        for i, raw in enumerate(raw_examples):
            family_name = self._get_family_name(raw)
            answer = raw.get("answer", "")
            task = raw.get("task", {})
            options = {opt["id"]: opt["label"] for opt in task.get("options", [])}

            # Build compact text summary
            answer_labels = " | ".join(
                f"{c} ({options.get(c, '?')})" for c in answer.split("|")
            )

            vec = self._extract_feature_vector(raw)
            vectors.append(vec)

            case = {
                "idx": i,
                "scenario_id": raw.get("scenario_id", ""),
                "family": family_name,
                "tag": raw.get("tag", "single-answer"),
                "answer": answer,
                "answer_labels": answer_labels,
                "options": options,
                "feature_vector": vec,
            }
            self.cases.append(case)
            self.family_indices.setdefault(family_name, []).append(i)

            if (i + 1) % 500 == 0:
                print(f"[RAG V2] Indexed {i + 1}/{len(raw_examples)}...")

        self.feature_matrix = np.stack(vectors, axis=0).astype(np.float32)

        with open(cache_path, "wb") as f:
            pickle.dump({"cases": self.cases, "family_indices": self.family_indices}, f)
        np.save(matrix_path, self.feature_matrix)
        print(f"[RAG V2] Index built: {len(self.cases)} cases, {self.FEATURE_DIM} features each.")

    # ──────────────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_feature_vector: np.ndarray,
        family_name: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top_k structurally similar training cases within the same family.

        Args:
            query_feature_vector: 13-dim normalised feature vector for the test scenario
            family_name:          "F1" | "F2" | "F3" | "F4"
            top_k:                number of cases to retrieve

        Returns:
            List of case dicts ordered by similarity (most similar first).
        """
        if self.feature_matrix is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Filter to same family
        family_idxs = self.family_indices.get(family_name, [])
        if not family_idxs:
            # Fallback: search all cases
            family_idxs = list(range(len(self.cases)))

        sub_matrix = self.feature_matrix[family_idxs]   # (M, FEATURE_DIM)
        q = query_feature_vector.reshape(1, -1).astype(np.float32)

        # Cosine similarity (vectors are already [0,1] so L2 is also reasonable)
        norms_sub = np.linalg.norm(sub_matrix, axis=1, keepdims=True) + 1e-8
        norms_q = np.linalg.norm(q) + 1e-8
        sims = (sub_matrix / norms_sub) @ (q / norms_q).T   # (M, 1)
        sims = sims.flatten()

        top_local = np.argsort(-sims)[:top_k]
        results = []
        for local_idx in top_local:
            global_idx = family_idxs[local_idx]
            case = dict(self.cases[global_idx])
            case["_similarity"] = float(sims[local_idx])
            results.append(case)
        return results

    def format_few_shots(self, retrieved: List[Dict[str, Any]]) -> str:
        """Format retrieved cases as few-shot context for the Qwen prompt."""
        if not retrieved:
            return ""
        lines = ["## Similar Solved Cases\n"]
        for i, case in enumerate(retrieved, 1):
            lines.append(
                f"### Case {i} [{case['family']}] (similarity={case.get('_similarity', 0):.3f})\n"
                f"**Correct answer:** {case['answer_labels']}\n"
            )
        return "\n".join(lines)

    def retrieve_and_format(
        self,
        query_feature_vector: np.ndarray,
        family_name: str,
        top_k: int = 3,
    ) -> str:
        retrieved = self.retrieve(query_feature_vector, family_name, top_k=top_k)
        return self.format_few_shots(retrieved)
