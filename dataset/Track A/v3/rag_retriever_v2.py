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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skills.scenario_parser import parse_option_label

try:
    import faiss as _faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _faiss = None
    _FAISS_AVAILABLE = False


_F1_ACTION_BUCKETS = [
    "lift_tilt",
    "insufficient_data",
    "add_neighbor",
    "pdcch_2sym",
    "check_server",
    "decrease_a3",
]

_CONFUSION_TRIGGER_ISSUES = frozenset({"coverage_hole", "backhaul_server"})

_ISSUE_DIM_NAMES = [
    "late_handover",
    "missing_neighbor",
    "interference",
    "coverage_hole",
    "inter_freq_load",
]


def _get_top_issue_from_query_vec(query_vec: np.ndarray) -> str:
    issue_scores = query_vec[8:13]
    if len(issue_scores) < 5:
        return "unknown"
    return _ISSUE_DIM_NAMES[int(np.argmax(issue_scores))]


@dataclass
class RAGHit:
    scenario_id: str
    features: Dict[str, float]
    gt_c_code: str
    gt_action_type: str
    similarity: float
    prompt_snippet: str
    is_confusion_pair: bool = False


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

    FEATURE_DIM = 16
    CACHE_DIR = ".rag_cache_v2"

    def __init__(self, train_json_path: str):
        self.train_json_path = train_json_path
        self.cases: List[Dict[str, Any]] = []
        self.family_indices: Dict[str, List[int]] = {}  # family_name → list of case indices
        self._action_indices: Dict[str, List[int]] = {}
        self._confusion_index: Dict[str, Dict[str, List[str]]] = {}
        self.feature_matrix: Optional[np.ndarray] = None  # shape (N, FEATURE_DIM)
        self._faiss_index = None

    # ──────────────────────────────────────────────────────────────────────────
    # Feature extraction (works on raw training JSON — no HTTP)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_feature_vector(raw: Dict[str, Any]) -> np.ndarray:
        """
        Extract a 16-dim numeric feature vector from a raw training example.
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

        # Proxy for missing_neighbor: count configured neighbors from net_config
        # and compare against visible MR neighbors; gap → missing neighbor signal.
        if not nc.empty and num_neighbors > 0:
            import re as _re
            nb_col = next(
                (c for c in nc.columns if "Neighbor Cell" in c and "gNodeB" in c), None
            )
            if nb_col:
                nb_str = str(nc[nb_col].iloc[0]) if len(nc) > 0 else ""
                num_configured = len(_re.findall(r'\[', nb_str))
                gap = max(0, num_neighbors - num_configured)
                if gap > 0:
                    missing_nb_score = min(1.0, gap / max(num_neighbors, 1))

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
            0.0,                         # outside_mainlobe_ratio (train data has no tool evidence)
            0.0,                         # tilt_angle_mean
            0.0,                         # overlap_ratio_scalar
        ], dtype=np.float32)

        # Clip to [0, 1]
        return np.clip(vec, 0.0, 1.0)

    @staticmethod
    def _coerce_feature_dim(vec: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Keep legacy 13-dim query vectors working against newer cached indices by
        zero-padding missing tail dimensions inside rag_retriever_v2.
        """
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.shape[0] == target_dim:
            return arr
        if arr.shape[0] < target_dim:
            return np.pad(arr, (0, target_dim - arr.shape[0]), mode="constant").astype(np.float32)
        return arr[:target_dim].astype(np.float32)

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

    @staticmethod
    def _sha256_file(path: str) -> str:
        import hashlib

        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _build_faiss_index(self, matrix: np.ndarray):
        """Build an optional FAISS cosine-similarity index."""
        if not _FAISS_AVAILABLE:
            return None
        norm_matrix = matrix.copy().astype(np.float32)
        _faiss.normalize_L2(norm_matrix)
        index = _faiss.IndexFlatIP(norm_matrix.shape[1])
        index.add(norm_matrix)
        return index

    def _cosine_sim_to_subset(self, query: np.ndarray, subset_idxs: List[int]) -> np.ndarray:
        """Return cosine similarities between query and feature_matrix[subset_idxs]."""
        if not subset_idxs:
            return np.array([], dtype=np.float32)
        sub = self.feature_matrix[subset_idxs].astype(np.float32)
        q = self._coerce_feature_dim(query, sub.shape[1]).reshape(1, -1)
        norms_sub = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8
        norm_q = float(np.linalg.norm(q)) + 1e-8
        return ((sub / norms_sub) @ (q / norm_q).T).flatten()

    @staticmethod
    def _sf_to_feature_vec(sf) -> np.ndarray:
        """
        Convert ScenarioFeatures-like input to the 16-dim float32 vector used by
        structured retrieval, with the last 3 dims reserved for tool evidence.
        """
        tp = sf.throughput
        mob = sf.mobility
        ifreq = sf.inter_freq
        base = np.array([
            (tp.worst_throughput_mbps or 0.0) / 500.0,
            (tp.throughput_drop_magnitude or 0.0) / 500.0,
            ((mob.serving_rsrp_dbm or -140.0) + 140.0) / 60.0,
            ((mob.serving_sinr_db or -20.0) + 20.0) / 40.0,
            (mob.neighbor_advantage_db or 0.0) / 20.0,
            ((mob.a3_margin_db or -10.0) + 10.0) / 20.0,
            (ifreq.max_dl_prb_util or 0.0) / 100.0,
            float(len(getattr(mob, "visible_neighbor_pcis", []) or [])) / 5.0,
            sf.score_late_handover,
            sf.score_missing_neighbor,
            sf.score_interference,
            sf.score_coverage_hole,
            sf.score_inter_freq_load,
        ], dtype=np.float32)

        tool_ev = getattr(sf, "tool_evidence", None)
        outside_ratio = 0.0
        tilt_mean = 0.0
        overlap_scalar = 0.0

        if tool_ev is not None:
            ml_flags = getattr(tool_ev, "mainlobe_flags", {}) or {}
            tilt_angles = getattr(tool_ev, "tilt_angles", {}) or {}
            raw_overlap = getattr(tool_ev, "overlap_ratio", None)

            n = max(len(ml_flags), 1)
            outside_ratio = sum(1 for v in ml_flags.values() if bool(v)) / n

            if tilt_angles:
                tilt_mean = sum(tilt_angles.values()) / len(tilt_angles) / 20.0

            if isinstance(raw_overlap, dict):
                overlap_scalar = float(max(raw_overlap.values(), default=0.0))
            elif isinstance(raw_overlap, (int, float)) and raw_overlap is not None:
                overlap_scalar = float(raw_overlap)

        tool_vec = np.array([outside_ratio, tilt_mean, overlap_scalar], dtype=np.float32)
        return np.clip(np.concatenate([base, tool_vec]), 0.0, 1.0)

    @staticmethod
    def _deduplicate(hits: List[dict], query_vec: np.ndarray, threshold: float = 0.98) -> List[dict]:
        """
        Drop hits whose feature_vector is too close to query_vec.
        This is mainly used to remove self-hits / near-duplicates.
        """
        kept = []
        for hit in hits:
            vec = hit["feature_vector"].astype(np.float32)
            aligned_query = StructuredCaseRAG._coerce_feature_dim(query_vec, vec.shape[0])
            q_norm = aligned_query / (np.linalg.norm(aligned_query) + 1e-8)
            v_norm = vec / (np.linalg.norm(vec) + 1e-8)
            if float(np.dot(q_norm, v_norm)) < threshold:
                kept.append(hit)
        return kept

    def _case_to_raghit(self, case: dict, sim: float, is_confusion: bool = False) -> "RAGHit":
        """
        Convert internal case dict → public RAGHit.
        Note: in the current codebase RAGHit.features is a compact dict, not a full
        ScenarioFeatures object.
        """
        vec = case["feature_vector"]
        features = {
            "worst_tp": float(vec[0]),
            "serving_rsrp": float(vec[2]),
            "serving_sinr": float(vec[3]),
            "neighbor_advantage": float(vec[4]),
            "max_prb_util": float(vec[6]),
            "score_late_handover": float(vec[8]),
            "score_missing_neighbor": float(vec[9]),
            "score_coverage_hole": float(vec[11]),
        }
        snippet = (
            f"[{case.get('family', '?')}] {case.get('action_type', '?')} "
            f"sim={sim:.3f} ans={case.get('answer', '?')}"
        )[:120]
        return RAGHit(
            scenario_id=case["scenario_id"],
            features=features,
            gt_c_code=case.get("answer", ""),
            gt_action_type=case.get("action_type", "unknown"),
            similarity=sim,
            prompt_snippet=snippet,
            is_confusion_pair=is_confusion,
        )

    def _build_confusion_index(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Build a seed -> opposite-class-neighbor mapping for the two most confusing
        action buckets: lift_tilt and insufficient_data.

        Returns:
            {
                "lift_tilt": {seed_sid: [opp_sid1, opp_sid2, ...]},
                "insufficient_data": {seed_sid: [opp_sid1, opp_sid2, ...]},
            }
        """
        result = {"lift_tilt": {}, "insufficient_data": {}}

        lt_idxs = self._action_indices.get("lift_tilt", [])
        id_idxs = self._action_indices.get("insufficient_data", [])
        if not lt_idxs or not id_idxs:
            return result

        lt_matrix = self.feature_matrix[lt_idxs].astype(np.float32)
        id_matrix = self.feature_matrix[id_idxs].astype(np.float32)

        lt_norms = np.linalg.norm(lt_matrix, axis=1, keepdims=True) + 1e-8
        id_norms = np.linalg.norm(id_matrix, axis=1, keepdims=True) + 1e-8
        sims = (lt_matrix / lt_norms) @ (id_matrix / id_norms).T

        for row_i, row in enumerate(sims):
            top2 = np.argsort(-row)[:2]
            seed_sid = self.cases[lt_idxs[row_i]]["scenario_id"]
            result["lift_tilt"][seed_sid] = [self.cases[id_idxs[j]]["scenario_id"] for j in top2]

        for row_i, row in enumerate(sims.T):
            top2 = np.argsort(-row)[:2]
            seed_sid = self.cases[id_idxs[row_i]]["scenario_id"]
            result["insufficient_data"][seed_sid] = [self.cases[lt_idxs[j]]["scenario_id"] for j in top2]

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Index building
    # ──────────────────────────────────────────────────────────────────────────

    def build_index(self, force_rebuild: bool = False) -> None:
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(self.CACHE_DIR, "rag_v2_cases.pkl")
        matrix_path = os.path.join(self.CACHE_DIR, "rag_v2_matrix.npy")
        confuse_path = os.path.join(self.CACHE_DIR, "confusion_index.pkl")
        hash_path = os.path.join(self.CACHE_DIR, "data_hash.txt")
        faiss_path = os.path.join(self.CACHE_DIR, "rag_v2_faiss.index")

        current_hash = self._sha256_file(self.train_json_path)
        cached_hash = ""
        if os.path.exists(hash_path):
            with open(hash_path, "r", encoding="utf-8") as f:
                cached_hash = f.read().strip()

        cache_valid = (
            not force_rebuild
            and current_hash == cached_hash
            and os.path.exists(cache_path)
            and os.path.exists(matrix_path)
        )

        if cache_valid:
            print("[RAG V2] Loading cached structured index...")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self.cases = data["cases"]
            self.family_indices = data["family_indices"]
            self._action_indices = data.get("action_indices", {})
            self.feature_matrix = np.load(matrix_path)
            if self.feature_matrix.ndim != 2 or self.feature_matrix.shape[1] != self.FEATURE_DIM:
                print(
                    "[RAG V2] Feature dim mismatch "
                    f"({self.feature_matrix.shape} vs *x{self.FEATURE_DIM}), rebuilding..."
                )
                return self.build_index(force_rebuild=True)
            if os.path.exists(confuse_path):
                with open(confuse_path, "rb") as f:
                    self._confusion_index = pickle.load(f)
            else:
                self._confusion_index = {}
            if _FAISS_AVAILABLE and os.path.exists(faiss_path):
                self._faiss_index = _faiss.read_index(faiss_path)
            else:
                self._faiss_index = None
            print(f"[RAG V2] Loaded {len(self.cases)} cases.")
            return

        print("[RAG V2] Building structured feature index...")
        with open(self.train_json_path, encoding="utf-8") as f:
            raw_examples = json.load(f)

        self.cases = []
        self.family_indices = {}
        self._action_indices = {}
        self._confusion_index = {}
        self._faiss_index = None
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
            first_c_code = answer.split("|")[0].strip() if answer else ""
            gt_label = options.get(first_c_code, "")
            parsed = parse_option_label(gt_label) if gt_label else {}
            action_type = parsed.get("action_type", "unknown")
            case["action_type"] = action_type
            global_idx = len(self.cases)
            self._action_indices.setdefault(action_type, []).append(global_idx)
            self.cases.append(case)
            self.family_indices.setdefault(family_name, []).append(i)

            if (i + 1) % 500 == 0:
                print(f"[RAG V2] Indexed {i + 1}/{len(raw_examples)}...")

        self.feature_matrix = np.stack(vectors, axis=0).astype(np.float32)

        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "cases": self.cases,
                    "family_indices": self.family_indices,
                    "action_indices": self._action_indices,
                },
                f,
            )
        np.save(matrix_path, self.feature_matrix)
        self._confusion_index = self._build_confusion_index()
        with open(confuse_path, "wb") as f:
            pickle.dump(self._confusion_index, f)
        self._faiss_index = self._build_faiss_index(self.feature_matrix)
        if self._faiss_index is not None:
            _faiss.write_index(self._faiss_index, faiss_path)
        with open(hash_path, "w", encoding="utf-8") as f:
            f.write(current_hash)
        print(f"[RAG V2] Index built: {len(self.cases)} cases, {self.FEATURE_DIM} features each.")

    # ──────────────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def _retrieve_raw(
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
        q = self._coerce_feature_dim(query_feature_vector, sub_matrix.shape[1]).reshape(1, -1)

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

    def _flat_retrieve(self, query_vec: np.ndarray, family_idxs: List[int], k: int) -> List[dict]:
        """Plain top-k cosine retrieval within the given family bucket."""
        sims = self._cosine_sim_to_subset(query_vec, family_idxs)
        top_local = np.argsort(-sims)[:k]
        out = []
        for li in top_local:
            gi = family_idxs[li]
            case = dict(self.cases[gi])
            case["_similarity"] = float(sims[li])
            out.append(case)
        return out

    def _stratified_retrieve(self, query_vec: np.ndarray, family_idxs: List[int], k: int) -> List[dict]:
        """
        Try to draw at least one example from each major F1 action bucket, then
        backfill by overall similarity.
        """
        family_set = set(family_idxs)
        quota = max(2, k // max(len(_F1_ACTION_BUCKETS), 1))

        collected: List[dict] = []
        used: set[int] = set()

        for bucket in _F1_ACTION_BUCKETS:
            bucket_idxs = [gi for gi in self._action_indices.get(bucket, []) if gi in family_set]
            if not bucket_idxs:
                continue
            sims = self._cosine_sim_to_subset(query_vec, bucket_idxs)
            top_local = np.argsort(-sims)[:quota]
            for li in top_local:
                gi = bucket_idxs[li]
                if gi in used:
                    continue
                case = dict(self.cases[gi])
                case["_similarity"] = float(sims[li])
                collected.append(case)
                used.add(gi)

        remaining = k - len(collected)
        if remaining > 0:
            all_sims = self._cosine_sim_to_subset(query_vec, family_idxs)
            order = np.argsort(-all_sims)
            for li in order:
                gi = family_idxs[li]
                if gi in used:
                    continue
                case = dict(self.cases[gi])
                case["_similarity"] = float(all_sims[li])
                collected.append(case)
                used.add(gi)
                remaining -= 1
                if remaining == 0:
                    break

        return collected

    def _get_confusion_pairs(self, query_vec: np.ndarray, family_idxs: List[int]) -> List["RAGHit"]:
        """
        Return up to 2 confusion exemplars, appended after the normal top-k hits.
        Each returned hit represents a seed case from one confusing class; its
        prompt_snippet can be enriched with one opposite-class sid from
        _confusion_index for boundary context.
        """
        family_set = set(family_idxs)
        out: List[RAGHit] = []

        for ctype in ("lift_tilt", "insufficient_data"):
            bucket_idxs = [gi for gi in self._action_indices.get(ctype, []) if gi in family_set]
            if not bucket_idxs:
                continue
            sims = self._cosine_sim_to_subset(query_vec, bucket_idxs)
            best_local = int(np.argmax(sims))
            gi = bucket_idxs[best_local]
            case = dict(self.cases[gi])
            hit = self._case_to_raghit(case, float(sims[best_local]), is_confusion=True)

            seed_sid = case["scenario_id"]
            paired = self._confusion_index.get(ctype, {}).get(seed_sid, [])
            if paired:
                hit.prompt_snippet = f"{hit.prompt_snippet} contrast={paired[0]}"[:120]
            out.append(hit)

        return out

    def retrieve(
        self,
        sf,
        k: int = 6,
        stratify_by_action: bool = True,
        include_confusion_pairs: bool = True,
        family_name: str = "F1",
    ) -> List["RAGHit"]:
        """
        Frozen public interface for Owner A / SFT integration.
        """
        if sf is None:
            return []
        if self.feature_matrix is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_vec = self._sf_to_feature_vec(sf)
        family_idxs = self.family_indices.get(family_name, list(range(len(self.cases))))

        if stratify_by_action:
            hits = self._stratified_retrieve(query_vec, family_idxs, k)
        else:
            hits = self._flat_retrieve(query_vec, family_idxs, k)

        hits = self._deduplicate(hits, query_vec, threshold=0.98)
        hits.sort(key=lambda h: -h["_similarity"])
        normal_hits = [self._case_to_raghit(h, h["_similarity"]) for h in hits[:k]]

        if include_confusion_pairs:
            if hasattr(sf, "top_issue") and callable(sf.top_issue):
                top_issue = sf.top_issue()[0]
            else:
                top_issue = _get_top_issue_from_query_vec(query_vec)
            if top_issue in _CONFUSION_TRIGGER_ISSUES:
                confusion_hits = self._get_confusion_pairs(query_vec, family_idxs)
                seen = {h.scenario_id for h in normal_hits}
                confusion_hits = [
                    h for h in confusion_hits
                    if h.scenario_id not in seen
                ]
                confusion_budget = min(len(confusion_hits), k)
                normal_budget = max(k - confusion_budget, 0)
                return (normal_hits[:normal_budget] + confusion_hits[:confusion_budget])[:k]

        return normal_hits[:k]

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
        retrieved = self._retrieve_raw(query_feature_vector, family_name, top_k=top_k)
        return self.format_few_shots(retrieved)
