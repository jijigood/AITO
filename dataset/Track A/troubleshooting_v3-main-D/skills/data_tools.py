#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
skills/data_tools.py — Owner D's data-analysis tooling library.

Pure-function scoring over training scenarios. No side effects, no decisions.
Owner A consumes these scores in curate_sft.py and decides the thresholds.

Deliverable: D5 in TEAM_DIVISION.md §5.4.
Target runtime: all four functions must complete over the 2000-scenario
training set in ≤ 30 s total (see D success criteria).
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np

_rag_singleton = None
_DEFAULT_TRAIN_JSON = os.path.join(
    os.path.dirname(__file__), "..", "data", "Phase_1", "train.json"
)


def _get_rag(train_json_path: Optional[str] = None, _rag=None):
    if _rag is not None:
        return _rag
    global _rag_singleton
    if _rag_singleton is not None:
        return _rag_singleton
    from rag_retriever_v2 import StructuredCaseRAG

    path = train_json_path or os.environ.get("RAG_TRAIN_JSON", _DEFAULT_TRAIN_JSON)
    rag = StructuredCaseRAG(train_json_path=path)
    rag.build_index()
    _rag_singleton = rag
    return _rag_singleton


def _reset_rag_singleton():
    global _rag_singleton
    _rag_singleton = None


def _inject_rag_singleton(rag_instance):
    global _rag_singleton
    _rag_singleton = rag_instance


def _sid_to_global_idx(r, scenario_id: str) -> Optional[int]:
    for idx, case in enumerate(r.cases):
        if case["scenario_id"] == scenario_id:
            return idx
    return None


def _max_cosine_to_indices(vec: np.ndarray, matrix: np.ndarray, indices: List[int]) -> float:
    if not indices:
        return 0.0
    sub = matrix[indices].astype(np.float32)
    v = vec.astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
    return float(np.max(sub @ v.reshape(-1, 1)))


def duplicate_score(scenario_id: str, cache: Optional[Dict] = None, _rag=None) -> float:
    """
    Returns the max cosine similarity between this scenario's feature vector
    and any *other* scenario in the training set.

    Range: [0.0, 1.0]. Higher = more duplicate-like.
    Suggested threshold (A will decide): > 0.98 → drop as near-duplicate.

    `cache` is an optional pre-computed similarity matrix to avoid recomputation
    across calls. If None, compute from scratch.
    """
    if cache is not None and scenario_id in cache:
        return float(cache[scenario_id])

    r = _get_rag(_rag=_rag)
    global_idx = _sid_to_global_idx(r, scenario_id)
    if global_idx is None:
        return 0.0

    family = r.cases[global_idx].get("family", "")
    family_idxs = r.family_indices.get(family, [])
    if len(family_idxs) < 2:
        return 0.0

    query_vec = r.feature_matrix[global_idx].astype(np.float32)
    sub = r.feature_matrix[family_idxs].astype(np.float32)

    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    norms = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8
    sims = ((sub / norms) @ q_norm.reshape(-1, 1)).flatten()

    try:
        local_self = family_idxs.index(global_idx)
        sims[local_self] = -1.0
    except ValueError:
        pass

    return float(np.max(sims))


def outlier_score(scenario_id: str, _rag=None) -> float:
    """
    Returns a distance-to-class-centroid score. Higher = more anomalous within
    its own class.

    Range: [0.0, ∞). Typically 0-5.
    Suggested threshold (A will decide): > 3.0 → flag for review.
    """
    r = _get_rag(_rag=_rag)
    global_idx = _sid_to_global_idx(r, scenario_id)
    if global_idx is None:
        return 0.0

    action_type = r.cases[global_idx].get("action_type", "unknown")
    action_idxs = r._action_indices.get(action_type, [])
    if not action_idxs:
        return 0.0

    centroid = r.feature_matrix[action_idxs].mean(axis=0)
    vec = r.feature_matrix[global_idx]
    return float(np.linalg.norm(vec - centroid))


def label_confidence(scenario_id: str, _rag=None) -> float:
    """
    Returns confidence that the ground-truth label is correct and unambiguous.

    Range: [0.0, 1.0]. Higher = more trustworthy label.
    Computed from:
      - Is gt in the option list? (hard check: 0.0 if no)
      - How well does this sample match its own action bucket?
      - How strongly is it competed by rival actions in the same family?
    """
    r = _get_rag(_rag=_rag)
    global_idx = _sid_to_global_idx(r, scenario_id)
    if global_idx is None:
        return 0.0

    case = r.cases[global_idx]
    vec = r.feature_matrix[global_idx]

    answer_str = case.get("answer", "")
    gt_c_code = answer_str.split("|")[0].strip() if answer_str else ""
    options = case.get("options", {})
    if not gt_c_code or gt_c_code not in options:
        return 0.0

    action_type = case.get("action_type", "unknown")
    same_action = [gi for gi in r._action_indices.get(action_type, []) if gi != global_idx]
    if same_action:
        same_support = _max_cosine_to_indices(vec, r.feature_matrix, same_action)
    else:
        same_support = 0.5

    family = case.get("family", "")
    rival_idxs = [
        gi for gi in r.family_indices.get(family, [])
        if gi != global_idx and r.cases[gi].get("action_type") != action_type
    ]
    rival_support = _max_cosine_to_indices(vec, r.feature_matrix, rival_idxs) if rival_idxs else 0.0

    margin = max(0.0, same_support - rival_support)
    return float(np.clip(0.4 + 0.4 * same_support + 0.2 * margin, 0.0, 1.0))


def ambiguity_score(scenario_id: str, _rag=None) -> float:
    """
    Returns how close the sample lies to a cross-action boundary.

    Range: [0.0, 1.0]. Higher = more ambiguous.
    Suggested use (A will decide): down-weight (not drop) in training loss.
    """
    r = _get_rag(_rag=_rag)
    global_idx = _sid_to_global_idx(r, scenario_id)
    if global_idx is None:
        return 0.0

    case = r.cases[global_idx]
    vec = r.feature_matrix[global_idx]
    action_type = case.get("action_type", "unknown")
    family = case.get("family", "")

    same_action = [gi for gi in r._action_indices.get(action_type, []) if gi != global_idx]
    rival_idxs = [
        gi for gi in r.family_indices.get(family, [])
        if gi != global_idx and r.cases[gi].get("action_type") != action_type
    ]

    same_support = _max_cosine_to_indices(vec, r.feature_matrix, same_action) if same_action else 0.0
    rival_support = _max_cosine_to_indices(vec, r.feature_matrix, rival_idxs) if rival_idxs else 0.0

    if same_support <= 1e-8:
        return float(np.clip(rival_support, 0.0, 1.0))
    return float(np.clip(rival_support / max(same_support, 1e-8), 0.0, 1.0))


def score_all_scenarios(scenario_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Convenience: run all four scorers and return a dict keyed by scenario_id.

    Output shape:
        {scenario_id: {
            "duplicate":  float,
            "outlier":    float,
            "confidence": float,
            "ambiguity":  float,
        }}
    """
    r = _get_rag()

    dup_cache: Dict[str, float] = {}
    for _, fidxs in r.family_indices.items():
        if len(fidxs) < 2:
            for gi in fidxs:
                dup_cache[r.cases[gi]["scenario_id"]] = 0.0
            continue
        sub = r.feature_matrix[fidxs].astype(np.float32)
        sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
        sim_matrix = sub @ sub.T
        np.fill_diagonal(sim_matrix, -1.0)
        max_sims = sim_matrix.max(axis=1)
        for li, gi in enumerate(fidxs):
            dup_cache[r.cases[gi]["scenario_id"]] = float(max_sims[li])

    try:
        from tqdm import tqdm

        iterator = tqdm(scenario_ids, desc="Scoring") if sys.stdout.isatty() else scenario_ids
    except ImportError:
        iterator = scenario_ids

    out: Dict[str, Dict[str, float]] = {}
    for sid in iterator:
        out[sid] = {
            "duplicate": duplicate_score(sid, cache=dup_cache, _rag=r),
            "outlier": outlier_score(sid, _rag=r),
            "confidence": label_confidence(sid, _rag=r),
            "ambiguity": ambiguity_score(sid, _rag=r),
        }
    return out
