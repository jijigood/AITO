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

from typing import Dict, List, Optional
import numpy as np

# TODO(Owner D): implement these four scoring functions.
# Return signatures are frozen; internals are yours to optimize.


def duplicate_score(scenario_id: str, cache: Optional[Dict] = None) -> float:
    """
    Returns the max cosine similarity between this scenario's feature vector
    and any *other* scenario in the training set.

    Range: [0.0, 1.0]. Higher = more duplicate-like.
    Suggested threshold (A will decide): > 0.98 → drop as near-duplicate.

    `cache` is an optional pre-computed similarity matrix to avoid recomputation
    across calls. If None, compute from scratch.
    """
    raise NotImplementedError("Owner D: D5 deliverable")


def outlier_score(scenario_id: str) -> float:
    """
    Returns a distance-to-class-centroid score. Higher = more anomalous within
    its own class.

    Range: [0.0, ∞). Typically 0-5.
    Suggested threshold (A will decide): > 3.0 → flag for review.
    """
    raise NotImplementedError("Owner D: D5 deliverable")


def label_confidence(scenario_id: str) -> float:
    """
    Returns confidence that the ground-truth label is correct and unambiguous.

    Range: [0.0, 1.0]. Higher = more trustworthy label.
    Computed from:
      - Is gt in the option list? (hard check: 0.0 if no)
      - How far is gt's score above the next-best option?
      - Does the feature pattern match the labeled action class?

    Suggested threshold (A will decide): < 0.3 → drop as likely label error.
    """
    raise NotImplementedError("Owner D: D5 deliverable")


def ambiguity_score(scenario_id: str) -> float:
    """
    Returns how close the second-best option is to the best, given the
    deterministic candidate generator's scoring.

    Range: [0.0, 1.0]. Higher = more ambiguous.
    Suggested use (A will decide): down-weight (not drop) in training loss.
    """
    raise NotImplementedError("Owner D: D5 deliverable")


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
    out = {}
    for sid in scenario_ids:
        out[sid] = {
            "duplicate":  duplicate_score(sid),
            "outlier":    outlier_score(sid),
            "confidence": label_confidence(sid),
            "ambiguity":  ambiguity_score(sid),
        }
    return out
