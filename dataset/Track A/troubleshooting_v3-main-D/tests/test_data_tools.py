#!/usr/bin/env python
# -*-coding:utf-8 -*-

import numpy as np

from skills.data_tools import (
    _inject_rag_singleton,
    _reset_rag_singleton,
    ambiguity_score,
    duplicate_score,
    label_confidence,
    outlier_score,
    score_all_scenarios,
)


class _FakeRAG:
    cases = [
        {
            "scenario_id": "s0",
            "family": "F1",
            "action_type": "lift_tilt",
            "answer": "C3",
            "options": {"C3": "Lift the tilt by 2 degrees"},
        },
        {
            "scenario_id": "s1",
            "family": "F1",
            "action_type": "lift_tilt",
            "answer": "C3",
            "options": {"C3": "Lift the tilt by 2 degrees"},
        },
        {
            "scenario_id": "s2",
            "family": "F1",
            "action_type": "lift_tilt",
            "answer": "C3",
            "options": {"C3": "Lift the tilt by 2 degrees"},
        },
        {
            "scenario_id": "s3",
            "family": "F1",
            "action_type": "check_server",
            "answer": "C5",
            "options": {"C5": "Check test server"},
        },
        {
            "scenario_id": "s4",
            "family": "F1",
            "action_type": "check_server",
            "answer": "C5",
            "options": {"C5": "Check test server"},
        },
        {
            "scenario_id": "s5",
            "family": "F1",
            "action_type": "add_neighbor",
            "answer": "C6",
            "options": {},
        },
    ]
    family_indices = {"F1": [0, 1, 2, 3, 4, 5]}
    _action_indices = {
        "lift_tilt": [0, 1, 2],
        "check_server": [3, 4],
        "add_neighbor": [5],
    }
    feature_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.9],
        [0.8, 0.2, 0.0],
    ], dtype=np.float32)


def test_d5a_duplicate_and_outlier_scores():
    _reset_rag_singleton()
    _inject_rag_singleton(_FakeRAG())

    try:
        assert duplicate_score("s1") > 0.9
        assert duplicate_score("missing") == 0.0
        assert outlier_score("s2") > 0.0
    finally:
        _reset_rag_singleton()


def test_d5b_label_confidence_ambiguity_and_batch_scoring():
    _reset_rag_singleton()
    _inject_rag_singleton(_FakeRAG())

    try:
        assert label_confidence("s5") == 0.0
        assert 0.0 <= label_confidence("s0") <= 1.0
        assert 0.0 <= ambiguity_score("s0") <= 1.0

        result = score_all_scenarios(["s0", "s1", "s5"])
        assert set(result.keys()) == {"s0", "s1", "s5"}
        for scores in result.values():
            assert set(scores.keys()) == {"duplicate", "outlier", "confidence", "ambiguity"}
    finally:
        _reset_rag_singleton()
