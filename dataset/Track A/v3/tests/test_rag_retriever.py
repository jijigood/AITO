#!/usr/bin/env python
# -*-coding:utf-8 -*-

from types import SimpleNamespace

import numpy as np

from main_v3 import _retrieve_rag_hits
from rag_retriever_v2 import StructuredCaseRAG


def _mock_sf():
    return SimpleNamespace(
        throughput=SimpleNamespace(
            worst_throughput_mbps=50.0,
            throughput_drop_magnitude=30.0,
        ),
        mobility=SimpleNamespace(
            serving_rsrp_dbm=-90.0,
            serving_sinr_db=5.0,
            neighbor_advantage_db=3.0,
            a3_margin_db=0.0,
            visible_neighbor_pcis=[1, 2, 3],
        ),
        inter_freq=SimpleNamespace(max_dl_prb_util=80.0),
        score_late_handover=0.1,
        score_missing_neighbor=0.05,
        score_interference=0.2,
        score_coverage_hole=0.7,
        score_inter_freq_load=0.15,
        tool_evidence=None,
        top_issue=lambda: ("coverage_hole", 0.7),
    )


def test_retrieve_respects_k_when_confusion_pairs_are_enabled():
    actions = [
        "lift_tilt",
        "lift_tilt",
        "insufficient_data",
        "insufficient_data",
        "add_neighbor",
        "add_neighbor",
        "pdcch_2sym",
        "pdcch_2sym",
        "check_server",
        "check_server",
        "decrease_a3",
        "decrease_a3",
    ]

    rag = StructuredCaseRAG.__new__(StructuredCaseRAG)
    rag.feature_matrix = np.eye(16, dtype=np.float32)[: len(actions)]
    rag.family_indices = {"F1": list(range(len(actions)))}
    rag._action_indices = {}
    rag._confusion_index = {"lift_tilt": {}, "insufficient_data": {}}
    rag.cases = []
    for idx, action in enumerate(actions):
        rag._action_indices.setdefault(action, []).append(idx)
        rag.cases.append(
            {
                "scenario_id": f"s{idx}",
                "family": "F1",
                "action_type": action,
                "answer": f"C{idx + 1}",
                "feature_vector": rag.feature_matrix[idx],
            }
        )

    hits = rag.retrieve(_mock_sf(), k=6, family_name="F1", include_confusion_pairs=True)

    assert len(hits) <= 6
    assert any(hit.is_confusion_pair for hit in hits)


def test_retrieve_bridge_defensively_truncates_to_top_k():
    class FakeRAG:
        def retrieve(self, sf, k, family_name):
            return [{"scenario_id": f"s{i}"} for i in range(k + 2)]

    hits = _retrieve_rag_hits(FakeRAG(), _mock_sf(), "F1", top_k=3)

    assert len(hits) == 3
