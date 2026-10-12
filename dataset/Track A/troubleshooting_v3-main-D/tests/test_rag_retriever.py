#!/usr/bin/env python
# -*-coding:utf-8 -*-

import json
from types import SimpleNamespace

import numpy as np

from rag_retriever_v2 import RAGHit, StructuredCaseRAG


def _write_train_json(path):
    examples = [
        {
            "scenario_id": "s1",
            "tag": "single-answer",
            "context": {"wireless_network_information": {"num_base_stations": "4"}},
            "answer": "C1",
            "task": {
                "options": [
                    {"id": "C1", "label": "Lift the tilt of cell 3279943_1 by 2 degrees"},
                    {"id": "C2", "label": "Check test server"},
                ]
            },
            "data": {},
        },
        {
            "scenario_id": "s2",
            "tag": "single-answer",
            "context": {"wireless_network_information": {"num_base_stations": "4"}},
            "answer": "C2",
            "task": {
                "options": [
                    {"id": "C1", "label": "Insufficient data"},
                    {"id": "C2", "label": "Check test server"},
                ]
            },
            "data": {},
        },
    ]
    path.write_text(json.dumps(examples), encoding="utf-8")


def _write_confusion_train_json(path):
    examples = [
        {
            "scenario_id": "lt0",
            "tag": "single-answer",
            "context": {"wireless_network_information": {"num_base_stations": "4"}},
            "answer": "C1",
            "task": {
                "options": [
                    {"id": "C1", "label": "Lift the tilt of cell 3279943_1 by 2 degrees"},
                    {"id": "C2", "label": "Insufficient data"},
                ]
            },
            "data": {},
        },
        {
            "scenario_id": "id0",
            "tag": "single-answer",
            "context": {"wireless_network_information": {"num_base_stations": "4"}},
            "answer": "C2",
            "task": {
                "options": [
                    {"id": "C1", "label": "Check test server"},
                    {"id": "C2", "label": "Insufficient data"},
                ]
            },
            "data": {},
        },
    ]
    path.write_text(json.dumps(examples), encoding="utf-8")


def _make_mock_sf(tool_evidence=None):
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
        tool_evidence=tool_evidence,
    )


def test_d4_public_interface_keeps_legacy_path(tmp_path):
    train_json = tmp_path / "train.json"
    _write_train_json(train_json)

    rag = StructuredCaseRAG(str(train_json))
    rag.CACHE_DIR = str(tmp_path / "cache")
    rag.build_index(force_rebuild=True)

    assert hasattr(rag, "_retrieve_raw")

    raw_hits = rag._retrieve_raw(rag.feature_matrix[0][:13], "F1", top_k=1)
    assert len(raw_hits) == 1
    assert raw_hits[0]["scenario_id"] == "s1"

    formatted = rag.retrieve_and_format(rag.feature_matrix[0][:13], "F1", top_k=1)
    assert "Similar Solved Cases" in formatted
    assert "Correct answer" in formatted

    empty = rag.retrieve(None, k=6)
    assert empty == []

    hit = RAGHit(
        scenario_id="sid",
        features={},
        gt_c_code="C1",
        gt_action_type="lift_tilt",
        similarity=0.9,
        prompt_snippet="snippet",
    )
    assert hit.is_confusion_pair is False


def test_d1a_build_index_adds_action_type_buckets(tmp_path):
    train_json = tmp_path / "train.json"
    _write_train_json(train_json)

    rag = StructuredCaseRAG(str(train_json))
    rag.CACHE_DIR = str(tmp_path / "cache")
    rag.build_index(force_rebuild=True)

    assert all("action_type" in case for case in rag.cases)
    assert rag._action_indices["lift_tilt"] == [0]
    assert rag._action_indices["check_server"] == [1]


def test_d1b_hash_cache_round_trip_loads_action_indices(tmp_path):
    train_json = tmp_path / "train.json"
    _write_train_json(train_json)
    cache_dir = tmp_path / "cache"

    rag = StructuredCaseRAG(str(train_json))
    rag.CACHE_DIR = str(cache_dir)
    rag.build_index(force_rebuild=True)

    assert (cache_dir / "rag_v2_cases.pkl").exists()
    assert (cache_dir / "rag_v2_matrix.npy").exists()
    assert (cache_dir / "data_hash.txt").exists()

    rag2 = StructuredCaseRAG(str(train_json))
    rag2.CACHE_DIR = str(cache_dir)
    rag2.build_index()

    assert rag2.feature_matrix is not None
    assert rag2.feature_matrix.shape == (2, rag2.FEATURE_DIM)
    assert rag2._action_indices == rag._action_indices


def test_d2a_helper_methods_on_synthetic_index():
    rag = StructuredCaseRAG.__new__(StructuredCaseRAG)
    rag.feature_matrix = np.eye(16, dtype=np.float32)

    query = np.ones(16, dtype=np.float32)
    sims = rag._cosine_sim_to_subset(query, [0, 1, 2])
    assert sims.shape == (3,)
    assert len(rag._cosine_sim_to_subset(query, [])) == 0

    sf = _make_mock_sf()
    vec = StructuredCaseRAG._sf_to_feature_vec(sf)
    assert vec.shape == (16,)
    assert vec.dtype == np.float32
    assert np.all(vec >= 0.0)
    assert np.all(vec <= 1.0)

    hits = [
        {"feature_vector": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)},
        {"feature_vector": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)},
    ]
    kept = StructuredCaseRAG._deduplicate(hits, hits[0]["feature_vector"], threshold=0.98)
    assert len(kept) == 1

    case = {
        "scenario_id": "s1",
        "family": "F1",
        "action_type": "lift_tilt",
        "answer": "C3",
        "feature_vector": np.zeros(16, dtype=np.float32),
    }
    hit = rag._case_to_raghit(case, 0.85)
    assert isinstance(hit, RAGHit)
    assert hit.similarity == 0.85
    assert hit.is_confusion_pair is False
    assert len(hit.prompt_snippet) <= 120


def test_d2b_retrieve_supports_stratified_and_flat_paths():
    class MockSF:
        class throughput:
            worst_throughput_mbps = 50.0
            throughput_drop_magnitude = 30.0

        class mobility:
            serving_rsrp_dbm = -90.0
            serving_sinr_db = 5.0
            neighbor_advantage_db = 3.0
            a3_margin_db = 0.0
            visible_neighbor_pcis = [1, 2, 3]

        class inter_freq:
            max_dl_prb_util = 80.0

        score_late_handover = 0.1
        score_missing_neighbor = 0.05
        score_interference = 0.2
        score_coverage_hole = 0.7
        score_inter_freq_load = 0.15

    rag = StructuredCaseRAG.__new__(StructuredCaseRAG)
    rag.feature_matrix = np.eye(16, dtype=np.float32)
    rag.family_indices = {"F1": [0, 1, 2, 3, 4, 5]}
    rag._action_indices = {
        "lift_tilt": [0],
        "insufficient_data": [1],
        "add_neighbor": [2],
        "pdcch_2sym": [3],
        "check_server": [4],
        "decrease_a3": [5],
    }
    rag.cases = [
        {
            "scenario_id": f"s{i}",
            "family": "F1",
            "action_type": action,
            "answer": f"C{i}",
            "feature_vector": rag.feature_matrix[i],
        }
        for i, action in enumerate(
            [
                "lift_tilt",
                "insufficient_data",
                "add_neighbor",
                "pdcch_2sym",
                "check_server",
                "decrease_a3",
            ]
        )
    ]

    hits = rag.retrieve(MockSF(), k=6, include_confusion_pairs=False)
    assert len(hits) <= 6
    assert all(isinstance(h, RAGHit) for h in hits)
    assert len({h.gt_action_type for h in hits}) >= 2

    flat_hits = rag.retrieve(MockSF(), k=3, stratify_by_action=False, include_confusion_pairs=False)
    assert len(flat_hits) <= 3
    assert all(isinstance(h, RAGHit) for h in flat_hits)


def test_d3a_build_confusion_index_and_cache_round_trip(tmp_path):
    rag = StructuredCaseRAG.__new__(StructuredCaseRAG)
    rag.cases = [
        {"scenario_id": "lt0"},
        {"scenario_id": "lt1"},
        {"scenario_id": "id0"},
        {"scenario_id": "id1"},
    ]
    rag._action_indices = {"lift_tilt": [0, 1], "insufficient_data": [2, 3]}
    rag.feature_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [1.0, 0.0, 0.1],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    ci = rag._build_confusion_index()
    assert "lift_tilt" in ci and "insufficient_data" in ci
    assert "lt0" in ci["lift_tilt"]
    assert "id0" in ci["insufficient_data"]

    train_json = tmp_path / "train.json"
    _write_confusion_train_json(train_json)
    cache_dir = tmp_path / "cache"

    rag2 = StructuredCaseRAG(str(train_json))
    rag2.CACHE_DIR = str(cache_dir)
    rag2.build_index(force_rebuild=True)

    assert (cache_dir / "confusion_index.pkl").exists()
    assert "lift_tilt" in rag2._confusion_index
    assert "insufficient_data" in rag2._confusion_index

    rag3 = StructuredCaseRAG(str(train_json))
    rag3.CACHE_DIR = str(cache_dir)
    rag3.build_index()
    assert rag3._confusion_index == rag2._confusion_index


def test_d3b_confusion_pairs_are_appended_when_triggered():
    class MockSF:
        class throughput:
            worst_throughput_mbps = 50.0
            throughput_drop_magnitude = 30.0

        class mobility:
            serving_rsrp_dbm = -120.0
            serving_sinr_db = -2.0
            neighbor_advantage_db = 3.0
            a3_margin_db = 0.0
            visible_neighbor_pcis = [1]

        class inter_freq:
            max_dl_prb_util = 20.0

        score_late_handover = 0.1
        score_missing_neighbor = 0.05
        score_interference = 0.2
        score_coverage_hole = 0.9
        score_inter_freq_load = 0.15

        def top_issue(self):
            return ("coverage_hole", 0.9)

    rag = StructuredCaseRAG.__new__(StructuredCaseRAG)
    rag.feature_matrix = np.eye(16, dtype=np.float32)
    rag.family_indices = {"F1": [0, 1]}
    rag._action_indices = {"lift_tilt": [0], "insufficient_data": [1]}
    rag._confusion_index = {
        "lift_tilt": {"s0": ["s1"]},
        "insufficient_data": {"s1": ["s0"]},
    }
    rag.cases = [
        {
            "scenario_id": "s0",
            "family": "F1",
            "action_type": "lift_tilt",
            "answer": "C0",
            "feature_vector": rag.feature_matrix[0],
        },
        {
            "scenario_id": "s1",
            "family": "F1",
            "action_type": "insufficient_data",
            "answer": "C1",
            "feature_vector": rag.feature_matrix[1],
        },
    ]

    pairs = rag._get_confusion_pairs(np.ones(16, dtype=np.float32), [0, 1])
    assert 1 <= len(pairs) <= 2
    assert all(h.is_confusion_pair for h in pairs)
    assert any("contrast=" in h.prompt_snippet for h in pairs)

    result = rag.retrieve(MockSF(), k=1, include_confusion_pairs=True)
    assert len(result) >= 2
    assert any(h.is_confusion_pair for h in result)


def test_d6_extract_feature_vector_shape_16():
    vec = StructuredCaseRAG._extract_feature_vector({"data": {}})
    assert vec.shape == (16,)
    assert np.all(vec[13:] == 0.0)


def test_d6_sf_to_feature_vec_tool_dims_nonzero():
    tool_evidence = SimpleNamespace(
        mainlobe_flags={1: True, 2: False},
        tilt_angles={1: 10.0},
        overlap_ratio=0.7,
    )
    vec = StructuredCaseRAG._sf_to_feature_vec(_make_mock_sf(tool_evidence=tool_evidence))
    assert vec.shape == (16,)
    assert abs(vec[13] - 0.5) < 1e-4
    assert abs(vec[14] - 0.5) < 1e-4
    assert abs(vec[15] - 0.7) < 1e-4


def test_d6_sf_to_feature_vec_none_tool_evidence():
    vec = StructuredCaseRAG._sf_to_feature_vec(_make_mock_sf(tool_evidence=None))
    assert vec.shape == (16,)
    assert vec[13] == 0.0
    assert vec[14] == 0.0
    assert vec[15] == 0.0


def test_d6_sf_to_feature_vec_missing_tool_evidence_attr():
    sf = _make_mock_sf()
    del sf.tool_evidence
    vec = StructuredCaseRAG._sf_to_feature_vec(sf)
    assert vec.shape == (16,)
    assert np.all(vec[13:] == 0.0)
