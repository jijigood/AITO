#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Regression coverage for the deterministic + tools F1 baseline.

These tests intentionally avoid a live FastAPI server. They pin the Owner A
integration contract around ToolEvidence -> feature augmentation -> candidate
ranking, using heldout scenarios that caught real regressions.
"""

import json
from pathlib import Path

from skills.candidate_generator import generate_single_candidates
from skills.feature_engine import run_all_skills
from skills.scenario_parser import parse_scenario
from skills.scenario_router import route
from skills.tool_caller import ToolCaller, ToolEvidence


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "Phase_1"
TRAIN_JSON = DATA_DIR / "train.json"


def _load_raw_scenario(scenario_id: str) -> dict:
    with TRAIN_JSON.open("r", encoding="utf-8") as fh:
        for raw in json.load(fh):
            if raw["scenario_id"] == scenario_id:
                return raw
    raise AssertionError(f"fixture scenario not found: {scenario_id}")


def _top_candidate_after_tools(scenario_id: str, evidence_builder):
    raw = _load_raw_scenario(scenario_id)
    ps = parse_scenario(raw)
    sf = run_all_skills(ps)
    evidence = evidence_builder(sf)
    sf = ToolCaller("http://unused").augment_features(sf, evidence)
    family = route(ps.tag, ps.num_base_stations)
    candidates = generate_single_candidates(ps, sf, family, top_n=5)
    assert candidates, f"no candidates generated for {scenario_id}"
    return raw["answer"], candidates[0], sf


def test_tools_promote_insufficient_data_from_sparse_mr():
    answer, top, sf = _top_candidate_after_tools(
        "e3e0a301-7a05-4b56-884c-4853b56c3c51",
        lambda sf: ToolEvidence(
            mainlobe_flags={sf.mobility.serving_pci_at_worst: True},
            mr_sample_count=5,
            tool_call_count=2,
        ),
    )

    assert sf.top_issue()[0] == "insufficient_data"
    assert top.action_type == "insufficient_data"
    assert top.c_code == answer == "C16"


def test_tools_promote_tertiary_lift_tilt_from_mainlobe_signal():
    answer, top, sf = _top_candidate_after_tools(
        "e66dbb45-38e7-440a-b173-124319e05954",
        lambda sf: ToolEvidence(
            mainlobe_flags={sf.mobility.serving_pci_at_worst: False},
            mr_sample_count=5,
            tool_call_count=2,
        ),
    )

    assert sf.geometry.recommended_action_family == "lift_tilt"
    assert sf.geometry.recommended_target_role == "tertiary"
    assert top.action_type == "lift_tilt"
    assert top.c_code == answer == "C18"


def test_sparse_mr_does_not_override_check_server():
    answer, top, sf = _top_candidate_after_tools(
        "e54d2633-e5c1-4fc6-813b-0a5a40e75349",
        lambda sf: ToolEvidence(
            mr_sample_count=5,
            tool_call_count=1,
        ),
    )

    assert sf.top_issue()[0] == "backhaul_server"
    assert top.action_type == "check_server"
    assert top.c_code == answer == "C2"
