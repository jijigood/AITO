#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Scenario Router — V2 Architecture

Classifies each scenario into one of four families BEFORE any solving begins.
Family assignment determines:
  - which skill bundle to run
  - expected answer cardinality (1 or 2–4)
  - retrieval filter for structured RAG

Families (from ARCHITECTURE_V2.md §5):
  F1:  4 BS + single-answer   (largest family, ~72% of training data)
  F2:  4 BS + multiple-answer (~11%)
  F3:  5 BS + multiple-answer (~4%)  — always exactly 4 answers
  F4: 13 BS + single-answer   (~12%) — inter-frequency / load focus

Critical rules derived from training data:
  - BS=5  → ALWAYS multiple-answer with exactly 4 options
  - BS=13 → ALWAYS single-answer
  - BS=4  → tag determines single vs multiple
"""

from dataclasses import dataclass
from typing import Literal

FamilyName = Literal["F1", "F2", "F3", "F4"]


@dataclass(frozen=True)
class Family:
    name: FamilyName
    num_bs: int
    is_multiple: bool
    expected_answer_count: int        # 1 for single; for multi: exact count or 0=variable
    description: str
    primary_skill_bundle: str         # which skill set to prioritise


FAMILIES: dict[FamilyName, Family] = {
    "F1": Family(
        name="F1",
        num_bs=4,
        is_multiple=False,
        expected_answer_count=1,
        description="4 BS + single-answer (mobility, coverage, PDCCH, server)",
        primary_skill_bundle="full",
    ),
    "F2": Family(
        name="F2",
        num_bs=4,
        is_multiple=True,
        expected_answer_count=2,      # usually 2, sometimes 4
        description="4 BS + multiple-answer (composite 2-action fix)",
        primary_skill_bundle="composite",
    ),
    "F3": Family(
        name="F3",
        num_bs=5,
        is_multiple=True,
        expected_answer_count=4,      # always exactly 4
        description="5 BS + multiple-answer (always 4 actions)",
        primary_skill_bundle="composite_4",
    ),
    "F4": Family(
        name="F4",
        num_bs=13,
        is_multiple=False,
        expected_answer_count=1,
        description="13 BS + single-answer (inter-freq / load balancing focus)",
        primary_skill_bundle="inter_freq",
    ),
}


def route(scenario_tag: str, num_base_stations: int) -> Family:
    """
    Assign a Family to a scenario based on its tag and BS count.

    Args:
        scenario_tag:       "single-answer" | "multiple-answer"
        num_base_stations:  integer from context.wireless_network_information

    Returns:
        Family dataclass instance
    """
    is_multiple = "multiple" in scenario_tag.lower()

    if num_base_stations == 5:
        return FAMILIES["F3"]

    if num_base_stations == 13:
        return FAMILIES["F4"]

    if num_base_stations == 4:
        return FAMILIES["F2"] if is_multiple else FAMILIES["F1"]

    # Fallback for unexpected BS counts: use tag to decide
    return FAMILIES["F2"] if is_multiple else FAMILIES["F1"]


def expected_answer_count(family: Family, actual_tag: str) -> tuple[int, int]:
    """
    Return (min_answers, max_answers) for a given family.

    Used by the candidate generator and multi-answer composer to know
    how many C-codes to emit.
    """
    if family.name == "F3":
        return (4, 4)
    if family.name == "F4" or family.name == "F1":
        return (1, 1)
    # F2: 2-option answers dominate (76%), 4-option answers are 24%
    # Default to 2 unless the candidate generator finds strong evidence for 4
    return (2, 4)
