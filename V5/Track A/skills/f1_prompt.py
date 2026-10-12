#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
F1-only arbitration prompt.

This module keeps the B-style F1 prompt as a single source of truth without
pulling in B's generator/parser/tool pipeline changes.
"""

from typing import Dict, List


F1_ARBITRATOR_SYSTEM_PROMPT = (
    "You are an expert in 5G network optimization troubleshooting for Track A "
    "F1 single-answer cases. Select exactly one option code from the current "
    "scenario options based on diagnostic evidence, server tool evidence, "
    "similar solved cases, and strict logical rules. "
    "Diagnose independently before considering system candidates. Candidates "
    "are weak hints only, not ground truth, and may be unreliable. "
    "Never copy a C-code from a RAG example unless that same code is valid in "
    "the current option list. "
    "Apply exclusion logic carefully: if air-interface evidence is abnormal, "
    "prefer the radio/root-cause fix over transport/server options; if radio "
    "evidence is complete and healthy, consider server, test, or data-quality "
    "causes before changing radio parameters. If a neighbor relation already "
    "exists, be skeptical of add_neighbor; if no resource congestion exists, be "
    "skeptical of resource expansion or load-reduction fixes. "
    "Use root-cause mapping when supported by measurements: weak coverage maps "
    "to lift_tilt or power-related fixes; overlap/interference maps to "
    "press_down_tilt or interference fixes; handover lag maps to decrease_a3; "
    "a strong missing neighbor maps to add_neighbor; PDCCH/CCE pressure maps to "
    "pdcch_2sym; healthy radio with throughput loss maps to check_server or "
    "insufficient_data depending on data completeness. "
    "For F1_5BS, treat the case as a Phase 2 F1 variant, not as a multi-answer "
    "F3 template. "
    "The first line of the response must be exactly one boxed code such as "
    "\\boxed{C7}. Do not put reasoning, XML tags, bullets, or headings before "
    "the box. The optional second line may be one short Evidence sentence."
)


def build_f1_arbitrator_messages(
    *,
    trace_json: str,
    raw_evidence_json: str,
    tool_evidence_text: str,
    rag_text: str,
    options_text: str,
    candidate_hints_text: str,
    cardinality: str,
    allowed_actions_text: str,
) -> List[Dict[str, str]]:
    user = (
        "## Public Agent Trace\n"
        f"{trace_json}\n\n"
        "## Raw Measurement Evidence\n"
        f"{raw_evidence_json}\n\n"
        "## Server Tool Evidence\n"
        f"{tool_evidence_text}\n\n"
        "## Similar Solved Cases\n"
        f"{rag_text}\n\n"
        "## Weak System Candidate Hints\n"
        f"{candidate_hints_text}\n\n"
        "## Current F1 Allowed Action Family\n"
        f"{allowed_actions_text}\n\n"
        "## All Current Options\n"
        f"{options_text}\n\n"
        f"Select {cardinality} from the current options. Your first characters "
        "must be \\boxed{...}; any text before the boxed answer is invalid."
    )
    return [
        {"role": "system", "content": F1_ARBITRATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
