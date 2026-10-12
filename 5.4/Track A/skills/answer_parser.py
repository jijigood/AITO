#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Robust answer extraction and validation for Track A.

The model is expected to output answers enclosed in \\boxed{...}.
This module provides:
  - extract_boxed_answer : primary extractor (last \\boxed{} wins)
  - validate_answer      : ensure format is valid C-codes in ascending order
  - iou_score            : compute IOU between predicted and ground-truth answer strings
"""

import re
from typing import Optional


# All valid option IDs (C1–C22)
VALID_OPTIONS = {f"C{i}" for i in range(1, 23)}


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the content of the LAST \\boxed{...} in the response.

    Handles:
      - \\boxed{C9}
      - \\boxed{C3|C7}
      - \\boxed{C5|C9|C11|C20}
      - Nested braces: \\boxed{{C9}}  (double-brace escaping from f-strings)
      - Whitespace inside braces
      - Trailing punctuation

    Returns:
        The extracted answer string (e.g. "C9" or "C3|C7"), or None if not found.
    """
    if not text:
        return None

    # Match \boxed{ ... } allowing nested single-level braces (for f-string escaping)
    matches = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text)
    if not matches:
        return None

    raw = matches[-1].strip()
    # Strip surrounding braces (double-brace escaping artefact)
    raw = re.sub(r'^\{|\}$', '', raw).strip()
    # Remove trailing punctuation
    raw = raw.rstrip('.,;:')
    # Collapse whitespace
    raw = re.sub(r'\s+', '', raw)

    return raw if raw else None


def validate_and_normalise(answer: str) -> str:
    """
    Validate that all codes in the answer are in C1–C22, sort them in ascending
    numeric order, and return the pipe-delimited string.

    Invalid codes are silently dropped. Returns "" if nothing valid remains.

    Examples:
        "C7|C3"       → "C3|C7"
        "C22|C1|C10"  → "C1|C10|C22"
        "C99|C3"      → "C3"
        ""            → ""
    """
    if not answer:
        return ""

    parts = [p.strip().upper() for p in answer.split("|") if p.strip()]
    valid_parts = [p for p in parts if p in VALID_OPTIONS]

    if not valid_parts:
        return ""

    # Sort by numeric suffix
    valid_parts.sort(key=lambda x: int(x[1:]))
    return "|".join(valid_parts)


def parse_answer(text: str) -> str:
    """
    Full pipeline: extract → validate → normalise.

    Returns a normalised answer string like "C3" or "C3|C7", or "" on failure.
    """
    raw = extract_boxed_answer(text)
    if not raw:
        return ""
    return validate_and_normalise(raw)


def iou_score(predicted: str, ground_truth: str) -> float:
    """
    Compute Intersection-over-Union between predicted and ground-truth answer strings.

    This mirrors the competition evaluation metric exactly:
        score = |intersection| / |union|

    Args:
        predicted:    e.g. "C3|C7"   or "C9"
        ground_truth: e.g. "C3|C7"   or "C9"

    Returns:
        Float in [0.0, 1.0].
    """
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0

    pred_set = set(predicted.upper().split("|"))
    gt_set = set(ground_truth.upper().split("|"))

    intersection = pred_set & gt_set
    union = pred_set | gt_set

    if not union:
        return 0.0

    return len(intersection) / len(union)


def batch_iou(predictions: list, ground_truths: list) -> dict:
    """
    Compute aggregate IOU stats over a list of (predicted, ground_truth) pairs.

    Returns:
        dict with 'mean_iou', 'exact_match_rate', 'scores' (list of per-sample IOU)
    """
    scores = [iou_score(p, g) for p, g in zip(predictions, ground_truths)]
    exact = sum(1 for s in scores if s == 1.0)
    return {
        "mean_iou": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "exact_match_rate": round(exact / len(scores), 4) if scores else 0.0,
        "scores": scores,
    }
