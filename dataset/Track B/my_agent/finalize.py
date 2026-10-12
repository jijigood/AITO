"""Strict final-answer emission for Track B.

This module deliberately keeps the final pass narrow: it receives the question,
validated CLI facts, and the exact output format, then asks the model to emit
only the competition answer. It should not receive the full ReAct transcript.
"""

from __future__ import annotations

import json
from typing import Callable

from answer_validation import extract_facts, normalize_answer, validate_answer
from prompts import _strip_thinking, extract_answer


LLMCall = Callable[[list[dict[str, str]], int], str | None]


def _clean_candidate(raw_content: str, family: str, question_text: str) -> tuple[str, tuple[str, ...]]:
    answer = extract_answer(_strip_thinking(raw_content or ""), family, question_text)
    answer = normalize_answer(answer, family, question_text) or answer.strip()
    validation = validate_answer(answer, family, question_text)
    return answer, validation.reasons


def strict_emit_answer(
    question_text: str,
    family: str,
    observations: list[dict],
    initial_answer: str = "",
    call_llm: LLMCall | None = None,
    max_tokens: int = 192,
) -> str:
    """
    Return a format-valid answer when possible.

    First trust an already-valid model answer. If it is invalid, ask for a
    single strict emission using only a compact fact bag.
    """
    answer, reasons = _clean_candidate(initial_answer, family, question_text)
    if not reasons:
        return answer
    if call_llm is None:
        return answer

    fact_bag = extract_facts(observations, family, question_text)
    messages = [
        {
            "role": "system",
            "content": (
                "You are the final-answer emitter for an exact-match networking "
                "competition. Output only one <answer>...</answer> block. Do not "
                "include reasoning, markdown, explanations, or tool calls."
            ),
        },
        {
            "role": "user",
            "content": (
                "Emit the final answer for this question using only the facts below.\n"
                "If the facts are insufficient, output the best answer that follows "
                "the required format exactly.\n\n"
                f"{json.dumps(fact_bag, ensure_ascii=False, indent=2)}\n\n"
                f"Previous invalid answer: {answer or initial_answer}\n"
                f"Validation problems: {'; '.join(reasons)}"
            ),
        },
    ]
    emitted = call_llm(messages, max_tokens)
    if emitted is None:
        return answer

    repaired, repaired_reasons = _clean_candidate(emitted, family, question_text)
    if not repaired_reasons:
        return repaired
    return repaired or answer
