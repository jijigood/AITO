"""Text-based ReAct agent loop — compatible with any OpenAI-compatible chat endpoint."""

import logging
import time
from typing import Optional

from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIError

from tools import NOCClient
from prompts import SYSTEM_PROMPT, detect_family, extract_answer, parse_tool_call, _strip_thinking

MAX_ITERATIONS = 40
# Stop querying 40 s before the 10-min competition limit
MAX_WALL_SECONDS = 560

log = logging.getLogger(__name__)


def _call_llm(
    client: OpenAI,
    messages: list,
    model_name: str,
    max_tokens: int = 1024,
) -> Optional[str]:
    """Call the LLM and return the assistant content string, or None on failure."""
    base_wait = 2.0
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
            if attempt == 3:
                log.error("LLM call failed after 3 attempts: %s", exc)
                return None
            wait = base_wait * (2 ** (attempt - 1))
            log.warning("LLM attempt %d failed (%s), retry in %.0fs", attempt, exc, wait)
            time.sleep(wait)
        except APIError as exc:
            if exc.status_code and 400 <= exc.status_code < 500:
                log.error("LLM client error (not retrying): %s", exc)
                return None
            if attempt == 3:
                log.error("LLM server error after 3 attempts: %s", exc)
                return None
            time.sleep(base_wait)
    return None


def solve(
    question_text: str,
    question_id: int,
    llm_client: OpenAI,
    noc_client: NOCClient,
    model_name: str = "qwen",
) -> str:
    """
    Run a text-based ReAct loop for one question.

    The LLM alternates between:
    - Outputting a <tool_call> block  → we execute the command and feed the result back
    - Outputting the final answer     → we extract and return it

    Competition constraint: all tool calls within a question are sequential.
    """
    family = detect_family(question_text)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
    ]
    t0 = time.time()

    for iteration in range(MAX_ITERATIONS):
        if time.time() - t0 > MAX_WALL_SECONDS:
            log.warning("Q%d: wall-clock budget exceeded at iteration %d", question_id, iteration)
            break

        content = _call_llm(llm_client, messages, model_name, max_tokens=2048)
        if content is None:
            log.error("Q%d: LLM returned None at iteration %d", question_id, iteration)
            break

        messages.append({"role": "assistant", "content": content})

        # Check if the LLM wants to call a tool
        tool_args = parse_tool_call(content)
        if tool_args is None:
            # No tool call → this is the final answer
            log.info(
                "Q%d: final answer after %d tool calls (%.0fs)",
                question_id, noc_client.call_count, time.time() - t0,
            )
            return extract_answer(_strip_thinking(content), family)

        device_name, command = tool_args
        log.info(
            "Q%d it=%d [%d calls] %s | %s",
            question_id, iteration, noc_client.call_count, device_name, command[:60],
        )

        # Execute the command (sequential, as required by competition rules)
        result = noc_client.execute(device_name, command)
        log.info(
            "Q%d it=%d result → %s",
            question_id, iteration, result[:120].replace("\n", "\\n"),
        )

        # Truncate very long outputs to prevent context overflow (routing tables can be 13KB+)
        MAX_OUTPUT = 1500
        if len(result) > MAX_OUTPUT:
            result = result[:MAX_OUTPUT] + f"\n[... output truncated at {MAX_OUTPUT} chars ...]"

        # Feed the result back as a user message
        messages.append({
            "role": "user",
            "content": f"Command output:\n{result}",
        })

    # Fallback: ask for best-effort answer with a short budget
    log.warning("Q%d: max iterations/timeout reached, requesting best-effort answer", question_id)
    messages.append({
        "role": "user",
        "content": (
            "You have reached the time limit. Output your best final answer right now "
            "in the EXACT required format. No tool_call, no explanation — answer only."
        ),
    })
    content = _call_llm(llm_client, messages, model_name, max_tokens=256)
    if content:
        return extract_answer(_strip_thinking(content), family)
    return ""
