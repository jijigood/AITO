"""Text-based ReAct agent loop — compatible with any OpenAI-compatible chat endpoint."""

import ipaddress
import logging
import os
import re
import time
from typing import Optional

from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIError

from tools import NOCClient, normalize_command
from prompts import REASONING_HINT, SYSTEM_PROMPT, detect_family, extract_answer, parse_tool_call, _strip_thinking
from answer_validation import normalize_answer, validate_answer
from finalize import strict_emit_answer
from playbooks.dispatcher import run_playbooks

MAX_ITERATIONS = int(os.environ.get("TRACKB_MAX_ITERATIONS", "20"))
# Stop querying 40 s before the 10-min competition limit
MAX_WALL_SECONDS = 560
TOOL_MAX_TOKENS = int(os.environ.get("TRACKB_TOOL_MAX_TOKENS", "768"))
FINAL_MAX_TOKENS = int(os.environ.get("TRACKB_FINAL_MAX_TOKENS", "192"))
MAX_OUTPUT_CHARS = int(os.environ.get("TRACKB_MAX_OUTPUT_CHARS", "1800"))
MAX_OBSERVATION_CHARS = int(os.environ.get("TRACKB_MAX_OBSERVATION_CHARS", "6500"))
ENABLE_DIRECT_TOPOLOGY = os.environ.get("TRACKB_ENABLE_DIRECT_TOPOLOGY", "1") != "0"
ENABLE_DIRECT_PATH = os.environ.get("TRACKB_ENABLE_DIRECT_PATH", "1") != "0"
ENABLE_PLAYBOOKS = os.environ.get("TRACKB_ENABLE_PLAYBOOKS", "1") != "0"

log = logging.getLogger(__name__)


def _record_llm(noc_client: NOCClient, phase: str, content: str, **extra) -> None:
    noc_client.llm_trace.append({
        "phase": phase,
        "content_preview": (content or "")[:1200],
        **extra,
    })


def _truncate_middle(text: str, limit: int) -> str:
    """Keep the beginning and end of long CLI output; both often carry useful facts."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = max(1, int(limit * 0.65))
    tail = max(1, limit - head)
    return (
        text[:head].rstrip()
        + f"\n[... output truncated to {limit} chars ...]\n"
        + text[-tail:].lstrip()
    )


def _question_ips(question_text: str) -> list[ipaddress.IPv4Address]:
    ips: list[ipaddress.IPv4Address] = []
    for raw in re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", question_text):
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            continue
        if isinstance(ip, ipaddress.IPv4Address) and ip not in ips:
            ips.append(ip)
    return ips


def _summarize_routing_table(output: str, question_text: str, limit: int) -> str | None:
    ips = _question_ips(question_text)
    if not ips:
        return None

    best: dict[ipaddress.IPv4Address, tuple[int, str]] = {}
    for line in output.splitlines():
        match = re.search(r"\b(\d{1,3}(?:\.\d{1,3}){3}/\d{1,2})\b", line)
        if not match:
            continue
        try:
            network = ipaddress.ip_network(match.group(1), strict=False)
        except ValueError:
            continue
        for ip in ips:
            if ip in network and network.prefixlen > best.get(ip, (-1, ""))[0]:
                best[ip] = (network.prefixlen, line.strip())

    if not best:
        return None

    lines = ["[ROUTING TABLE SUMMARY] Longest-prefix matches for IPs mentioned in the question:"]
    for ip in ips:
        if ip in best:
            lines.append(f"{ip}: {best[ip][1]}")
    lines.append("")
    lines.append("[ROUTING TABLE HEAD]")
    lines.extend(output.splitlines()[:12])
    return _truncate_middle("\n".join(lines), limit)


def _summarize_output(output: str, command: str, question_text: str, limit: int) -> str:
    normalized_command = " ".join(command.lower().split())
    if (
        normalized_command.startswith("display ip routing-table")
        or normalized_command.startswith("show ip route")
        or normalized_command.startswith("show route")
    ):
        summary = _summarize_routing_table(output, question_text, limit)
        if summary:
            return summary
    return _truncate_middle(output, limit)


def _build_messages(question_text: str, observations: list[dict]) -> list[dict]:
    family = detect_family(question_text)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
    ]
    hint = REASONING_HINT.get(family)
    if hint:
        messages.append({"role": "user", "content": hint})
    if not observations:
        return messages

    blocks = []
    budget = MAX_OBSERVATION_CHARS
    for obs in reversed(observations):
        block = (
            f"Observation {obs['idx']}:\n"
            f"device: {obs['device']}\n"
            f"command: {obs['command']}\n"
            f"output:\n{obs['output']}"
        )
        if len(block) > budget and blocks:
            break
        blocks.append(block[-budget:] if len(block) > budget else block)
        budget -= len(blocks[-1])
        if budget <= 0:
            break

    observations_text = "\n\n".join(reversed(blocks))
    messages.append({
        "role": "user",
        "content": (
            "Command observations collected so far are below. Use them as the current "
            "state, avoid repeating identical commands unless necessary, and either "
            "issue one next <tool_call> or provide the final <answer>.\n\n"
            f"{observations_text}"
        ),
    })
    return messages


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


def _extract_validated_answer(
    raw_content: str,
    family: str,
    question_text: str,
) -> tuple[str, tuple[str, ...]]:
    answer = extract_answer(_strip_thinking(raw_content), family, question_text)
    answer = normalize_answer(answer, family, question_text) or answer.strip()
    validation = validate_answer(answer, family, question_text)
    return answer, validation.reasons


def _finalize_answer(
    client: OpenAI,
    model_name: str,
    question_text: str,
    observations: list[dict],
    raw_content: str,
    family: str,
    question_id: int,
    llm_trace: list[dict] | None = None,
) -> str:
    """Validate final output; use strict emit before saving."""
    answer, reasons = _extract_validated_answer(raw_content, family, question_text)
    if not reasons:
        return answer

    log.warning(
        "Q%d: invalid final answer format (%s); attempting strict emit",
        question_id,
        "; ".join(reasons),
    )

    def emit_call(messages: list[dict[str, str]], max_tokens: int) -> str | None:
        emitted = _call_llm(client, messages, model_name, max_tokens=max_tokens)
        if llm_trace is not None and emitted is not None:
            llm_trace.append({
                "phase": "strict_emit",
                "content_preview": emitted[:1200],
            })
        return emitted

    emitted_answer = strict_emit_answer(
        question_text=question_text,
        family=family,
        observations=observations,
        initial_answer=raw_content,
        call_llm=emit_call,
        max_tokens=FINAL_MAX_TOKENS,
    )
    validation = validate_answer(emitted_answer, family, question_text)
    if validation.ok:
        log.info("Q%d: strict emit produced a valid answer", question_id)
    else:
        log.warning(
            "Q%d: strict emit answer is still invalid (%s)",
            question_id,
            "; ".join(validation.reasons),
        )
    return emitted_answer


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
    if ENABLE_PLAYBOOKS and (
        (family == "topology" and ENABLE_DIRECT_TOPOLOGY)
        or (family == "path" and ENABLE_DIRECT_PATH)
        or family == "fault"
    ):
        playbook_answer = run_playbooks(question_text, noc_client, family=family)
        if playbook_answer:
            log.info("Q%d: playbook answer after %d tool calls", question_id, noc_client.call_count)
            return playbook_answer

    observations: list[dict] = []
    executed_commands: set[tuple[str, str]] = set()
    duplicate_attempts = 0
    malformed_tool_attempts = 0
    t0 = time.time()

    for iteration in range(MAX_ITERATIONS):
        if time.time() - t0 > MAX_WALL_SECONDS:
            log.warning("Q%d: wall-clock budget exceeded at iteration %d", question_id, iteration)
            break

        messages = _build_messages(question_text, observations)
        content = _call_llm(llm_client, messages, model_name, max_tokens=TOOL_MAX_TOKENS)
        if content is None:
            log.error("Q%d: LLM returned None at iteration %d", question_id, iteration)
            break
        _record_llm(noc_client, "react", content, iteration=iteration)

        # Check if the LLM wants to call a tool
        tool_args = parse_tool_call(content)
        if tool_args is None:
            if "<tool_call" in content.lower():
                malformed_tool_attempts += 1
                log.warning(
                    "Q%d it=%d malformed tool_call; asking model to resend",
                    question_id,
                    iteration,
                )
                observations.append({
                    "idx": len(observations) + 1,
                    "device": "SYSTEM",
                    "command": "malformed tool_call",
                    "output": (
                        "[MALFORMED TOOL CALL] The previous response contained a "
                        "<tool_call> tag but did not include both 'device:' and "
                        "'command:'. Resend exactly one complete tool_call block, "
                        "or provide the final <answer> if enough information is known."
                    ),
                })
                if malformed_tool_attempts < 3:
                    continue
                log.warning("Q%d: too many malformed tool_call responses", question_id)

            # No tool call → this is the final answer
            log.info(
                "Q%d: final answer after %d tool calls (%.0fs)",
                question_id, noc_client.call_count, time.time() - t0,
            )
            return _finalize_answer(
                llm_client,
                model_name,
                question_text,
                observations,
                content,
                family,
                question_id,
                noc_client.llm_trace,
            )

        device_name, command = tool_args
        malformed_tool_attempts = 0
        command_key = (device_name.lower(), " ".join(normalize_command(command, device_name).lower().split()))
        if command_key in executed_commands:
            duplicate_attempts += 1
            log.warning(
                "Q%d it=%d duplicate command skipped: %s | %s",
                question_id, iteration, device_name, command[:60],
            )
            observations.append({
                "idx": len(observations) + 1,
                "device": device_name,
                "command": command,
                "output": (
                    "[REPEATED COMMAND SKIPPED] This exact command was already "
                    "executed. Choose a different command or provide the final answer."
                ),
            })
            if duplicate_attempts >= 3:
                break
            continue
        executed_commands.add(command_key)
        duplicate_attempts = 0

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

        observations.append({
            "idx": len(observations) + 1,
            "device": device_name,
            "command": command,
            "output": _summarize_output(result, command, question_text, MAX_OUTPUT_CHARS),
        })

    # Fallback: ask for best-effort answer with a short budget
    log.warning("Q%d: max iterations/timeout reached, requesting best-effort answer", question_id)
    messages = _build_messages(question_text, observations)
    messages.append({
        "role": "user",
        "content": (
            "Output your best final answer right now in the EXACT required format. "
            "No tool_call, no explanation — answer only."
        ),
    })
    content = _call_llm(llm_client, messages, model_name, max_tokens=FINAL_MAX_TOKENS)
    if content:
        _record_llm(noc_client, "best_effort", content)
        return _finalize_answer(
            llm_client,
            model_name,
            question_text,
            observations,
            content,
            family,
            question_id,
            noc_client.llm_trace,
        )
    return ""
