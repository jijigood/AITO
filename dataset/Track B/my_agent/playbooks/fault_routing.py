"""High-confidence routing-fault playbook."""

import ipaddress

from tools import NOCClient

from playbooks.fault_common import (
    RouteHop,
    configured_static_routes_for,
    parse_fault_question,
    route_trace,
    validate_fault_answer,
)


def should_run(question_text: str, family: str) -> bool:
    lower = question_text.lower()
    if "pj area" in lower or "pjgfa area" in lower:
        return False
    return family == "fault" and "routing fault" in lower


def run(question_text: str, noc_client: NOCClient) -> str | None:
    parsed = parse_fault_question(question_text, noc_client)
    if parsed.source is None or parsed.destination_ip is None:
        return None

    trace = route_trace(noc_client, parsed.source, parsed.destination_ip)
    if not trace:
        return None

    loop_answer = _detect_loop(trace, parsed.destination_ip, question_text)
    if loop_answer:
        return loop_answer

    last = trace[-1]
    if last.route is None and last.neighbor is None and last.device != parsed.destination_device:
        cause = "static route error" if configured_static_routes_for(
            noc_client,
            last.device,
            parsed.destination_ip,
        ) else "missing static route"
        answer = f"{last.device};{parsed.destination_ip};{cause}"
        return validate_fault_answer(answer, question_text)

    return None


def _detect_loop(
    trace: list[RouteHop],
    destination_ip: ipaddress.IPv4Address,
    question_text: str,
) -> str | None:
    seen: set[str] = set()
    for hop in trace:
        if hop.neighbor and hop.neighbor in seen:
            answer = f"{hop.device};{destination_ip};routing loop"
            return validate_fault_answer(answer, question_text)
        seen.add(hop.device)
    return None
