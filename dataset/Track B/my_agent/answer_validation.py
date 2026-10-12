"""Answer validation and repair prompts for Track B exact-match outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


NODE_RE = r"[A-Za-z][A-Za-z0-9]*(?:-(?!>)[A-Za-z0-9]+)*"
PORT_RE = r"[A-Za-z][A-Za-z0-9/.-]*"
IP_RE = r"\d{1,3}(?:\.\d{1,3}){3}"

TOPOLOGY_LINE_RE = re.compile(rf"^({NODE_RE})\(({PORT_RE})\)->({NODE_RE})\(({PORT_RE})\)$")
PATH_LINE_RE = re.compile(rf"^{NODE_RE}(?:->{NODE_RE})+$")
FAULT_LINE_RE = re.compile(rf"^({NODE_RE});([^;\n]+);([^;\n]+)$")

DEVICE_NAMES = [
    "Alpha-Center-01", "Alpha-Center-02",
    "Beta-Portal-01", "Beta-Portal-02", "Beta-Aegis-01", "Beta-Aegis-02",
    "Beta-Axis-01", "Beta-Axis-02", "Beta-Node-01", "Beta-Node-02",
    "Beta-Node-03", "Beta-Node-04",
    "Gamma-Portal-01", "Gamma-Portal-02", "Gamma-Aegis-01", "Gamma-Aegis-02",
    "Gamma-Axis-01", "Gamma-Axis-02", "Gamma-Node-01", "Gamma-Node-02",
    "Gamma-Node-03", "Gamma-Node-04",
    "Delta-Portal-01", "Delta-Portal-02", "Delta-Aegis-01", "Delta-Aegis-02",
    "Delta-Axis-01", "Delta-Axis-02", "Delta-Node-01", "Delta-Node-02",
    "Delta-Node-03", "Delta-Node-04",
    "Demeter-Prime-01", "Demeter-Prime-02", "Atlas-Prime-01", "Atlas-Prime-02",
    "Janus-Prime-01", "Janus-Prime-02", "Aegis-Prime-01", "Aegis-Prime-02",
    "Hermes-Prime-01", "Hermes-Prime-02", "PX1",
    "Eon-Node-01", "Gaia-Node-01", "Aegis-Node-01", "Janus-Node-01",
    "Janus-Node-02", "Atlas-Node-01", "Atlas-Node-02", "Demeter-Node-01",
    "Demeter-Node-02", "Hermes-Node-01", "Hermes-Node-02",
]

ROUTING_CAUSES = {
    "blackhole route",
    "missing static route",
    "static route error",
    "ARP configuration error",
    "routing loop",
    "BGP configuration error",
    "OSPF configuration error",
    "loopback IP configuration conflict",
    "VXLAN configuration error",
    "L3VPN configuration error",
    "L2VPN configuration error",
    "IS-IS configuration error",
    "SRV6-Policy tunnel planning error",
}

PORT_CAUSES = {
    "shutdown",
    "interface IP error",
    "traffic congestion on port bandwidth",
    "MAC address configuration error",
    "VPN configuration missing",
    "OSPF configuration error",
    "MTU value configuration error",
    "host information collection function missing",
}

FAULT_CAUSES = ROUTING_CAUSES | PORT_CAUSES

CAUSE_ALIASES = {
    "incorrect static route": "static route error",
    "static route incorrect": "static route error",
    "incorrect arp configuration": "ARP configuration error",
    "arp configuration error": "ARP configuration error",
    "incorrect arp": "ARP configuration error",
    "loop": "routing loop",
    "routing loop": "routing loop",
    "incorrect interface ip": "interface IP error",
    "interface ip error": "interface IP error",
    "traffic congestion causing port bandwidth to be overused": "traffic congestion on port bandwidth",
    "port bandwidth overused": "traffic congestion on port bandwidth",
    "bandwidth congestion": "traffic congestion on port bandwidth",
    "host information collection missing": "host information collection function missing",
}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reasons: tuple[str, ...] = ()


def split_answer_lines(answer: str) -> list[str]:
    return [line.strip() for line in answer.strip().splitlines() if line.strip()]


def infer_first_device(question_text: str) -> str | None:
    positions: list[tuple[int, str]] = []
    for name in DEVICE_NAMES:
        pos = question_text.find(name)
        if pos >= 0:
            positions.append((pos, name))
    if not positions:
        return infer_dynamic_device(question_text)
    return min(positions)[1]


def infer_dynamic_device(question_text: str) -> str | None:
    """Best-effort target/source device extraction for unseen Phase 2/3 names."""
    device = r"[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+|[A-Z][A-Za-z]+Leaf\d+|[A-Z][A-Za-z]+Spine\d+|PSS|PX\d+"
    ignored = {
        "LocalNodeName", "RemoteNodeName", "PeerNodeName", "LocalPortNumber",
        "RemotePortNumber", "PeerPortNumber", "FaultyNode",
    }
    patterns = [
        rf"planning data of ({device})",
        rf"planned data of ({device})",
        rf"link planning data of ({device})",
        rf"from ({device}) to",
        rf"({device}) is addressing",
        rf"ping(?:ing)? .*? from ({device})",
        rf"({device}) pings",
        rf"({device}) received",
        rf"on the ({device}) node",
    ]
    for pattern in patterns:
        match = re.search(pattern, question_text, re.I)
        if match:
            candidate = match.group(1)
            if candidate not in ignored:
                return candidate
    match = re.search(device, question_text)
    if match and match.group(0) not in ignored:
        return match.group(0)
    return None


def normalize_fault_cause(cause: str) -> str:
    stripped = " ".join(cause.strip().strip(" .。;；").split())
    stripped = re.sub(r"^\(?\d+\)?\s*", "", stripped)
    alias = CAUSE_ALIASES.get(stripped.lower())
    return alias or stripped


def normalize_answer(answer: str, family: str, question_text: str = "") -> str:
    """Apply safe formatting normalizations without inventing missing facts."""
    lines = split_answer_lines(answer)
    if not lines:
        return ""

    if family == "topology":
        target = infer_first_device(question_text)
        normalized: list[str] = []
        seen: set[str] = set()
        for line in lines:
            if re.search(r"LocalNode|RemoteNode|LocalPort|RemotePort", line, re.I):
                continue
            match = TOPOLOGY_LINE_RE.fullmatch(line)
            if not match:
                continue
            local, local_port, remote, remote_port = match.groups()
            if target and remote == target and local != target:
                local, local_port, remote, remote_port = remote, remote_port, local, local_port
            if target and local != target:
                continue
            item = f"{local}({local_port})->{remote}({remote_port})"
            if item not in seen:
                normalized.append(item)
                seen.add(item)
        return "\n".join(normalized)

    if family == "path":
        candidates = [line.rstrip(".,;，。；") for line in lines if PATH_LINE_RE.fullmatch(line.rstrip(".,;，。；"))]
        candidates = [line for line in candidates if not re.search(IP_RE, line)]
        if candidates:
            return max(candidates, key=len)
        return "\n".join(lines)

    if family == "fault":
        normalized = []
        seen = set()
        for line in lines:
            match = FAULT_LINE_RE.fullmatch(line)
            if not match:
                continue
            node, target, cause = match.groups()
            causes = [normalize_fault_cause(part) for part in cause.split(",")]
            for normalized_cause in causes:
                item = f"{node};{target.strip()};{normalized_cause}"
                if item not in seen:
                    normalized.append(item)
                    seen.add(item)
        return "\n".join(normalized)

    return "\n".join(lines)


def validate_answer(answer: str, family: str, question_text: str = "") -> ValidationResult:
    lines = split_answer_lines(answer)
    reasons: list[str] = []
    if not lines:
        return ValidationResult(False, ("empty",))
    if "<tool_call" in answer.lower():
        reasons.append("contains tool_call")
    if re.search(r"Observation|Let's|I need|Based on|Analyze|Actually|Maybe|However|\*\*|`|^\d+\.", answer, re.I | re.M):
        reasons.append("contains explanation/markdown")

    if family == "topology":
        target = infer_first_device(question_text)
        for line in lines:
            if re.search(r"LocalNode|RemoteNode|LocalPort|RemotePort", line, re.I):
                reasons.append("contains placeholder")
            match = TOPOLOGY_LINE_RE.fullmatch(line)
            if not match:
                reasons.append(f"bad topology line: {line[:80]}")
                continue
            if target and match.group(1) != target:
                reasons.append(f"not from target perspective: {line[:80]}")

    elif family == "path":
        if len(lines) != 1:
            reasons.append("path must be exactly one line")
        line = lines[0] if lines else ""
        if not PATH_LINE_RE.fullmatch(line):
            reasons.append(f"bad path line: {line[:80]}")
        if re.search(IP_RE, line):
            reasons.append("path contains IP address")

    elif family == "fault":
        for line in lines:
            match = FAULT_LINE_RE.fullmatch(line)
            if not match:
                reasons.append(f"bad fault line: {line[:80]}")
                continue
            cause = normalize_fault_cause(match.group(3))
            if cause not in FAULT_CAUSES:
                reasons.append(f"unknown fault cause: {match.group(3)[:80]}")

    return ValidationResult(not reasons, tuple(dict.fromkeys(reasons)))


def format_requirements(family: str) -> str:
    if family == "topology":
        return (
            "Each line must be exactly LocalNode(LocalPort)->RemoteNode(RemotePort). "
            "The local node must be the target node from the question. No placeholder line."
        )
    if family == "path":
        return "Output exactly one line: NodeA->NodeB->NodeC. Node names only; no IP addresses."
    if family == "fault":
        causes = sorted(FAULT_CAUSES)
        return (
            "Each line must be exactly FaultyNode;Target;FaultCause. "
            "FaultCause must be one of: " + "; ".join(causes)
        )
    return "Output only the final answer."


def _fact_line_category(line: str) -> str | None:
    """Classify CLI output lines worth preserving for final-answer emission."""
    text = line.strip()
    if not text:
        return None
    lower = text.lower()
    if text.startswith("["):
        return "status"
    if "->" in text:
        return "link_or_path"
    if re.search(rf"\b{IP_RE}/\d{{1,2}}\b", text):
        return "route_or_interface_ip"
    if re.search(rf"\b{IP_RE}\b", text) and any(token in lower for token in ("arp", "mac", "next", "gateway")):
        return "ip_mapping"
    if any(token in lower for token in (
        "ip route-static",
        "ip route ",
        "shutdown",
        "ospf",
        "bgp",
        "isis",
        "is-is",
        "vxlan",
        "vpn",
        "srv6",
        "segment-routing",
        "blackhole",
    )):
        return "config_or_protocol"
    if re.search(r"\b(?:GE|XGE|Eth|Ethernet|GigabitEthernet|Vlanif|LoopBack)\S*", text):
        if any(token in f" {lower} " for token in (" up ", " down ", " admin", "current state", "line protocol")):
            return "interface_status"
        if "from_" in lower or lower.startswith("to-") or " to-" in lower:
            return "interface_description"
    return None


def extract_facts(
    observations: list[dict[str, Any]],
    family: str,
    question_text: str = "",
    max_items: int = 180,
) -> dict[str, Any]:
    """
    Build a compact fact bag for the strict final-answer emission stage.

    The goal is to keep facts from CLI output while excluding the long ReAct context
    that tends to make the model continue reasoning or emit markdown.
    """
    items: list[dict[str, str | int]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for obs in observations[-18:]:
        device = str(obs.get("device", ""))
        command = str(obs.get("command", ""))
        output = str(obs.get("output", ""))
        for raw_line in output.splitlines():
            line = " ".join(raw_line.strip().split())
            if not line:
                continue
            category = _fact_line_category(line)
            if category is None:
                continue
            key = (device, command, category, line)
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "device": device,
                "command": command,
                "category": category,
                "line": line[:500],
            })
            if len(items) >= max_items:
                break
        if len(items) >= max_items:
            break

    return {
        "family": family,
        "required_format": format_requirements(family),
        "question": question_text,
        "facts": items,
    }


def build_repair_prompt(family: str, invalid_answer: str, reasons: tuple[str, ...]) -> str:
    reason_text = "; ".join(reasons) if reasons else "format mismatch"
    return (
        "Your previous response is not a valid final answer for this competition.\n"
        f"Problems: {reason_text}\n"
        f"Required format: {format_requirements(family)}\n\n"
        "Rewrite the final answer using ONLY facts already present in the observations. "
        "Do not call tools. Do not explain. Do not use markdown. Wrap the final answer "
        "inside <answer>...</answer>.\n\n"
        "Previous response:\n"
        f"{invalid_answer}"
    )
