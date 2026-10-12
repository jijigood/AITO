"""Shared helpers for fault playbooks."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass

from answer_validation import DEVICE_NAMES, validate_answer
from deterministic import (
    NODE_RE,
    PORT_RE,
    _interface_ip,
    _is_error,
    _lldp_commands_for,
    _route_for_ip,
    parse_description_links,
    parse_lldp_links,
)
from tools import NOCClient


QUESTION_PORT_RE = r"\b(?:GE|XGE|Eth|Ethernet|GigabitEthernet)\d+(?:/\d+)+\b"


@dataclass(frozen=True)
class FaultQuestion:
    body: str
    source: str | None
    destination_ip: ipaddress.IPv4Address | None
    destination_device: str | None
    destination_port: str | None


@dataclass(frozen=True)
class InterfaceInfo:
    device: str
    port: str
    ip: ipaddress.IPv4Address | None
    physical: str
    protocol: str
    raw: str


@dataclass(frozen=True)
class RouteHop:
    device: str
    route: tuple[str, str, str] | None
    neighbor: str | None


def question_body(question_text: str) -> str:
    """Remove format text and examples; keep the actual task sentence."""
    if "Port fault examples:" not in question_text:
        return question_text.strip()
    text = question_text.split("Port fault examples:", 1)[1]
    lines = text.splitlines()
    start = 0
    for idx, line in enumerate(lines):
        if "interface IP error" in line or "traffic congestion" in line:
            start = idx + 1
    return "\n".join(lines[start:]).strip()


def parse_fault_question(question_text: str, noc_client: NOCClient | None = None) -> FaultQuestion:
    body = question_body(question_text)
    source = _parse_source(body)
    destination_device = _parse_destination_device(body, source)
    destination_port = _parse_destination_port(body)
    destination_ip = _parse_destination_ip(body)
    if destination_ip is None and noc_client and destination_device and destination_port:
        output = noc_client.execute(destination_device, "display ip interface brief")
        if not _is_error(output):
            destination_ip = _interface_ip(output, destination_port)
    return FaultQuestion(
        body=body,
        source=source,
        destination_ip=destination_ip,
        destination_device=destination_device,
        destination_port=destination_port,
    )


def _parse_source(body: str) -> str | None:
    patterns = [
        rf"ping(?:ing)?\s+\d{{1,3}}(?:\.\d{{1,3}}){{3}}\s+from\s+({NODE_RE})",
        rf"({NODE_RE})\s+(?:is\s+)?addressing\b",
        rf"({NODE_RE})\s+pings\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, body, re.I)
        if match:
            return match.group(1)
    return None


def _parse_destination_device(body: str, source: str | None) -> str | None:
    patterns = [
        rf"{QUESTION_PORT_RE}\s+(?:port\s+)?of\s+({NODE_RE})",
        rf"({NODE_RE})'s\s+{QUESTION_PORT_RE}",
    ]
    for pattern in patterns:
        match = re.search(pattern, body, re.I)
        if match and match.group(1) != source:
            return match.group(1)
    return None


def _parse_destination_port(body: str) -> str | None:
    match = re.search(QUESTION_PORT_RE, body)
    return match.group(0) if match else None


def _parse_destination_ip(body: str) -> ipaddress.IPv4Address | None:
    matches = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", body)
    for raw in reversed(matches):
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            continue
        if isinstance(ip, ipaddress.IPv4Address):
            return ip
    return None


def parse_interface_table(output: str, device: str) -> list[InterfaceInfo]:
    interfaces: list[InterfaceInfo] = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        port = parts[0]
        if not re.match(r"^(?:GE|XGE|Eth|Ethernet|GigabitEthernet|Vlanif|LoopBack|Gi)\S*$", port):
            continue
        ip: ipaddress.IPv4Address | None = None
        physical = ""
        protocol = ""
        if "/" in parts[1] or parts[1] == "unassigned":
            if parts[1] != "unassigned":
                try:
                    parsed = ipaddress.ip_address(parts[1].split("/", 1)[0])
                except ValueError:
                    parsed = None
                if isinstance(parsed, ipaddress.IPv4Address):
                    ip = parsed
            physical = parts[2] if len(parts) > 2 else ""
            protocol = parts[3] if len(parts) > 3 else ""
        else:
            physical = parts[1]
            protocol = parts[2]
        interfaces.append(InterfaceInfo(device, port, ip, physical, protocol, " ".join(parts)))
    return interfaces


def get_interface_info(noc_client: NOCClient, device: str, port: str) -> InterfaceInfo | None:
    for command in ("display ip interface brief", "display interface brief", "show ip interface brief", "show interface brief"):
        output = noc_client.execute(device, command)
        if _is_error(output):
            continue
        for info in parse_interface_table(output, device):
            if info.port == port:
                return info
    return None


def find_ip_owner(noc_client: NOCClient, destination_ip: ipaddress.IPv4Address) -> InterfaceInfo | None:
    for device in DEVICE_NAMES:
        output = noc_client.execute(device, "display ip interface brief")
        if _is_error(output):
            continue
        for info in parse_interface_table(output, device):
            if info.ip == destination_ip:
                return info
    return None


def is_shutdown(info: InterfaceInfo) -> bool:
    text = f"{info.physical} {info.protocol} {info.raw}".lower()
    return "*down" in text or "admin-down" in text or "administratively down" in text


def route_trace(
    noc_client: NOCClient,
    source: str,
    destination_ip: ipaddress.IPv4Address,
    max_hops: int = 10,
) -> list[RouteHop]:
    trace: list[RouteHop] = []
    current = source
    seen: set[str] = set()
    for _ in range(max_hops):
        if current in seen:
            trace.append(RouteHop(current, None, current))
            break
        seen.add(current)
        routes = noc_client.execute(current, "display ip routing-table")
        if _is_error(routes):
            routes = noc_client.execute(current, "show ip route")
        route = _route_for_ip(routes, destination_ip) if not _is_error(routes) else None
        if route is None:
            trace.append(RouteHop(current, None, None))
            break
        _next_hop, egress_port, _proto = route
        neighbor = neighbor_by_port(noc_client, current, egress_port)
        trace.append(RouteHop(current, route, neighbor))
        if neighbor is None:
            break
        current = neighbor
    return trace


def configured_static_routes_for(
    noc_client: NOCClient,
    device: str,
    destination_ip: ipaddress.IPv4Address,
) -> list[str]:
    output = noc_client.execute(device, "display current-configuration")
    if _is_error(output):
        output = noc_client.execute(device, "show running-config")
    if _is_error(output):
        return []

    matches: list[str] = []
    for line in output.splitlines():
        text = line.strip()
        if not text.startswith("ip route"):
            continue
        parts = text.split()
        for idx in range(len(parts) - 1):
            try:
                network = ipaddress.ip_network(f"{parts[idx]}/{parts[idx + 1]}", strict=False)
            except ValueError:
                continue
            if destination_ip in network:
                matches.append(text)
                break
    return matches


def neighbor_by_port(noc_client: NOCClient, device: str, port: str) -> str | None:
    for command in _lldp_commands_for(device):
        output = noc_client.execute(device, command)
        if _is_error(output):
            continue
        for link in parse_lldp_links(output, device):
            match = re.match(rf"^({NODE_RE})\(({PORT_RE})\)->({NODE_RE})\(({PORT_RE})\)$", link)
            if match and match.group(1) == device and match.group(2) == port:
                return match.group(3)
        break

    output = noc_client.execute(device, "display interface description")
    if not _is_error(output):
        for link in parse_description_links(output, device):
            match = re.match(rf"^({NODE_RE})\(({PORT_RE})\)->({NODE_RE})\(({PORT_RE})\)$", link)
            if match and match.group(1) == device and match.group(2) == port:
                return match.group(3)
    return None


def validate_fault_answer(answer: str, question_text: str) -> str | None:
    return answer if validate_answer(answer, "fault", question_text).ok else None
