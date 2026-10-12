"""PJ/PJGFA business-IP path playbook."""

from __future__ import annotations

from collections import deque
import re

from answer_validation import validate_answer
from deterministic import (
    NODE_RE,
    PORT_RE,
    _is_error,
    _lldp_commands_for,
    _path_source_device,
    _question_destination_ip,
    parse_lldp_links,
)
from tools import NOCClient


PJ_GRAPH_DEVICES = (
    "Demeter-Prime-01", "Demeter-Prime-02",
    "Atlas-Prime-01", "Atlas-Prime-02",
    "Janus-Prime-01", "Janus-Prime-02",
    "Aegis-Prime-01", "Aegis-Prime-02",
    "Hermes-Prime-01", "Hermes-Prime-02",
    "PX1", "PSS",
    "Eon-Node-01", "Gaia-Node-01", "Aegis-Node-01",
    "Janus-Node-01", "Janus-Node-02",
    "Atlas-Node-01", "Atlas-Node-02",
    "Demeter-Node-01", "Demeter-Node-02",
    "Hermes-Node-01", "Hermes-Node-02",
)


def should_run(question_text: str, family: str) -> bool:
    lower = question_text.lower()
    return family == "path" and ("pj area" in lower or "pjgfa area" in lower)


def _interface_owner(noc_client: NOCClient, destination_ip: str) -> str | None:
    candidate = _candidate_owner_from_ip(destination_ip)
    if candidate and _device_has_ip(noc_client, candidate, destination_ip):
        return candidate

    for device in PJ_GRAPH_DEVICES:
        if device == candidate:
            continue
        if _device_has_ip(noc_client, device, destination_ip):
            return device
    return None


def _candidate_owner_from_ip(destination_ip: str) -> str | None:
    parts = destination_ip.split(".")
    if len(parts) != 4:
        return None
    first, second, _third, last = parts
    if first == "10" and second == "1":
        if last == "10":
            return "Hermes-Prime-01"
        if last == "20":
            return "Hermes-Prime-02"
    if first == "20" and second == "1":
        if last == "10":
            return "Hermes-Node-01"
        if last == "20":
            return "Hermes-Node-02"
    return None


def _device_has_ip(noc_client: NOCClient, device: str, destination_ip: str) -> bool:
    output = noc_client.execute(device, "display ip interface brief")
    if _is_error(output):
        return False
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].split("/", 1)[0] == destination_ip:
            return True
    return False


def _neighbors(
    noc_client: NOCClient,
    device: str,
    cache: dict[str, set[str]],
) -> set[str]:
    if device in cache:
        return cache[device]

    neighbors: set[str] = set()
    for command in _lldp_commands_for(device):
        output = noc_client.execute(device, command)
        if _is_error(output):
            continue
        for link in parse_lldp_links(output, device):
            match = re.match(rf"^({NODE_RE})\(({PORT_RE})\)->({NODE_RE})\(({PORT_RE})\)$", link)
            if not match:
                continue
            local, _local_port, remote, _remote_port = match.groups()
            if local == device:
                neighbors.add(remote)
        break

    cache[device] = neighbors
    return neighbors


def _shortest_path(noc_client: NOCClient, source: str, target: str) -> list[str] | None:
    queue: deque[list[str]] = deque([[source]])
    seen = {source}
    neighbor_cache: dict[str, set[str]] = {}
    while queue:
        path = queue.popleft()
        current = path[-1]
        if current == target:
            return path
        for neighbor in sorted(_neighbors(noc_client, current, neighbor_cache)):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(path + [neighbor])
    return None


def _build_lldp_graph(noc_client: NOCClient) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for device in PJ_GRAPH_DEVICES:
        for command in _lldp_commands_for(device):
            output = noc_client.execute(device, command)
            if _is_error(output):
                continue
            for link in parse_lldp_links(output, device):
                match = re.match(rf"^({NODE_RE})\(({PORT_RE})\)->({NODE_RE})\(({PORT_RE})\)$", link)
                if not match:
                    continue
                local, _local_port, remote, _remote_port = match.groups()
                graph.setdefault(local, set()).add(remote)
                graph.setdefault(remote, set()).add(local)
            break
    return graph


def _shortest_path_graph(graph: dict[str, set[str]], source: str, target: str) -> list[str] | None:
    queue: deque[list[str]] = deque([[source]])
    seen = {source}
    while queue:
        path = queue.popleft()
        current = path[-1]
        if current == target:
            return path
        for neighbor in sorted(graph.get(current, ())):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(path + [neighbor])
    return None


def run(question_text: str, noc_client: NOCClient) -> str | None:
    source = _path_source_device(question_text)
    destination = _question_destination_ip(question_text)
    if source is None or destination is None:
        return None

    owner = _interface_owner(noc_client, str(destination))
    if owner is None:
        return None
    if owner == source:
        return None

    path = _shortest_path(noc_client, source, owner)
    if path is None:
        path = _shortest_path_graph(_build_lldp_graph(noc_client), source, owner)
    if not path:
        return None
    answer = "->".join(path)
    if validate_answer(answer, "path", question_text).ok:
        return answer
    return None
