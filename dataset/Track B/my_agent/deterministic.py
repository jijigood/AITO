"""Small deterministic solvers for Track B tasks with reliable table structure."""

from __future__ import annotations

import re

import ipaddress
from collections import deque

from answer_validation import DEVICE_NAMES, infer_first_device, normalize_answer, validate_answer
from tools import KNOWN_HUAWEI_DEVICES, NOCClient


PORT_RE = r"[A-Za-z][A-Za-z0-9/.-]*"
NODE_RE = r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*"
IP_RE = r"\d{1,3}(?:\.\d{1,3}){3}"
LLDP_COMMANDS = (
    "display lldp neighbor brief",
    "show lldp neighbors",
    "display lldp neighbor-information",
    "display lldp neighbor-list",
)
DESCRIPTION_COMMANDS = (
    "display interface description",
    "show interface description",
)
PJ_TOPOLOGY_DEVICES = (
    "Demeter-Prime-01", "Demeter-Prime-02",
    "Atlas-Prime-01", "Atlas-Prime-02",
    "Janus-Prime-01", "Janus-Prime-02",
    "Aegis-Prime-01", "Aegis-Prime-02",
    "Hermes-Prime-01", "Hermes-Prime-02",
    "PX1",
    "Eon-Node-01", "Gaia-Node-01", "Aegis-Node-01",
    "Janus-Node-01", "Janus-Node-02",
    "Atlas-Node-01", "Atlas-Node-02",
    "Demeter-Node-01", "Demeter-Node-02",
    "Hermes-Node-01", "Hermes-Node-02",
)


def _lldp_commands_for(device_name: str) -> tuple[str, ...]:
    if device_name in KNOWN_HUAWEI_DEVICES:
        return ("display lldp neighbor brief",)
    if device_name == "PSS":
        return ("show lldp neighbors",)
    return LLDP_COMMANDS


def _description_commands_for(device_name: str) -> tuple[str, ...]:
    if device_name in KNOWN_HUAWEI_DEVICES:
        return ("display interface description",)
    if device_name == "PSS":
        return ("show interface description",)
    return DESCRIPTION_COMMANDS


def _is_error(output: str) -> bool:
    return output.startswith((
        "[PERMISSION DENIED]",
        "[COMMAND ERROR]",
        "[DEVICE NOT FOUND]",
        "[REQUEST FAILED]",
        "[HTTP ",
    ))


def parse_lldp_links(output: str, local_device: str) -> list[str]:
    links: list[str] = []
    for line in output.splitlines():
        # Huawei:
        # GE1/0/0 101 GE1/0/4 Gamma-Portal-01
        match = re.match(rf"^\s*({PORT_RE})\s+\d+\s+({PORT_RE})\s+({NODE_RE})\s*$", line)
        if match:
            local_port, remote_port, remote_device = match.groups()
            links.append(f"{local_device}({local_port})->{remote_device}({remote_port})")
            continue

        # Cisco:
        # BoardLeaf1 GigabitEthernet0/0/0/0 120 B,R GE1/0/1
        match = re.match(rf"^\s*({NODE_RE})\s+({PORT_RE})\s+\d+\s+\S+\s+({PORT_RE})\s*$", line)
        if match:
            remote_device, local_port, remote_port = match.groups()
            if "/" in local_port and "/" in remote_port:
                links.append(f"{local_device}({local_port})->{remote_device}({remote_port})")
    return links


def parse_description_links(output: str, target_device: str) -> list[str]:
    links: list[str] = []
    for line in output.splitlines():
        if " up " not in f" {line} ":
            continue
        # Example:
        # GE1/0/2 up up From_A_GE1/0/2_To_B_GE1/0/6
        match = re.search(
            rf"^\s*({PORT_RE})\s+up\s+up\s+From_({NODE_RE})_({PORT_RE})_To_({NODE_RE})_({PORT_RE})",
            line,
        )
        if match:
            _shown_port, left_node, left_port, right_node, right_port = match.groups()
            if left_node == target_device:
                links.append(f"{left_node}({left_port})->{right_node}({right_port})")
            elif right_node == target_device:
                links.append(f"{right_node}({right_port})->{left_node}({left_port})")
            continue

        # Example:
        # GE1/0/2 up up To-Janus-Prime-01-GE1/0/0
        match = re.search(rf"^\s*({PORT_RE})\s+up\s+up\s+To-({NODE_RE})-({PORT_RE})", line)
        if match:
            local_port, remote_node, remote_port = match.groups()
            links.append(f"{target_device}({local_port})->{remote_node}({remote_port})")
    return links


def _neighbor_by_port(output: str, local_device: str, local_port: str) -> str | None:
    for link in parse_lldp_links(output, local_device) + parse_description_links(output, local_device):
        match = re.match(rf"^{NODE_RE}\(({PORT_RE})\)->({NODE_RE})\(({PORT_RE})\)$", link)
        if match and match.group(1) == local_port:
            return match.group(2)
    return None


def _devices_in_question(question_text: str) -> list[str]:
    positions = []
    for name in DEVICE_NAMES:
        pos = question_text.find(name)
        if pos >= 0:
            positions.append((pos, name))
    return [name for _, name in sorted(positions)]


def _device_pattern() -> str:
    return "|".join(re.escape(name) for name in sorted(DEVICE_NAMES, key=len, reverse=True))


def _path_source_device(question_text: str) -> str | None:
    device = _device_pattern()
    patterns = [
        rf"addressed from ({device})",
        rf"addressing from ({device})",
        rf"path from ({device})\b",
        rf"from ({device}) to\b",
        rf"ping(?:ing)? .*? from ({device})\b",
        rf"({device}) is trying to reach",
        rf"({device}) addressing\b",
        rf"({device}) routes? to\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, question_text, re.I)
        if match:
            return match.group(1)
    return infer_first_device(question_text)


def _destination_device(question_text: str, source: str) -> str | None:
    port = _destination_port(question_text)
    devices = _devices_in_question(question_text)
    if port:
        port_pos = question_text.find(port)
        after = question_text[port_pos: port_pos + 180]
        for name in devices:
            if name != source and name in after:
                return name
        before = question_text[max(0, port_pos - 120): port_pos]
        before_devices = [name for name in devices if name != source and name in before]
        if before_devices:
            return before_devices[-1]

    remaining = [name for name in devices if name != source]
    return remaining[0] if remaining else None


def _destination_port(question_text: str) -> str | None:
    match = re.search(r"\b(?:GE|XGE|Eth|Ethernet|GigabitEthernet)\d+(?:/\d+)+\b", question_text)
    return match.group(0) if match else None


def _question_destination_ip(question_text: str) -> ipaddress.IPv4Address | None:
    for raw in re.findall(rf"\b{IP_RE}\b", question_text):
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            continue
        if isinstance(ip, ipaddress.IPv4Address):
            return ip
    return None


def _interface_ip(output: str, port: str) -> ipaddress.IPv4Address | None:
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == port and "/" in parts[1]:
            raw_ip = parts[1].split("/", 1)[0]
            try:
                ip = ipaddress.ip_address(raw_ip)
            except ValueError:
                return None
            if isinstance(ip, ipaddress.IPv4Address):
                return ip
    return None


def _route_for_ip(output: str, destination: ipaddress.IPv4Address) -> tuple[str, str, str] | None:
    best: tuple[int, str, str, str] | None = None
    for line in output.splitlines():
        match = re.search(rf"\b({IP_RE}/\d{{1,2}})\b", line)
        if not match:
            continue
        try:
            network = ipaddress.ip_network(match.group(1), strict=False)
        except ValueError:
            continue
        if destination not in network:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        next_hop = parts[-2]
        interface = parts[-1]
        proto = parts[1] if len(parts) > 1 else ""
        candidate = (network.prefixlen, next_hop, interface, proto)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None
    return best[1], best[2], best[3]


def _topology_scan_candidates(question_text: str, target: str) -> list[str]:
    lower = question_text.lower()
    if "pj area" in lower or "pjgfa area" in lower or "-prime-" in target.lower():
        return [name for name in PJ_TOPOLOGY_DEVICES if name in DEVICE_NAMES and name != target]
    return [name for name in DEVICE_NAMES if name != target]


def _parse_ip_interface_rows(output: str) -> list[tuple[str, ipaddress.IPv4Interface, str, str]]:
    rows: list[tuple[str, ipaddress.IPv4Interface, str, str]] = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 4 or "/" not in parts[1]:
            continue
        try:
            interface = ipaddress.ip_interface(parts[1])
        except ValueError:
            continue
        if not isinstance(interface.ip, ipaddress.IPv4Address):
            continue
        rows.append((parts[0], interface, parts[2].lower(), parts[3].lower()))
    return rows


def _up_l3_peer_links(
    question_text: str,
    noc_client: NOCClient,
    target: str,
    candidates: list[str],
) -> list[str]:
    target_output = noc_client.execute(target, "display ip interface brief")
    if _is_error(target_output):
        target_output = noc_client.execute(target, "show ip interface brief")
    if _is_error(target_output):
        return []

    target_rows = [
        row for row in _parse_ip_interface_rows(target_output)
        if row[2] == "up"
        and row[3].startswith("up")
        and row[1].network.prefixlen == 30
        and not row[0].lower().startswith(("loopback", "vlanif", "vbdif"))
    ]
    if not target_rows:
        return []

    peer_by_ip: dict[ipaddress.IPv4Address, tuple[str, str]] = {}
    for device in candidates:
        if device == target:
            continue
        output = noc_client.execute(device, "display ip interface brief")
        if _is_error(output):
            output = noc_client.execute(device, "show ip interface brief")
        if _is_error(output):
            continue
        for port, address, physical, protocol in _parse_ip_interface_rows(output):
            if physical == "up" and protocol.startswith("up"):
                peer_by_ip[address.ip] = (device, port)

    links: list[str] = []
    for local_port, address, _physical, _protocol in target_rows:
        peer_hosts = [ip for ip in address.network.hosts() if ip != address.ip]
        if len(peer_hosts) != 1:
            continue
        peer = peer_by_ip.get(peer_hosts[0])
        if peer:
            remote_device, remote_port = peer
            links.append(f"{target}({local_port})->{remote_device}({remote_port})")

    answer = normalize_answer("\n".join(links), "topology", question_text)
    return answer.splitlines() if validate_answer(answer, "topology", question_text).ok else []


def _edges_by_lldp_or_description(noc_client: NOCClient, device: str) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    for command in _lldp_commands_for(device):
        output = noc_client.execute(device, command)
        if _is_error(output):
            continue
        for link in parse_lldp_links(output, device):
            match = re.match(rf"^({NODE_RE})\(({PORT_RE})\)->({NODE_RE})\(({PORT_RE})\)$", link)
            if match and match.group(1) == device:
                _local, local_port, remote, remote_port = match.groups()
                edges.append((remote, local_port, remote_port))
        if edges:
            return edges

    for command in _description_commands_for(device):
        output = noc_client.execute(device, command)
        if _is_error(output):
            continue
        for link in parse_description_links(output, device):
            match = re.match(rf"^({NODE_RE})\(({PORT_RE})\)->({NODE_RE})\(({PORT_RE})\)$", link)
            if match and match.group(1) == device:
                _local, local_port, remote, remote_port = match.groups()
                edges.append((remote, local_port, remote_port))
        if edges:
            return edges

    return edges


def _complete_path_by_lldp(
    noc_client: NOCClient,
    current: str,
    destination_device: str,
    existing_path: list[str],
    destination_port: str | None = None,
    max_hops: int = 6,
) -> list[str] | None:
    queue: deque[list[str]] = deque([[current]])
    seen = set(existing_path)
    seen.discard(current)
    seen.add(current)
    while queue:
        segment = queue.popleft()
        node = segment[-1]
        if node == destination_device:
            return existing_path[:-1] + segment
        if len(segment) > max_hops:
            continue
        edges = sorted(_edges_by_lldp_or_description(noc_client, node), key=lambda item: item[0])
        for neighbor, _local_port, remote_port in edges:
            if neighbor in seen:
                continue
            if neighbor == destination_device and destination_port and remote_port != destination_port:
                continue
            seen.add(neighbor)
            queue.append(segment + [neighbor])
    return None


def _question_devices(question_text: str, target: str) -> list[str]:
    devices = [name for name in DEVICE_NAMES if name != target and name in question_text]
    if devices:
        return devices

    # Conservative fallback: same zone/area family only.
    if "-" in target:
        prefix = target.split("-", 1)[0]
        devices = [name for name in DEVICE_NAMES if name != target and name.startswith(prefix + "-")]
    if "Prime" in target:
        devices.extend([name for name in DEVICE_NAMES if name != target and "-Prime-" in name])
    return list(dict.fromkeys(devices))


def solve_topology(question_text: str, noc_client: NOCClient) -> str | None:
    """Return a validated topology answer when LLDP/description parsing is enough."""
    target = infer_first_device(question_text)
    if not target:
        return None

    for command in _lldp_commands_for(target):
        direct = noc_client.execute(target, command)
        if not _is_error(direct):
            answer = normalize_answer("\n".join(parse_lldp_links(direct, target)), "topology", question_text)
            if validate_answer(answer, "topology", question_text).ok:
                return answer

    for command in _description_commands_for(target):
        desc = noc_client.execute(target, command)
        if not _is_error(desc):
            answer = normalize_answer("\n".join(parse_description_links(desc, target)), "topology", question_text)
            if validate_answer(answer, "topology", question_text).ok:
                return answer

    inferred: list[str] = []
    candidates = _question_devices(question_text, target)
    scan_all_candidates = _is_error(direct) and _is_error(desc)
    if scan_all_candidates:
        candidates = _topology_scan_candidates(question_text, target)

    for candidate in candidates:
        if candidate == target:
            continue
        for command in _lldp_commands_for(candidate):
            output = noc_client.execute(candidate, command)
            if _is_error(output):
                continue
            for link in parse_lldp_links(output, candidate):
                match = re.match(rf"^({NODE_RE})\(({PORT_RE})\)->({NODE_RE})\(({PORT_RE})\)$", link)
                if not match:
                    continue
                local, local_port, remote, remote_port = match.groups()
                if remote == target:
                    inferred.append(f"{target}({remote_port})->{local}({local_port})")
            break
        for command in _description_commands_for(candidate):
            output = noc_client.execute(candidate, command)
            if _is_error(output):
                continue
            for link in parse_description_links(output, candidate):
                match = re.match(rf"^({NODE_RE})\(({PORT_RE})\)->({NODE_RE})\(({PORT_RE})\)$", link)
                if not match:
                    continue
                local, local_port, remote, remote_port = match.groups()
                if remote == target:
                    inferred.append(f"{target}({remote_port})->{local}({local_port})")
            break
        if inferred and not scan_all_candidates:
            break

    answer = normalize_answer("\n".join(dict.fromkeys(inferred)), "topology", question_text)
    if validate_answer(answer, "topology", question_text).ok:
        return answer

    peer_links = _up_l3_peer_links(
        question_text,
        noc_client,
        target,
        _topology_scan_candidates(question_text, target),
    )
    answer = normalize_answer("\n".join(peer_links), "topology", question_text)
    if validate_answer(answer, "topology", question_text).ok:
        return answer
    return None


def solve_path(question_text: str, noc_client: NOCClient) -> str | None:
    """Trace physical-interface path using routing-table egress interfaces plus LLDP."""
    source = _path_source_device(question_text)
    if not source:
        return None

    destination_device = _destination_device(question_text, source)
    destination = _question_destination_ip(question_text)
    if destination is None and destination_device:
        port = _destination_port(question_text)
        if not port:
            return None
        output = noc_client.execute(destination_device, "display ip interface brief")
        if _is_error(output):
            return None
        destination = _interface_ip(output, port)

    if destination is None or destination_device is None:
        return None

    path = [source]
    current = source
    for _ in range(12):
        if current == destination_device:
            answer = "->".join(path)
            if validate_answer(answer, "path", question_text).ok:
                return answer
            return None

        routes = noc_client.execute(current, "display ip routing-table")
        if _is_error(routes):
            completed = _complete_path_by_lldp(
                noc_client,
                current,
                destination_device,
                path,
                destination_port=_destination_port(question_text),
            )
            if completed:
                answer = "->".join(completed)
                if validate_answer(answer, "path", question_text).ok:
                    return answer
            return None
        route = _route_for_ip(routes, destination)
        if route is None:
            completed = _complete_path_by_lldp(
                noc_client,
                current,
                destination_device,
                path,
                destination_port=_destination_port(question_text),
            )
            if completed:
                answer = "->".join(completed)
                if validate_answer(answer, "path", question_text).ok:
                    return answer
            return None
        _next_hop, egress_port, _proto = route

        neighbor = None
        for command in _lldp_commands_for(current):
            lldp = noc_client.execute(current, command)
            if not _is_error(lldp):
                neighbor = _neighbor_by_port(lldp, current, egress_port)
                if neighbor:
                    break
        if neighbor is None:
            for command in _description_commands_for(current):
                desc = noc_client.execute(current, command)
                if not _is_error(desc):
                    neighbor = _neighbor_by_port(desc, current, egress_port)
                    if neighbor:
                        break
        if neighbor is None:
            completed = _complete_path_by_lldp(
                noc_client,
                current,
                destination_device,
                path,
                destination_port=_destination_port(question_text),
            )
            if completed:
                answer = "->".join(completed)
                if validate_answer(answer, "path", question_text).ok:
                    return answer
            return None
        if neighbor in path:
            return None

        path.append(neighbor)
        current = neighbor

    return None
