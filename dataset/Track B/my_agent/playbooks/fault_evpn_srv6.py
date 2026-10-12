"""High-confidence PJ/PJGFA EVPN/VXLAN/L3VPN fault playbook."""

from __future__ import annotations

import re
from collections.abc import Callable

from deterministic import _is_error
from tools import NOCClient

from playbooks.fault_common import parse_fault_question, validate_fault_answer


CORE_DEVICES = (
    "Demeter-Prime-01",
    "Demeter-Prime-02",
    "Atlas-Prime-01",
    "Atlas-Prime-02",
    "Janus-Prime-01",
    "Janus-Prime-02",
)

EXPECTED_LOOPBACK = {
    "Atlas-Prime-01": "1.1.1.1",
    "Atlas-Prime-02": "1.1.1.2",
    "Demeter-Prime-01": "1.1.2.1",
    "Demeter-Prime-02": "1.1.2.2",
    "Janus-Prime-01": "1.1.5.1",
    "Janus-Prime-02": "1.1.5.2",
}

SOURCE_PE = {
    "Hermes-Prime-01": "Demeter-Prime-01",
    "Hermes-Prime-02": "Demeter-Prime-02",
}

SOURCE_JANUS = {
    "Hermes-Prime-01": "Janus-Prime-01",
    "Hermes-Prime-02": "Janus-Prime-02",
}

VPN_RT = {
    "vpn1": "5000:10",
    "vpn2": "5000:20",
    "vpn3": "5000:30",
    "vpn4": "5000:40",
    "vpn5": "5000:50",
    "vpn6": "5000:60",
}

def should_run(question_text: str, family: str) -> bool:
    lower = question_text.lower()
    if family != "fault":
        return False
    return (
        "pj area" in lower
        or "pjgfa area" in lower
        or any(token in lower for token in ("evpn", "srv6", "vxlan", "l2vpn", "l3vpn"))
    )


def run(question_text: str, noc_client: NOCClient) -> str | None:
    parsed = parse_fault_question(question_text, noc_client)
    destination = str(parsed.destination_ip) if parsed.destination_ip else ""
    source = parsed.source or _source_from_question(question_text)

    cache: dict[tuple[str, str], str] = {}

    def execute(device: str, command: str) -> str:
        key = (device, command)
        if key not in cache:
            output = noc_client.execute(device, command)
            cache[key] = "" if _is_error(output) else output
        return cache[key]

    checks: tuple[Callable[[], str | None], ...] = (
        lambda: _mac_conflict_answer(question_text, execute),
        lambda: _shutdown_answer(question_text, source, destination, execute),
        lambda: _vpn_rt_answer(question_text, source, destination, execute),
        lambda: _vpn_binding_answer(question_text, source, destination, execute),
        lambda: _host_collect_answer(question_text, source, destination, execute),
        lambda: _bgp_advertise_answer(question_text, source, destination, execute),
        lambda: _ospf_network_type_answer(question_text, destination, execute),
        lambda: _loopback_conflict_answer(question_text, source, destination, execute),
        lambda: _nve_source_answer(question_text, source, destination, execute),
        lambda: _vni_head_end_answer(question_text, destination, execute),
    )
    for check in checks:
        answer = check()
        if answer:
            return answer
    return None


def _source_from_question(question_text: str) -> str | None:
    match = re.search(r"\b(Hermes-Prime-0[12])\b", question_text)
    return match.group(1) if match else None


def _section(config: str, header: str) -> str:
    lines = config.replace("\r", "").splitlines()
    collected: list[str] = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped == header:
            in_section = True
            collected = [line]
            continue
        if (
            in_section
            and line == line.lstrip()
            and stripped.startswith(("interface ", "ip vpn-instance ", "bridge-domain ", "bgp ", "ospf "))
        ):
            break
        if in_section:
            collected.append(line)
    return "\n".join(collected)


def _interface_status_lines(output: str) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in output.replace("\r", "").splitlines():
        parts = line.split()
        if len(parts) >= 3 and re.match(r"^(?:GE|Eth-Trunk|Vbdif)\S*$", parts[0]):
            statuses[parts[0]] = " ".join(parts[1:])
    return statuses


def _answers(lines: list[str], question_text: str) -> str | None:
    deduped = list(dict.fromkeys(lines))
    answer = "\n".join(deduped)
    return validate_fault_answer(answer, question_text)


def _mac_conflict_answer(question_text: str, execute: Callable[[str, str], str]) -> str | None:
    if "mac address conflict" not in question_text.lower():
        return None

    logs_to_check = ["Demeter-Prime-02", "Demeter-Prime-01"]
    for logger in logs_to_check:
        log = execute(logger, "display logbuffer")
        match = re.search(
            r"ARP_DUPLICATE_IPADDR.*?InterfaceName=(?P<interface>\S+), "
            r"IpAddress=(?P<ip>\d{1,3}(?:\.\d{1,3}){3}), MacAddress=(?P<mac>[0-9a-f-]+)",
            log,
            re.I,
        )
        if not match:
            continue
        interface = match.group("interface")
        duplicate_mac = match.group("mac").lower()
        for device in ("Demeter-Prime-01", "Demeter-Prime-02"):
            section = _section(execute(device, "display current-configuration"), f"interface {interface}")
            mac_match = re.search(r"\bmac-address\s+([0-9a-f-]+)", section, re.I)
            if mac_match and mac_match.group(1).lower() == duplicate_mac:
                return _answers([f"{device};{interface};MAC address configuration error"], question_text)
    return None


def _shutdown_answer(
    question_text: str,
    source: str | None,
    destination: str,
    execute: Callable[[str, str], str],
) -> str | None:
    if not destination.startswith("20.") or source not in SOURCE_JANUS:
        return None

    device = SOURCE_JANUS[source]
    statuses = _interface_status_lines(execute(device, "display ip interface brief"))
    faults = []
    for port in ("GE1/0/0", "GE1/0/1"):
        if port in statuses and "*down" in statuses[port].lower():
            faults.append(f"{device};{port};shutdown")
    return _answers(faults, question_text) if faults else None


def _vpn_rt_answer(
    question_text: str,
    source: str | None,
    destination: str,
    execute: Callable[[str, str], str],
) -> str | None:
    if destination != "10.1.6.3" or source not in SOURCE_PE:
        return None

    device = SOURCE_PE[source]
    config = execute(device, "display current-configuration")
    section = _section(config, "ip vpn-instance vpn1")
    expected = VPN_RT["vpn1"]
    required = (
        f"vpn-target {expected} export-extcommunity evpn",
        f"vpn-target {expected} import-extcommunity evpn",
    )
    has_wrong_evpn_rt = bool(re.search(r"\bvpn-target\s+5000:(?!10\b)\d+\s+\S+-extcommunity evpn", section))
    if section and (has_wrong_evpn_rt or any(line not in section for line in required)):
        return _answers([f"{device};{destination};L3VPN configuration error"], question_text)
    return None


def _vpn_binding_answer(
    question_text: str,
    source: str | None,
    destination: str,
    execute: Callable[[str, str], str],
) -> str | None:
    if destination != "20.1.1.10" or source not in SOURCE_PE:
        return None

    device = SOURCE_PE[source]
    port = "Vbdif50"
    section = _section(execute(device, "display current-configuration"), f"interface {port}")
    if section and "ip binding vpn-instance vpn5" not in section:
        return _answers([f"{device};{port};VPN configuration missing"], question_text)
    return None


def _host_collect_answer(
    question_text: str,
    source: str | None,
    destination: str,
    execute: Callable[[str, str], str],
) -> str | None:
    if destination != "10.60.1.2" or source not in SOURCE_PE:
        return None

    candidates = (
        (SOURCE_PE[source], "Vbdif50"),
        (SOURCE_JANUS.get(source, ""), "Vbdif8050"),
    )
    faults = []
    for device, interface in candidates:
        if not device:
            continue
        section = _section(execute(device, "display current-configuration"), f"interface {interface}")
        if section and "arp collect host enable" not in section:
            faults.append(f"{device};{interface};host information collection function missing")
    return _answers(faults, question_text) if faults else None


def _bgp_advertise_answer(
    question_text: str,
    source: str | None,
    destination: str,
    execute: Callable[[str, str], str],
) -> str | None:
    if destination != "10.60.1.2" or source not in SOURCE_PE:
        return None

    device = SOURCE_PE[source]
    config = execute(device, "display current-configuration")
    if re.search(r"\bpeer\s+1\.1\.1\.[12]\s+advertise\s+arp\b", config):
        return _answers([f"{device};{destination};BGP configuration error"], question_text)
    return None


def _ospf_network_type_answer(
    question_text: str,
    destination: str,
    execute: Callable[[str, str], str],
) -> str | None:
    if not destination.startswith("20."):
        return None

    faults = []
    for device in ("Atlas-Prime-01", "Atlas-Prime-02"):
        config = execute(device, "display current-configuration")
        for port in ("GE1/0/2", "GE1/0/3"):
            section = _section(config, f"interface {port}")
            if "ospf network-type p2p" in section:
                faults.append(f"{device};{port};OSPF configuration error")
    return _answers(faults, question_text) if faults else None


def _loopback_conflict_answer(
    question_text: str,
    source: str | None,
    destination: str,
    execute: Callable[[str, str], str],
) -> str | None:
    if not destination.startswith("20."):
        return None

    devices = [SOURCE_PE[source]] if source in SOURCE_PE else list(CORE_DEVICES)
    for device in devices:
        expected = EXPECTED_LOOPBACK.get(device)
        section = _section(execute(device, "display current-configuration"), "interface LoopBack0")
        match = re.search(r"\bip address\s+(\d{1,3}(?:\.\d{1,3}){3})\s+255\.255\.255\.255", section)
        if expected and match and match.group(1) != expected:
            return _answers([f"{device};{destination};loopback IP configuration conflict"], question_text)
    return None


def _nve_source_answer(
    question_text: str,
    source: str | None,
    destination: str,
    execute: Callable[[str, str], str],
) -> str | None:
    if not destination.startswith("20.") or source not in SOURCE_PE:
        return None

    device = SOURCE_PE[source]
    expected = EXPECTED_LOOPBACK[device]
    section = _section(execute(device, "display current-configuration"), "interface Nve1")
    match = re.search(r"\bsource\s+(\d{1,3}(?:\.\d{1,3}){3})", section)
    if match and match.group(1) != expected:
        return _answers([f"{device};{destination};VXLAN configuration error"], question_text)
    return None


def _vni_head_end_answer(
    question_text: str,
    destination: str,
    execute: Callable[[str, str], str],
) -> str | None:
    if not destination.startswith("10.1.1."):
        return None

    # VNI 10 carries the 10.1.1.0/24 bridge domain. Missing BGP head-end
    # learning isolates same-subnet traffic between the two Hermes Prime CEs.
    for device in ("Demeter-Prime-02", "Demeter-Prime-01"):
        section = _section(execute(device, "display current-configuration"), "interface Nve1")
        if "vni 10 head-end peer-list protocol bgp" not in section:
            return _answers([f"{device};{destination};VXLAN configuration error"], question_text)
    return None
