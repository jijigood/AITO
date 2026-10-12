"""System prompt, tool call parsing, family detection, and answer extraction."""

import json
import re

from answer_validation import normalize_answer, normalize_fault_cause

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior IP network O&M engineer participating in a network troubleshooting competition.

## Task
Answer network questions by querying device CLIs and analyzing the output.

## How to Use Tools
When you need to run a CLI command on a device, output a tool call block and NOTHING ELSE in that response:

<tool_call>
device: DeviceName
command: display lldp neighbor brief
</tool_call>

You will receive the command output in the next message. Then continue analyzing and call more tools or give the final answer.

When you have enough information, output ONLY the final answer — no tool_call block, no explanation.

## All Devices (Phase 1)
- Alpha zone: Alpha-Center-01, Alpha-Center-02
- Beta zone (Development & Testing): Beta-Portal-01, Beta-Portal-02, Beta-Aegis-01, Beta-Aegis-02, Beta-Axis-01, Beta-Axis-02, Beta-Node-01, Beta-Node-02, Beta-Node-03, Beta-Node-04
- Gamma zone (Big Data): Gamma-Portal-01, Gamma-Portal-02, Gamma-Aegis-01, Gamma-Aegis-02, Gamma-Axis-01, Gamma-Axis-02, Gamma-Node-01, Gamma-Node-02, Gamma-Node-03, Gamma-Node-04
- Delta zone (Management): Delta-Portal-01, Delta-Portal-02, Delta-Aegis-01, Delta-Aegis-02, Delta-Axis-01, Delta-Axis-02, Delta-Node-01, Delta-Node-02, Delta-Node-03, Delta-Node-04
- PJ area: Demeter-Prime-01, Demeter-Prime-02, Atlas-Prime-01, Atlas-Prime-02, Janus-Prime-01, Janus-Prime-02, Aegis-Prime-01, Aegis-Prime-02, Hermes-Prime-01, Hermes-Prime-02, PX1, PSS
- PJGFA area: Eon-Node-01, Gaia-Node-01, Aegis-Node-01, Janus-Node-01, Janus-Node-02, Atlas-Node-01, Atlas-Node-02, Demeter-Node-01, Demeter-Node-02, Hermes-Node-01, Hermes-Node-02
- PSS is Cisco. Most other Phase 1 devices are Huawei.

## Vendor Command Equivalents
Use the command syntax for the device vendor. If one syntax returns [COMMAND ERROR], switch to the corresponding vendor syntax.

Huawei:
display current-configuration
display logbuffer
display alarm active
display memory-usage
display lldp neighbor brief
display interface brief
display interface description
display eth-trunk <trunk_id>             ← requires numeric ID, e.g. "display eth-trunk 1"
display vlan
display mac-address
display stp brief
display stp interface brief
display ip interface brief
display ipv6 interface <intf>            ← requires interface name, e.g. "display ipv6 interface GE1/0/1"
display arp
display ipv6 neighbors
display ip routing-table                        ← NO destination IP argument; search the output yourself
display ip routing-table vpn-instance <name>   ← requires VPN name
display ipv6 routing-table
display ospf peer
display ospf routing
display ospf lsdb
display bgp evpn all routing-table
display bgp vpnv4 all routing-table
display vxlan tunnel
display vxlan troubleshooting
display vrrp verbose
display bfd session all
display ip pool
display srv6-te policy status
display srv6-te policy
display segment-routing ipv6 local-sid end forwarding
display segment-routing ipv6 local-sid end-x forwarding
ping -a <src_ip> <dst_ip>               ← source IP first, then destination IP
tracert -a <src_ip> <dst_ip>

Cisco:
show running-config
show logging
show lldp neighbors
show interface description
show ip interface brief
show interface brief
show ip arp
show ip route
show ip route ospf
show ip ospf neighbor
show ip ospf database
show bgp l2vpn evpn
show bgp vpnv4 unicast all
show nve vni
show nve peers
show vrrp detail
show bfd neighbors
ping <dst_ip> source <src_ip>
traceroute <dst_ip> source <src_ip>

H3C:
display current-configuration
display logbuffer
display alarm active
display memory
display lldp neighbor-information
display interface brief
display interface description
display ip interface brief
display arp all
display ip routing-table
display ospf peer
display ospf routing
display ospf lsdb
display bgp l2vpn evpn
display vxlan tunnel

NOTE: "display arp all" is valid for H3C but NOT for Huawei. Huawei uses "display arp".
NOTE: Commands with pipes such as "| include" are NOT available. Use the full display/show command and inspect the output.
NOTE: Interface-specific brief/detail commands are often unavailable. Prefer summary tables such as "display interface brief", "display interface description", "show interface description", and "show ip interface brief".

## Task Types and Strategies

### TYPE 1 — TOPOLOGY RECONSTRUCTION
Triggered by: "accidentally deleted" + "restore the link connections" / "UP link connections"
Goal: list all UP links of a target node in the format LocalNode(port)->RemoteNode(port)

Strategy:
1. Run `display lldp neighbor brief` on the target device.
   - Each row shows: LocalPort | NeighborDevice | NeighborPort
   - These are the UP physical links.
2. If you get [PERMISSION DENIED]:
   - The question will list candidate neighbor devices.
   - Query `display lldp neighbor brief` on EACH candidate neighbor.
   - When you find a row `LocalPort | <TargetDevice> | TargetPort`, the link is:
     TargetDevice(TargetPort)->QueryingNeighborDevice(LocalPort)
   - ALWAYS report from the TARGET device's perspective. The target is the local node, neighbors are remote.
3. If LLDP is unavailable on neighbors too:
   - Query `display interface description` on the target (shows configured remote node/port).
   - Cross-check with `display arp` on both sides to confirm the port mapping.
   - `display interface brief` to confirm which ports are UP.

Answer format — one link per line, no blank lines, no extra spaces:
LocalNodeName(LocalPort)->RemoteNodeName(RemotePort)

Wrap the final answer in <answer>...</answer> tags.

### TYPE 2 — PATH QUERY
Triggered by: source node "addressing" a destination IP/port
Goal: the ordered sequence of nodes the traffic traverses

Strategy:
1. If the destination IP is not given: run `display ip interface brief` on the destination device to find the IP of the named interface.
2. On the source device: `display ip routing-table` → find the route matching the destination → get next-hop IP and egress interface.
3. Identify which device owns that next-hop IP: query `display ip interface brief` on likely candidates in the same zone.
4. Repeat hop-by-hop until you reach the destination device.
5. The path is: source → hop1 → hop2 → ... → destination.

Answer format — single line, no spaces around arrows:
SourceNode->IntermediateNode->DestinationNode

Wrap the final answer in <answer>...</answer> tags.

### TYPE 3 — FAULT DIAGNOSIS
Triggered by: "two major categories of faults: routing faults and port faults"
Goal: identify all faults causing the described connectivity failure

Strategy:
1. Trace the expected path from source to destination via routing tables.
2. At each hop, check:
   a. `display ip routing-table` — missing route, blackhole (next-hop unreachable), incorrect destination, loop
   b. `display interface brief` — port status (shutdown = Admin DOWN, down = link DOWN)
   c. `display current-configuration` — static route pointing to a shutdown interface → blackhole route
   d. `display arp` — incorrect ARP binding (wrong IP-to-MAC mapping)
3. Report every fault found.

Routing fault output format: FaultyNode;DestinationIP;FaultCause
Port fault output format:    FaultyNode;FaultyPort;FaultCause

Valid routing fault causes (use EXACTLY as written):
  blackhole route
  missing static route
  static route error
  ARP configuration error
  routing loop
  BGP configuration error
  OSPF configuration error
  loopback IP configuration conflict
  VXLAN configuration error
  L3VPN configuration error
  L2VPN configuration error
  IS-IS configuration error
  SRV6-Policy tunnel planning error

Valid port fault causes (use EXACTLY as written):
  shutdown
  interface IP error
  traffic congestion on port bandwidth
  MAC address configuration error
  VPN configuration missing
  OSPF configuration error
  MTU value configuration error
  host information collection function missing

Answer format — one fault per line, no blank lines:
FaultyNode;Target;cause

Wrap the final answer in <answer>...</answer> tags.

## Error Handling
- [PERMISSION DENIED]: That command is blocked for this question. Use an alternative method.
- [COMMAND ERROR]: You used wrong syntax. Check the command list above and fix before retrying.
- [DEVICE NOT FOUND]: The device name is incorrect. Check spelling against the inventory list.

## Output Rules (CRITICAL — competition uses exact string match)
When you have the final answer, wrap it in <answer> tags:

<answer>
NodeA(Port)->NodeB(Port)
NodeA(Port)->NodeC(Port)
</answer>

Rules:
1. Put ONLY the answer inside <answer> tags — no explanation, no "Based on my analysis", no markdown
2. Match the format exactly: correct separators (-> or ;), correct capitalization, correct node/port names
3. No trailing newline, no leading/trailing spaces on any line
4. Multi-line answers: one item per line, NO blank lines between items
5. Stop querying and answer as soon as you have sufficient information
6. Do NOT include tool_call inside or after the <answer> block
"""


REASONING_HINT = {
    "topology": (
        "Topology hint: identify the target node first. Prefer LLDP. If target "
        "LLDP is denied, query candidate neighbors and report links from the "
        "target node perspective only."
    ),
    "path": (
        "Path hint: find the destination IP if only a port is named, then use "
        "the source routing table, map each egress port to its LLDP neighbor, "
        "and stop when the destination node is reached."
    ),
    "fault": (
        "Fault hint: ignore example fault lines in the question; they are not "
        "facts. Extract the actual source and destination from the final task "
        "sentence. Check destination interface status first, then trace routes "
        "hop by hop. If a route is absent from the active table and no matching "
        "static route is configured, use missing static route. If a matching "
        "static route is configured but not active or points to an impossible "
        "next hop, use static route error. If forwarding returns to a visited "
        "node, use routing loop. Output only FaultyNode;Target;FaultCause."
    ),
}


# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------

def detect_family(question_text: str) -> str:
    """Classify a question into topology / path / fault."""
    lower = question_text.lower()

    # Fault: explicit category mention
    if "two major categories of faults" in lower or "two major types of faults" in lower:
        return "fault"

    # Topology: data was deleted and needs to be restored
    if ("accidentally deleted" in lower or "planning data" in lower) and (
        "restore" in lower or "supplement" in lower or "up link" in lower
    ):
        return "topology"

    # Path: routing path from source to destination
    if "path" in lower and ("addressing" in lower or "address" in lower):
        return "path"
    if "addressing" in lower:
        return "path"

    return "fault"  # safe default


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tool call parsing (text-based ReAct)
# ---------------------------------------------------------------------------

_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_CALL_RE = re.compile(
    r"device(?:_name)?\s*:\s*(.+?)\s*\n\s*(?:command\s*:\s*)?(.+)$",
    re.IGNORECASE | re.DOTALL,
)


def parse_tool_call(text: str):
    """
    Return (device_name, command) if the text contains a <tool_call> block,
    otherwise return None.
    """
    text = _strip_thinking(text)
    for block_match in _TOOL_CALL_BLOCK_RE.finditer(text):
        block = block_match.group(1).strip()
        if not block:
            continue

        # Some models emit JSON inside <tool_call>.
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            device = payload.get("device") or payload.get("device_name")
            command = payload.get("command")
            if device and command:
                return str(device).strip(), str(command).strip()

        m = _TOOL_CALL_RE.search(block)
        if m:
            return m.group(1).strip(), m.group(2).strip()

        device_match = re.search(r"device(?:_name)?\s*[:=]\s*[\"']?([^\"'\n]+)", block, re.I)
        command_match = re.search(r"command\s*[:=]\s*[\"']?([^\"'\n]+)", block, re.I)
        if device_match and command_match:
            return device_match.group(1).strip(), command_match.group(1).strip()
    return None


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks produced by Qwen's CoT mode."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _normalized_or_original(candidate: str, family: str, question_text: str) -> str:
    normalized = normalize_answer(candidate, family, question_text)
    return normalized if normalized.strip() else candidate.strip()


def extract_answer(text: str, family: str, question_text: str = "") -> str:
    """
    Extract the final answer from raw LLM output.
    Returns a clean string suitable for writing to result.csv.
    """
    text = _strip_thinking(text)
    if not text.strip():
        return ""

    # Prefer content inside <answer>...</answer> tags
    answer_block = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if answer_block:
        text = answer_block.group(1).strip()
        if text:
            # Clean and return lines directly (no secondary regex needed for tagged answers)
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            return _normalized_or_original("\n".join(lines), family, question_text)

    if family == "topology":
        # Pattern: NodeA(port)->NodeB(port)
        # Node name: letters/digits/hyphens; port: GE1/0/1, XGE1/0/1, Eth0/0/1, etc.
        _NODE = r"[A-Za-z][A-Za-z0-9]*(?:-(?!>)[A-Za-z0-9]+)*"
        _PORT = r"[A-Za-z0-9/]+"
        pattern = rf"({_NODE})\(({_PORT})\)->({_NODE})\(({_PORT})\)"
        matches = re.findall(pattern, text)
        if matches:
            candidate = "\n".join(f"{a}({b})->{c}({d})" for a, b, c, d in matches)
            return _normalized_or_original(candidate, family, question_text)

    elif family == "path":
        # Pattern: NodeA->NodeB->NodeC (no parentheses, 2+ nodes connected by ->)
        # Use (?!>) lookahead after each '-' inside a node name to avoid consuming
        # the '-' that is part of the '->' arrow separator.
        _NODE = r"[A-Za-z][A-Za-z0-9]*(?:-(?!>)[A-Za-z0-9]+)*"
        _PATH_RE = re.compile(rf"({_NODE}(?:->{_NODE})+)")
        best = ""
        for line in text.split("\n"):
            line = line.strip()
            if "(" in line:
                continue
            line = re.sub(r"[.。，,;；\s]+$", "", line)
            m = _PATH_RE.search(line)
            if m and len(m.group(1)) > len(best):
                best = m.group(1)
        if best:
            return _normalized_or_original(best, family, question_text)

    elif family == "fault":
        # Pattern: node;target;cause
        # node: device name (word chars + hyphens)
        # target: IP address or port name (no semicolons)
        # cause: fault description (no semicolons, no newlines)
        pattern = r"([\w][\w-]*);([^;\n]+);([^;\n]+)"
        matches = re.findall(pattern, text)
        if matches:
            lines = []
            for node, target, cause in matches:
                lines.append(f"{node.strip()};{target.strip()};{normalize_fault_cause(cause)}")
            return _normalized_or_original("\n".join(lines), family, question_text)

    # Fallback: return the last non-empty block of text
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        last = paragraphs[-1]
        lines = [ln.strip() for ln in last.split("\n") if ln.strip()]
        return _normalized_or_original("\n".join(lines), family, question_text)

    return _normalized_or_original(text.strip(), family, question_text)
