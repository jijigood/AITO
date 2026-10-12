# Track B Agent System Overview

This document describes the overall system design, workflow, key constraints, and future extension direction of `my_agent`. It targets Track B of the IP network troubleshooting challenge. The goal is to generate a stable and valid `result.csv` while complying with the competition rules.

## 1. System Goal

Track B requires the agent to solve IP network tasks by calling simulated network device CLI interfaces exposed by the Agent Tool Server. The tasks fall into three main categories:

- Topology restoration: complete the links connected to the UP interfaces of a given node.
- Path query: provide the hop-by-hop path from a source node to a target IP or target port.
- Fault localization: identify routing faults or port faults that cause service interruption, and output the causes in the required format.

The final submission file is:

```text
id,prediction
1,...
2,...
```

The primary scoring metric is exact-match accuracy. Therefore, the system must pay close attention to formatting, spaces, fault reason strings, node names, and port names.

## 2. System Components

The current `my_agent` implementation consists of four core Python modules:

| File | Purpose |
|---|---|
| `main.py` | Batch entry point; loads questions, schedules concurrent workers, saves results, and records progress |
| `agent_loop.py` | ReAct loop; lets the model alternate between tool calls and final-answer generation |
| `tools.py` | NOC API client; calls the official `server.py` endpoint `/api/agent/execute` |
| `prompts.py` | System prompt, task-family detection, tool-call parsing, and answer extraction |

Official components are kept unchanged:

| Official Component | Purpose |
|---|---|
| `server.py` | Local Agent Tool Server that simulates multi-vendor device CLIs |
| `devices_outputs/` | Offline CLI outputs indexed by question, device, and command |
| `question_limits_config.json` | Defines per-question no-permission command restrictions |
| `data/Phase_1/test.json` | The 50 Phase 1 test questions |

## 3. Architecture

```text
test.json
   |
   v
main.py
   |
   | Per-question scheduling, max concurrency = 2
   v
agent_loop.py
   |
   | Sends the question and conversation history
   v
OpenAI-compatible LLM
   |
   | Outputs a <tool_call> block or the final answer
   v
tools.py / NOCClient
   |
   | POST /api/agent/execute
   v
official server.py
   |
   | Reads CLI output by question_number + device_name + command
   v
devices_outputs/
   |
   v
agent_loop.py
   |
   | Iterative analysis until the final answer is produced
   v
result.csv + detail.jsonl + progress.json
```

## 4. Per-Question Workflow

Each question is processed as follows:

1. `main.py` loads the question from `test.json` and creates a `NOCClient` for the question id.
2. `prompts.detect_family()` roughly classifies the task as `topology`, `path`, or `fault`.
3. `agent_loop.solve()` sends the system prompt and question to the LLM.
4. If the LLM outputs a `<tool_call>` block, the system extracts the device name and command.
5. `NOCClient.execute()` calls the official server and automatically includes `question_number`.
6. The CLI output is fed back to the LLM as a new user message.
7. The LLM either calls another tool or outputs the final answer.
8. `prompts.extract_answer()` extracts a clean answer according to the task family.
9. `main.py` writes the prediction to `result.csv` and records the task family, call count, duration, and answer in `detail.jsonl`.

## 5. Competition Constraint Mapping

| Competition Constraint | Current System Design |
|---|---|
| Tool calls within one question must be sequential | `agent_loop.py` parses and executes only one `<tool_call>` at a time |
| Overall concurrency is limited to 2 questions | `main.py --concurrency` defaults to 2 |
| Each request must include the question number | `NOCClient` automatically sends `question_number` |
| Phase 2 tie-break may consider API call count | `detail.jsonl` records per-question `call_count` |
| Output is judged by exact match | `extract_answer()` cleans model output |
| Some commands may be blocked by permission restrictions | `NOCClient` wraps HTTP 403 as `[PERMISSION DENIED]` |
| Some commands may have syntax errors | `NOCClient` wraps HTTP 422 as `[COMMAND ERROR]` |

## 6. Strategies by Task Type

### 6.1 Topology Restoration

Preferred commands:

```text
display lldp neighbor brief
display interface brief
display interface description
display arp
```

Normal procedure:

1. Query LLDP on the target node.
2. If LLDP on the target node is blocked, query LLDP on each candidate neighbor.
3. If LLDP is still unavailable, cross-check interface descriptions, ARP entries, and interface states.
4. Output `LocalNode(port)->RemoteNode(port)`.

### 6.2 Path Query

Preferred commands:

```text
display ip interface brief
display ip routing-table
display arp
```

Normal procedure:

1. Identify the target port IP or the subnet containing the target IP.
2. Perform longest-prefix matching in the source node's routing table to find the next hop.
3. Use interface IP tables to identify which device owns the next-hop IP.
4. Repeat hop by hop until reaching the destination node.
5. Output `NodeA->NodeB->NodeC`.

### 6.3 Fault Localization

Preferred commands:

```text
display ip routing-table
display interface brief
display current-configuration
display arp
display logbuffer
display alarm active
display bgp evpn all routing-table
display bgp vpnv4 all routing-table
display vxlan tunnel
display srv6-te policy
```

Normal procedure:

1. Infer the expected forwarding path first.
2. Check whether each hop has missing routes, wrong routes, blackhole routes, or routing loops.
3. Check whether related interfaces are shut down, have incorrect IP addresses, or show bandwidth overuse.
4. Check ARP, BGP, OSPF, VXLAN, L3VPN, L2VPN, SRv6, and other protocol or configuration states.
5. Use the exact fault reason strings listed in the current question.

## 7. Current Risks

The current version still has the following risks and should address them early in the plan:

- The default server port must be aligned with the official `server.py`.
- The `--resume` logic must not clear previous results.
- Fault reasons must use the exact strings from the question, not a small fixed legacy vocabulary.
- `PSS` is a Cisco device, so Cisco command guidance must be added.
- Long routing tables and configurations are currently sent directly to the LLM, which may bloat the context.
- Pure ReAct has weak control over call count, so deterministic parsing and rule-based solvers should be introduced later.

## 8. Target Design

The ideal final system should not be a pure LLM-driven blind exploration loop. It should become:

```text
question parser
  + deterministic CLI parser
  + topology/routing graph builder
  + rule-based fault detector
  + LLM-assisted judgment
  + strict answer formatter
```

This design can improve accuracy, stability, and API-call efficiency at the same time.
