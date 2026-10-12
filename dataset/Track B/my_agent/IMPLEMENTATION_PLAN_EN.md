# Track B Agent Implementation Plan

This document describes the implementation plan for `my_agent`. The guiding principle is to first make the system stable and runnable, then gradually reduce API calls and improve exact-match accuracy, and finally evolve it into a reusable deterministic network reasoning framework.

## Overall Principles

1. Stability first: ensure that all questions can finish and produce a valid `result.csv`.
2. Exact formatting first: Track B uses exact match, so a formatting error is effectively a wrong answer.
3. Reduce calls without being reckless: Phase 2 tie-breaks may consider API call count, but accuracy remains the top priority.
4. Combine rules with the LLM: deterministic network facts should be parsed by code, while complex judgments can be assisted by the LLM.
5. Every step must be verifiable: each stage should be tested on both a small smoke set and the full 50-question set.

## Step 1: Fix Basic Runtime Issues

### Prior Statement

This step is not intended to significantly improve accuracy. Its goal is to make the baseline reliable to run, resume, and submit.

### Work Items

1. Change the default `server_url` in `main.py` from `http://localhost:7861` to `http://localhost:7860`.
2. Fix the issue where `--resume` clears `result.csv` and `detail.jsonl`.
3. Sort `result.csv` by `id` before final output.
4. Add startup checks:
   - Whether `test.json` exists.
   - Whether the Agent Tool Server is reachable.
   - Whether the LLM endpoint is reachable.
5. Add per-question command caching inside `NOCClient` to avoid repeated calls to the same command.

### Expected Effect

- The runtime entry point is aligned with the official server default.
- Interrupted runs can resume without losing completed answers.
- The submission file has stable ordering.
- Repeated command calls are reduced.
- After this step, the system should be able to reliably finish all 50 questions.

### Verification

```bash
python my_agent/main.py --questions 1,2,7 --server_url http://localhost:7860
python my_agent/main.py --resume --server_url http://localhost:7860
```

Check:

- Whether `my_agent/results/result.csv` exists.
- Whether the CSV header is `id,prediction`.
- Whether `detail.jsonl` records call count and duration.
- Whether old results remain after resume.

## Step 2: Fix Prompts and Output Vocabulary

### Prior Statement

This step primarily addresses exact-match risk. Even if the network reasoning is correct, using the wrong fault reason string will still lead to a wrong answer.

### Work Items

1. State in the prompt that fault reasons must use the exact strings listed in the current question.
2. Add all fault reasons that appear in Phase 1 questions:
   - `blackhole route`
   - `missing static route`
   - `static route error`
   - `ARP configuration error`
   - `routing loop`
   - `BGP configuration error`
   - `OSPF configuration error`
   - `loopback IP configuration conflict`
   - `VXLAN configuration error`
   - `L3VPN configuration error`
   - `L2VPN configuration error`
   - `IS-IS configuration error`
   - `SRV6-Policy tunnel planning error`
   - `shutdown`
   - `interface IP error`
   - `traffic congestion on port bandwidth`
   - `MAC address configuration error`
   - `VPN configuration missing`
   - `MTU value configuration error`
   - `host information collection function missing`
3. Add that `PSS` is a Cisco device and must use `show ...` commands.
4. Add lightweight validation after `extract_answer()`:
   - Topology answers must match `A(port)->B(port)`.
   - Path answers must match `A->B->C`.
   - Fault answers must match `node;target;cause`.

### Expected Effect

- Fault reason strings become closer to the question wording, reducing cases where the meaning is right but the string is wrong.
- Questions involving `PSS` should have fewer command syntax errors.
- Answer extraction becomes more robust when the model includes explanations.

### Verification

First run a sample set covering all task types:

```bash
python my_agent/main.py --questions 1,2,7,17,29,34,39 --server_url http://localhost:7860
```

Check:

- Whether every fault reason comes exactly from the question text.
- Whether multi-line answers contain no blank lines.
- Whether calls related to `PSS` use Cisco commands.

## Step 3: Add a Command Output Compression Layer

### Prior Statement

The goal of this step is to reduce context bloat. It does not change the official server output; it only compresses information before sending it to the LLM.

### Work Items

Add `summarize_output(command, raw_output, question_context)` after `NOCClient.execute()`:

1. `display lldp neighbor brief`: parse into an edge table.
2. `display interface brief`: keep only interface name, physical state, protocol state, utilization, and error counters.
3. `display ip interface brief`: keep only interface, IP, mask, and state.
4. `display ip routing-table`: prioritize routes related to the target subnet, default routes, static routes, and blackhole routes.
5. `display current-configuration`: extract relevant blocks by keywords such as interface, ip route-static, bgp, ospf, vpn-instance, vxlan, and srv6.
6. `display arp`: structure as IP, MAC, and interface.

### Expected Effect

- The LLM context becomes shorter, making long questions less likely to forget earlier information.
- Per-question runtime decreases.
- Complex fault questions become easier to diagnose because key configurations are highlighted.

### Verification

Compare before and after compression:

- Per-question runtime.
- Whether LLM calls hit context-length issues.
- Average `call_count` in `detail.jsonl`.
- Whether answers for key sample questions do not regress.

## Step 4: Deterministic Solver for Topology Restoration

### Prior Statement

Topology restoration is the easiest task type to encode deterministically. The goal is to solve questions 1-6 and 29-33 with minimal reliance on LLM reasoning.

### Work Items

1. Extract the target node and candidate neighbors from the question text.
2. First call `display lldp neighbor brief` on the target node.
3. If the target node is blocked, iterate over candidate neighbors and query their LLDP tables.
4. If LLDP is unavailable, cross-check interface descriptions, interface brief, and ARP.
5. Generate `LocalNode(port)->RemoteNode(port)` directly in code.

### Expected Effect

- Topology answers become more stable.
- API call count becomes controllable.
- Output formatting errors are largely eliminated.

### Verification

```bash
python my_agent/main.py --questions 1,2,3,4,5,6,29,30,31,32,33 --server_url http://localhost:7860
```

Check:

- Whether every line strictly matches `A(port)->B(port)`.
- Whether no-permission questions automatically use reverse LLDP.

## Step 5: Deterministic Solver for Path Queries

### Prior Statement

Path questions can be inferred hop by hop from routing tables. The goal is to make code handle most path reasoning, with the LLM only as fallback.

### Work Items

1. Extract the source node, target node, target port, or target IP from the question.
2. If the target IP is not given, query `display ip interface brief` on the target node.
3. Perform longest-prefix matching in the source node's `display ip routing-table`.
4. Reverse-map the next-hop IP to the owning device.
5. Repeat hop by hop until reaching the target node.
6. Generate `A->B->C`.

### Expected Effect

- Path-query call count decreases.
- Path output formatting becomes stable.
- For known no-permission route-table questions, the solver can switch early to neighbor/interface-IP inference.

### Verification

```bash
python my_agent/main.py --questions 7,8,9,10,11,12,13,14,15,16,34,35,36,37,38 --server_url http://localhost:7860
```

Check:

- Whether the path reaches the target node.
- Whether the answer incorrectly includes interface names or IP addresses.
- Whether there are extra spaces.

## Step 6: Strengthen Fault Localization Rules

### Prior Statement

Fault questions are the most complex and should not be fully rule-based in one jump. Start with high-confidence rules, then let the LLM make the final judgment when protocol-level reasoning is needed.

### Work Items

1. Static route faults:
   - Missing route: `missing static route`
   - Wrong next hop: `static route error`
   - Unreachable next hop or egress interface shut down: `blackhole route` or port `shutdown`
2. Interface faults:
   - `*down` or configured `shutdown`: `shutdown`
   - Interface IP inconsistent with peer or planning data: `interface IP error`
   - Abnormal utilization: `traffic congestion on port bandwidth`
3. ARP faults:
   - Abnormal IP-to-interface or IP-to-MAC binding: `ARP configuration error`
4. Protocol faults:
   - For BGP, OSPF, VXLAN, L3VPN, L2VPN, and SRv6, first extract key configuration and state, then pass structured evidence to the LLM.

### Expected Effect

- Simple fault questions no longer rely on free-form LLM reasoning.
- Complex protocol questions have structured evidence, reducing misjudgment.
- Fault reason strings stay aligned with the question text.

### Verification

```bash
python my_agent/main.py --questions 17,18,19,20,21,22,23,24,25,26,27,28,39,40,41,42,43,44,45,46,47,48,49,50 --server_url http://localhost:7860
```

Check:

- Whether each line contains exactly one fault.
- Whether routing-fault and port-fault formats are mixed up.
- Whether any output uses a fault reason not listed in the question.

## Step 7: Full Regression and API Call Optimization

### Prior Statement

This step avoids major strategy changes. It focuses on regression, statistics, and small targeted improvements.

### Work Items

1. Run all 50 questions.
2. Summarize each question by:
   - Task family.
   - Call count.
   - Duration.
   - Whether the answer is empty.
   - Whether the answer format is invalid.
3. Optimize questions with unusually high call counts.
4. Improve `extract_answer()` for questions with invalid formatting.
5. Freeze a smoke-test question set.

### Expected Effect

- A stable baseline is formed.
- API-call bottlenecks become visible.
- Phase 2 migration has clear debugging entry points.

### Verification

```bash
python my_agent/main.py --input data/Phase_1/test.json --server_url http://localhost:7860 --concurrency 2
```

Output files to check:

```text
my_agent/results/result.csv
my_agent/results/detail.jsonl
my_agent/results/progress.json
```

## Recommended Execution Order

The five highest-priority tasks are:

1. Fix the default server port.
2. Fix resume so it does not clear results.
3. Make fault reasons prefer the exact wording from the question.
4. Add PSS/Cisco command guidance.
5. Add per-question command caching.

After these five items are done, move on to deterministic solvers. This will quickly produce a runnable, resumable, and measurable Track B baseline.

## 2026-04-29 Second Competition-Goal Check

This pass fixed several low-risk gaps against the official objectives:

1. Added `host information collection function missing` to the fault-cause vocabulary.
2. Fault-answer normalization now strips numbered prefixes such as `(3)` and splits comma-separated multi-cause answers into separate lines.
3. H3C LLDP now prefers the README command `display lldp neighbor-information`.
4. Route-table summaries now also cover Cisco `show ip route` / `show route`.
5. `detail.jsonl` trace records include `question_number`, `started_at`, and `elapsed_ms` for Phase 2 trace review.
6. Final `result.csv` sorting removes duplicate ids; `validate_results.py` now reports duplicate ids; `--resume` uses non-empty CSV rows to avoid treating empty failures as completed work.

Remaining core gaps:

1. Path and fault questions still need deterministic solvers; current accuracy still relies heavily on LLM reasoning.
2. If Phase 2 publishes a strict execution-trace upload schema, `detail.jsonl` may need a final adapter.
3. Phase 3 SRv6/EVPN/ISIS/BGP faults still need specialized parsers beyond the current prompt and command coverage.

## 2026-04-29 Test Record

Actual test results:

1. Static compilation passed for `agent_loop.py`, `answer_validation.py`, `deterministic.py`, `main.py`, `prompts.py`, `tools.py`, and `validate_results.py`.
2. Offline fixtures:
   - topology: 11/11 format-valid deterministic answers.
   - physical-interface path questions Q7-Q16: format-valid deterministic answers.
3. First live smoke (Q1/Q2/Q7/Q17/Q34/Q39/Q42):
   - Q1/Q2 topology succeeded.
   - path/fault exposed repeated tool calls and empty final answers from the model.
4. Fixes added:
   - `llm_trace` records raw model-response previews for debugging.
   - malformed `<tool_call>` is treated as a retryable model-format error.
   - deterministic path solver traces physical paths via longest-prefix routing matches plus LLDP egress ports.
5. Q7 retest:
   - solved by deterministic path without LLM.
   - valid answer:
     `Beta-Node-01->Beta-Axis-02->Beta-Portal-02->Alpha-Center-02->Gamma-Portal-02->Gamma-Axis-02->Gamma-Node-01`

Conclusion:

- Topology and Q7-Q16-style physical-interface path questions now have deterministic-first coverage.
- Fault questions and PJ-area path questions with only service IPs remain the next priority.
