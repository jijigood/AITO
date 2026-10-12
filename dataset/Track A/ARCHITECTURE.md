# System Architecture — Track A Agent

## Design Goals

1. **Accuracy** — correctly identify the root cause(s) from 22 options using structured RF analysis
2. **Speed** — answer each question in < 5 minutes to qualify for the full score discount in Phase 3
3. **Generalisability** — perform well on unseen test data from the same distribution as training

---

## High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                         Orchestrator  (main.py)                        │
│                                                                         │
│  for each scenario in test.json:                                        │
│    1. fetch few-shot context from RAG Module                            │
│    2. build system prompt = diagnostic protocol + few-shot context      │
│    3. run Diagnostic Agent loop (max 8 turns)                           │
│    4. extract \boxed{answer} → write row to result.csv                  │
└────────────────┬───────────────────────────────┬────────────────────────┘
                 │                               │
    ┌────────────▼────────────┐    ┌─────────────▼─────────────┐
    │      RAG Module          │    │     Diagnostic Agent        │
    │   (rag_retriever.py)     │    │        (main.py)            │
    │                          │    │                             │
    │  FAISS index of 2000     │    │  Fine-tuned Qwen3.5-35B    │
    │  training embeddings     │    │  calls server.py tools      │
    │  → top-k few-shots       │    │  in structured order        │
    └────────────┬────────────┘    └─────────────┬───────────────┘
                 │                               │
                 └───────────┬───────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Answer Parser   │
                    │ (answer_parser)  │
                    │                  │
                    │  \boxed{} → C3   │
                    │  validate, sort  │
                    └────────┬────────┘
                             │
                        result.csv
```

---

## Component Detail

### 1. Orchestrator (`main.py`)

The top-level loop. For each of the 500 test scenarios:

- Calls the RAG Module to retrieve 2–3 similar past cases
- Builds the system message: `SYSTEM_PROMPT + few-shot examples`
- Instantiates the Diagnostic Agent conversation with `[system, user]`
- Runs the tool-calling loop (up to 8 iterations)
- If no `\boxed{}` is found after the loop, sends one final forced-answer prompt
- Parses the answer, writes to `result.csv`
- Saves partial results every 10 scenarios (crash recovery)

**Key design decision — system message placement:** Injecting the diagnostic protocol as a `system` role message (rather than prepending it to the user message) ensures it is treated as persistent instruction by the model rather than part of the problem description.

---

### 2. RAG Module (`rag_retriever.py`)

Provides semantic few-shot retrieval so the model sees concrete solved examples that resemble the current problem before it starts calling tools.

**Index construction (once at startup):**

```
train.json (2000 examples)
    │
    ▼  for each example:
    │  text = description + option_labels
    │
    ▼
sentence-transformers/all-MiniLM-L6-v2
    │
    ▼
768-dim L2-normalised embeddings
    │
    ▼
FAISS IndexFlatIP  ──cached to .rag_cache/──►  reused on next run
```

**Retrieval (per scenario at inference):**

```
test question text
    │
    ▼
embed with same model
    │
    ▼
FAISS nearest-neighbour search (cosine similarity)
    │
    ▼
top-k training examples
    │
    ▼
format as:
  ### Example 1 [single-answer] (similarity=0.94)
  Options: C1: ..., C2: ..., ...
  Key RF observations: worst_t=..., serving PCI=..., RSRP=..., SINR=...
  Correct answer: C9 (Decrease A3 Offset threshold for 3225568_1)
    │
    ▼
appended to SYSTEM_PROMPT
```

**Why cosine similarity over BM25:** The question text for all 2000 examples is nearly identical (same problem template). The discriminating signal is in the option labels — specifically the cell IDs and parameter names. Dense embeddings capture this semantic similarity; BM25 would not differentiate well given the templated structure.

---

### 3. Diagnostic Agent (tool-calling loop in `main.py`)

The core reasoning engine. It is the model (Qwen3.5-35B-A3B, base or fine-tuned) interacting with `server.py` through the OpenAI tool-use API.

**Conversation structure:**

```
[system]    SYSTEM_PROMPT + few-shot examples
[user]      "Analyze 5G network drive test data... Options: C1:... C22:..."
[assistant] → tool_calls: [get_throughput_logs]
[tool]      {"Logs": "Timestamp|DL_TP\n..."}
[assistant] → tool_calls: [get_serving_cell_pci(time=worst_t)]
[tool]      {"PCI": 328}
[assistant] → tool_calls: [get_cell_info(328), get_serving_cell_rsrp(t),
                            get_serving_cell_sinr(t), get_neighboring_cells_pci(t)]
[tool×4]    {cell config}, {RSRP}, {SINR}, {neighbor PCIs}
[assistant] → tool_calls: [get_neighboring_cell_rsrp(t, pci) × N]
[tool×N]    {neighbor RSRP values}
[assistant] "Evidence: ... Findings: ... \boxed{C9}"
```

**Parallel tool calls:** In turns 3 and 4, the model is instructed (via the system prompt) to issue multiple tool calls in a single response. The existing loop already iterates over `msg.tool_calls` as a list, so all calls in one turn are executed before the next model call. This reduces 8–10 sequential round-trips to 4–5 turns, cutting wall-clock time by roughly 50%.

**Diagnostic protocol (encoded in SYSTEM_PROMPT):**

| Step | Tool(s) called | Decision extracted |
|---|---|---|
| 1 | `get_throughput_logs` | Worst degradation timestamp |
| 2 | `get_serving_cell_pci(t)` | Serving cell PCI |
| 3 | `get_cell_info` + `get_serving_cell_rsrp` + `get_serving_cell_sinr` + `get_neighboring_cells_pci` (parallel) | A3 threshold, neighbor list, RSRP, SINR, visible neighbor PCIs |
| 4 | `get_neighboring_cell_rsrp × N` (parallel) | Strongest neighbor RSRP |
| 5 (optional) | `get_signaling_plane_event_log` / `get_kpi_data` / `get_mr_data` | HO failures, cell load, aggregate MR data |

**Root cause mapping (also in SYSTEM_PROMPT):**

| RF condition | Root cause | Answer type |
|---|---|---|
| neighbor RSRP − serving RSRP > A3Offset×0.5 + A3Hyst×0.5, neighbor in config | Late handover | Decrease A3 Offset for serving cell |
| Strong neighbor NOT in neighbor list | Missing relation | Add neighbor relationship |
| SINR < 0 dB, RSRP > −110 dBm | Interference | Tilt down / reduce power |
| RSRP < −110 dBm | Coverage hole | Increase power / lift tilt |
| RF healthy, throughput low | Backhaul/server | Check test server |
| High PRB utilization in KPI | Cell overloaded | Decrease CovInterFreq thresholds |

---

### 4. Skills (`skills/`)

Stateless utility modules. Not called by the agent at runtime — used by the fine-tuning pipeline and offline evaluation.

**`skills/rf_analysis.py`**
- `check_a3_handover_condition` — exact 3GPP A3 computation (values in 0.5 dB units)
- `check_neighbor_in_config` — parses the `PCell Neighbor Cell` string to check PCI presence
- `classify_rf_issue` — heuristic classifier mapping RF snapshot → issue category + reasoning string

**`skills/answer_parser.py`**
- `parse_answer` — `\boxed{}` extraction + C1–C22 validation + ascending sort
- `iou_score` — IOU metric matching the competition evaluator exactly
- `batch_iou` — aggregate accuracy reporting for offline evaluation runs

---

### 5. Fine-Tuned Qwen3.5-35B-A3B (`finetune/`)

The fine-tuned model replaces the base model for Phase 2 and Phase 3. It is trained to reliably output structured `\boxed{answer}` responses after a minimal, well-ordered set of tool calls, reducing both the number of iterations needed and the rate of malformed outputs.

**Dataset generation (`finetune/prepare_data.py`):**

For each of the 2000 training examples, the CSV data embedded in `train.json` is parsed directly (no HTTP server needed) to simulate the tool call responses and construct the full multi-turn conversation. Output: 1800 train + 200 validation examples in JSONL.

**Training (`finetune/llamafactory_config.yaml` + `finetune/train.py`):**

```
Qwen3.5-35B-A3B (frozen)
    │
    ▼ LoRA adapter (trainable)
    │  rank=16, alpha=32
    │  target: all projection layers (q,k,v,o,gate,up,down)
    │
    ▼ 1800 SFT examples
    │  3 epochs, lr=2e-4, cosine schedule
    │  bf16, flash-attn2
    │
    ▼
lora_output/   ← adapter weights only (~1 GB)
    │
    ▼ (optional merge)
merged_model/  ← full model for vLLM serving
```

**Inference with fine-tuned model:**

Serve the merged model with vLLM and point `main.py` at it:
```bash
vllm serve merged_model/ --enable-auto-tool-choice --tool-call-parser hermes
python main.py --model_url http://localhost:8000/v1 --model_name merged_model
```

---

## Data Flow (End-to-End)

```
                         OFFLINE (one-time)
─────────────────────────────────────────────────────────────────────
train.json ──► prepare_data.py ──► sft_train.jsonl / sft_val.jsonl
                                        │
                                        ▼
                              llamafactory-cli train
                                        │
                                        ▼
                                   lora_output/
                                        │
                              llamafactory-cli export
                                        │
                                        ▼
                                  merged_model/
                                        │
                                        ▼
                                  vLLM server

train.json ──► rag_retriever.build_index() ──► .rag_cache/


                         ONLINE (per test scenario)
─────────────────────────────────────────────────────────────────────
test.json (500 scenarios)
    │
    ▼ for each scenario
    │
    ├──► RAG: embed question ──► FAISS search ──► top-2 few-shots
    │
    ├──► build system message = SYSTEM_PROMPT + few-shots
    │
    └──► Diagnostic Agent loop
              Turn 1: get_throughput_logs
              Turn 2: get_serving_cell_pci
              Turn 3: [parallel] get_cell_info + get_rsrp + get_sinr + get_neighbors
              Turn 4: [parallel] get_neighboring_cell_rsrp × N
              Turn 5: (optional) get_kpi_data / get_signaling_plane_event_log
              Final:  \boxed{C9}
                │
                ▼
         answer_parser: extract → validate → sort
                │
                ▼
          result.csv row: scenario_id, answers
```

---

## Phase-by-Phase Deployment

| Phase | Model | RAG | Notes |
|---|---|---|---|
| Phase 1 (now → May 4) | Base Qwen3.5-35B-A3B via OpenRouter | Yes (2 few-shots) | Unlimited submissions; iterate on prompt |
| Phase 2 (May 4–18) | LoRA fine-tuned, self-hosted | Yes (2 few-shots) | 3 submissions max; use best |
| Phase 3 (May 18–29) | LoRA fine-tuned, organiser-hosted | Yes | Speed matters: < 5 min = 100% discount |

---

## Ablation Flags (for iteration)

The following CLI flags allow controlled experiments to measure the contribution of each component:

```bash
# Baseline (no system prompt, no RAG)
python main.py --no_system_prompt --no_rag

# System prompt only
python main.py --no_rag

# System prompt + RAG (full system, default)
python main.py

# Vary RAG depth
python main.py --rag_top_k 1
python main.py --rag_top_k 3
```

Compare `mean_iou` across runs on the training set (use `DATA_SPLIT=train` on the server) to quantify each component's contribution.
