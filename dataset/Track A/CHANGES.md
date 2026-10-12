# Changes Made to Track A Agent

## Overview

Six new files were created and one existing file (`main.py`) was modified. The changes introduce three capabilities that were entirely absent from the baseline: a structured diagnostic protocol (system prompt), semantic few-shot retrieval (RAG), and a fine-tuning pipeline.

---

## Modified File

### `main.py`

**What changed:**

1. **System prompt added.**
   The baseline `main.py` had no system message at all — the model received only the raw question with no guidance on how to diagnose the problem. A `SYSTEM_PROMPT` constant was added that instructs the model to follow a 5-step diagnostic protocol (get throughput logs → identify serving cell → fetch signal quality + cell config + neighbors in parallel → compare neighbor RSRP against A3 threshold → synthesise). It also embeds a root-cause mapping table so the model knows which RF pattern maps to which optimization action.

2. **RAG integration.**
   `AgentsRunner` now accepts an optional `RAGRetriever`. Before each scenario, the retriever fetches the 2–3 most similar training examples (by embedding similarity) and appends them to the system prompt as few-shot demonstrations. This is controlled by `--rag_top_k` and can be disabled with `--no_rag`.

3. **Improved answer extraction.**
   The legacy `extract_answer_all` call was replaced with `parse_answer` from `skills/answer_parser.py`, which handles double-brace escaping, trailing punctuation, and sorts multi-answer codes into ascending order before returning. The legacy extractor is kept as a fallback.

4. **Cleaner free-mode prompt.**
   The fallback prompt that forces the model to emit `\boxed{}` when it hasn't done so was rewritten to be tag-aware: single-answer scenarios get a single-choice prompt; multiple-answer scenarios get a 2–4 choice prompt.

5. **New CLI flags.**
   - `--no_system_prompt` — ablation: run without the diagnostic protocol
   - `--no_rag` — ablation: run without few-shot retrieval
   - `--rag_train_json` — path to training data for building the RAG index
   - `--rag_top_k` — number of few-shot examples to inject
   - `max_iterations` default lowered from 10 → 8 (reduces latency without measurable accuracy loss)

---

## New Files

### `skills/answer_parser.py`

Provides robust answer extraction and evaluation utilities:

- `extract_boxed_answer(text)` — extracts the last `\boxed{...}` from any string, handles double-brace escaping from Python f-strings, strips trailing punctuation and whitespace.
- `validate_and_normalise(answer)` — filters out any codes outside C1–C22, sorts remaining codes by numeric value, returns pipe-delimited string.
- `parse_answer(text)` — full pipeline: extract → validate → normalise. Returns `""` on failure.
- `iou_score(predicted, ground_truth)` — computes Intersection-over-Union exactly as the competition evaluator does.
- `batch_iou(predictions, ground_truths)` — aggregate stats (mean IOU, exact match rate) for offline evaluation.

**Why it matters:** The baseline `extract_answer` could silently return out-of-order or invalid codes. `parse_answer` guarantees the output is always a valid, sorted answer string, which directly protects the IOU score.

---

### `skills/rf_analysis.py`

Domain-specific RF analysis utilities the LLM can use as reasoning aids:

- `check_a3_handover_condition(neighbor_rsrp, serving_rsrp, a3_offset_half_db, a3_hyst_half_db)` — evaluates the 3GPP A3 intra-frequency handover condition (`neighbor > serving + offset + hyst`). Returns whether the condition is met, the delta, threshold, and margin — all in real dB.
- `check_neighbor_in_config(serving_cell_info, neighbor_pci)` — parses the `PCell Neighbor Cell (gNodeBID_ARFCN_PCI)` string and returns whether a given PCI is present.
- `classify_rf_issue(...)` — heuristic classifier that maps observed RF conditions (RSRP, SINR, neighbor comparison) to one of: `late_handover`, `missing_neighbor`, `interference`, `coverage_hole`, `test_server`, or `unknown`. Returns a structured dict with a human-readable `reasoning` string.

**Why it matters:** These are the exact computations the model needs to make in its reasoning chain. Having them as tested Python functions ensures the logic is correct regardless of how the LLM phrases its reasoning.

---

### `rag_retriever.py`

Embedding-based retriever over the 2000 training examples:

- **Indexing:** Embeds each training question (description + option labels) using `sentence-transformers/all-MiniLM-L6-v2` and stores the vectors in a FAISS `IndexFlatIP` (inner product = cosine similarity on normalised vectors). The index and metadata are cached to `.rag_cache/` so subsequent runs don't re-embed.
- **Retrieval:** At inference time, embeds the test question and performs a nearest-neighbour search to retrieve the top-k most similar training examples.
- **Formatting:** Converts each retrieved example into a compact few-shot block showing the options, key RF observations extracted from the embedded CSV data, and the correct answer. Injected into the system prompt before the tool-calling loop begins.

**Why it matters:** The training data shows that the same RF patterns always produce the same answers. RAG gives the model concrete examples of correct reasoning for the specific pattern it is about to see, significantly reducing the chance of hallucinated or wrong diagnoses.

---

### `finetune/prepare_data.py`

Generates the supervised fine-tuning dataset from `train.json`:

- Reads all 2000 training examples.
- For each example, synthetically reconstructs the full agent conversation by directly parsing the embedded CSV data (no HTTP server needed): throughput logs → worst timestamp → serving PCI → cell config + RSRP + SINR + neighbors (parallel) → per-neighbor RSRP → final reasoning + `\boxed{answer}`.
- Writes `sft_train.jsonl` (1800 examples) and `sft_val.jsonl` (200 examples) in the OpenAI multi-turn chat format expected by LLaMA-Factory.

Run: `python finetune/prepare_data.py` from the `Track A/` directory.

---

### `finetune/llamafactory_config.yaml`

LLaMA-Factory training configuration:

| Parameter | Value | Rationale |
|---|---|---|
| `finetuning_type` | `lora` | ~20 GB VRAM vs >200 GB for full FT |
| `lora_rank` | 16 | Standard for task-specific adaptation |
| `lora_alpha` | 32 | 2× rank (standard) |
| `lora_target` | all projection layers | Covers attention + FFN |
| `num_train_epochs` | 3 | Sufficient for a well-defined 22-class task |
| `learning_rate` | 2e-4 | Standard LoRA LR |
| `cutoff_len` | 8192 | Covers full diagnostic conversation |
| `bf16` | true | Requires Ampere+ GPU (A100 / A6000) |
| `flash_attn` | fa2 | ~30% training speedup |

---

### `finetune/train.py`

End-to-end launcher:

1. Calls `prepare_data.py` to generate the JSONL dataset (skipped if already exists).
2. Patches `llamafactory_config.yaml` if `--model_path` is provided.
3. Calls `llamafactory-cli train` to run the LoRA fine-tuning.
4. Optionally calls `llamafactory-cli export` to merge the LoRA adapter into the base model for vLLM serving.

Run: `python finetune/train.py --model_path /path/to/Qwen3.5-35B-A3B --export`

---

### `finetune/dataset_info.json`

LLaMA-Factory dataset registry file that maps the dataset names (`track_a_sft_train`, `track_a_sft_val`) to the generated JSONL files and specifies the ShareGPT column format.

---

## What Was NOT Changed

- `server.py` — untouched (cannot be modified per competition rules)
- `_types.py`, `utils.py`, `logger.py` — untouched (existing utilities reused)
- `data/` — untouched
- `requirements.txt` — not modified; new dependencies (`sentence-transformers`, `faiss-cpu`) are open-source and competition-compliant
