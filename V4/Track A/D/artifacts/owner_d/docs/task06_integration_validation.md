# Task 06 Integration And Validation

## Integration Summary

Owner D artifacts are currently consumable from both A and C side through the existing retrieval path:

- Primary retriever class: `rag_retriever_v2.StructuredCaseRAG`
- Stable legacy entrypoint: `rag.retrieve(sf, k=..., family_name=...)`
- Prompt-safe entrypoint for direct context injection: `rag.retrieve_trajectory_context(sf, branch, k=3, family_name=...)`

No caller is required to migrate immediately. Existing call sites in `phase2_run.py`, `main_v3.py`, `build_sft_v3.py`, and `eval_ckpt.py` still work with the same `StructuredCaseRAG(str(train_json)); build_index(); retrieve(...)` flow.

## A/C Consumption Notes

### For Owner A

- `phase2_run.py` can keep using `rag.retrieve(...)`.
- Branch-aware retrieval is now applied automatically when `phase2_run.py` can infer a high-risk branch from feature scores.
- Debug diagnostics may still store internal fields such as `gt_c_code`, but prompt formatting no longer injects them into the LLM-facing text.

### For Owner C

- Preferred prompt-safe interface is `retrieve_trajectory_context(sf, branch, k=3, family_name=...)`.
- If C consumes RAG hits directly, only these fields should be used in prompt text:
  - `branch`
  - `root_cause`
  - `gt_action_type` / `action`
  - `target_role`
  - `confidence`
  - `key_evidence`
  - `tool_sequence`
- `trajectory_prompt_snippets.md` is safe to quote or adapt into branch-specific prompts.

## Minimal Validation

All checks below were run in conda env `tele`.

### 1. Retriever construction

- Built `StructuredCaseRAG('data/Phase_1/train.json')`
- `build_index()` completed successfully
- Cached index load also succeeded

### 2. Prompt-safe context

- Called `retrieve_trajectory_context()` on an F1 high-risk query
- Output contained:
  - `branch`
  - `root_cause`
  - `action`
  - `target_role`
  - `confidence`
  - `key_evidence`
  - `tool_sequence`
- Output did not contain:
  - `gt_c_code`
  - `answer_labels`
  - `Correct answer`
  - `C1-C22` training answer identifiers

### 3. Existing A-side path

- `phase2_run._format_rag_hits_for_prompt()` rendered safe text from current `RAGHit` objects
- `main_v3._build_f1_prompt()` assembled an F1 prompt successfully using current RAG hits
- `eval_ckpt.py` style path (`StructuredCaseRAG -> build_index`) remained usable

## Current Limits

- Probe-derived memory ingestion is implemented, but the current workspace does not include real Phase 2 probe `.diagnostics.jsonl` artifacts, so `trajectory_probe_memory.jsonl` is currently empty.
- `trajectory_memory_combined.jsonl` therefore still contains only `phase1_train` records at the moment.
- C can consume prompt-safe context now, but true probe-specific few-shot coverage depends on A exporting diagnostics files into the shared tree.

## Residual Risks

- `main_v3.py` still prints the current scenario's option list with local `C1-C22` codes, which is expected for solving the live scenario; this is not training leakage, but it means prompt scans must distinguish local option codes from retrieved training answer codes.
- Internal diagnostics dictionaries still retain debug-only fields such as `gt_c_code`. This is acceptable for audit/debug, but any future prompt formatter must continue filtering them out.
- Probe branch coverage is structurally ready but data-limited. High-risk branches with real Phase 2 tool traces are not yet represented as stored probe memory in this repo snapshot.

## Recommended Next Use

1. A can continue calling `rag.retrieve(...)` without code changes.
2. C can start using `retrieve_trajectory_context()` for direct prompt injection.
3. Once A exports probe diagnostics JSONL into `results/phase2/...`, rerun:
   - `python build_probe_trajectory_memory.py`
   - `python build_combined_trajectory_memory.py`
   - `python build_trajectory_coverage_report.py`

