# v5_20 Qwen Sample-20 Audit

Date: 2026-05-18

## Setup

- Sample: 20 rows from the 60 `v5_20` F1/F1_5BS flips.
- Composition: 6 `F1_5BS -> decrease_a3` rows plus 14 `F1 -> press_down_tilt` rows.
- Model: local vLLM `qwen` base model, not `diag` LoRA.
- API path: real Phase 2 server diagnostics path via `phase2_run.py`.
- Proxy: `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY` cleared; `NO_PROXY=*`; local/server clients bypass env proxy.
- Final successful run: `--llm_policy arbitrate --f1_llm_first --no_llm_thinking --llm_max_tokens 512`.
- Attempted extended-thinking run with `--llm_max_tokens 2048`, but one request exceeded practical runtime and retried, so it was stopped before any completed row.

## Core Result

Raw first-answer agreement with `v5_20`:

- Overall: 6/20
- `decrease_a3` subset: 6/6
- `press_down_tilt` subset: 0/14

Final LLM answer agreement with `v5_20`:

- Overall: 9/20
- `decrease_a3` subset: 6/6
- `press_down_tilt` subset: 3/14

The final-answer count is less important than raw first-answer count because the confirmation step compares against the deterministic pipeline answer. For validating whether qwen independently supports `v5_20`, the raw first answer is the key signal.

## Action Distribution

`v5_20` target actions in the sample:

- `press_down_tilt`: 14
- `decrease_a3`: 6

Qwen raw first-answer actions:

- `decrease_a3`: 16
- `check_server`: 3
- `pdcch_2sym`: 1

Qwen final-answer actions:

- `decrease_a3`: 14
- `press_down_tilt`: 3
- `add_neighbor`: 1
- `pdcch_2sym`: 1
- `check_server`: 1

## Interpretation

The 6 `F1_5BS/decrease_a3` flips are strongly supported by qwen: 6/6 raw agreement and 6/6 final agreement.

The 54-row `F1/press_down_tilt` part of `v5_20` is not supported by this audit. In the 14-row stratified sample, qwen never independently chose `press_down_tilt` as its first answer. It mostly chose `decrease_a3`, and occasionally `check_server` or `pdcch_2sym`.

This weakens the claim that `v5_20` should move the score toward 0.172-0.182. That score range requires roughly 40-50 of the 60 flips to become correct; this audit supports the 6 `decrease_a3` flips but does not support bulk-applying the 54 `press_down_tilt` flips.

## Recommendation

Do not submit full `v5_20` as the primary safe candidate.

A safer derivative would be:

- Start from `v5_11`.
- Apply only the 6 `F1_5BS -> decrease_a3` flips from `v5_20`.
- Do not apply the 54 `F1 -> press_down_tilt` flips unless a stronger LLM/manual audit supports them.

Artifacts:

- Diagnostics: `results/phase2/f1_v5_20_audit/qwen_sample20_nothink.diagnostics.jsonl`
- Summary CSV: `results/phase2/f1_v5_20_audit/qwen_sample20_nothink_summary.csv`
- Output CSV from the audit run: `results/phase2/f1_v5_20_audit/qwen_sample20_nothink.csv`
