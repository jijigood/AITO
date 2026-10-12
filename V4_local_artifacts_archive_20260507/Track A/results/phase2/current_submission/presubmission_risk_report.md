# Phase 2 Presubmission Risk Report

Generated after live-server pre-check, full run, low-confidence repair, and targeted risk probes.

## Current Track A Candidate

- Candidate CSV: `results/phase2/current_submission/result_full_uncertainfix.csv`
- Audit: `results/phase2/current_submission/result_full_uncertainfix.audit.json`
- Rows: 500 / 500
- Hard errors: 0
- Warnings: 0
- API errors: 0
- Low-confidence rows: 0
- Lowest top-candidate score: 0.30
- Track B: intentionally blank; this team is not participating in Track B.

## Later Experiments Not Promoted

Two later Qwen35-assisted full runs were also generated and audited:

- `archive/phase2_old_runs_20260507/result_fix67_qwen35_full.csv`
- `archive/phase2_old_runs_20260507/result_fix67b_qwen35_full.csv`

Both files are format-valid and cover all 500 Track A scenarios, but they are
not promoted as the current submission candidate in this report.  Their audit
minimum top-candidate score is 0.12, lower than the 0.30 floor for
`result_full_uncertainfix.csv`, and the LoRA/vLLM heldout report still shows
the current `v4_fast_balanced` checkpoint fails both high-risk F1 classes
(`lift_tilt` and `insufficient_data`).  Treat the Qwen35 files as experiments
unless an external leaderboard result or targeted validation explicitly favors
them.

## What Was Actually Validated

- Live API schema was exercised on all 500 Phase 2 scenarios.
- Core data sources were present for every scenario: user plane, net config, KPI/traffic, MR.
- Signaling was absent/empty in live probes and was deliberately skipped in the full run.
- The original 5 low-confidence F1 rows were re-probed with mainlobe + MR evidence and repaired.
- A second targeted probe was run on the remaining behavioral risk classes:
  - 28 `F1_5BS` rows
  - 7 `F2` rows with all-zero local feature scores

## Distribution Check

Phase 1 training action distribution:

- F1: balanced among `lift_tilt`, `insufficient_data`, `pdcch_2sym`, `add_neighbor`, `check_server`, `decrease_a3`
- F2: only two patterns, `adjust_azimuth + increase_power` and `decrease_power + press_down_tilt`
- F3: fixed 4-action pattern, `decrease_power + press_down_tilt + increase_a3 + increase_a3`
- F4: always `decrease_interfreq`

Phase 2 current prediction distribution:

- F2 preserves the same two-pattern structure as Phase 1.
- F3/F4/F4M follow the stable Phase 1 family priors.
- F1_5BS is a new Phase 2 family; all 28 predictions are `press_down_tilt`.

## Remaining Risk Classes

### 1. F1_5BS New Family

Risk:

- This family was absent from Phase 1 training.
- Current solver predicts `press_down_tilt` for all 28 rows.

Mitigation / Evidence:

- All 28 rows were re-probed with mainlobe, tilt, and overlap tools.
- Serving SINR is consistently mid/poor: 1.42 to 4.93 dB.
- Serving RSRP is strong: -89.18 to -81.18 dBm.
- PRB is low: 11.66 to 19.82.
- This is consistent with overlap/interference geometry rather than capacity/server.
- No higher-confidence alternative rule emerged from the probes.

Decision:

- Do not change before first submission unless new external evidence appears.

### 2. F2 All-Zero Local Feature Rows

Risk:

- 7 F2 rows had all-zero local issue scores before template composition.
- They use the `decrease_power + press_down_tilt` fallback pair.

Mitigation / Evidence:

- All 7 were re-probed with mainlobe, tilt, and overlap tools.
- Overlap is uniformly low: 0.02 to 0.29.
- F2 Phase 1 labels use only two action-pair families, and the current overall F2 distribution still matches that structure.
- Switching these 7 rows would be a distributional guess, not a tool-supported correction.

Decision:

- Do not patch before first submission.

### 3. F3 / F4 Template Priors

Risk:

- These families are heavily prior/template-based.

Mitigation / Evidence:

- Phase 1 training distribution is extremely regular:
  - F3 always uses the same 4-action family.
  - F4 always uses `decrease_interfreq`.
- Phase 2 F4M is a new cardinality variant, but still uses the same 13-BS inter-frequency structure.

Decision:

- Accept for first submission.

## Recommendation

This candidate is not merely a format-valid file. It has passed live-server behavior checks and targeted probes on the highest-risk branches. There is still unavoidable label-free uncertainty, especially in `F1_5BS`, but no additional deterministic patch currently has positive expected value.

Recommended Track A submission file:

`results/phase2/current_submission/result_full_uncertainfix.csv`

Keep the `Track B` column empty in this CSV; no Track B merge is needed.
