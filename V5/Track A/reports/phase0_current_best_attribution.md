# Phase 0 Current-Best Attribution

Date: 2026-05-18

## Baseline Lock

Official scored baseline for the current discussion:

- File: `submission_600_diag_lora_boundary_resubmit.csv`
- Public score: `0.130199999`
- Track A effective score: about `0.2604` because Track B is blank and the platform averages tracks.
- Format check: READY
  - 600 rows
  - 500 Track A answers
  - 0 Track B answers

SHA256:

```text
d569e769aa2ef21d4711ac360fbe62e1ba600a3eb1bca485408a7fe5477d2aa5  submission_600_diag_lora_boundary_resubmit.csv
```

## Phase 2 Family Counts

| Family | Rows |
|---|---:|
| F1 | 241 |
| F2 | 145 |
| F4 | 29 |
| F3_5BS_MULTI | 29 |
| F1_5BS | 28 |
| F4M_13BS_MULTI | 28 |

## Official Baseline Structure

Official baseline action/cardinality observations:

- F1: 241 single-answer rows.
- F1_5BS: 28 single-answer rows.
- F2: 145 rows, but only 137 are 2-code answers; 8 are 4-code answers.
- F3_5BS_MULTI: 29 rows, all 4-code.
- F4: 29 rows, all `decrease_interfreq`.
- F4M_13BS_MULTI: 28 rows, all 2-code `decrease_interfreq|decrease_interfreq`.

F1/F1_5BS official baseline action distribution:

| Action | Count |
|---|---:|
| lift_tilt | 71 |
| adjust_azimuth | 60 |
| decrease_a3 | 54 |
| insufficient_data | 45 |
| pdcch_2sym | 26 |
| add_neighbor | 13 |
| check_server | 0 |

This confirms two major structural gaps:

- `adjust_azimuth` appears 60 times in F1/F1_5BS single-answer, even though Phase1 F1 ground truth has no such action.
- `check_server` is absent in F1/F1_5BS, even though Phase1 F1 has a large `check_server` class.

## Legality Audit

Allowed final F1/F1_5BS actions:

```text
check_server / decrease_a3 / lift_tilt / insufficient_data / pdcch_2sym / add_neighbor
```

F2 legal action pairs:

```text
{adjust_azimuth, increase_power}
{decrease_power, press_down_tilt}
```

| File | F1/F1_5BS illegal actions | F2 bad structures | F3 bad card | F4 bad | F4M bad |
|---|---:|---:|---:|---:|---:|
| official_026 | 60 | 10 | 0 | 0 | 0 |
| v5_11 | 60 | 0 | 0 | 0 | 0 |
| v5_21_safe | 54 | 0 | 0 | 0 | 0 |
| v5_22_qwen_raw54 | 1 | 0 | 0 | 0 | 0 |

Interpretation:

- Official baseline has 60 F1/F1_5BS `adjust_azimuth` illegal-action rows.
- Official baseline has 10 F2 bad structures.
- `v5_11` fixes only the 10 F2 bad structures.
- `v5_21_safe` fixes those 10 F2 rows plus 6 F1_5BS `adjust_azimuth -> decrease_a3` rows.
- `v5_22_qwen_raw54` fixes those 10 F2 rows plus all 60 F1/F1_5BS `adjust_azimuth` rows, but leaves one F1 `press_down_tilt` illegal-action row.

The remaining `v5_22` F1 illegal row:

```text
041efc72-b61d-4c14-b8af-6a75a2ec272d -> C23 / press_down_tilt
```

## Diff Summary Versus Official Baseline

| Candidate | Diff Rows | Family Diff |
|---|---:|---|
| v5_11 | 10 | F2: 10 |
| v5_21_safe | 16 | F2: 10, F1_5BS: 6 |
| v5_22_qwen_raw54 | 70 | F1: 54, F1_5BS: 6, F2: 10 |
| v5_20_guard_patch | 70 | F1: 54, F1_5BS: 6, F2: 10 |

Important: `v5_22` is not a pure F1 patch relative to the official baseline. It also includes the 10 F2 structural fixes from `v5_11`.

## v5_22 Diff Details

The 60 F1/F1_5BS single-answer illegal actions are changed as follows:

| Old Action | New Action | Count |
|---|---|---:|
| adjust_azimuth | decrease_a3 | 50 |
| adjust_azimuth | check_server | 6 |
| adjust_azimuth | pdcch_2sym | 2 |
| adjust_azimuth | lift_tilt | 1 |
| adjust_azimuth | press_down_tilt | 1 |

The 10 non-F1 drift rows in `v5_22` are all F2 structural fixes:

| Family | Rows | Description |
|---|---:|---|
| F2 | 10 | Converts bad 4-code or illegal action-pair answers to legal 2-code F2 pairs |

Therefore, future V6 work should use explicit base switches:

- `base_A`: official baseline only.
- `base_B`: official baseline + 10 F2 structural fixes.
- `base_C`: `base_B` + validated F1/F1_5BS changes.

Do not silently inherit the 10 F2 changes when testing F1 changes.

## Override Check

Both audited LoRA overrides are intact in all checked files:

| File | 6060dca7 | 1784b64c |
|---|---|---|
| official_026 | C22 OK | C20 OK |
| v5_11 | C22 OK | C20 OK |
| v5_21_safe | C22 OK | C20 OK |
| v5_22_qwen_raw54 | C22 OK | C20 OK |

## Phase 0 Decision

Locked baseline:

```text
submission_600_diag_lora_boundary_resubmit.csv
```

Recommended next bases:

```text
base_A = official_026
base_B = official_026 + 10 F2 structural fixes
```

Next step:

- Build Phase 0.5 validation harness on Phase1.
- Evaluate baseline-equivalent, F1 action guard, and F1 action-first/check_server increments before composing any V6 submission.

Immediate guard implication:

- `v5_22` has one remaining F1 illegal action (`press_down_tilt`), so any V6 candidate must set F1/F1_5BS illegal action count to 0.
