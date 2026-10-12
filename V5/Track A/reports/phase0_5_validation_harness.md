# Phase 0.5 Phase1 Validation Harness Report

## Holdout

- Seed: `20260518`
- Requested sample size: `300`
- Actual sample size: `300`

| Family | Phase1 Available | Phase2 Weight | Target | Selected | Low Confidence |
|---|---:|---:|---:|---:|---|
| F1 | 1456 | 0.482 | 145 | 154 | False |
| F1_5BS | 0 | 0.056 | 17 | 0 | True |
| F2 | 228 | 0.290 | 87 | 96 | False |
| F3_5BS_MULTI | 71 | 0.058 | 17 | 25 | False |
| F4 | 245 | 0.058 | 17 | 25 | False |
| F4M_13BS_MULTI | 0 | 0.056 | 17 | 0 | True |

## Mechanism Summary

| Mechanism | N | Mean IOU | Exact | F1 Illegal | F2 Bad | F3 Bad Card | F4 Bad | F4M Bad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deterministic_current | 300 | 0.8567 | 0.8567 | 0 | 0 | 0 | 0 | 0 |
| strict_f1_allowed_guard | 300 | 0.8567 | 0.8567 | 0 | 0 | 0 | 0 | 0 |

## deterministic_current Per-Family

| Family | N | Mean IOU | Exact | F1 Illegal | F2 Bad |
|---|---:|---:|---:|---:|---:|
| F1 | 154 | 0.7922 | 0.7922 | 0 | 0 |
| F2 | 96 | 0.8854 | 0.8854 | 0 | 0 |
| F3_5BS_MULTI | 25 | 1.0000 | 1.0000 | 0 | 0 |
| F4 | 25 | 1.0000 | 1.0000 | 0 | 0 |

## deterministic_current Single-Action Precision/Recall

| Action | GT | Pred | TP | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| add_neighbor | 25 | 20 | 20 | 1.0000 | 0.8000 |
| check_server | 22 | 22 | 22 | 1.0000 | 1.0000 |
| decrease_a3 | 20 | 25 | 20 | 0.8000 | 1.0000 |
| decrease_interfreq | 25 | 25 | 25 | 1.0000 | 1.0000 |
| insufficient_data | 27 | 0 | 0 | 0.0000 | 0.0000 |
| lift_tilt | 28 | 55 | 28 | 0.5091 | 1.0000 |
| pdcch_2sym | 32 | 32 | 32 | 1.0000 | 1.0000 |

## strict_f1_allowed_guard Per-Family

| Family | N | Mean IOU | Exact | F1 Illegal | F2 Bad |
|---|---:|---:|---:|---:|---:|
| F1 | 154 | 0.7922 | 0.7922 | 0 | 0 |
| F2 | 96 | 0.8854 | 0.8854 | 0 | 0 |
| F3_5BS_MULTI | 25 | 1.0000 | 1.0000 | 0 | 0 |
| F4 | 25 | 1.0000 | 1.0000 | 0 | 0 |

## strict_f1_allowed_guard Single-Action Precision/Recall

| Action | GT | Pred | TP | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| add_neighbor | 25 | 20 | 20 | 1.0000 | 0.8000 |
| check_server | 22 | 22 | 22 | 1.0000 | 1.0000 |
| decrease_a3 | 20 | 25 | 20 | 0.8000 | 1.0000 |
| decrease_interfreq | 25 | 25 | 25 | 1.0000 | 1.0000 |
| insufficient_data | 27 | 0 | 0 | 0.0000 | 0.0000 |
| lift_tilt | 28 | 55 | 28 | 0.5091 | 1.0000 |
| pdcch_2sym | 32 | 32 | 32 | 1.0000 | 1.0000 |
