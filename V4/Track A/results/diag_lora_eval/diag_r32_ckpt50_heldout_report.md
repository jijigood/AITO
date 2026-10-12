# Diagnostic LoRA r32 checkpoint-50 Heldout Gate

Run time: 2026-05-14 11:29-13:46 CST

## Model

- Train config: `configs/lora_diag_1epoch_2gpu.yaml`
- DeepSpeed config: `configs/ds_z2_b1_gas8.json`
- Checkpoint: `checkpoints/diag_lora_r32_1epoch/checkpoint-50`
- Training stopped after checkpoint-50 for evaluation.
- Teacher-forced eval loss at step 50: 0.002443
- Eval output: `results/diag_lora_eval/diag_r32_ckpt50_heldout200.jsonl`

## Full Heldout Result

| Metric | Result |
| --- | ---: |
| Evaluated | 200 |
| JSON valid | 194/200 = 0.970 |
| C-code leaks | 0/200 |
| API errors | 0/200 |
| Deterministic IOU | 0.7700 |
| Mapped LoRA IOU | 0.8500 |
| High-only gated IOU | 0.6900 |

Full/global gate does not pass:

- JSON valid is below the 0.98 gate.
- High-only global override is worse than deterministic because `add_neighbor`/`decrease_a3` is unsafe.
- `decrease_a3` action accuracy is high, but mapped C-code is poor, indicating target/relation mapping mismatch.

## Per Action

| GT action | n | deterministic IOU | mapped LoRA IOU | action accuracy |
| --- | ---: | ---: | ---: | ---: |
| add_neighbor | 38 | 0.6842 | 0.6842 | 0.6053 |
| check_server | 38 | 1.0000 | 1.0000 | 1.0000 |
| decrease_a3 | 35 | 0.9429 | 0.4857 | 0.9429 |
| insufficient_data | 32 | 0.0000 | 1.0000 | 1.0000 |
| lift_tilt | 26 | 1.0000 | 1.0000 | 0.9231 |
| pdcch_2sym | 31 | 1.0000 | 1.0000 | 1.0000 |

## Policy Simulation

| Policy | Accepted | Overall IOU |
| --- | ---: | ---: |
| deterministic only | 0/200 | 0.7700 |
| high confidence globally | 138/200 | 0.6900 |
| medium/high only for `insufficient_data` or `lift_tilt` | 56/200 | 0.9300 |

## Decision

Do not use this LoRA as a global reranker and do not enable it for `add_neighbor` vs `decrease_a3`.

It is suitable only for the `lift_tilt_vs_insufficient_data` branch, where the subtype arbitration rule is stable. The production overlay gate has been narrowed accordingly in `skills/agentic_overlay.py`.

The invalid JSON rows are mostly max-token truncations under the eval default of 256 tokens; `eval_diag_lora.py` now defaults to 384 tokens to match Phase 2 runtime. This does not change the main decision because the unsafe behavior is concentrated in `add_neighbor`/`decrease_a3`, not JSON leakage.
