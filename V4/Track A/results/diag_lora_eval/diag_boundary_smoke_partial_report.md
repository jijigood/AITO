# Diagnostic Boundary Smoke Partial Report

- checkpoint: `checkpoints/diag_lora_boundary_smoke_r16_qlora/checkpoint-25`
- train_runtime: `0:25:51.18`
- train_loss: `0.6499`
- eval_loss: `0.381953`
- eval command was stopped early because the lift_tilt pass criterion became mathematically unreachable.

## Partial Inference Result

- evaluated rows: `20/58`
- JSON valid: `20/20`
- insufficient_data action accuracy so far: `11/11`
- lift_tilt action accuracy so far: `0/9`
- pass criterion for lift_tilt was `>=20/26`; after 9 lift_tilt misses, the maximum possible score was `17/26`.

## Decision

Do not proceed to `lora_diag_1epoch.yaml` from this data/prompt version. The smoke adapter over-corrects the boundary and predicts `insufficient_data` for lift_tilt cases.
