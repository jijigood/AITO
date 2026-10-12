# Diagnostic LoRA Boundary Subtype Smoke

Run time: 2026-05-14 09:30-10:35 CST

## Training

- Config: `configs/lora_diag_boundary_subtype_smoke.yaml`
- Output: `checkpoints/diag_lora_boundary_subtype_smoke_r16_qlora/checkpoint-25`
- GPU: single A100 on `CUDA_VISIBLE_DEVICES=0`
- Steps: 25 optimizer steps on 200 smoke samples
- Train loss: 0.5058
- Teacher-forced eval loss: 0.3433
- JSON/C-code target format: diagnostic JSON only

## Boundary Inference Eval

- Eval file: `results/diag_lora_eval/diag_boundary_subtype_smoke_lift_insuff.jsonl`
- Scope: heldout `insufficient_data` + `lift_tilt` boundary rows only
- Rows: 58
- JSON valid: 58/58
- C-code leakage: 0/58
- API errors: 0/58
- Action accuracy: 58/58
- Mapped LoRA IOU: 1.0000
- Deterministic IOU on same rows: 0.4483

## Per Action

| GT action | n | deterministic IOU | mapped LoRA IOU | action accuracy |
| --- | ---: | ---: | ---: | ---: |
| insufficient_data | 32 | 0.0000 | 1.0000 | 1.0000 |
| lift_tilt | 26 | 1.0000 | 1.0000 | 1.0000 |

## Confidence

| GT action | high | medium | low |
| --- | ---: | ---: | ---: |
| insufficient_data | 0 | 32 | 0 |
| lift_tilt | 0 | 15 | 11 |

Default high-only gate accepts 0/58, so it preserves deterministic behavior and does not fix insufficient_data. A medium+high boundary gate accepts 47/58 and reaches 1.0000 IOU on this smoke set because all insufficient_data rows are medium and all low-confidence lift_tilt rows are already correct deterministically.

## Decision

The subtype arbitration signal is learnable and fixed the previous overcorrection. This smoke passes the direction check:

- insufficient_data: 32/32, threshold was >= 15/32
- lift_tilt: 26/26, threshold was >= 20/26
- JSON valid: 58/58

Do not treat this as full heldout gate approval yet. The next safe step is full-rank or full-sample training only after preserving the subtype prompt/data rules and evaluating all 200 heldout rows with the production branch-scoped medium override.
