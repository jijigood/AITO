# Phase 2 Closeout: Diagnostic LoRA Boundary Work

Date: 2026-05-14

## Final Phase 2 Submission Posture

Use:

`results/phase2/sub2_lora_boundary/submission_600_diag_lora_boundary_only.csv`

Do not use the runtime-fix outputs for Phase 2 submission.

## Why This Is The Recommended File

The recommended file is based on `sub2c` and applies only two strict,
independently-audited LoRA overrides:

- `6060dca7-32c8-4cd8-adef-425aa1aec774`: C23 -> C22
- `1784b64c-3a27-4056-9aec-aaac8cdd284e`: C21 -> C20

Both are on the `lift_tilt_vs_insufficient_data` branch, both map to
`insufficient_data`, and subtype leaks were 0.

Reference audit:

`results/phase2/sub2_lora_boundary/diff_diag_lora_boundary_report.md`

## Runtime-Fix Experiment Result

Runtime bugs were confirmed and fixed in `skills/agentic_overlay.py`:

- Step 2 no longer vetoes insufficient_data for MR count 3-8.
- Step 2 now receives deterministic candidates with subtype reason strings.

However, the existing checkpoint still has a strong `lift_tilt` prior on
Phase 2 generic-primary boundary cases. A targeted 10-row test from the 30
generic-primary high-PRB lift_tilt cases produced:

- completed calls: 10/10
- diagnosis actions: lift_tilt=9, insufficient_data=1
- final answer changes: 1/10

This is below the threshold needed to justify using runtime-fix output as a
submission candidate. The runtime fix should stay for Phase 3, but it does not
rescue the current checkpoint.

Reference audit:

`results/phase2/sub2_lora_boundary_runtimefix_t120/runtimefix_targeted_report.md`

## Do Not Accidentally Submit

Do not submit:

- `results/phase2/sub2_lora_boundary_runtimefix/result_diag_lora_boundary_runtimefix.csv`
- `results/phase2/sub2_lora_boundary_runtimefix_t120/result_diag_lora_boundary_runtimefix_t120.csv`
- `results/phase2/sub2_lora_boundary_runtimefix_t120/generic_primary_30_run/result_generic_primary_30_runtimefix.csv`

These are diagnostic experiments, not submission artifacts.
