# A7 Ablation Report — 2026-04-26

This report records the current auditable Owner A evaluation evidence for the
200-scenario F1 heldout split.

Source CSV: `ablation_report_20260426.csv`

## Summary

| Run | Overall F1 IOU | add_neighbor | insufficient_data | lift_tilt | Avg latency | Log |
|---|---:|---:|---:|---:|---:|---|
| deterministic, no LoRA/tools | 0.5750 | 0.3421 | 0.0000 | 0.0000 | 0.0s | `logs/eval_a3_corrected_deterministic.log` |
| ckpt-477 fixed, LoRA-only | 0.6000 | 0.4737 | 0.0000 | 0.0000 | 19.4s | `logs/eval_ckpt477_fixed_t768.log` |
| corrected-246 fixed, LoRA-only | 0.6050 | 0.5000 | 0.0000 | 0.0000 | 40.8s | `logs/eval_corrected_fixed_t768.log` |
| deterministic + tools | 0.8500 | 0.3421 | 0.9062 | 1.0000 | 0.0s | `logs/eval_deterministic_tools_full_20260426_2021_rerun.log` |

## Readout

The LoRA-only track was completed and evaluated, but it did not meet the A3
promotion target of F1 IOU >= 0.70. The best LoRA-only checkpoint is
`checkpoints/v3_a3_corrected/checkpoint-246` at `0.6050`; the operational
latency reference remains `checkpoints/v3_zero3/checkpoint-477` at `0.6000`.

The tool path is the largest verified gain: `deterministic + tools` reproduces
`0.8500` overall on two independent runs. It closes the previous zero-recall
families, reaching `insufficient_data=0.9062` and `lift_tilt=1.0000`.

The remaining blocker is `add_neighbor`: it stays at `0.3421` in the current
tools baseline and fails the `>=0.65` per-action gate. This is not a LoRA
training-only problem based on the observed plateau/regression across LoRA
checkpoints.

## Promotion Status

`deterministic + tools` passes the overall IOU, latency, `lift_tilt`, and
`insufficient_data` gates, but it does not pass the `add_neighbor` gate. Treat it
as the current strongest auditable baseline, not as a fully green submission
configuration.
