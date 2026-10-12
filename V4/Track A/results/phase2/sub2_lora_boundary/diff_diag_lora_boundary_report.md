# diag_lora_boundary strict submission audit

## Files
- recommended submission: `results/phase2/sub2_lora_boundary/submission_600_diag_lora_boundary_only.csv`
- full raw-run overlay backup: `results/phase2/sub2_lora_boundary/submission_600_diag_lora_boundary_fullrun_overlay.csv`
- phase2 raw: `results/phase2/sub2_lora_boundary/result_diag_lora_boundary_only.csv`
- diagnostics: `results/phase2/sub2_lora_boundary/result_diag_lora_boundary_only.diagnostics.jsonl`
- detailed diff: `results/phase2/sub2_lora_boundary/diff_diag_lora_boundary_vs_baselines.csv`

## Shape
- recommended submission rows: 600
- phase2 raw rows: 500
- diagnostics rows: 500
- base template: sub2c
- strict policy: apply only valid LoRA overrides, keep all other sub2c rows

## Agentic override audit
- active overrides in raw run: 2
- valid overrides applied to recommended submission: 2
- suppressed overrides: 9
- subtype leaks: 0
- overrides by branch: {'lift_tilt_vs_insufficient_data': 2}
- overrides by action: {'insufficient_data': 2}
  - `6060dca7-32c8-4cd8-adef-425aa1aec774`: C23 -> C22 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis
  - `1784b64c-3a27-4056-9aec-aaac8cdd284e`: C21 -> C20 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis

## Pairwise Track A diffs vs recommended submission
- uncertainfix: 337/600 differ (337/500 on Phase2 IDs)
- sub2: 25/600 differ (25/500 on Phase2 IDs)
- sub2c: 2/600 differ (2/500 on Phase2 IDs)
- postB: 154/600 differ (154/500 on Phase2 IDs)
- postCD_f1guard: 154/600 differ (154/500 on Phase2 IDs)

## Rows where any baseline differs
- total: 368/600
