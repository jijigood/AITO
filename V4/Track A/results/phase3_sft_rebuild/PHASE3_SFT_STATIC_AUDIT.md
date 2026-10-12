# Phase 3 SFT Static Audit

Date: 2026-05-14

## Files

- train: `results/phase3_sft_rebuild/sft_diag_train.jsonl`
- val: `results/phase3_sft_rebuild/sft_diag_val.jsonl`
- plan: `results/phase3_sft_rebuild/sft_diag_plan.jsonl`
- dataset info: `results/phase3_sft_rebuild/dataset_info.json`
- builder report: `results/phase3_sft_rebuild/sft_diag_report.md`

## Shape

- train records: 3106
- val records: 200
- plan records: 544

## Phase 3 Changes Confirmed

- Prompt now includes compact all-options block with Track A C-codes.
- Prompt now includes deterministic candidates with option, action, score,
  target role/cell, and subtype reason.
- Prompt now includes a tool-evidence placeholder so offline training and
  runtime Step 2 share the same high-level prompt shape.
- Assistant JSON now includes `should_override`.
- Assistant JSON has 0 C-code leaks across train/val/plan.
- Candidate ordering is deployable deterministic ordering in train and val;
  label-aware promotion is disabled.

## should_override Distribution

Train:

- `should_override=false`: 2041
- `should_override=true`: 1065

Validation:

- `should_override=false`: 154
- `should_override=true`: 46

Plan:

- `should_override=false`: 508
- `should_override=true`: 36

## Boundary Sanity Checks

Validation insufficient_data:

- rows: 32
- deterministic top1 action: lift_tilt for 32/32
- `should_override=true`: 32/32
- top deterministic subtype: `F1 high-PRB healthy-radio -> lift_tilt/primary`

Validation lift_tilt:

- rows: 26
- deterministic top1 action: lift_tilt for 26/26
- `should_override=false`: 26/26
- top deterministic subtype: `F1 coverage hypothesis -> lift_tilt/tertiary`

This is the desired Phase 3 supervision geometry: the model sees the same
action family on both sides, but the subtype and should_override label separate
the boundary.

## Training Configs

Smoke:

`configs/lora_diag_phase3_smoke.yaml`

Full one-epoch run:

`configs/lora_diag_phase3_1epoch.yaml`

Both configs point to `results/phase3_sft_rebuild` so they do not accidentally
train on the older root-level SFT JSONL files.

## Next Gate

After smoke training, evaluate with:

`eval_diag_lora.py --val-jsonl results/phase3_sft_rebuild/sft_diag_val.jsonl --train data/Phase_1/train.json --gate-mode should_override`

Pass signal for smoke, not final integration:

- invalid JSON near zero
- insufficient_data action no longer collapses to lift_tilt
- lift_tilt does not collapse to insufficient_data
- should_override recall improves on deterministic-wrong boundary rows
- should_override precision does not create broad unsafe overrides
