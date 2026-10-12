# Diagnostic SFT Report

## Inputs
- train examples: 2000
- heldout IDs loaded: 200
- family distribution: `{'F1': 1456, 'F2': 228, 'F3': 71, 'F4': 245}`
- raw F1 action distribution: `{'add_neighbor': 240, 'check_server': 239, 'decrease_a3': 231, 'insufficient_data': 252, 'lift_tilt': 254, 'pdcch_2sym': 240}`

## Outputs
- train records: 3106
- val records: 200
- non-F1 diagnosis_plan records: 544
- train action distribution: `{'add_neighbor': 606, 'check_server': 201, 'decrease_a3': 196, 'insufficient_data': 880, 'lift_tilt': 912, 'pdcch_2sym': 311}`
- val action distribution: `{'add_neighbor': 38, 'check_server': 38, 'decrease_a3': 35, 'insufficient_data': 32, 'lift_tilt': 26, 'pdcch_2sym': 31}`
- train augmentation distribution: `{'hard_negative:add_neighbor_vs_decrease_a3': 202, 'hard_negative:insufficient_data_vs_lift_tilt': 220, 'hard_negative:lift_tilt_vs_insufficient_data': 228, 'none': 1256, 'upsample': 1200}`
- train confidence distribution: `{'high': 1129, 'low': 3, 'medium': 1974}`
- train boundary-pair records: `{'False': 1289, 'True': 1817}`
- val boundary-pair records: `{'False': 140, 'True': 60}`

## Safety Checks
- Assistant outputs diagnostic JSON only; no boxed answers.
- User prompts are guarded against Track A C-code leakage.
- Heldout F1 IDs are excluded from the training split.
- Non-F1 rows are written as diagnosis_plan data for later optional training, not mixed into first-round F1 SFT.
- Heldout candidate ordering is not label-promoted; this avoids leaking the answer through validation prompts.

## Boundary Notes
- lift_tilt/insufficient_data action selection is based on lift_tilt candidate subtype, not MR count alone.
- F1 coverage hypothesis with isolated tertiary target supports lift_tilt.
- Generic high-PRB healthy-radio/primary fallback supports insufficient_data.
- MR count affects confidence calibration only.
- A high-only Phase 2 override gate will not accept insufficient_data medium-confidence outputs.
- Use action-level smoke tests first; only add submission overrides after the gate policy is updated deliberately.

## Recommended Next Gate
- Train a 100-200 sample smoke adapter before the full rank-32 run.
- Evaluate all heldout lift_tilt and insufficient_data rows for action accuracy and confidence calibration.
- Block Phase 2 connection if boundary action accuracy regresses or medium-confidence policy is not handled safely.
