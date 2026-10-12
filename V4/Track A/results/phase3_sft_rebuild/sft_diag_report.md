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
- train should_override distribution: `{'False': 2041, 'True': 1065}`
- val should_override distribution: `{'False': 154, 'True': 46}`
- train action-level override distribution: `{'False': 2041, 'True': 1065}`

## Safety Checks
- Assistant outputs diagnostic JSON only; no boxed answers.
- User prompts include compact Track A option codes for prompt parity.
- Assistant outputs are guarded against Track A C-code leakage.
- Heldout F1 IDs are excluded from the training split.
- Non-F1 rows are written as diagnosis_plan data for later optional training, not mixed into first-round F1 SFT.
- Candidate ordering is deployable deterministic ordering in train and validation; no label-aware promotion.

## Boundary Notes
- lift_tilt/insufficient_data action selection is based on lift_tilt candidate subtype, not MR count alone.
- F1 coverage hypothesis with isolated tertiary target supports lift_tilt.
- Generic high-PRB healthy-radio/primary fallback supports insufficient_data.
- MR count affects confidence calibration only.
- should_override is the intended Phase 3 override signal; confidence is diagnostic metadata.
- Use action-level and should_override smoke tests first; only add submission overrides after the gate passes.

## Recommended Next Gate
- Train a 100-200 sample smoke adapter before the full rank-32 run.
- Evaluate all heldout lift_tilt and insufficient_data rows for action accuracy and confidence calibration.
- Track should_override precision/recall on deterministic-wrong boundary rows.
- Block integration if boundary action accuracy regresses or should_override precision is unsafe.
