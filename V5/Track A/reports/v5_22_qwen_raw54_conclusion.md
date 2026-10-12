# v5_22 Qwen Raw-54 Conclusion

Date: 2026-05-18

## Build

`v5_22` is built from `v5_21_safe`:

- Base: `submission_600_v5_21_safe_f1_5bs_decrease_a3.csv`
- Keep the 6 validated `F1_5BS -> decrease_a3` flips.
- Replace the 54 F1 `adjust_azimuth` rows with qwen raw first answers.
- Do not use the confirmation/final LLM answer for these 54 rows; the raw first answer is the independent diagnosis signal.

Output:

- `submissions/submission_600_v5_22_qwen_raw54_plus_f1_5bs.csv`

## Validation

Format smoke:

- 600 rows
- 500 Track A answers
- 0 Track B answers
- Format valid
- Both audited LoRA overrides preserved:
  - `6060dca7... -> C22`
  - `1784b64c... -> C20`

Diffs:

- vs `v5_11`: 60 changes
- vs `v5_21_safe`: 54 changes
- vs `v5_20`: 53 changes

## Qwen Raw-54 Distribution

For the 54 F1 rows:

- `decrease_a3`: 44
- `check_server`: 6
- `pdcch_2sym`: 2
- `lift_tilt`: 1
- `press_down_tilt`: 1
- `adjust_azimuth`: 0

Raw first-answer agreement:

- With `v5_20` press-down patch: 1/54
- With `v5_11` adjust-azimuth baseline: 0/54

This confirms the earlier sample-20 result: qwen strongly rejects bulk `press_down_tilt`, but also never returns to `adjust_azimuth`.

## Interpretation

`v5_21_safe` is the conservative fallback. It only captures the 6 F1_5BS cases that were independently validated by qwen.

`v5_22` is the higher-EV candidate. It avoids the near-certain bad `adjust_azimuth` prior and uses qwen's per-row raw diagnosis for the 54 F1 cases. It is still risky because Phase 2 labels are hidden and the local diagnostics classified these rows as `overshoot_overlap`; however, qwen's raw evidence reading consistently points elsewhere, mostly toward `decrease_a3`.

## Recommendation

Submission priority:

1. `v5_22_qwen_raw54_plus_f1_5bs` as the main score-improvement candidate.
2. `v5_21_safe_f1_5bs_decrease_a3` as fallback if the team wants the lowest-risk variant.
3. Do not submit full `v5_20`; its 54-row `press_down_tilt` patch is not supported by qwen raw diagnostics.

Artifacts:

- Submission: `submissions/submission_600_v5_22_qwen_raw54_plus_f1_5bs.csv`
- Summary CSV: `results/phase2/f1_v5_20_audit/qwen_raw54_plus_6_summary.csv`
- Remaining-40 diagnostics: `results/phase2/f1_v5_20_audit/qwen_remaining40_nothink.diagnostics.jsonl`
- Sample-20 diagnostics: `results/phase2/f1_v5_20_audit/qwen_sample20_nothink.diagnostics.jsonl`
