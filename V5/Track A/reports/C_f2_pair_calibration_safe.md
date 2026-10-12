# C F2 Pair Calibration Safe Report

## Phase1 Distribution

| top_issue | count | azimuth | tilt | majority | majority_rate |
|---|---:|---:|---:|---|---:|
| overshoot_overlap | 178 | 142 | 36 | azimuth | 0.798 |
| late_handover | 25 | 0 | 25 | tilt | 1.000 |
| uncertain | 25 | 0 | 25 | tilt | 1.000 |

## Phase1 Crossval Smoke

- rows: 228
- old_accuracy: 0.8421
- candidate_flips: 0
- flip_new_precision: 0.0000
- flip_improves: 0
- flip_hurts: 0
- net_gain_rows: 0

## Phase2 Candidates

- diagnostic_f2_rows: 145
- candidates: 3
- applied: 3
- skipped: {'target_score_drop_too_large': 54, 'already_majority_pair': 68, 'no_phase1_bucket': 13, 'top_score_too_low': 7}

| scenario_id | top_issue | current_pair | target_pair | current_score | target_score | majority_rate | priority |
|---|---|---|---|---:|---:|---:|---:|
| a4176c2c-458b-4c9c-b89b-c0c873747d0b | late_handover | azimuth | tilt | 0.6856 | 0.6111 | 1.000 | 0.4554 |
| f02d340d-18c0-43c9-ae9c-c6132c5fe719 | late_handover | azimuth | tilt | 0.6620 | 0.5901 | 1.000 | 0.4246 |
| 32ad6a89-3f8c-4e03-9936-5d1be80459b4 | overshoot_overlap | tilt | azimuth | 0.4510 | 0.4510 | 0.798 | 0.1979 |

## F2 Legality Audit

- bad_count: 0
- pair_counts: {'tilt': 70, 'azimuth': 75}
