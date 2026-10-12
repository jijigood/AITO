# C — F2 Pair Calibration Report

Generated: 2026-05-16

## Parameters

- Confidence threshold: 0.7
- Max flips: 25
- Base CSV: D:\Eng_tools\AI_coding\race_telco\V5\Track A\submissions\submission_600_v5_10_adj_only_plus_f2_pair_fix.csv

## Phase 1 Analysis

Total F2 rows: 228
Valid F2 rows with legal pairs: 228

### top_issue -> pair distribution

| top_issue | total | azimuth | tilt | azimuth % | tilt % | majority | confidence |
|---|---|---|---|---|---|---|---|
| overshoot_overlap | 178 | 142 | 36 | 79.8% | 20.2% | azimuth | 0.60 |
| uncertain | 25 | 0 | 25 | 0.0% | 100.0% | tilt | 1.00 |
| late_handover | 25 | 0 | 25 | 0.0% | 100.0% | tilt | 1.00 |

## Phase 2 Evaluation

Total F2 rows: 145
Cardinality-2 F2 rows: 145
High-confidence flip candidates: 76

### Recommended Flips (sorted by flip_score)

| scenario_id | top_issue | current_pair | target_pair | confidence | flip_score |
|---|---|---|---|---|---|
| f00df504-5e00-49c3-b040-29c62021793d | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| 1f872048-7692-4443-bbda-324975dbef07 | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| f86420da-3884-4a89-985e-13b870108ac8 | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| f428f517-0bcd-4d13-b31a-929eb3584363 | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| a5aa02e2-3000-4e1b-b1e1-56b629e9c2d4 | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| 863e801a-af00-411d-95c8-9795bf7859fd | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| 815f0842-32f3-440f-837d-42abdd09f248 | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| 71f8e7bd-15b6-490d-adf4-b4746615987a | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| e72d1cd8-f9ec-495b-81fa-8469bbc665ac | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| de896155-b409-4446-8679-e1bb486e4882 | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| 732ac9cf-5ad2-4934-a5f6-f41d6edde04e | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| a23ab670-fd49-4272-b64c-fcbf270dae96 | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| f765cbbc-ad37-492b-bfb2-b4844b28e1d7 | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| ff500b22-2cdd-48b8-9029-a19556a0284b | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| d1db7c9d-0831-43cc-8458-2c592f8ddd11 | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| ddb040fe-708c-472d-8ef6-584b8ec5b7ec | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| 57ff95b0-924b-44f1-b5a8-4603b422af24 | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| f3d7cd99-2ea9-46f9-91f6-79ea7bd303fd | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| d06a257c-34d3-4641-a17b-6ec6284a1f20 | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| f02d340d-18c0-43c9-ae9c-c6132c5fe719 | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| 5137d812-1c90-4200-98c7-b1d4f44bb3d5 | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| e83a09dd-795a-4066-9f5c-2de6da46dfea | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| a4176c2c-458b-4c9c-b89b-c0c873747d0b | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| 337083f8-8abe-425c-b1bb-9ff292a5517b | uncertain | azimuth | tilt | 1.00 | 0.0000 |
| f88ec12a-82ad-4f49-be0a-c278cc038ae6 | uncertain | azimuth | tilt | 1.00 | 0.0000 |

## Patch Result

Flips applied: 25
Skipped: 0

### Applied Flips

| scenario_id | top_issue | old | old_pair | new | new_pair | confidence |
|---|---|---|---|---|---|---|
| f00df504-5e00-49c3-b040-29c62021793d | uncertain | C12|C22 | azimuth | C1|C6 | tilt | 1.00 |
| 1f872048-7692-4443-bbda-324975dbef07 | uncertain | C1|C2 | azimuth | C3|C9 | tilt | 1.00 |
| f86420da-3884-4a89-985e-13b870108ac8 | uncertain | C3|C8 | azimuth | C1|C7 | tilt | 1.00 |
| f428f517-0bcd-4d13-b31a-929eb3584363 | uncertain | C2|C18 | azimuth | C6|C9 | tilt | 1.00 |
| a5aa02e2-3000-4e1b-b1e1-56b629e9c2d4 | uncertain | C2|C11 | azimuth | C4|C9 | tilt | 1.00 |
| 863e801a-af00-411d-95c8-9795bf7859fd | uncertain | C3|C23 | azimuth | C14|C16 | tilt | 1.00 |
| 815f0842-32f3-440f-837d-42abdd09f248 | uncertain | C2|C20 | azimuth | C3|C9 | tilt | 1.00 |
| 71f8e7bd-15b6-490d-adf4-b4746615987a | uncertain | C1|C24 | azimuth | C5|C8 | tilt | 1.00 |
| e72d1cd8-f9ec-495b-81fa-8469bbc665ac | uncertain | C2|C5 | azimuth | C1|C4 | tilt | 1.00 |
| de896155-b409-4446-8679-e1bb486e4882 | uncertain | C2|C7 | azimuth | C4|C13 | tilt | 1.00 |
| 732ac9cf-5ad2-4934-a5f6-f41d6edde04e | uncertain | C8|C9 | azimuth | C6|C12 | tilt | 1.00 |
| a23ab670-fd49-4272-b64c-fcbf270dae96 | uncertain | C1|C7 | azimuth | C6|C15 | tilt | 1.00 |
| f765cbbc-ad37-492b-bfb2-b4844b28e1d7 | uncertain | C6|C9 | azimuth | C2|C3 | tilt | 1.00 |
| ff500b22-2cdd-48b8-9029-a19556a0284b | uncertain | C3|C13 | azimuth | C1|C18 | tilt | 1.00 |
| d1db7c9d-0831-43cc-8458-2c592f8ddd11 | uncertain | C2|C10 | azimuth | C6|C16 | tilt | 1.00 |
| ddb040fe-708c-472d-8ef6-584b8ec5b7ec | uncertain | C4|C10 | azimuth | C2|C9 | tilt | 1.00 |
| 57ff95b0-924b-44f1-b5a8-4603b422af24 | uncertain | C3|C21 | azimuth | C4|C7 | tilt | 1.00 |
| f3d7cd99-2ea9-46f9-91f6-79ea7bd303fd | uncertain | C1|C7 | azimuth | C2|C11 | tilt | 1.00 |
| d06a257c-34d3-4641-a17b-6ec6284a1f20 | uncertain | C7|C10 | azimuth | C2|C8 | tilt | 1.00 |
| f02d340d-18c0-43c9-ae9c-c6132c5fe719 | uncertain | C7|C9 | azimuth | C10|C15 | tilt | 1.00 |
| 5137d812-1c90-4200-98c7-b1d4f44bb3d5 | uncertain | C5|C23 | azimuth | C9|C12 | tilt | 1.00 |
| e83a09dd-795a-4066-9f5c-2de6da46dfea | uncertain | C6|C16 | azimuth | C3|C11 | tilt | 1.00 |
| a4176c2c-458b-4c9c-b89b-c0c873747d0b | uncertain | C2|C10 | azimuth | C12|C14 | tilt | 1.00 |
| 337083f8-8abe-425c-b1bb-9ff292a5517b | uncertain | C2|C8 | azimuth | C14|C15 | tilt | 1.00 |
| f88ec12a-82ad-4f49-be0a-c278cc038ae6 | uncertain | C3|C4 | azimuth | C1|C6 | tilt | 1.00 |
