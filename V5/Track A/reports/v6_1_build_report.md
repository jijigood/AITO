# V6.1 Build Report

## Inputs

- Base: `submission_600_diag_lora_boundary_resubmit.csv`
- Raw-54 source: `results/phase2/f1_v5_20_audit/qwen_raw54_plus_6_summary.csv`
- F1_5BS six-fix source: `submissions/submission_600_v5_21_safe_f1_5bs_decrease_a3.csv`
- F2 structural-fix source: `submissions/submission_600_v5_11_baseline_plus_f2_pair_fix.csv`

## Fix Maps

- Raw-54 F1 fixes: `54`
- Strict Raw-54 F1 fixes: `54`
- F1_5BS validated fixes: `6`
- F2 structural fixes: `10`

## Strict F1 Fallbacks

- 041efc72-b61d-4c14-b8af-6a75a2ec272d: raw C23/press_down_tilt -> fallback C5/decrease_a3

## Output Audits
| Variant                  | Rows | Track A | Track B | Diff vs Base | F1 Illegal | F2 Bad | LoRA OK |
| ------------------------ | ---- | ------- | ------- | ------------ | ---------- | ------ | ------- |
| v6_1_raw54_base          | 600  | 500     | 0       | 54           | 7          | 10     | True    |
| v6_1_strict60_base       | 600  | 500     | 0       | 60           | 0          | 10     | True    |
| v6_1_strict60_plus_f2fix | 600  | 500     | 0       | 70           | 0          | 0      | True    |

## v6_1_raw54_base

Family counts:
- `F1`: 241
- `F1_5BS`: 28
- `F2`: 145
- `F3_5BS_MULTI`: 29
- `F4`: 29
- `F4M_13BS_MULTI`: 28

F1/F1_5BS action distribution:
- `F1:add_neighbor`: 8
- `F1:check_server`: 6
- `F1:decrease_a3`: 94
- `F1:insufficient_data`: 32
- `F1:lift_tilt`: 72
- `F1:pdcch_2sym`: 28
- `F1:press_down_tilt`: 1
- `F1_5BS:add_neighbor`: 5
- `F1_5BS:adjust_azimuth`: 6
- `F1_5BS:decrease_a3`: 4
- `F1_5BS:insufficient_data`: 13

F1 illegal rows:
- `61b818e5-0414-471d-a5c5-94f0c1e2e242` `F1_5BS` `C2` `adjust_azimuth`
- `ad7ec8da-deaf-4cca-8cea-9e5fc290a17d` `F1_5BS` `C18` `adjust_azimuth`
- `2d688d4d-e9fe-4aab-a3c7-aec1b4b5a947` `F1_5BS` `C11` `adjust_azimuth`
- `9fc4ad17-21b3-4474-adb7-a939a04bc843` `F1_5BS` `C7` `adjust_azimuth`
- `adda43e7-3f09-44ab-8665-1f1c24be70b6` `F1_5BS` `C12` `adjust_azimuth`
- `c4fb29f5-147e-40c6-92b3-ae66de737fd3` `F1_5BS` `C8` `adjust_azimuth`
- `041efc72-b61d-4c14-b8af-6a75a2ec272d` `F1` `C23` `press_down_tilt`

F2 bad rows:
- `f428f517-0bcd-4d13-b31a-929eb3584363` `C1|C6|C9|C20` `['add_neighbor', 'decrease_power', 'press_down_tilt', 'decrease_a3']`
- `732ac9cf-5ad2-4934-a5f6-f41d6edde04e` `C6|C12|C15|C19` `['decrease_power', 'press_down_tilt', 'decrease_a3', 'add_neighbor']`
- `13482c57-c7d2-40b3-8020-c45ff3a396f4` `C1|C4|C6|C22` `['decrease_power', 'decrease_a3', 'add_neighbor', 'press_down_tilt']`
- `ddb040fe-708c-472d-8ef6-584b8ec5b7ec` `C1|C2|C12|C20` `['decrease_a3', 'decrease_power', 'add_neighbor', 'press_down_tilt']`
- `f02d340d-18c0-43c9-ae9c-c6132c5fe719` `C10|C15|C20|C22` `['press_down_tilt', 'decrease_power', 'add_neighbor', 'decrease_a3']`
- `e83a09dd-795a-4066-9f5c-2de6da46dfea` `C13|C14|C15|C23` `['decrease_a3', 'press_down_tilt', 'add_neighbor', 'decrease_power']`
- `a4176c2c-458b-4c9c-b89b-c0c873747d0b` `C12|C13|C19|C20` `['press_down_tilt', 'add_neighbor', 'decrease_power', 'decrease_a3']`
- `acb8a1db-6d3f-4949-b26f-bdab3ec0008b` `C6|C24` `['decrease_a3', 'add_neighbor']`
- `20bee2a8-0b4f-40a8-9406-46d9ea820a8f` `C5|C13` `['add_neighbor', 'decrease_a3']`
- `ba05f47d-e2a7-4f60-b05a-a521d84bbeef` `C11|C16|C18|C19` `['press_down_tilt', 'add_neighbor', 'decrease_a3', 'decrease_power']`

LoRA:
- `6060dca7`: C22 expected C22 (OK)
- `1784b64c`: C20 expected C20 (OK)

## v6_1_strict60_base

Family counts:
- `F1`: 241
- `F1_5BS`: 28
- `F2`: 145
- `F3_5BS_MULTI`: 29
- `F4`: 29
- `F4M_13BS_MULTI`: 28

F1/F1_5BS action distribution:
- `F1:add_neighbor`: 8
- `F1:check_server`: 6
- `F1:decrease_a3`: 95
- `F1:insufficient_data`: 32
- `F1:lift_tilt`: 72
- `F1:pdcch_2sym`: 28
- `F1_5BS:add_neighbor`: 5
- `F1_5BS:decrease_a3`: 10
- `F1_5BS:insufficient_data`: 13

F2 bad rows:
- `f428f517-0bcd-4d13-b31a-929eb3584363` `C1|C6|C9|C20` `['add_neighbor', 'decrease_power', 'press_down_tilt', 'decrease_a3']`
- `732ac9cf-5ad2-4934-a5f6-f41d6edde04e` `C6|C12|C15|C19` `['decrease_power', 'press_down_tilt', 'decrease_a3', 'add_neighbor']`
- `13482c57-c7d2-40b3-8020-c45ff3a396f4` `C1|C4|C6|C22` `['decrease_power', 'decrease_a3', 'add_neighbor', 'press_down_tilt']`
- `ddb040fe-708c-472d-8ef6-584b8ec5b7ec` `C1|C2|C12|C20` `['decrease_a3', 'decrease_power', 'add_neighbor', 'press_down_tilt']`
- `f02d340d-18c0-43c9-ae9c-c6132c5fe719` `C10|C15|C20|C22` `['press_down_tilt', 'decrease_power', 'add_neighbor', 'decrease_a3']`
- `e83a09dd-795a-4066-9f5c-2de6da46dfea` `C13|C14|C15|C23` `['decrease_a3', 'press_down_tilt', 'add_neighbor', 'decrease_power']`
- `a4176c2c-458b-4c9c-b89b-c0c873747d0b` `C12|C13|C19|C20` `['press_down_tilt', 'add_neighbor', 'decrease_power', 'decrease_a3']`
- `acb8a1db-6d3f-4949-b26f-bdab3ec0008b` `C6|C24` `['decrease_a3', 'add_neighbor']`
- `20bee2a8-0b4f-40a8-9406-46d9ea820a8f` `C5|C13` `['add_neighbor', 'decrease_a3']`
- `ba05f47d-e2a7-4f60-b05a-a521d84bbeef` `C11|C16|C18|C19` `['press_down_tilt', 'add_neighbor', 'decrease_a3', 'decrease_power']`

LoRA:
- `6060dca7`: C22 expected C22 (OK)
- `1784b64c`: C20 expected C20 (OK)

## v6_1_strict60_plus_f2fix

Family counts:
- `F1`: 241
- `F1_5BS`: 28
- `F2`: 145
- `F3_5BS_MULTI`: 29
- `F4`: 29
- `F4M_13BS_MULTI`: 28

F1/F1_5BS action distribution:
- `F1:add_neighbor`: 8
- `F1:check_server`: 6
- `F1:decrease_a3`: 95
- `F1:insufficient_data`: 32
- `F1:lift_tilt`: 72
- `F1:pdcch_2sym`: 28
- `F1_5BS:add_neighbor`: 5
- `F1_5BS:decrease_a3`: 10
- `F1_5BS:insufficient_data`: 13

LoRA:
- `6060dca7`: C22 expected C22 (OK)
- `1784b64c`: C20 expected C20 (OK)
