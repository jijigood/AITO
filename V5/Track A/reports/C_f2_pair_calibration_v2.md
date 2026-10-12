# C — F2 Pair Calibration Report (v2)

## Changes vs V5_C original

- Base CSV: v5_11 (baseline + F2 structural fixes only; no v5_07 F1 flips)
- Excluded: 6 D-keep rows where D's API-based analysis contradicts C's offline top_issue
- flip_score fixed: uncertain cases now score at phase1_confidence (1.0), not 0.0
- Tilt code selection sorted numerically for deterministic output

## Parameters

- Confidence threshold: 0.7
- Max flips: 999
- Base CSV: submission_600_v5_11_baseline_plus_f2_pair_fix.csv
- Excluded IDs: 6

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
Cardinality-2 F2 rows: 139
Excluded (D-keep conflicts): 6
Flip candidates: 70

### Candidates (sorted by flip_score)

| scenario_id | top_issue | current_pair | target_pair | confidence | flip_score |
|---|---|---|---|---|---|
| f00df504-5e00-49c3-b040-29c62021793d | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 1f872048-7692-4443-bbda-324975dbef07 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| f86420da-3884-4a89-985e-13b870108ac8 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| a5aa02e2-3000-4e1b-b1e1-56b629e9c2d4 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 863e801a-af00-411d-95c8-9795bf7859fd | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 815f0842-32f3-440f-837d-42abdd09f248 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 71f8e7bd-15b6-490d-adf4-b4746615987a | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| e72d1cd8-f9ec-495b-81fa-8469bbc665ac | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| de896155-b409-4446-8679-e1bb486e4882 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| a23ab670-fd49-4272-b64c-fcbf270dae96 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| f765cbbc-ad37-492b-bfb2-b4844b28e1d7 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| ff500b22-2cdd-48b8-9029-a19556a0284b | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| d1db7c9d-0831-43cc-8458-2c592f8ddd11 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 57ff95b0-924b-44f1-b5a8-4603b422af24 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| f3d7cd99-2ea9-46f9-91f6-79ea7bd303fd | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| d06a257c-34d3-4641-a17b-6ec6284a1f20 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 5137d812-1c90-4200-98c7-b1d4f44bb3d5 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 337083f8-8abe-425c-b1bb-9ff292a5517b | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| f88ec12a-82ad-4f49-be0a-c278cc038ae6 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| e3481556-e1e4-49f2-9d9d-a047297cc242 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 371ea7fb-d6f9-45e8-b502-81c4804af9ba | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 40400e43-4205-47bb-86cb-4e7e9653a9fe | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 201bf04f-a03e-4087-86a2-cc0361166b18 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| d319eb75-be5c-4aa9-81f6-9632f2177a8f | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 46b455b4-f618-4336-9c53-a73628bbfd74 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| da205a6c-0beb-4ed8-b2e8-e7bdb94f88db | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 8663a98f-734e-46a9-a5bd-23bec79731f4 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 9ccd2a41-433f-4293-b19b-40661e6c298c | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 054a2600-cf4c-46b4-a65b-527106768a53 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 2895f71c-6fdf-406b-9728-2a657bd876b6 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| ce7e2a00-38b3-4eb2-9981-792e13619013 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| ac809e50-4372-497b-9e50-f2658003fb46 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 16e2e629-c315-424a-837c-375eab4be2ad | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| e470b801-ca77-4c4e-b8fd-96679001c6c8 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 952a1d11-1f7e-4d8d-88f7-354f207b64a1 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| dca250b6-783f-4774-bd73-06af30cc776d | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 69b07994-9618-4d73-8e19-9f7a6f517994 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| cc3421e4-1646-41cb-b31d-01129c51751a | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| b4b57660-d3d8-4059-90dd-07d26947d1bd | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| ba05f47d-e2a7-4f60-b05a-a521d84bbeef | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| bc2e61ee-93fb-451f-8656-9c6590f4b002 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 0723bd3a-60d4-4f7c-a441-e9c14cd6c6ca | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| b71c86a7-c89b-4bc1-b58a-f437de5c68d6 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| b019d631-6f59-49cf-a489-85a561281d51 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| d65f5bd9-e955-465b-be4d-b07c76690b95 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| ca969cc9-78df-47ae-aefc-2972691e9ff4 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| aa7c719f-2ecf-4b31-9b9a-4c024de0d769 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| fbc7fa7e-6437-4fe5-bd7c-aafa090091ff | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 2fe0c43d-2183-4a2b-9a33-4e09d10cea30 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 9e62763e-ffa5-4ecf-82f5-8f693064b32f | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 8f8e3ba8-c35a-41ec-8870-9e0602450a5d | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| d0e7a28f-2474-4d35-8487-f0fb0fe21429 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 1c30a245-d4e1-4f4f-9e1e-e6c94edaf8e6 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 2c9e99d0-bc53-4a96-a942-a8f84c768029 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| cfcbe96d-8979-41f3-8946-419d7812cb81 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 1500e616-4f0d-48cb-a67b-cf225a2bfd85 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| c7855316-97b9-48ed-8919-c5853aca58d3 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 132c624a-c39d-4adf-bd70-653b8a7acdd8 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 8537f835-5853-41f7-8498-bf789bf86166 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| c2c80a6c-d00d-4227-b741-14fb30170130 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| b4e916dd-ffce-435e-ba64-301706376b84 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 814a0b57-3698-47e3-a2c7-b6b5e83528ce | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| b12d176f-8116-4c29-a900-ec41abfb6b60 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 3ccc4eba-6648-4901-b1c3-e3ba6dae218d | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 9cdd5c86-a7fa-4e60-b78c-e461f2243efe | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 484a4598-4f13-4c08-b48c-0722092438f8 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| ddc54a58-8604-40f1-8e79-d2b976dd4a85 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 732eafb6-9a40-4e36-a9b5-d78595e6e9c9 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| 9d0594c2-592b-468c-98c4-93a6226e1592 | uncertain | azimuth | tilt | 1.00 | 1.0000 |
| faab3b03-cfab-4216-849f-0a66c0f72fa0 | uncertain | azimuth | tilt | 1.00 | 1.0000 |

## Patch Result

Flips applied: 70
Skipped: 0

### Applied Flips

| scenario_id | top_issue | old | old_pair | new | new_pair | confidence |
|---|---|---|---|---|---|---|
| f00df504-5e00-49c3-b040-29c62021793d | uncertain | C12|C22 | azimuth | C1|C6 | tilt | 1.00 |
| 1f872048-7692-4443-bbda-324975dbef07 | uncertain | C1|C2 | azimuth | C3|C9 | tilt | 1.00 |
| f86420da-3884-4a89-985e-13b870108ac8 | uncertain | C3|C8 | azimuth | C1|C7 | tilt | 1.00 |
| a5aa02e2-3000-4e1b-b1e1-56b629e9c2d4 | uncertain | C2|C11 | azimuth | C4|C9 | tilt | 1.00 |
| 863e801a-af00-411d-95c8-9795bf7859fd | uncertain | C3|C23 | azimuth | C14|C16 | tilt | 1.00 |
| 815f0842-32f3-440f-837d-42abdd09f248 | uncertain | C2|C20 | azimuth | C3|C9 | tilt | 1.00 |
| 71f8e7bd-15b6-490d-adf4-b4746615987a | uncertain | C1|C24 | azimuth | C5|C8 | tilt | 1.00 |
| e72d1cd8-f9ec-495b-81fa-8469bbc665ac | uncertain | C2|C5 | azimuth | C1|C4 | tilt | 1.00 |
| de896155-b409-4446-8679-e1bb486e4882 | uncertain | C2|C7 | azimuth | C4|C13 | tilt | 1.00 |
| a23ab670-fd49-4272-b64c-fcbf270dae96 | uncertain | C1|C7 | azimuth | C6|C15 | tilt | 1.00 |
| f765cbbc-ad37-492b-bfb2-b4844b28e1d7 | uncertain | C6|C9 | azimuth | C2|C3 | tilt | 1.00 |
| ff500b22-2cdd-48b8-9029-a19556a0284b | uncertain | C3|C13 | azimuth | C1|C18 | tilt | 1.00 |
| d1db7c9d-0831-43cc-8458-2c592f8ddd11 | uncertain | C2|C10 | azimuth | C6|C16 | tilt | 1.00 |
| 57ff95b0-924b-44f1-b5a8-4603b422af24 | uncertain | C3|C21 | azimuth | C4|C7 | tilt | 1.00 |
| f3d7cd99-2ea9-46f9-91f6-79ea7bd303fd | uncertain | C1|C7 | azimuth | C2|C11 | tilt | 1.00 |
| d06a257c-34d3-4641-a17b-6ec6284a1f20 | uncertain | C7|C10 | azimuth | C2|C8 | tilt | 1.00 |
| 5137d812-1c90-4200-98c7-b1d4f44bb3d5 | uncertain | C5|C23 | azimuth | C9|C12 | tilt | 1.00 |
| 337083f8-8abe-425c-b1bb-9ff292a5517b | uncertain | C2|C8 | azimuth | C14|C15 | tilt | 1.00 |
| f88ec12a-82ad-4f49-be0a-c278cc038ae6 | uncertain | C3|C4 | azimuth | C1|C6 | tilt | 1.00 |
| e3481556-e1e4-49f2-9d9d-a047297cc242 | uncertain | C12|C15 | azimuth | C1|C4 | tilt | 1.00 |
| 371ea7fb-d6f9-45e8-b502-81c4804af9ba | uncertain | C11|C14 | azimuth | C7|C17 | tilt | 1.00 |
| 40400e43-4205-47bb-86cb-4e7e9653a9fe | uncertain | C9|C10 | azimuth | C5|C15 | tilt | 1.00 |
| 201bf04f-a03e-4087-86a2-cc0361166b18 | uncertain | C3|C9 | azimuth | C5|C6 | tilt | 1.00 |
| d319eb75-be5c-4aa9-81f6-9632f2177a8f | uncertain | C12|C23 | azimuth | C1|C11 | tilt | 1.00 |
| 46b455b4-f618-4336-9c53-a73628bbfd74 | uncertain | C9|C13 | azimuth | C1|C5 | tilt | 1.00 |
| da205a6c-0beb-4ed8-b2e8-e7bdb94f88db | uncertain | C5|C23 | azimuth | C4|C10 | tilt | 1.00 |
| 8663a98f-734e-46a9-a5bd-23bec79731f4 | uncertain | C9|C18 | azimuth | C8|C13 | tilt | 1.00 |
| 9ccd2a41-433f-4293-b19b-40661e6c298c | uncertain | C16|C21 | azimuth | C6|C10 | tilt | 1.00 |
| 054a2600-cf4c-46b4-a65b-527106768a53 | uncertain | C16|C22 | azimuth | C3|C7 | tilt | 1.00 |
| 2895f71c-6fdf-406b-9728-2a657bd876b6 | uncertain | C9|C16 | azimuth | C1|C2 | tilt | 1.00 |
| ce7e2a00-38b3-4eb2-9981-792e13619013 | uncertain | C2|C16 | azimuth | C13|C15 | tilt | 1.00 |
| ac809e50-4372-497b-9e50-f2658003fb46 | uncertain | C4|C20 | azimuth | C2|C5 | tilt | 1.00 |
| 16e2e629-c315-424a-837c-375eab4be2ad | uncertain | C3|C6 | azimuth | C5|C12 | tilt | 1.00 |
| e470b801-ca77-4c4e-b8fd-96679001c6c8 | uncertain | C10|C23 | azimuth | C2|C9 | tilt | 1.00 |
| 952a1d11-1f7e-4d8d-88f7-354f207b64a1 | uncertain | C1|C3 | azimuth | C4|C15 | tilt | 1.00 |
| dca250b6-783f-4774-bd73-06af30cc776d | uncertain | C8|C10 | azimuth | C3|C5 | tilt | 1.00 |
| 69b07994-9618-4d73-8e19-9f7a6f517994 | uncertain | C6|C9 | azimuth | C1|C2 | tilt | 1.00 |
| cc3421e4-1646-41cb-b31d-01129c51751a | uncertain | C1|C6 | azimuth | C10|C15 | tilt | 1.00 |
| b4b57660-d3d8-4059-90dd-07d26947d1bd | uncertain | C8|C10 | azimuth | C4|C9 | tilt | 1.00 |
| ba05f47d-e2a7-4f60-b05a-a521d84bbeef | uncertain | C5|C9 | azimuth | C11|C14 | tilt | 1.00 |
| bc2e61ee-93fb-451f-8656-9c6590f4b002 | uncertain | C4|C12 | azimuth | C6|C15 | tilt | 1.00 |
| 0723bd3a-60d4-4f7c-a441-e9c14cd6c6ca | uncertain | C12|C18 | azimuth | C3|C11 | tilt | 1.00 |
| b71c86a7-c89b-4bc1-b58a-f437de5c68d6 | uncertain | C1|C11 | azimuth | C7|C12 | tilt | 1.00 |
| b019d631-6f59-49cf-a489-85a561281d51 | uncertain | C5|C24 | azimuth | C2|C4 | tilt | 1.00 |
| d65f5bd9-e955-465b-be4d-b07c76690b95 | uncertain | C8|C9 | azimuth | C6|C14 | tilt | 1.00 |
| ca969cc9-78df-47ae-aefc-2972691e9ff4 | uncertain | C5|C14 | azimuth | C1|C6 | tilt | 1.00 |
| aa7c719f-2ecf-4b31-9b9a-4c024de0d769 | uncertain | C4|C8 | azimuth | C5|C11 | tilt | 1.00 |
| fbc7fa7e-6437-4fe5-bd7c-aafa090091ff | uncertain | C11|C17 | azimuth | C6|C15 | tilt | 1.00 |
| 2fe0c43d-2183-4a2b-9a33-4e09d10cea30 | uncertain | C7|C24 | azimuth | C3|C11 | tilt | 1.00 |
| 9e62763e-ffa5-4ecf-82f5-8f693064b32f | uncertain | C3|C19 | azimuth | C1|C13 | tilt | 1.00 |
| 8f8e3ba8-c35a-41ec-8870-9e0602450a5d | uncertain | C17|C20 | azimuth | C2|C15 | tilt | 1.00 |
| d0e7a28f-2474-4d35-8487-f0fb0fe21429 | uncertain | C2|C22 | azimuth | C9|C11 | tilt | 1.00 |
| 1c30a245-d4e1-4f4f-9e1e-e6c94edaf8e6 | uncertain | C1|C14 | azimuth | C3|C4 | tilt | 1.00 |
| 2c9e99d0-bc53-4a96-a942-a8f84c768029 | uncertain | C3|C18 | azimuth | C2|C4 | tilt | 1.00 |
| cfcbe96d-8979-41f3-8946-419d7812cb81 | uncertain | C6|C8 | azimuth | C5|C11 | tilt | 1.00 |
| 1500e616-4f0d-48cb-a67b-cf225a2bfd85 | uncertain | C2|C3 | azimuth | C6|C7 | tilt | 1.00 |
| c7855316-97b9-48ed-8919-c5853aca58d3 | uncertain | C2|C18 | azimuth | C3|C14 | tilt | 1.00 |
| 132c624a-c39d-4adf-bd70-653b8a7acdd8 | uncertain | C9|C19 | azimuth | C2|C4 | tilt | 1.00 |
| 8537f835-5853-41f7-8498-bf789bf86166 | uncertain | C1|C2 | azimuth | C5|C12 | tilt | 1.00 |
| c2c80a6c-d00d-4227-b741-14fb30170130 | uncertain | C3|C12 | azimuth | C13|C14 | tilt | 1.00 |
| b4e916dd-ffce-435e-ba64-301706376b84 | uncertain | C2|C7 | azimuth | C9|C13 | tilt | 1.00 |
| 814a0b57-3698-47e3-a2c7-b6b5e83528ce | uncertain | C4|C11 | azimuth | C3|C5 | tilt | 1.00 |
| b12d176f-8116-4c29-a900-ec41abfb6b60 | uncertain | C8|C22 | azimuth | C5|C11 | tilt | 1.00 |
| 3ccc4eba-6648-4901-b1c3-e3ba6dae218d | uncertain | C9|C19 | azimuth | C2|C3 | tilt | 1.00 |
| 9cdd5c86-a7fa-4e60-b78c-e461f2243efe | uncertain | C6|C14 | azimuth | C10|C15 | tilt | 1.00 |
| 484a4598-4f13-4c08-b48c-0722092438f8 | uncertain | C2|C7 | azimuth | C12|C19 | tilt | 1.00 |
| ddc54a58-8604-40f1-8e79-d2b976dd4a85 | uncertain | C4|C5 | azimuth | C14|C16 | tilt | 1.00 |
| 732eafb6-9a40-4e36-a9b5-d78595e6e9c9 | uncertain | C18|C19 | azimuth | C9|C11 | tilt | 1.00 |
| 9d0594c2-592b-468c-98c4-93a6226e1592 | uncertain | C14|C15 | azimuth | C7|C11 | tilt | 1.00 |
| faab3b03-cfab-4216-849f-0a66c0f72fa0 | uncertain | C10|C21 | azimuth | C13|C14 | tilt | 1.00 |
