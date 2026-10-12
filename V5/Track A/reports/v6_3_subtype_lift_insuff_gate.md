# V6.3 Subtype lift_tilt/insufficient_data Gate

## Decision
- Use the diagnostic SFT boundary subtype rule directly; no model call required.
- Rule: generic high-PRB primary fallback -> insufficient_data; coverage tertiary hypothesis -> keep lift_tilt.
- Scope: Phase2 F1/F1_5BS single-answer rows where v6_2 currently answers lift_tilt.

## Phase1 Gate
- DEV: `{'n': 303, 'hits': 303, 'accuracy': 1.0, 'gt_counts': {'insufficient_data': 151, 'lift_tilt': 152}}`
- TEST: `{'n': 203, 'hits': 203, 'accuracy': 1.0, 'gt_counts': {'insufficient_data': 101, 'lift_tilt': 102}}`
- subtype table: `{'dev:coverage_hypothesis_tertiary': {'lift_tilt': 152}, 'dev:generic_high_prb_primary_fallback': {'insufficient_data': 151}, 'test:coverage_hypothesis_tertiary': {'lift_tilt': 102}, 'test:generic_high_prb_primary_fallback': {'insufficient_data': 101}}`

## Phase2 Application
- base: `/home/huangxiaolin/AITO/V5/Track A/submissions/submission_600_v6_2_addneighbor_gate.csv`
- output: `/home/huangxiaolin/AITO/V5/Track A/submissions/submission_600_v6_3_subtype_lift_insuff_gate.csv`
- changed rows: `28`
- Phase2 lift subtypes: `{'other:primary:geometry recommendation': 14, 'coverage_hypothesis_tertiary': 29, 'generic_high_prb_primary_fallback': 28, 'unknown': 1}`
- protected raw54 intersection: `[]`

## Structural Audit
- rows: `600`
- Track A rows: `500`
- Track B nonblank: `0`
- F1 illegal: `0`
- F2 bad: `0`
- LoRA overrides: `{'6060dca7': 'OK', '1784b64c': 'OK'}`

## Changed Rows

| id | old | new | reason |
|---|---:|---:|---|
| 95cda870-3964-4e6b-b661-46ff57c94aba | C10 | C2 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| 0025e3f1-6a6a-4146-8fce-c0cf97816995 | C14 | C13 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| da968423-724d-4d1d-a4b7-17541f21d116 | C8 | C15 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| 4ff7e5c7-e7cf-406f-ad73-a1e3438caccb | C11 | C3 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| eb91eb27-7ab5-4f60-bb38-7d71c58e02a8 | C1 | C12 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| f3a881f3-2a7c-4065-85b6-fc210962372f | C16 | C24 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| 10f63584-28ba-4a37-b5ad-084c2418800c | C5 | C6 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| 4977d843-18af-47e7-87dd-f3a99ad8306c | C7 | C13 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| fcff6672-f6eb-46d4-83e9-5c6d86e4d90d | C16 | C22 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| b63bb459-a261-4901-98a7-945f320dae84 | C16 | C4 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| f43ee4bb-ca0a-4771-b20e-cd36cd093281 | C17 | C2 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| 72d41141-5395-417b-ae27-744051a16651 | C12 | C21 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| 3de3a430-2e56-4ebf-a849-674ecae67d4b | C16 | C18 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| 689ff6bc-b877-47c5-92c0-f66e168a9129 | C11 | C23 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| 0b5ba024-af2f-4d50-9c87-83803cd4c61d | C11 | C1 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| 3eccaf22-9c49-4e87-8ac7-19bb514b1b04 | C2 | C12 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| 44f266a0-8295-4d30-9854-ba1cc449901d | C13 | C24 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| 5622037e-9be6-4408-958b-6ac8d158af1b | C19 | C7 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| f2cb0cff-fafa-494e-9d5e-364a685b742a | C20 | C4 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| 4285e888-22c3-4caa-a245-81d76d891757 | C24 | C1 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| 999685d2-dd51-4856-8dd7-e15fdf02513f | C4 | C20 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| 37020348-6b81-4098-9e53-8ece23158960 | C8 | C18 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| 35463299-505f-4fd8-b0f1-d6f8aee6aa53 | C1 | C15 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| 0e4eb0cd-19c2-4f4d-b045-7e25a7a4ba12 | C16 | C4 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| ed2c5525-94c7-4e25-bcd8-d447fd77358d | C8 | C6 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| ec7a63d4-a986-4436-af45-2ae8f799cedf | C1 | C21 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| b0719a09-7ef9-427a-b23a-8336baa61d37 | C18 | C11 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
| f37d0166-c908-4e30-a5ad-288188c44a9d | C12 | C16 | F1 high-PRB healthy-radio → lift_tilt/primary hypothesis |
