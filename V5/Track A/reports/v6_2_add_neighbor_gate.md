# V6.2 Add-Neighbor Gate

- Selected rule: `a2d=1:m>=0.70:l<=0.35:a3<=3.00|d2a=1:m<=0.30:l>=0.65:a3>=2.50`
- Gate passed: `True`

| Split | N | Baseline IOU | New IOU | Delta | Changed | Helped | Harmed | decrease_a3 P old->new | add_neighbor R old->new |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | 872 | 0.7764 | 0.7924 | 0.0161 | 14 | 14 | 0 | 0.7701->0.8214 | 0.7222->0.7917 |
| test | 584 | 0.7671 | 0.7842 | 0.0171 | 10 | 10 | 0 | 0.7438->0.7949 | 0.6667->0.7396 |
| holdout | 154 | 0.7922 | 0.7987 | 0.0065 | 1 | 1 | 0 | 0.8000->0.8333 | 0.8000->0.8400 |
| all | 1456 | 0.7727 | 0.7891 | 0.0165 | 24 | 24 | 0 | 0.7593->0.8105 | 0.7000->0.7708 |

## Rule Actions

- `dev`: `{'decrease_a3_to_add_neighbor': 10, 'add_neighbor_to_decrease_a3': 4}`
- `test`: `{'decrease_a3_to_add_neighbor': 7, 'add_neighbor_to_decrease_a3': 3}`
- `holdout`: `{'decrease_a3_to_add_neighbor': 1}`
- `all`: `{'add_neighbor_to_decrease_a3': 7, 'decrease_a3_to_add_neighbor': 17}`

## Phase2 Candidate

- Submission: `submissions/submission_600_v6_2_addneighbor_gate.csv`
- Protected F1/F1_5BS v6_1 fix rows: `60`
- Phase2 rule changes: `9`
- Audit: `{'rows': 600, 'track_a': 500, 'track_b': 0, 'diff_vs_baseline': 79, 'illegal_f1': 0, 'f2_bad': 0, 'lora_ok': True}`
- `c827524a-bdd2-4003-b2ed-2850289f6be4` C1/decrease_a3 -> C11/add_neighbor (decrease_a3_to_add_neighbor, a3=2.76)
- `da28c256-f98d-4248-b7cc-81cd293f3321` C17/decrease_a3 -> C15/add_neighbor (decrease_a3_to_add_neighbor, a3=4.94)
- `c7af696b-7e5c-421c-964f-63adb0ed71c4` C9/decrease_a3 -> C3/add_neighbor (decrease_a3_to_add_neighbor, a3=3.09)
- `19cb5e13-425a-4c57-9f24-65b11f8fba2d` C6/add_neighbor -> C5/decrease_a3 (add_neighbor_to_decrease_a3, a3=2.83)
- `d2a453a0-33af-47ee-bf03-cb931e5c62be` C22/decrease_a3 -> C8/add_neighbor (decrease_a3_to_add_neighbor, a3=7.73)
- `76aba6e6-a6eb-4454-8e81-6a8d2c5c5896` C2/decrease_a3 -> C3/add_neighbor (decrease_a3_to_add_neighbor, a3=8.08)
- `e06b2ff2-12ae-4fa1-b5d4-b424e9c3c2c9` C14/decrease_a3 -> C1/add_neighbor (decrease_a3_to_add_neighbor, a3=4.86)
- `12aadc27-7b18-4b83-bb8a-5b8c40fdde7c` C1/decrease_a3 -> C3/add_neighbor (decrease_a3_to_add_neighbor, a3=2.98)
- `a2107abb-46c7-4e67-a011-07aaf2339ba0` C2/decrease_a3 -> C23/add_neighbor (decrease_a3_to_add_neighbor, a3=4.08)
