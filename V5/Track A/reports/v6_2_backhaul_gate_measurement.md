# V6.2 Backhaul Check-Server Gate Measurement

## Phase1 F1 Signal

- Rows measured: `1456`
- Current mean IOU: `0.7727`

| Threshold | Selected | GT check_server | Signal precision | Already pred check | Impacted | Impact precision | Delta IOU | Helped | Harmed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 985 | 239 | 0.2426 | 239 | 746 | 0.0000 | -0.3393 | 0 | 494 |
| 0.20 | 985 | 239 | 0.2426 | 239 | 746 | 0.0000 | -0.3393 | 0 | 494 |
| 0.30 | 985 | 239 | 0.2426 | 239 | 746 | 0.0000 | -0.3393 | 0 | 494 |
| 0.40 | 985 | 239 | 0.2426 | 239 | 746 | 0.0000 | -0.3393 | 0 | 494 |
| 0.45 | 246 | 239 | 0.9715 | 239 | 7 | 0.0000 | -0.0021 | 0 | 3 |
| 0.48 | 241 | 239 | 0.9917 | 239 | 2 | 0.0000 | -0.0007 | 0 | 1 |
| 0.49 | 240 | 239 | 0.9958 | 239 | 1 | 0.0000 | 0.0000 | 0 | 0 |
| 0.50 | 239 | 239 | 1.0000 | 239 | 0 | NA | 0.0000 | 0 | 0 |

## Phase2 Diagnostic Distribution

- Rows measured: `241`
- Backhaul histogram: `{'0.0': 123, '0.2': 25, '0.3': 5, '0.4': 85, '0.5': 3}`

| Threshold | Selected | Families | Current actions |
|---:|---:|---|---|
| 0.10 | 118 | `{'F1': 118}` | `{'pdcch_2sym': 26, 'lift_tilt': 57, 'insufficient_data': 32, 'decrease_a3': 3}` |
| 0.20 | 118 | `{'F1': 118}` | `{'pdcch_2sym': 26, 'lift_tilt': 57, 'insufficient_data': 32, 'decrease_a3': 3}` |
| 0.30 | 93 | `{'F1': 93}` | `{'pdcch_2sym': 26, 'lift_tilt': 57, 'insufficient_data': 7, 'decrease_a3': 3}` |
| 0.40 | 88 | `{'F1': 88}` | `{'pdcch_2sym': 26, 'lift_tilt': 57, 'insufficient_data': 2, 'decrease_a3': 3}` |
| 0.45 | 3 | `{'F1': 3}` | `{'lift_tilt': 3}` |
| 0.48 | 3 | `{'F1': 3}` | `{'lift_tilt': 3}` |
| 0.49 | 2 | `{'F1': 2}` | `{'lift_tilt': 2}` |
| 0.50 | 0 | `{}` | `{}` |

## Decision

- Phase1 does not support a broad `backhaul_server >= 0.4` promotion.
- At `>=0.4`, Phase1 signal precision is `0.2426`, while Phase2 would promote/check `88` rows.
- At `>=0.5`, Phase1 signal precision is `1.0000`, but Phase2 only has `0` rows.
