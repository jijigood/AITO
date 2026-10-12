# Diagnostic Training Ceiling Audit

- scenarios: 2000
- heldout ids loaded: 200
- candidate top_k: 5
- errors: 0

## GT coverage bucket

| item | count | pct |
|---|---:|---:|
| top1 | 1633 | 81.7% |
| top3 | 362 | 18.1% |
| top5 | 4 | 0.2% |
| options_not_top5 | 1 | 0.1% |

## GT coverage by family

| item | count | pct |
|---|---:|---:|
| F1 / top1 | 1125 | 56.2% |
| F1 / top3 | 326 | 16.3% |
| F4 / top1 | 245 | 12.2% |
| F2 / top1 | 192 | 9.6% |
| F3 / top1 | 71 | 3.5% |
| F2 / top3 | 36 | 1.8% |
| F1 / top5 | 4 | 0.2% |
| F1 / options_not_top5 | 1 | 0.1% |

## Single-answer GT action coverage

| item | count | pct |
|---|---:|---:|
| lift_tilt / top1 | 254 | 14.9% |
| insufficient_data / top3 | 248 | 14.6% |
| decrease_interfreq / top1 | 245 | 14.4% |
| pdcch_2sym / top1 | 240 | 14.1% |
| check_server / top1 | 239 | 14.1% |
| decrease_a3 / top1 | 224 | 13.2% |
| add_neighbor / top1 | 168 | 9.9% |
| add_neighbor / top3 | 71 | 4.2% |
| decrease_a3 / top3 | 7 | 0.4% |
| insufficient_data / top5 | 4 | 0.2% |
| add_neighbor / options_not_top5 | 1 | 0.1% |

## Override supervision ceiling

- exact answer differs from deterministic top1: 367 / 2000 (18.4%)
- single-answer action differs from deterministic top action: 331 / 1701 (19.5%)

## Insufficient data failure audit

- insufficient_data GT rows: 252

### Category

| item | count | pct |
|---|---:|---:|
| top_lift_tilt_generic_primary | 252 | 100.0% |

### GT coverage for insufficient_data

| item | count | pct |
|---|---:|---:|
| top3 | 248 | 98.4% |
| top5 | 4 | 1.6% |

### Risk branch for insufficient_data

| item | count | pct |
|---|---:|---:|
| high / lift_tilt_vs_insufficient_data | 252 | 100.0% |

### Current subtype gate would invoke overlay

| item | count | pct |
|---|---:|---:|
| True | 252 | 100.0% |

### Insufficient data by family/category

| item | count | pct |
|---|---:|---:|
| F1 / top_lift_tilt_generic_primary | 252 | 100.0% |
