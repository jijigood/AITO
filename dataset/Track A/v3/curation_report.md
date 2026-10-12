# Curation Report

- Mode: `fallback`
- Train source: `data/Phase_1/train.json`
- Total F1 examples: `1456`
- Held-out validation size: `200`
- Training candidates before curation: `1256`
- Kept training IDs: `1112`
- Dropped training IDs: `144`
- Curated IDs: `curated_train_ids.txt`
- Held-out val IDs: `heldout_f1_val_ids.txt`
- Weight metadata: `curation_weights.json`

## Drop Counts

- `DROP_DUPLICATE`: 144
- `DROP_LOW_CONFIDENCE`: 0
- `KEEP`: 1103
- `KEEP_FLAGGED`: 9

## Weight Counts

- `0.5`: 1069
- `1.0`: 43

## F1 Action Distribution (pre-curation train split)

- `add_neighbor`: 202
- `check_server`: 201
- `decrease_a3`: 196
- `insufficient_data`: 220
- `lift_tilt`: 228
- `pdcch_2sym`: 209

## Thresholds

- `duplicate_drop`: 0.98
- `confidence_drop`: 0.3
- `outlier_flag`: 3.0
- `ambiguity_weight`: 0.7
