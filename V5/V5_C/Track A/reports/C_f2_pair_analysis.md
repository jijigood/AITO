# C — F2 Multi-Answer Pair Analysis

生成时间：2026-05-15

## TL;DR

Phase 2 F2 (4-BS, multi-answer) has 145 rows. Two findings:

1. **10 rows have illegal action types or wrong cardinality.** Fixed by
   `scripts/C_f2_pair_repair.py`. Output:
   `submissions/submission_600_v5_10_adj_only_plus_f2_pair_fix.csv`.
2. **Pair-selection in the remaining 137 card-2 rows is skewed**: 50.4%
   azimuth-pair / 48.2% tilt-pair, vs Phase 1's 62% / 38%. Likely 10–15
   tilt-pair rows should be azimuth-pair. Real but harder to fix per-row.

## 1. Phase 1 F2 ground rules

From `data/Phase_1/train.json` (228 F2 rows):

- Every answer has cardinality exactly 2.
- Every answer is one of two legal action pairs:
  - **azimuth pair**: `{adjust_azimuth, increase_power}` — 142/228 = **62.3%**
  - **tilt pair**: `{decrease_power, press_down_tilt}` — 86/228 = **37.7%**

No other action types or cardinalities exist in Phase 1 F2.

## 2. Phase 2 violations of the rule (10 rows)

From `result_diag_lora_boundary_only.diagnostics.jsonl`:

- **8 rows with cardinality 4** — all containing the same illegal action
  composition `{add_neighbor, decrease_a3, decrease_power, press_down_tilt}`.
  All 8 have `top_issue = late_handover`.
- **2 rows with cardinality 2 but illegal actions** `{add_neighbor, decrease_a3}`
  (no F2-legal codes at all). Both have `top_issue = missing_neighbor`.

Common diagnosis: the candidate generator leaked F1 single-answer actions
(`add_neighbor`, `decrease_a3`) into F2 multi-answer outputs. Both action
types are correct outputs for F1 but never appear in Phase 1 F2 answers.

## 3. The fix applied

For each violator, take the highest-scoring 2-code candidate whose action
set is one of the two legal F2 pairs:

| sid | old | old actions | new | new pair | top_issue |
|---|---|---|---|---|---|
| f428f517 | C1\|C6\|C9\|C20 | F1-leak | **C2\|C18** | azimuth (0.66) | late_handover |
| 732ac9cf | C6\|C12\|C15\|C19 | F1-leak | **C8\|C9** | azimuth (0.66) | late_handover |
| 13482c57 | C1\|C4\|C6\|C22 | F1-leak | **C3\|C17** | tilt (0.74) | late_handover |
| ddb040fe | C1\|C2\|C12\|C20 | F1-leak | **C4\|C10** | azimuth (0.66) | late_handover |
| f02d340d | C10\|C15\|C20\|C22 | F1-leak | **C7\|C9** | azimuth (0.66) | late_handover |
| e83a09dd | C13\|C14\|C15\|C23 | F1-leak | **C6\|C16** | azimuth (0.66) | late_handover |
| a4176c2c | C12\|C13\|C19\|C20 | F1-leak | **C2\|C10** | azimuth (0.69) | late_handover |
| ba05f47d | C11\|C16\|C18\|C19 | F1-leak | **C5\|C9** | azimuth (0.66) | late_handover |
| acb8a1db | C6\|C24 | F1-leak | **C2\|C11** | tilt (0.74) | missing_neighbor |
| 20bee2a8 | C5\|C13 | F1-leak | **C1\|C20** | tilt (0.74) | missing_neighbor |

Pair split in the fix: 7 azimuth / 3 tilt. (Close to Phase 1's 62/38.)

### Why I trust the candidate-generator score for pair choice

For the 8 card-4 rows, the candidate list already contains *both* the
azimuth pair and a tilt-pair subset. The score gap is informative:

- For 7/8 rows, the best azimuth-pair candidate scores ~0.66, beating the
  best tilt-pair candidate (~0.51–0.59) by 7–15 points.
- For 1 row (`13482c57`), the best tilt-pair candidate scores 0.74, beating
  the best azimuth-pair candidate (0.61) by 13 points.

The candidate generator's pair-level scoring is the only per-row signal we
have offline. Using it is strictly better than coin-flip and aligned with
the Phase 1 prior on average.

### Expected impact (bounded)

Per row, current IOU is **at most 0.5** (4 codes including at most 2 correct).
Replacement candidate is either correct (IOU → 1.0, +0.5) or wrong (IOU → 0,
−0.5). If the candidate-generator score gives ~60–65% pair-selection
accuracy:

- Expected gain ≈ 10 × (0.6 × 0.5 + 0.4 × −0.5) = **+1.0 IOU on 500 rows**
  ≈ **+0.002 raw / +0.001 public score**.

Tiny but free.  Downside is bounded by ~−0.001 in the worst case.

## 4. The bigger issue C should investigate next

Phase 2 card-2 F2 pair split is **50.4% azimuth / 48.2% tilt** (137 rows).
Phase 1 expects **62% / 38%**. Difference suggests ~10–15 rows currently
labeled tilt-pair should actually be azimuth-pair.

`top_issue` distribution per chosen pair (Phase 2 ours):

| top_issue | azimuth-pair chosen | tilt-pair chosen |
|---|---|---|
| overshoot_overlap | 49 | 39 |
| late_handover | 11 | 18 |
| uncertain | 7 | 0 |
| missing_neighbor | 0 | 9 |
| backhaul_server | 2 | 0 |

The most interesting bucket is the 18 rows with `top_issue = late_handover`
chosen as tilt-pair. `late_handover` is a handover-mobility signal, not a
coverage-tilt signal. In Phase 1 F2, what's the empirical pair distribution
for rows whose deterministic top_issue is `late_handover`? If it leans
azimuth, this is a 10–18 row addressable surface.

### Suggested next analysis (no submission needed)

1. Run the same feature pipeline over Phase 1 F2 to compute
   `top_issue → pair` conditional distribution (this is a calibration step,
   analogous to what B did for KNN).
2. For each conditional bucket whose Phase 1 majority pair differs from the
   row's current Phase 2 pair, list those rows as candidates.
3. Apply only the high-confidence subset (e.g., buckets where Phase 1
   majority is ≥ 70%).

This is the F2 analogue of B's KNN calibration: holdout-validated, per-row
attributable, bounded downside.

## 5. Recommended use of the 10-row patch

The `submission_600_v5_10_adj_only_plus_f2_pair_fix.csv` is a strict
superset of `v5_07_adj_only.csv` (60 F1 changes + 10 F2 changes). All
10 F2 changes are independent of the F1 hypothesis — they correct
violations of the Phase 1 F2 contract regardless of whether the adj_az
hypothesis holds.

Recommendation:

- **Sub1** = `v5_07_adj_only` (clean isolated test of F1 structural hypothesis).
  Do not bundle the F2 fix in sub1, because then sub1's public-score delta
  conflates two hypotheses.
- **Sub2** integration: if sub1 ↑, use `v5_10` or
  `v5_10 + calibrated KNN-60 flips`. The F2 fix should always be included
  in sub2 — it's free expected value.
- If sub1 ↓ unexpectedly, sub2 should be `v5_00_baseline + F2 fix only`
  (10 changes from the 0.1302 baseline). That's the safest fallback.

## 6. Files produced

- `scripts/C_f2_pair_repair.py` — repair script.
- `reports/C_f2_pair_repair.json` — per-row change log (10 rows).
- `submissions/submission_600_v5_10_adj_only_plus_f2_pair_fix.csv` — F2 fix
  applied on top of `v5_07`.

To produce a baseline + F2-only variant (for sub2 fallback scenario):

```bash
/home/huangxiaolin/AITO/bin/python3 scripts/C_f2_pair_repair.py \
  --base_csv submissions/submission_600_v5_00_base_01302.csv \
  --out_csv submissions/submission_600_v5_11_baseline_plus_f2_pair_fix.csv \
  --report_json reports/C_f2_pair_repair_on_baseline.json
```
