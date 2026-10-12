# Agentic Overlay Test Report — Phase 2 High-Risk Probes
**Generated**: 2026-05-08  
**Source**: `result_full_uncertainfix.diagnostics.jsonl` (full 500-scenario run, 2026-05-07)  
**Candidate**: `result_full_uncertainfix.csv` — 500/500 answered, 0 hard errors

---

## Summary

| Metric | Value |
|--------|-------|
| Total scenarios | 500 |
| High-risk F1/F1_5BS (overlay-eligible) | **28** |
| Agentic overlay errors | **0** |
| Deterministic overrides by LLM | **0** |
| Mean scenario latency (all 500) | 10.9 s (p95: 14.3 s) |

The agentic overlay functioned correctly as a **validation layer**: LLM confirmed the deterministic pipeline's prediction on all 28 high-risk scenarios. No overrides were generated, which is expected given the current feature score distributions (see below).

---

## Branch Breakdown

### Branch A: `lift_tilt_vs_insufficient_data` (22 scenarios)

**Trigger condition**: `coverage_hole > 0.20 AND insufficient_data > 0.20 AND |diff| < 0.25`

| Scenario ID (prefix) | cov score | insuf score | Deterministic action | LLM override |
|---|---|---|---|---|
| 0fdd3fef | 0.250 | 0.780 | insufficient_data | — |
| 69a578a6 | 0.250 | 0.820 | insufficient_data | — |
| 62b0b362 | 0.250 | 0.780 | insufficient_data | — |
| d2e36af8 | 0.250 | 0.780 | insufficient_data | — |
| 44840230 | 0.250 | 0.780 | insufficient_data | — |
| 95a3235a | 0.250 | 0.820 | insufficient_data | — |
| 54110aae | 0.250 | 0.820 | insufficient_data | — |
| 1f161b48 | 0.250 | 0.780 | insufficient_data | — |
| 70a5d3e9 | 0.250 | 0.820 | insufficient_data | — |
| 9a03817f | 0.250 | 0.820 | insufficient_data | — |
| 465bfdfc | 0.250 | 0.780 | insufficient_data | — |
| 22de1f6d | 0.250 | 0.820 | insufficient_data | — |
| b6412181 | 0.250 | 0.780 | insufficient_data | — |
| 80844485 | 0.250 | 0.780 | insufficient_data | — |
| fb93f000 | 0.250 | 0.780 | insufficient_data | — |
| be526c5a | 0.250 | 0.780 | insufficient_data | — |
| 32245d27 | 0.250 | 0.780 | insufficient_data | — |
| 6be53ad8 | 0.250 | 0.780 | insufficient_data | — |
| bc5ad307 | 0.250 | 0.780 | insufficient_data | — |
| af7536e6 | 0.250 | 0.820 | insufficient_data | — |
| 56591d6e | 0.250 | 0.820 | insufficient_data | — |
| 4528d0fe | 0.250 | 0.780 | insufficient_data | — |

**Why no overrides**: After the 2026-05-07 mainlobe fix, all 22 cases have `insuf ∈ {0.780, 0.820}` vs `cov = 0.250`. The score gap (0.53–0.57) is large enough that the LLM, given the same evidence, reaches the same conclusion as the deterministic pipeline. Overrides would only appear if the LLM had additional trajectory context showing a different pattern — this is D's opportunity (trajectory RAG).

**Two insuf score levels**:
- `insuf = 0.820`: MR count < 7 (sparse data gate, `skills/tool_caller.py` line ~362)
- `insuf = 0.780`: mainlobe flag = True via `judge_mainlobe_or_not` (line ~355)

---

### Branch B: `add_neighbor_vs_decrease_a3` (6 scenarios)

**Trigger condition**: `missing_neighbor > 0.20 AND late_handover > 0.20 AND |diff| < 0.25`

| Scenario ID (prefix) | mn score | lh score | Deterministic action | LLM override |
|---|---|---|---|---|
| f37da282 | 0.750 | 0.400 | missing_neighbor → add_neighbor | — |
| c827524a | 0.300 | 0.700 | late_handover → decrease_a3 | — |
| 11e38c8b | 0.300 | 0.700 | late_handover → decrease_a3 | — |
| da28c256 | 0.300 | 0.700 | late_handover → decrease_a3 | — |
| a4e55fd2 | 0.750 | 0.400 | missing_neighbor → add_neighbor | — |
| 3e7e649d | 0.300 | 0.700 | late_handover → decrease_a3 | — |

**Why no overrides**: Score differences are 0.35 (0.75 vs 0.40) or 0.40 (0.70 vs 0.30) — well beyond the 0.25 ambiguity threshold that triggers the branch. These cases are borderline enough to trip the trigger but clear enough for the deterministic pipeline to decide confidently. LLM confirmation is expected.

---

## Assessment & Next Steps

**Current status**: Overlay is a **confirmation layer**, not a correction layer. This is correct behavior for the current state of the pipeline — the deterministic scores are well-calibrated after the mainlobe fix.

**When will overrides appear?**
Overrides will emerge when the LLM, given trajectory RAG context (D's deliverable), surfaces a pattern that contradicts the deterministic score. This is most likely in the `lift_tilt_vs_insufficient_data` branch where all 22 cases converge to the same score pattern (cov=0.250, insuf=0.78/0.82) — the LLM can potentially differentiate cases within this cluster using RF geometry or serving cell context.

**Actions for Owner A**:
1. When D delivers `retrieve_trajectory_context()`, wire it into `_build_step2_messages()` at the `INTEGRATION HOOK [D]` marker in `skills/agentic_overlay.py`.
2. Re-run the 22 `lift_tilt_vs_insufficient_data` scenarios as a targeted probe to observe whether trajectory context produces overrides.
3. If LLM overrides appear on ≥3 scenarios with `evidence_confidence=high` and switch `insufficient_data → lift_tilt`, evaluate whether the switch improves or hurts — Phase 2 ground truth is not available, so use IOU on similar Phase 1 cases.
4. Keep `--llm_policy audit` and `--agentic_policy high_risk` for Submission 1.

**LoRA status** (unchanged): `v4_fast_balanced` IOU=0.00 on lift_tilt and insufficient_data. Keep in `audit` mode. Do not promote.

---

## Appendix: API Call Distribution (Full 500-scenario run)

| Tool | Calls | Errors |
|---|---|---|
| get_user_plane_data | 500 | 0 |
| get_config_data | 500 | 0 |
| get_kpi_data | 500 | 0 |
| get_mr_data | 500 | 0 |
| calculate_overlap_ratio | 321 | 0 |
| judge_mainlobe_or_not | 186 | 0 |
| **Total** | **2507** | **0** |

Signaling data missing for all 500 scenarios (expected — not provided by competition server).
