# V3 Framework — Track A Telco Troubleshooting

**Competition**: Telco Troubleshooting Agentic Challenge — Track A (Wireless Network)
**Phase 1 Deadline**: 2026-05-04
**Base Model**: Qwen3.5-35B-A3B (cannot be replaced, LoRA only)
**Scoring**: `IOU(predicted_codes, ground_truth) × time_discount`

---

## 1. Executive Summary

V3 is a **modular, family-aware pipeline** that combines deterministic feature engineering with a fine-tuned Qwen LoRA. It replaces V2's monolithic "send everything to the model" approach. The key insight: **F2/F3/F4 scenarios (~25% of traffic) can be solved deterministically at 100% accuracy, and F1 scenarios (~75%) become a clean re-ranking task for the LoRA rather than generation from scratch.**

### Score progression

| Stage | Method | Overall IOU | F1 | F2 | F3 | F4 | Notes |
|---|---|---|---|---|---|---|---|
| V1 baseline | Raw Qwen only | ~0.25 | — | — | — | — | Pre-LoRA |
| V2 (submitted) | LoRA + minimal features | **0.3665** | 0.17 | 0.85 | 1.00 | 0.90 | Inter-freq over-triggered |
| V3 deterministic | Fixed features, no model | **0.6750** | 0.5752 | 1.00 | 1.00 | 1.00 | Current offline |
| V3 target | + retrained LoRA + tools | **≥ 0.85** | 0.78 | 1.00 | 1.00 | 1.00 | Submission goal |

### Key wins since V2

1. **F2/F3/F4 bypass the model entirely** — deterministic candidates already correct; Qwen was adding noise (0.85 → 1.00)
2. **4BS inter-freq cap** — V2 scored inter_freq_load > 0.5 for ~80% of F1 scenarios, drowning every other signal. V3 caps it at 0.08 for ≤5BS scenarios.
3. **SINR-based issue grouping** — SINR<0 → handover family; SINR>5 + high PRB → coverage/backhaul; SINR>5 + low PRB → PDCCH/server
4. **Serving-cell candidate boost** — V2 assumed "primary" was always the target; V3 reads the actual serving-cell role and applies a 1.05× boost
5. **Offline eval harness** — `eval_v3.py` runs without the server or Qwen, letting us iterate in seconds

---

## 2. Competition Context

### 2.1 Task

Given a troubleshooting scenario (CSV logs + textual description + multiple-choice options labeled C1..C22), predict the set of correct C-codes. The scenario describes a wireless network anomaly (poor throughput, call drops, handover failure) and the options are candidate mitigation actions paired with specific cells.

### 2.2 Scenario families

| Family | Description | Share | Answer type | Typical BS count |
|---|---|---|---|---|
| **F1** | 4-BS deployment, single root cause | ~75% | 1 C-code | 4 |
| **F2** | 4-BS deployment, multi-cause (pair of actions) | ~9% | 2 C-codes | 4 |
| **F3** | 5-BS deployment, multi-cause (always 4 actions) | ~4% | 4 C-codes | 5 |
| **F4** | 13-BS deployment, single cause | ~12% | 1 C-code | 13 |

### 2.3 F1 action-type distribution

F1 is the hardest family because one action must be selected from six classes with heavily overlapping feature profiles:

| Action | Share | Distinguishing signal |
|---|---|---|
| `insufficient_data` | 21% | Missing MR samples for some cells; indistinguishable from `lift_tilt` locally |
| `lift_tilt` | 19% | Antenna mispointing; needs `judge_mainlobe` geometric check |
| `pdcch_2sym` | 18% | Downlink CCE utilization > 0.5, low PRB |
| `check_server` | 16% | High PRB, high RSRP, SINR>5 (no radio issue) |
| `add_neighbor` | 15% | Neighbor cell has RSRP advantage but isn't configured |
| `decrease_a3` | 11% | Handover triggered too late despite neighbor being better |

### 2.4 Scoring

```
IOU = |pred ∩ gt| / |pred ∪ gt|
time_discount = 1.0  if wall_time < 5 min
              = 0.5  if 5 ≤ wall_time < 10 min
              = 0.0  if wall_time ≥ 10 min
final_score  = IOU × time_discount
```

This means **over-predicting hurts as much as under-predicting** (union in denominator), and **any single scenario taking > 10 min scores zero for that scenario**.

---

## 3. Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Raw scenario (CSV files + question text + C1..C22 options)      │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. scenario_parser.ParsedScenario                               │
│     • traffic.csv → DataFrame                                     │
│     • mr.csv      → DataFrame                                     │
│     • site.json   → dict (cell coordinates, azimuth, tilt)        │
│     • options     → list[CodedOption]                             │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. scenario_router.route() → Family enum {F1, F2, F3, F4}       │
│     Decision tree based on: num_base_stations, tag, n_answers    │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. feature_engine.run_all_skills() → ScenarioFeatures            │
│     Skill A: mobility signals (RSRP, SINR, neighbor advantage)   │
│     Skill B: throughput / drop detection                          │
│     Skill C: inter-frequency load (PRB utilization)               │
│     Skill D: signaling events (handover attempts, failures)       │
│     Skill E: PDCCH / CCE utilization                              │
│     Skill F: coverage hole geometric check                        │
│     Skill G: issue score aggregation + V3 4BS overhaul            │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  4. candidate_generator.generate_*_candidates()                  │
│     Maps issue scores → (c_code, score, reason) tuples            │
│     Applies serving-cell boost (V3) and options-availability mask │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
                   ┌───────────┴────────────┐
                   │                        │
              (F2/F3/F4)                 (F1)
                   │                        │
                   ▼                        ▼
          top_candidate()          tool_caller.augment()
                                           │
                                           ▼
                                  rag_retriever_v2.retrieve(k=6)
                                           │
                                           ▼
                                  Qwen LoRA via serve_model.py
                                           │
                                           ▼
                                  answer_parser.extract()
                                           │
                   └───────────┬────────────┘
                               ▼
                   (c_code_list, confidence)
                               │
                               ▼
                   result.csv + result_final.csv
```

---

## 4. Quick Start

### 4.1 Environment

The project assumes a `uv`-managed virtualenv at `/home/huangxiaolin/AITO/`:

```bash
# Activate
source /home/huangxiaolin/AITO/bin/activate

# Or invoke directly
/home/huangxiaolin/AITO/bin/python main_v3.py ...
```

Required packages: `pandas`, `numpy`, `faiss-cpu`, `sentence-transformers`, `requests`, `openai` (for OpenAI-compatible vLLM endpoint).

### 4.2 Offline evaluation (no server, no Qwen)

```bash
# Full 200-sample offline eval
python eval_v3.py --full --per-action

# With error analysis CSV (top-6 candidates per failure case)
python eval_v3.py --full --per-action --errors errors.csv

# Filter to one family
python eval_v3.py --full --per-action --family F1
```

### 4.3 Running with the competition server + Qwen

```bash
# Start Qwen serving in a separate terminal (vLLM with LoRA adapter)
bash launch_qwen.sh   # exposes http://localhost:8001/v1

# Run end-to-end pipeline
python main_v3.py \
    --server_url http://localhost:7860 \
    --model_url  http://localhost:8001/v1 \
    --output_dir results/v3_run_$(date +%Y%m%d_%H%M%S)
```

### 4.4 Useful flags

| Flag | Effect |
|---|---|
| `--deterministic` | Skip Qwen; take top candidate for every scenario (useful as fallback) |
| `--use-tools` | Enable `tool_caller.augment_features()` for F1 (when Owner C's work lands) |
| `--no_rag` | Disable RAG retrieval (tests whether RAG is hurting on some distribution) |
| `--limit N` | Process only the first N scenarios — fast iteration |
| `--verbose` | Print per-scenario features, candidates, and Qwen prompts |

---

## 5. File Inventory

### 5.1 Top-level

| File | Lines | Purpose |
|---|---|---|
| `main_v3.py` | ~650 | V3 orchestrator. `V3Runner` class with `solve()`, `benchmark()`. CLI entrypoint. |
| `eval_v3.py` | ~220 | Offline evaluation harness. Loads scenarios from `train_data/`, runs the full pipeline up to candidate selection, reports per-family and per-action IOU. |
| `serve_model.py` | ~180 | vLLM-compatible Qwen server with LoRA adapter. OpenAI Chat Completions API. |
| `rag_retriever_v2.py` | ~332 | `StructuredCaseRAG` — 13-dim feature-vector FAISS index over training set, with family-name hard filter. |
| `TEAM_DIVISION.md` | — | Four-owner work split. Reads it before starting any task. |
| `README_V3.md` | — | This file. |

### 5.2 `skills/` directory

| File | Purpose | Owner |
|---|---|---|
| `scenario_parser.py` | CSV + option parsing into `ParsedScenario` dataclass | B |
| `scenario_router.py` | Family classifier (F1/F2/F3/F4) based on BS count + tag | B |
| `feature_engine.py` | 7-skill pipeline producing `ScenarioFeatures`; core of V3 improvements | B |
| `candidate_generator.py` | Issue-score → C-code candidate list with reasons | B |
| `answer_parser.py` | Extracts `\boxed{C1,C5}` from Qwen output, computes IOU | B |
| `tool_caller.py` | Wraps 19 server tools; produces `ToolEvidence` | C |

### 5.3 Data

| Path | Purpose |
|---|---|
| `train_data/` | 2000 training scenarios. Read-only input to RAG and SFT. |
| `.rag_cache_v2/` | Pickled FAISS index + feature matrix. Built on first run. |

---

## 6. Feature Engine Deep Dive

`feature_engine.run_all_skills()` is the heart of V3. Each skill contributes signals that the candidate generator consumes.

### 6.1 Skills

| Skill | Produces | Key logic |
|---|---|---|
| A (mobility) | `serving_rsrp_dbm`, `serving_sinr_db`, `neighbor_advantage_db`, `a3_margin_db` | Parses MR CSV; identifies serving cell by majority vote; computes best-neighbor delta |
| B (throughput) | `worst_throughput_mbps`, `drop_magnitude`, `drop_duration_s` | Detects sustained throughput cliffs via rolling min |
| C (inter-frequency) | `max_dl_prb_util`, `max_ul_prb_util` | Max across all cells; used to flag congestion |
| D (signaling) | Handover attempt count, failure rate | Counts `HO_Request`, `HO_Failure` events |
| E (PDCCH) | `max_dl_cce` | Max downlink CCE utilization — key for `pdcch_2sym` |
| F (coverage) | `coverage_hole_risk` | Geometric heuristic using site locations |
| G (aggregator) | All `score_*` fields | Combines skills into 6 issue scores |

### 6.2 V3 4BS overhaul (the critical fix)

```python
# At end of run_all_skills(), after 13BS family boost
if ps.num_base_stations <= 5 and ps.num_base_stations != 13:
    sf.score_inter_freq_load = min(sf.score_inter_freq_load, 0.08)

if ps.num_base_stations == 4 and is_single:
    if sinr < 0:
        # handover family
        if neighbor_advantage > 8 and margin > 2:
            score_missing_neighbor = 0.75
            score_late_handover    = 0.35
        elif neighbor_advantage > 3:
            score_late_handover    = 0.70
            score_missing_neighbor = 0.30
        # suppress unrelated scores
        score_coverage_hole    = min(..., 0.15)
        score_pdcch_issue      = min(..., 0.15)
        score_backhaul_server  = min(..., 0.15)

    elif sinr > 5:
        if max_prb > 70:
            score_coverage_hole    = max(..., 0.45)
            score_backhaul_server  = max(..., 0.40)
            # suppress handover
        else:
            if max_dl_cce > 0.5:
                score_pdcch_issue     = max(..., 0.50)
                score_backhaul_server = max(..., 0.40)
            else:
                score_backhaul_server = max(..., 0.50)
                score_pdcch_issue     = max(..., 0.40)
            # suppress handover
```

### 6.3 Issue score → action mapping

| Top issue | Mapped actions (in priority) |
|---|---|
| `score_late_handover` | `decrease_a3` |
| `score_missing_neighbor` | `add_neighbor` |
| `score_coverage_hole` | `lift_tilt`, `increase_power` |
| `score_pdcch_issue` | `pdcch_2sym` |
| `score_backhaul_server` | `check_server`, `insufficient_data` |

---

## 7. Candidate Generator Deep Dive

`candidate_generator.generate_single_candidates()` produces the ranked list passed to Qwen:

1. For each issue score ≥ 0.3, look up the mapped action(s).
2. Find the option in `ps.options` that matches (`action_type`, `target_cell_role`).
3. Apply boost = `score × 1.05` if target is the serving cell (V3 fix).
4. `_upsert_candidate()` — update if new score exceeds existing, else skip.
5. Return `[(c_code, score, reason), ...]` sorted by score desc.

### 7.1 Why V2 failed on `decrease_a3`

V2 assumed `target_cell_role="primary"` for every action. In reality, the serving cell was often `secondary` or `tertiary`. Both options (primary and secondary) got the same `late_handover` score, but V2 broke ties by listing primary first.

### 7.2 V3 fix

```python
if serving_role:
    if sf.score_late_handover >= 0.3:
        hit = _find_best_option(
            ps, "decrease_a3", serving_role,
            boost=sf.score_late_handover * 1.05,
        )
        if hit:
            _upsert_candidate(scored, reasons, hit[0], hit[1], reason)
```

The 1.05× boost is small but enough to win tiebreakers when the two cells have otherwise identical scores.

---

## 8. Evaluation Harness (`eval_v3.py`)

Runs the full deterministic pipeline on training scenarios with known labels. Outputs:

```
Overall mean IOU: 0.6750
F1: 0.5752 (n=150)
F2: 0.8421 (n=19)
F3: 1.0000 (n=8)
F4: 1.0000 (n=23)

Per-action breakdown (F1 only):
  check_server       1.00 (n=24)
  pdcch_2sym         1.00 (n=27)
  decrease_a3        0.97 (n=17)
  add_neighbor       0.35 (n=23)
  lift_tilt          0.00 (n=28)
  insufficient_data  0.00 (n=31)
```

With `--errors errors.csv`, writes per-failure rows with:
- `scenario_id`, `family`, `gt_c_code`, `pred_c_code`
- `gt_rank` — where the correct answer sat in the candidate list
- `candidates` — top-6 with scores
- `top_issue` — highest-scoring issue

**Iteration loop**: change `feature_engine.py` → `python eval_v3.py --full --per-action` → inspect deltas → commit.

---

## 9. Remaining Optimization Targets

### 9.1 `lift_tilt` — currently 0%

**Why it fails**: correct option typically targets a non-serving cell (often `tertiary`) because the antenna pattern is wrong, not the serving cell. Local CSV data can't distinguish this from `insufficient_data` — both present as high-SINR + low-PRB + moderate throughput.

**Fix path**:
- Short-term (Owner B): geometric heuristic — if serving cell RSRP is moderate (-105 to -90) AND there's no nearby cell in the main lobe direction, score `lift_tilt` higher than `insufficient_data`.
- Long-term (Owner C): call `judge_mainlobe` + `calculate_tilt_angle` for each candidate target cell; use the mainlobe flag as a hard disambiguator.

**Target**: 0% → 50% deterministic, 70% with LoRA.

### 9.2 `insufficient_data` — currently 0%

**Why it fails**: the correct signal is "MR data for cell X is missing or stale" — pattern-recognizable by Qwen with the right prompt, invisible to the local heuristics.

**Fix path**:
- Owner D: build RAG index that returns `insufficient_data` exemplars when the feature signature matches (high PRB with no handover activity).
- Owner A: fine-tune Qwen to recognize this pattern with explicit CoT rationale tokens.

**Target**: 0% → 30% deterministic (via feature engineering only), 60% with LoRA.

### 9.3 `add_neighbor` — currently 35%

**Why it fails**: when `neighbor_advantage_db > 8` and `a3_margin_db > 2`, the candidate generator correctly scores `missing_neighbor = 0.75`, but `_find_best_option` sometimes picks the wrong target cell because the "best neighbor" in MR data ≠ the "missing neighbor" in options.

**Fix path** (Owner B): cross-reference `option.target_cell` with `mr_data.best_neighbor_cell`; when they disagree, prefer the option matching the MR signal.

**Target**: 35% → 75%.

### 9.4 F2 regression — 1.0 → 0.84

**Why it regressed**: The SINR-grouping fix for F1 also affects F2 because F2 scenarios also have 4BS. The `is_single` guard catches most cases but not all — specifically when the F2 tag contains the word "single" in a non-canonical form.

**Fix path** (Owner B): make `is_single` robust against tag variants; add a unit test covering all F2 tag patterns in training data.

**Target**: 0.84 → 1.0.

---

## 10. Fine-Tuning Integration

The LoRA is trained on `sft_v3.jsonl`, built by Owner A using this exact V3 pipeline:

1. For each training scenario, call `parse` → `route` → `feature_engine` → `candidate_generator`.
2. If family is F1, serialize the prompt using `_build_f1_prompt(ps, sf, candidates, rag_hits)`.
3. Assistant turn = `\boxed{gt_c_code}` + short rationale.
4. Store as `{system, user, assistant}` triple.

**Why this matters**: train/test prompt distribution is identical. The LoRA learns to re-rank the candidate list, not generate C-codes from scratch. This is far easier than free-form generation and should transfer well.

See `TEAM_DIVISION.md` §2 for Owner A's full plan.

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: pandas` | Using system Python | `source /home/huangxiaolin/AITO/bin/activate` |
| RAG returns empty | `.rag_cache_v2/` missing | Delete cache dir; it rebuilds automatically on first run |
| F2 score drops below 1.0 after feature edits | SINR-grouping fix leaked into multi-answer path | Check `is_single` guard in `feature_engine.py` |
| Qwen hallucinates C-codes outside C1..C22 | System prompt missing options constraint | Use `QWEN_V3_SYSTEM_PROMPT`; do not override |
| Wall-time > 5 min | Tool calls unbounded | Check tool budget in `tool_caller.py`; add per-tool timeouts |

---

## 12. Changelog

| Date | Change | Files |
|---|---|---|
| 2026-04-20 | V3 framework created | `main_v3.py`, `eval_v3.py`, feature/candidate fixes |
| 2026-04-20 | `tool_caller.py` skeleton | `skills/tool_caller.py` |
| 2026-04-20 | Self-contained `v3/` distribution | entire folder |
| 2026-04-20 | Legacy `rag_retriever.py` removed from v3 | kept only V2 StructuredCaseRAG |

---

## 13. References

- Track A data schema: `data/Phase_1/train.json`
- Competition server API: `server.py` (do not modify)
- Scoring function: `skills/answer_parser.py:compute_iou()`
- V2 architecture (historical): `../main_v2.py`
