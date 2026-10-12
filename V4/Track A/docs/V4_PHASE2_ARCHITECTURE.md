## 1. System Architecture

### 1.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2 test.json  (500 scenarios, data fields are stubs)       │
└─────────────────────────────┬───────────────────────────────────┘
                              │ scenario_id + task description
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  APIDataGatherer  (skills/api_data_gatherer.py)                  │
│  4 core calls to competition server:                             │
│    GET /user-plane-data   → user_plane_data CSV + worst_time     │
│    GET /config-data       → network_configuration_data CSV       │
│    GET /get_kpi_data      → traffic_data CSV                     │
│    GET /get_mr_data       → mr_data CSV                          │
│  0–1 optional:                                                   │
│    GET /signaling-plane-event-log  (skipped: always empty)       │
│  0–2 conditional disambiguation calls:                           │
│    GET /judge_mainlobe?pci=…&time=…                              │
│    GET /calculate_overlap_ratio?pci_serving=…&pci_neighbor=…     │
└─────────────────────────────┬───────────────────────────────────┘
                              │ raw API JSON
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ScenarioParser  (skills/scenario_parser.py)                     │
│  • Reconstructs 5 pipe-CSV DataFrames from raw JSON              │
│  • Parses 22+ option labels → action_type, target_cell, cells    │
│  • Infers cell roles (primary / secondary / tertiary)            │
│  • Builds pci ↔ cell_id maps                                     │
│  → ParsedScenario                                                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ScenarioRouter  (skills/scenario_router.py)                     │
│  + Phase 2 family override in phase2_run.py                      │
│                                                                  │
│  BS=4, single-answer  → F1    (classic 4-BS troubleshoot)        │
│  BS=4, multi-answer   → F2    (2-action composite fix)           │
│  BS=5, single-answer  → F1_5BS (Phase 2 new; uses F1 logic)      │
│  BS=5, multi-answer   → F3    (4-action multi-select)            │
│  BS=13, single-answer → F4    (inter-frequency/load)             │
│  BS=13, multi-answer  → F4M   (Phase 2 new; 13-BS pair)          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FeatureEngine  (skills/feature_engine.py)                       │
│  run_all_skills(ps) runs 7 skills:                               │
│    mobility  → serving RSRP/SINR, neighbor advantage, A3 margin  │
│    throughput → worst throughput, drop magnitude/duration        │
│    inter-freq → max PRB utilization per cell                     │
│    interference → SINR anomaly detection                         │
│    PDCCH     → max downlink CCE utilization                      │
│    geometry  → coverage hole risk                                │
│    aggregator → 9 issue scores in [0, 1]                         │
│  → ScenarioFeatures                                              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼  (Phase 2) conditional tool checks
┌─────────────────────────────────────────────────────────────────┐
│  Phase2ToolChecks + ToolCaller                                   │
│  _apply_phase2_tool_checks() in phase2_run.py builds             │
│  ToolEvidence, then reuses ToolCaller.augment_features():        │
│                                                                  │
│  Conditional extra calls:                                        │
│    judge_mainlobe_or_not → Outside_Mainlobe_Flag                 │
│    calculate_overlap_ratio → inter-cell overlap                  │
│                                                                  │
│  No extra HTTP call for MR/KPI: they are already fetched by      │
│  APIDataGatherer and reused from ParsedScenario.                 │
│  Signaling is optional via --fetch_signaling and currently       │
│  disabled because live responses are empty.                      │
│                                                                  │
│  ToolCaller.augment_features() adjusts ScenarioFeatures:         │
│    mainlobe=True  → score_insufficient_data = max(cur, 0.78)     │
│    mainlobe=False → no-op  (Phase 2 fix: constant False ignored) │
│    mr_count < 7   → score_insufficient_data = max(cur, 0.82)     │
│    overlap > 0.3  → score_overshoot_overlap = max(cur, 0.55)     │
│  → ScenarioFeatures (augmented)                                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  CandidateGenerator  (skills/candidate_generator.py)             │
│  Issue score → C-code ranking:                                   │
│                                                                  │
│  Score              → Action                                     │
│  late_handover      → decrease_a3 (serving cell)                 │
│  missing_neighbor   → add_neighbor (best unconfig'd neighbor)    │
│  coverage_hole      → lift_tilt / increase_power                 │
│  overshoot_overlap  → press_down_tilt / decrease_power           │
│  pdcch_issue        → pdcch_2sym                                 │
│  backhaul_server    → check_server                               │
│  insufficient_data  → insufficient_data                          │
│  inter_freq_load    → decrease_interfreq                         │
│                                                                  │
│  F2/F3/F4/F4M: template priors; F1/F1_5BS: scored candidates    │
│  → List[RankedCandidate]                                         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               │                             │
         F2/F3/F4/F4M                  F1/F1_5BS
         top_candidate()          agentic_overlay (high-risk only)
         → format answer          → controlled 2-call LLM loop
               │                             │
               │           ┌─────────────────┘
               │           │
               │           ▼
               │    AgenticOverlay  (skills/agentic_overlay.py)
               │    Triggers only when risk level = "high":
               │      Branch 1: lift_tilt_vs_insufficient_data
               │        tool menu: judge_mainlobe_or_not,
               │                   calculate_tilt_angle,
               │                   get_user_location
               │      Branch 2: add_neighbor_vs_decrease_a3
               │        tool menu: get_neighboring_cells_pci,
               │                   get_neighboring_cell_rsrp,
               │                   get_cell_info
               │    Step 1: LLM chooses one tool OR skips
               │    Step 2: LLM produces structured diagnosis
               │    Maps diagnosis → C-code or falls back to
               │    deterministic answer
               │           │
               └─────┬─────┘
                     │
                     ▼  (optional, currently "audit" mode)
              FineTunedVerifier
              phase2_run._apply_fine_tuned_verifier()
              Policy "audit": logs LoRA answer, keeps deterministic
              Policy "gate":  allows low-risk gated overrides
              NOTE: v4_fast_balanced IOU=0 on lift_tilt and
              insufficient_data — do not promote to "gate" or "always"
                     │
                     ▼
              diagnostics dict → result.csv
              (ID, Track A, Track B)
```

### 1.2 Phase 2 Family Routing Matrix

| BS | tag | Family | Answer count | Treatment |
|---|---|---|---|---|
| 4 | single-answer | F1 | 1 | Feature scores + agentic overlay |
| 4 | multiple-answer | F2 | 2 | Deterministic pair template |
| 5 | single-answer | F1_5BS | 1 | F1 logic; current candidate predicts `press_down_tilt`; evidence-supported, no Phase 2 GT |
| 5 | multiple-answer | F3 | 4 | Fixed 4-action template |
| 13 | single-answer | F4 | 1 | decrease_interfreq prior |
| 13 | multiple-answer | F4M | 2 | 13-BS inter-frequency pair template |

### 1.3 Phase 2 F1 Issue Score -> Action Map

| Score field | Triggered when | Maps to action |
|---|---|---|
| `score_late_handover` | SINR < 0, neighbor advantage > 3 dB | `decrease_a3` |
| `score_missing_neighbor` | Best neighbor not in config, advantage > 3 dB | `add_neighbor` |
| `score_coverage_hole` | Low SINR + low RSRP / geometry-power evidence after ignoring constant mainlobe=False | `lift_tilt` or `increase_power` |
| `score_overshoot_overlap` | Overlap ratio > 0.3 between serving and neighbor | `press_down_tilt` / `decrease_power` |
| `score_pdcch_issue` | High downlink CCE utilization | `pdcch_2sym` |
| `score_backhaul_server` | KPI anomaly, no radio fault | `check_server` |
| `score_insufficient_data` | MR count < 7 OR `Outside_Mainlobe_Flag=True` | `insufficient_data` |
| `score_inter_freq_load` | High PRB utilization (13-BS) | `decrease_interfreq` |

### 1.4 Agentic Overlay Risk Triggers

The overlay fires only when `risk["level"] == "high"`. Trigger conditions:

**Branch `lift_tilt_vs_insufficient_data`**:
```
coverage_hole > 0.20 AND insufficient_data > 0.20
  AND ( |coverage_hole − insufficient_data| < 0.25
        OR (single-answer AND insufficient_data ≥ 0.65 AND coverage_hole ≥ 0.20) )
```

**Branch `add_neighbor_vs_decrease_a3`**:
```
missing_neighbor > 0.20 AND late_handover > 0.20
  AND ( |missing_neighbor − late_handover| < 0.25
        OR (single-answer AND missing_neighbor ≥ 0.25 AND late_handover ≥ 0.35) )
```

### 1.5 Key Phase 2 Code Fix (2026-05-07)

**Problem**: `judge_mainlobe_or_not` returns `Outside_Mainlobe_Flag=False` for all observed Phase 2 coverage-like probes (constant output). The old `augment_features()` treated `False` as a signal to boost `coverage_hole → lift_tilt`, which over-selected `lift_tilt` in 161 cases.

**Fix in `skills/tool_caller.py`**:
```python
# OLD (buggy):
if serving_flag is False:          # constant False in observed Phase 2 probes → always boosted
    sf.score_coverage_hole = max(sf.score_coverage_hole, 0.74)
    # ... forced lift_tilt

# NEW (fixed):
if serving_flag is True:           # only non-constant signal
    sf.score_insufficient_data = max(current_insufficient, 0.78)
    sf.score_coverage_hole = min(sf.score_coverage_hole, 0.25)
# serving_flag is False → no-op; let raw feature scores decide

# Also fixed MR count guard:
if evidence.mr_sample_count < self.mr_sample_threshold \
        and serving_flag is not True:   # was "not False" — always blocked in Phase 2
    sf.score_insufficient_data = max(..., 0.82)
```

**Effect confirmed by probe** (28 high-risk scenarios): the previously forced coverage cases now show `insufficient_data/0.780` instead of `coverage_hole/0.74`. The archived `result_full_uncertainfix.csv` snapshot was generated with this fix applied.

---

## 2. Key Files

| File | Owner | Purpose |
|---|---|---|
| `phase2_run.py` | A | Phase 2 main runner; orchestrates all stages; writes result CSV |
| `phase2_audit.py` | A | Pre-submission validator; checks format, coverage, cardinality |
| `phase2_select_probe.py` | A | Selects representative probe IDs for live testing |
| `main.py` | — | Official Phase 2 Environment class; API HTTP layer |
| `skills/api_data_gatherer.py` | A | Fetches 4 core data sources + optional disambiguation calls |
| `skills/agentic_overlay.py` | A/C | Controlled 2-call LLM loop for high-risk F1 branches; consumes B/D artifacts |
| `skills/scenario_parser.py` | B | Raw JSON/CSV → ParsedScenario |
| `skills/scenario_router.py` | B | (tag, BS count) → Family |
| `skills/feature_engine.py` | B | ParsedScenario → ScenarioFeatures (9 issue scores) |
| `skills/candidate_generator.py` | B | ScenarioFeatures → ranked C-code candidates |
| `skills/tool_caller.py` | B | Tool evidence → augmented ScenarioFeatures |
| `skills/answer_parser.py` | B | C-code extraction and IOU scoring |
| `skills/data_tools.py` | D | CSV utilities and scenario analysis helpers |
| `skills/rf_analysis.py` | B | RF / path-loss helpers |
| `rag_retriever_v2.py` | D | Trajectory RAG / memory retrieval over Phase 1 and probe traces |
| `build_sft_v3.py` | A | SFT dataset builder for Phase 1 LoRA training |
| `curate_sft.py` | A | SFT curation, balancing, and hard-negative preparation |
| `eval_ckpt.py` | A | LoRA checkpoint heldout evaluation |

---

## 3. Workflow

### 3.1 Full Phase 2 Run

```bash
# Model server (Qwen3.5-35B-A3B + LoRA, runs separately)
CUDA_VISIBLE_DEVICES=0,1 \
NO_PROXY=127.0.0.1,localhost \
/home/huangxiaolin/dify_vllm_uv310/bin/vllm serve /home/huangxiaolin/AITO/model \
  --host 0.0.0.0 --port 8001 --served-model-name qwen \
  --tensor-parallel-size 2 --enable-lora \
  --lora-modules v4=/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track\ A/checkpoints/v4_fast_balanced \
  --max-lora-rank 32 --gpu-memory-utilization 0.82 \
  --max-model-len 8192 --reasoning-parser qwen3 --language-model-only

# Phase 2 full run (500 scenarios). If proxy variables are set in the shell,
# unset them here so live-server requests use the intended route.
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    AGENT_API_KEY=<AGENT_API_KEY> \
    NO_PROXY=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
    no_proxy=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
python phase2_run.py \
  --server_url  https://120.46.145.77/no \
  --test_json   data/Phase_2/test.json \
  --out         results/phase2/current_submission/result_full_uncertainfix_next.csv \
  --model_url   http://127.0.0.1:8001/v1 \
  --llm_policy  audit \
  --agentic_policy high_risk \
  --save_freq   10
```

### 3.2 Representative Probe (18 scenarios)

```bash
# Select probe IDs
python phase2_select_probe.py \
  --test_json data/Phase_2/test.json \
  --out_ids   results/phase2/probe_selection/probe_ids.txt \
  --per_group 3

# Run probe
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    AGENT_API_KEY=<AGENT_API_KEY> \
    NO_PROXY=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
    no_proxy=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
python phase2_run.py \
  --server_url https://120.46.145.77/no \
  --test_json  data/Phase_2/test.json \
  --ids        results/phase2/probe_selection/probe_ids.txt \
  --out        results/phase2/agentic_overlay_tests/probe_agentic_representative_next.csv \
  --model_url  http://127.0.0.1:8001/v1 \
  --llm_policy audit --agentic_policy high_risk
```

### 3.3 High-Risk Probe (28 scenarios)

```bash
# Run against pre-selected high-risk IDs
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    AGENT_API_KEY=<AGENT_API_KEY> \
    NO_PROXY=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
    no_proxy=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
python phase2_run.py \
  --server_url https://120.46.145.77/no \
  --test_json  data/Phase_2/test.json \
  --ids        results/phase2/agentic_overlay_tests/high_risk_ids.txt \
  --out        results/phase2/agentic_overlay_tests/high_risk_agentic_probe_next.csv \
  --model_url  http://127.0.0.1:8001/v1 \
  --llm_policy audit --agentic_policy high_risk
```

### 3.4 Pre-Submission Audit

```bash
python phase2_audit.py \
  --test_json  data/Phase_2/test.json \
  --result_csv results/phase2/current_submission/result_full_uncertainfix_next.csv
```

### 3.5 LLM Verifier Policies

| `--llm_policy` | Behavior | Recommended |
|---|---|---|
| `off` | Deterministic only, no LLM | Fast reproduction |
| `audit` | Calls LoRA, logs answer, keeps deterministic | **Current Phase 2 default** |
| `gate` | Allows low-risk gated overrides | Only after checkpoint passes heldout gates |
| `always` | Always uses LoRA answer | Debug only |
| `arbitrate` | LLM+RAG as final arbiter for high-risk cases | Not yet enabled (RAG not integrated) |

### 3.6 Key CLI Flags

| Flag | Effect |
|---|---|
| `--agentic_policy high_risk` | Enable agentic overlay for high-risk F1 branches |
| `--llm_policy audit` | Log LoRA answer but keep deterministic result |
| `--fetch_signaling` | Fetch signaling events (disabled: returns empty) |
| `--no_rag` | Disable RAG retrieval |
| `--save_freq N` | Checkpoint output CSV every N scenarios |
| `--ids file.txt` | Run only specified scenario IDs (probe mode) |
| `--token TOKEN` | Competition server bearer token (or set `AGENT_API_KEY`) |

---

## 4. Owner Interface Summary

Detailed team ownership, dates, and deliverables live in `PHASE2_TEAM_DIVISION.md`. This architecture document keeps only the technical integration contract:

```text
B -> A/C/D: branch_spec + evidence_schema + tool_menu
C -> A: prompt/CoT/loop package
D -> A/C: retrieve_trajectory_context() + trajectory examples
A -> all: probe/full-run diagnostics when needed for iteration
```

---

## 5. Current Submission State (2026-05-07)

### 5.1 Archived Candidate Snapshot

```
/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track A/results/phase2/current_submission/result_full_uncertainfix.csv
```

| Metric | Value |
|---|---|
| Total rows | 500 / 500 |
| Hard errors | 0 |
| API errors | 0 |
| Top-candidate score floor | 0.30 |
| Mean run latency | ~11 s/scenario |
| Families covered | F1, F1_5BS, F2, F3, F4, F4M |

### 5.2 Phase 2 F1 Action Distribution (after fix)

| Top issue | Count | Primary action |
|---|---|---|
| overshoot_overlap | 89 | press_down_tilt |
| coverage_hole | 102 | increase_power (73) + lift_tilt (29) |
| insufficient_data | 30 | insufficient_data |
| late_handover | 18 | decrease_a3 |
| missing_neighbor | 2 | add_neighbor |
| F1_5BS (separate) | 28 | press_down_tilt |

### 5.3 Remaining Risk Classes

| Risk | Assessment | Decision |
|---|---|---|
| F1_5BS all → `press_down_tilt` | Probe-stable and evidence-supported; serving SINR 1–5 dB + strong RSRP + low PRB supports overlap diagnosis, but Phase 2 GT is unavailable | Accept for first submission; revisit after leaderboard feedback |
| 73 coverage_hole → `increase_power` | Raw score driven; no mainlobe evidence to prefer `lift_tilt`; Phase 2 GT unknown | Accept; revisit after leaderboard feedback |
| 7 F2 all-zero local score rows | Probe showed overlap 0.02–0.29 (low); switching would be a guess | Accept |
| LoRA v4 failing lift_tilt/insufficient_data | IOU=0.0 on both classes; not used as verifier | Keep `audit` policy |

### 5.4 Agentic Overlay Status

- 28 high-risk probes: **0 errors, 0 overrides**.
- 0 overrides is **expected for the current candidate** — the fixed deterministic pipeline already predicts `insufficient_data` for these cases and the LLM confirms the same diagnosis.
- Overlay is functioning as a validation layer. It will produce overrides only when the LLM's evidence points to a different diagnosis than the deterministic prediction.
- Next improvement: add trajectory RAG to the overlay prompt to give the LLM additional context for genuinely ambiguous cases.

---

## 6. Related Documents

| Document | Purpose |
|---|---|
| `README.md` | Documentation index and reading order |
| `PHASE2_PROJECT.md` | Project overview and current candidate summary |
| `PHASE2_TEAM_DIVISION.md` | A/B/C/D ownership, timeline, and submission budget |

## 7. Glossary

| Term | Definition |
|---|---|
| **C-code** | One of C1..C24 in the current Phase 2 data, a coded option in the scenario's multiple-choice list |
| **F1/F2/F3/F4/F1_5BS/F4M** | Scenario families (see §1.2 routing matrix) |
| **ScenarioFeatures** | Output of `feature_engine.run_all_skills()`; contains 9 issue scores |
| **ToolEvidence** | Structured results from conditional server API calls |
| **ParsedScenario** | Output of `scenario_parser`; DataFrames + parsed option objects |
| **RankedCandidate** | Tuple of (c_code, score, reason) from `candidate_generator` |
| **IOU** | Intersection-over-union: `|pred ∩ gt| / |pred ∪ gt|` |
| **Time discount** | Score multiplier: <5 min → 1.0, <10 min → 0.5, ≥10 min → 0.0 |
| **Agentic overlay** | Controlled 2-call LLM loop that may override the deterministic answer for high-risk cases |
| **Audit mode** | LoRA policy that logs the model's answer but uses the deterministic result |
| **MR data** | Measurement Reports — UE-side radio measurements per cell |
| **Outside_Mainlobe_Flag** | API result: True = UE is outside antenna main lobe; False = inside |
| **Worst time** | Timestamp of lowest throughput; used as anchor for all API queries |
| **PRB utilization** | Physical Resource Block usage rate; high = radio congestion |
| **CCE utilization** | Downlink Control Channel Element usage; high = PDCCH bottleneck |
| **A3 event** | 3GPP handover trigger when neighbor signal exceeds serving cell by A3 margin |
