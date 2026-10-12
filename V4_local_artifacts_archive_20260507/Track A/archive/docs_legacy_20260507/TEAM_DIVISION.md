# V3 Team Division of Labor

**Competition**: Telco Troubleshooting Agentic Challenge — Track A (Wireless)
**Phase 1 Deadline**: 2026-05-04 (≈2 weeks from D-day = 2026-04-20)
**Base Model**: Qwen3.5-35B-A3B (LoRA only; cannot swap base)

### Current baselines

| Version | Overall | F1 | F2 | F3 | F4 | Notes |
|---|---|---|---|---|---|---|
| V2 submission | 0.3665 | 0.17 | 0.85 | 1.00 | 0.90 | Fine-tuning only, no feature engine |
| V3 deterministic (today) | 0.6750 | 0.5752 | 1.00 | 1.00 | 1.00 | Feature engine + no model |
| V3 deterministic + tools (audited) | 0.8500 F1-heldout only | 0.8500 | — | — | — | Strongest current baseline; add_neighbor gate still fails |
| **V3 target** | **≥ 0.85** | **≥ 0.78** | **1.00** | **1.00** | **1.00** | With retrained LoRA + tool evidence |

### Why we split 4 ways

The V3 pipeline has four loosely-coupled layers:
1. Feature / candidate layer (local, deterministic)
2. Tool-evidence layer (server-side, authoritative)
3. Retrieval layer (RAG over 2000 training cases)
4. Model layer (Qwen LoRA fine-tuning)

Each layer has an independent optimization axis and minimal shared state once interfaces are frozen. This lets four people work in parallel through the remaining two weeks without stepping on each other.

---

## 1. Architecture → Ownership map

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  raw scenario                                                      │
│        │                                                           │
│        ▼                                                           │
│  scenario_parser ──► ParsedScenario                                │
│        │                                                           │
│        ▼                                                           │
│  scenario_router ──► Family {F1|F2|F3|F4}                          │
│        │                                                           │
│        ▼                                                           │
│  feature_engine ──► ScenarioFeatures                 [Owner B]     │
│        │                                                           │
│        ▼                                                           │
│  candidate_generator ──► [(c_code, score, reason), ...][Owner B]   │
│        │                                                           │
│        ├─────────────┬──────────────┐                             │
│        │             │              │                             │
│      F2/F3/F4       F1 (~75%)      F1 (fallback)                   │
│        │             │              │                             │
│        ▼             ▼              ▼                             │
│   top_cand()    tool_caller    top_cand()           [Owner C]     │
│                      │                                             │
│                      ▼                                             │
│                 ToolEvidence                                       │
│                      │                                             │
│                      ▼                                             │
│                 ScenarioFeatures++                                 │
│                      │                                             │
│                      ▼                                             │
│                 rag_retriever_v2 ──► RAGHit[k=6]     [Owner D]     │
│                      │                                             │
│                      ▼                                             │
│                 _build_f1_prompt()                                 │
│                      │                                             │
│                      ▼                                             │
│                 Qwen LoRA (vLLM)                     [Owner A]     │
│                      │                                             │
│                      ▼                                             │
│                 answer_parser ──► c_code_list        [Owner B]     │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Summary table

| Owner | Layer | Files | Measurable goal |
|---|---|---|---|
| **A (me)** | Fine-tuning + SFT data curation + submission | `build_sft_v3.py`, `curate_sft.py`, LoRA config, `main_v3.py` integration | F1 IOU ≥ 0.78 with LoRA |
| **B** | Features + candidates + parsers | `skills/feature_engine.py`, `skills/candidate_generator.py`, `skills/scenario_parser.py`, `skills/scenario_router.py`, `skills/answer_parser.py` | F1 deterministic ≥ 0.70; F2/F3/F4 ≥ 0.98 |
| **C** | Server tool integration | `skills/tool_caller.py` | lift_tilt ≥ 0.60, insufficient_data ≥ 0.60 |
| **D** | Inference-time RAG + data-analysis tooling | `rag_retriever_v2.py`, `skills/data_tools.py` | 60%+ RAG hit-rate on lift_tilt / insufficient_data probes |

---

## 2. Owner A — Fine-tuning & Submission *(me)*

### 2.1 Responsibility Charter

Own everything from "raw training scenarios" to "result submitted". This includes **SFT data curation** (using Owner D's analysis tooling), dataset construction, LoRA training, checkpoint selection, integration with `main_v3.py`, and the actual upload to the competition server.

**Why A owns curation, not D**: curation = deciding which scenarios go into `sft_v3.jsonl`. That is a modeling decision tied to class balance, hard negatives, and val-split hygiene — all of which A already owns. D provides the tooling (dedup scoring, outlier detection, label-error flagging) as a library; A calls it.

### 2.2 Entry conditions (what I need from others before starting)

| From | What | Deadline |
|---|---|---|
| Owner B | Frozen `ScenarioFeatures` schema | D+2 (2026-04-22) |
| Owner B | `feature_engine.py` stable enough that feature vectors don't shift | D+2 |
| Owner C | Initial `ToolEvidence` schema (can be empty values for pass 1) | D+2 |
| Owner D | `retrieve(sf, k=6) → list[RAGHit]` signature | D+2 |
| Owner D | `data_tools.py` — dedup, outlier, label-error scoring functions | D+3 |

### 2.3 Deliverables

| # | Artifact | Detailed spec | ETA |
|---|---|---|---|
| A1a | `curate_sft.py` | Consumes D's `data_tools.py` scores. Decides inclusion: drops duplicates (cosine < 0.02), drops obvious label errors, down-weights ambiguous cases. Writes `curated_train_ids.txt` + `curation_report.md` with counts dropped per reason. | Done; current report used fallback scoring |
| A1b | `build_sft_v3.py` | Iterates `curated_train_ids.txt` → for each F1 scenario builds `(system, user, assistant)` using `_build_f1_prompt()`. Writes `sft_v3.jsonl`. Applies class-balanced upsampling (§2.5). Outputs histogram. | Done |
| A2 | LoRA training config | `llama-factory` YAML: r=32, alpha=64, dropout=0.05, target_modules={q,k,v,o,gate,up,down}, lr=1e-4 cosine, 3% warmup, bs=32, epochs=3, max_len=4096. DeepSpeed zero-2. | Done |
| A3 | First SFT checkpoint | Trained on current feature set (no tool evidence). Benchmark F1 ≥ 0.70. Save checkpoints every 500 steps. Pick best on val split. | Done; target missed (`0.6050` best LoRA-only) |
| A4 | Retrain with tool evidence | After Owner C lands C4. Expected delta: F1 0.70 → 0.75. | Pending; do only after C interface is stable |
| A5 | Retrain with full RAG + confusion pairs | After Owner D lands D3. Expected delta: F1 0.75 → 0.78. | Pending; first require D ablation on tools baseline |
| A6 | Submission pipeline | `submit.py` — runs `main_v3.py` against eval server, writes `result_final.csv`, uploads via competition portal. Includes dry-run mode. | Dry-run done; real upload not configured |
| A7 | Ablation report | CSV: {checkpoint × per-action IOU × wall-time}. Proves which components move which numbers. | Done for current evidence |

### 2.3.1 Current A posture — integration guard, not blind retraining

As of 2026-04-26, A's SFT/LoRA loop has been executed, but longer training on
the same data is not the highest-ROI next step. The strongest audited baseline is
`deterministic + tools = 0.8500` on the 200-scenario F1 heldout split, while
LoRA-only tops out at `0.6050`.

A is therefore in integration-guard mode:
- Preserve the audited `deterministic + tools` baseline.
- Run `owner_a_verify.py` after any B/C/D handoff.
- Retrain LoRA only after B/C/D changes the input distribution: candidate set,
  tool evidence schema/values, RAG evidence, or prompt format.
- Keep `ckpt-477 fixed` and `corrected-246 fixed` as LoRA-only references, not
  as the active path to the next score jump.

### 2.4 Training data construction spec

```python
# build_sft_v3.py pseudocode
for scenario_id in train_ids:
    ps = parse(scenario_id)
    family = route(ps)
    if family != Family.F1:
        continue                           # F1 only
    sf = run_all_skills(ps)
    candidates = generate_single_candidates(ps, sf)[:6]
    rag_hits = rag.retrieve(sf, k=6)

    system = QWEN_V3_SYSTEM_PROMPT         # frozen prompt
    user   = _build_f1_prompt(ps, sf, candidates, rag_hits)
    gt     = ground_truth[scenario_id]     # one C-code for F1
    rationale = _build_rationale(sf, gt)   # e.g. "SINR=6.2, PRB=82% → coverage_hole → lift_tilt/secondary"
    assistant = f"\\boxed{{{gt}}}\n# {rationale}"

    write_jsonl({system, user, assistant})
```

Rationale tokens improve learning; they can be stripped at inference for latency.

### 2.5 Class-balanced sampling

F1 action histogram is imbalanced (21/19/18/16/15/11). Upsample the three the deterministic pipeline fails on:

| Action | Raw % | Sampled % | Rationale |
|---|---|---|---|
| lift_tilt | 19 | **30** | Deterministic 0% — critical to shift the needle here |
| insufficient_data | 21 | **25** | Deterministic 0% |
| add_neighbor | 15 | **18** | Deterministic 35% — moderate upsample |
| pdcch_2sym | 18 | 12 | Deterministic 100% — don't waste capacity |
| check_server | 16 | 10 | Deterministic 100% |
| decrease_a3 | 11 | 5 | Deterministic 97% |

Hard-negative pairs: for every `lift_tilt` positive, include 2 `insufficient_data` negatives (and vice versa) — forces the model to learn the boundary.

### 2.6 Evaluation protocol

Gates before promoting a checkpoint:
1. F1 IOU ≥ 0.75 on 200-sample held-out set (no overlap with train)
2. Per-action: lift_tilt ≥ 0.60, insufficient_data ≥ 0.60, add_neighbor ≥ 0.65
3. F2/F3/F4 unchanged (they bypass the model; should stay 1.0)
4. Mean wall-time per F1 scenario ≤ 90 seconds

Tool: `python eval_v3.py --full --per-action --model http://localhost:8001/v1 --lora ckpt_xxx`.

### 2.7 Non-goals

- Will not modify `feature_engine.py`, `candidate_generator.py`, `tool_caller.py`, or `rag_retriever_v2.py`. If I see a bug, I open an issue with a minimal repro for the respective owner.
- Will not touch the F2/F3/F4 deterministic paths — they're already 1.0.
- Will not re-engineer the base model or try non-LoRA adapters — scope creep.

### 2.8 Risks I own

| Risk | Mitigation |
|---|---|
| SFT overfits to RAG exemplars | Stratified train/val split; early-stop on val perplexity |
| Prompt drift between train and inference | `_build_f1_prompt()` is a single function used in both paths |
| Feature schema churn invalidates the dataset | Freeze at D+2 (B1 gate); block further changes |
| LoRA inference too slow | Measure tokens/s at D+3; if > 30s per scenario, switch to vLLM with AWQ quantization |

---

## 3. Owner B — Feature Engineering & Deterministic Pipeline

### 3.1 Responsibility Charter

Own the backbone: everything from raw scenario → candidate list. This layer determines both the deterministic baseline AND the input distribution for Owner A's LoRA. When this is correct, F2/F3/F4 are free (100%) and F1 becomes a clean re-ranking task.

### 3.2 Files owned

| File | Current state | Key functions |
|---|---|---|
| `skills/feature_engine.py` | ~1200 lines, recently overhauled with V3 4BS fix | `run_all_skills()`, `_get_max_dl_cce()`, all `skill_*` functions |
| `skills/candidate_generator.py` | ~600 lines, V3 serving-cell boost added | `generate_single_candidates()`, `generate_multi_candidates()`, `_find_best_option()`, `_upsert_candidate()` |
| `skills/scenario_parser.py` | Stable | `parse()`, `ParsedScenario`, `CodedOption` |
| `skills/scenario_router.py` | Stable | `route()`, `Family` enum |
| `skills/answer_parser.py` | Stable | `extract_boxed()`, `compute_iou()` |

### 3.3 Priority bugs

| Pri | Bug | Current → Target |
|---|---|---|
| P0 | `add_neighbor` cell-role targeting misses 65% of cases — picks primary when tertiary is correct | 0.35 → 0.75 |
| P0 | F2 regressed from 1.0 to 0.84 after SINR-grouping fix leaked into multi-answer path | 0.84 → 1.00 |
| P1 | `lift_tilt` indistinguishable from `insufficient_data` with local data only | 0% → 40% (rest via tools) |
| P2 | Feature schema needs hard freeze so Owner A's dataset doesn't invalidate | Freeze at D+2 |

### 3.4 Deliverables

| # | Artifact | Detailed spec | ETA |
|---|---|---|---|
| B1 | Schema freeze | Lock `ScenarioFeatures` dataclass. Any new field requires explicit team sign-off + Owner A retrain. Document this in `SCHEMA.md`. | D+2 |
| B2 | Fix add_neighbor targeting | In `_find_best_option`, cross-check `option.target_cell_id` against `mr_data.best_neighbor_cell`. When they match, apply 1.15× boost. | D+3 |
| B3 | Fix F2 regression | Root-cause: `is_single` guard missed some tag variants. Make it parse the answer count directly: `n_answers = len(gt_answers); is_single = (n_answers == 1)`. Add test fixtures covering all F2 tag variants. | D+3 |
| B4 | Geometric lift_tilt heuristic | In `feature_engine.py`, add: if serving RSRP ∈ (-105, -90) AND no site within 30° of serving cell's azimuth within 500m, score `coverage_hole_tilt = 0.6` (distinct from `coverage_hole_power`). | D+5 |
| B5 | Unit tests | Pytest covering all 6 F1 action classes × 3 scenarios each (18 tests minimum). Run on every commit. | D+5 |
| B6 | Refactor: extract 4BS overhaul into `_apply_4bs_overhaul()` | Current inline block is hard to review. Move to named helper with its own docstring. No behavior change. | D+6 |

### 3.5 Success criteria

- F2/F3/F4 each ≥ 0.98 on full training set
- F1 deterministic baseline ≥ 0.70 (up from 0.5752)
- `eval_v3.py --full` runs without exceptions
- All B5 tests pass on main

### 3.6 Testing expectations

`skills/tests/test_feature_engine.py` should include:
```python
def test_f1_lift_tilt_tertiary_target():
    ps = load_fixture("f1_lift_tilt_tertiary_001")
    sf = run_all_skills(ps)
    candidates = generate_single_candidates(ps, sf)
    assert candidates[0][0] == ps.options["lift_tilt/tertiary"].c_code

def test_f2_regression_not_masked_by_is_single():
    ps = load_fixture("f2_multi_drops")
    sf = run_all_skills(ps)
    assert sf.score_inter_freq_load > 0.3  # not capped to 0.08
```

### 3.7 Risks I own

| Risk | Mitigation |
|---|---|
| Fix for P0 add_neighbor introduces new regression | Add regression suite covering all passing actions before changing logic |
| Schema freeze blocks obvious improvement | Freeze is soft — emergency unblock possible with team consensus + Owner A retrain budget |
| Geometric heuristic overfits to training set | Hold out 100 scenarios as B-specific val set, never touched during development |

---

## 4. Owner C — Tool Integration (19 server tools)

### 4.1 Responsibility Charter

Bridge the local pipeline with the 19 competition server tools. The deterministic pipeline reads CSV fields directly — `tool_caller.py` adds the **geometric and causal** signals that CSVs don't contain, and that the model alone cannot derive.

### 4.2 The 19 tools

| Tool | Purpose | When to call |
|---|---|---|
| `judge_mainlobe` | Is UE in main lobe of cell X? | Always for SINR>5 + low PRB scenarios (lift_tilt vs check_server) |
| `calculate_tilt_angle` | Current antenna tilt | When `judge_mainlobe=False` — decides if tilt needs adjusting |
| `calculate_overlap_ratio` | Inter-cell coverage overlap | SINR<0 handover scenarios |
| `get_mr_data` | Raw MR samples | When local MR looks sparse (catches `insufficient_data`) |
| `get_kpi_data` | Historical KPIs for a cell | Chronic vs transient diagnosis |
| `get_cell_config` | Configured neighbors, A3 params | `add_neighbor` / `decrease_a3` verification |
| `get_signaling_trace` | Raw signaling events | Rare — only when other tools don't resolve |
| `get_traffic_pattern` | Temporal traffic shape | Distinguish congestion vs outage |
| ... (11 others, see server.py) | ... | Opportunistic |

### 4.3 Files owned

`skills/tool_caller.py` — currently a skeleton with `ToolEvidence` dataclass and `TODO` comments in `augment_features()`. Must become a production module.

### 4.4 Why this is the highest-leverage work

The deterministic pipeline fails on two F1 classes (lift_tilt, insufficient_data) for the same reason: **local features can't distinguish them**. Both present as high-SINR + moderate-PRB + moderate-throughput. The only signals that separate them are:

- `lift_tilt`: `judge_mainlobe=False` for the problem cell
- `insufficient_data`: `get_mr_data` returns < N samples for the problem cell

No amount of feature engineering or fine-tuning can fix this without the tools.

### 4.5 Deliverables

| # | Artifact | Detailed spec | ETA |
|---|---|---|---|
| C1 | Freeze `ToolEvidence` schema | Dataclass with {mainlobe_flags, tilt_angles, overlap_ratio, kpi_data_raw, signaling_events, mr_sample_counts}. Document each field. | D+2 |
| C2 | `call_targeted()` for SINR<0 group | When top_issue ∈ {late_handover, missing_neighbor}: call `get_cell_config`, `calculate_overlap_ratio` for serving + top-2 neighbors. | D+3 |
| C3 | `call_targeted()` for SINR>5+low-PRB group | When top_issue ∈ {coverage_hole, backhaul_server}: call `judge_mainlobe` for each candidate target, `get_mr_data` for insufficient_data detection. | D+4 |
| C4 | `augment_features()` | Reads `ToolEvidence`, updates `ScenarioFeatures`:<br>• `mainlobe=False` → `score_coverage_hole += 0.3`, target=that cell<br>• `mr_sample_count < 10` → `score_insufficient_data = 0.7` (new field)<br>• `overlap_ratio > 0.7` → `score_late_handover += 0.2` | D+4 |
| C5 | Tool-call budget enforcement | Per-scenario wall-time tracker; abort further tool calls if budget exceeded. Fallback = original features. Default budget: 90s. | D+5 |
| C6 | Integration tests | Against real server at `--server_url`. Cover each tool at least once. Marks tests `@pytest.mark.integration` so unit tests still run without server. | D+6 |
| C7 | Response caching | `@lru_cache` keyed by `(tool_name, scenario_id, args_hash)`. Speeds up re-runs during eval. | D+7 |

### 4.6 Constraints

- **Time budget**: total tool-call latency per scenario ≤ 90 s (leaving 210 s for Qwen + parsing). This is enforced by C5.
- **Failure handling**: any tool timeout, HTTP error, or schema mismatch must fall back to original features. Never crash the pipeline.
- **Rate limits**: the server may rate-limit; respect 429 responses with exponential backoff.

### 4.7 Success criteria

- lift_tilt recall ≥ 0.60 *before* any fine-tuning improvement
- insufficient_data recall ≥ 0.60 *before* fine-tuning improvement
- Mean tool-call latency ≤ 90s per F1 scenario (p95 ≤ 150s)
- Zero pipeline crashes attributable to tool failures over a 200-sample eval run

### 4.8 Risks I own

| Risk | Mitigation |
|---|---|
| Server rate-limits during bulk eval | Response cache (C7); exponential backoff; per-scenario parallelism capped at 4 |
| Tool response schema mismatch | Schema validation with Pydantic at boundary; soft-fail if unexpected |
| Tool latency eats time-discount budget | C5 enforces per-scenario budget; can drop to deterministic fallback |
| I block Owner A's retrain | Ship C1+C4 early with empty/partial evidence; iterate on quality |

---

## 5. Owner D — Inference-time RAG + Data-Analysis Tooling

### 5.1 Responsibility Charter

Own two distinct-but-related products:
1. **Inference-time RAG** — the live retrieval system that runs every time Qwen is called. This is the main product; it directly affects both F1 IOU and tokens-per-prompt.
2. **Data-analysis tooling** — a library of scoring functions (dedup, outlier, label-error) that Owner A calls from `curate_sft.py` to decide what goes into `sft_v3.jsonl`.

**What D does NOT own**: the actual curation decisions. D computes the scores; A decides the thresholds and drops. This split exists because curation thresholds trade off against class balance, val-split hygiene, and hard-negative mining — all modeling concerns that belong with A.

### 5.2 Files owned

- `rag_retriever_v2.py` — `StructuredCaseRAG`. Builds a FAISS index over 13-dim feature vectors with a family-name hard filter.
- `skills/data_tools.py` — new library. Pure-function scoring over the training set. No side effects, no decisions; A consumes the output.

### 5.3 Current gaps

| Pri | Problem | Action |
|---|---|---|
| P0 | Retrieval not balanced across action types → top-k monopolized by dominant classes | Stratified retrieval: k/6 per action type |
| P0 | Near-duplicate scenarios dominate top-k | Cosine-based dedup on feature vectors |
| P1 | No confusion-pair mining for lift_tilt / insufficient_data | Build an explicit confusion index pairing them |
| P1 | Feature vector doesn't include tool-evidence fields yet | Coordinate with Owner C: extend to 16-dim after C1 |
| P1 | No tooling for A to detect bad training labels | Ship `data_tools.py` as a scoring library |

### 5.4 Deliverables

| # | Artifact | Detailed spec | ETA |
|---|---|---|---|
| D1 | Feature-vector FAISS index | Rebuilt on top of 13-dim vectors from `_sf_to_feature_vec()`. Cached to `.rag_cache_v2/`. Rebuild trigger: hash of train_data. | D+3 |
| D2 | Stratified top-k retrieval | `retrieve(sf, k=6) → list[RAGHit]`: first partitions by action type, returns k/6 per type, falls back to similarity within each bucket. | D+4 |
| D3 | Confusion-pair index | For each lift_tilt example, precompute its 2 nearest insufficient_data neighbors (and vice versa). Returned alongside the stratified top-k so the model sees the boundary explicitly at both train and inference. | D+5 |
| D4 | Frozen retrieval signature | `retrieve(sf: ScenarioFeatures, k: int = 6) → list[RAGHit]`. No changes after D+2 without sign-off. | D+2 |
| D5 | `skills/data_tools.py` (tooling library for A) | Pure functions returning per-scenario scores:<br>• `duplicate_score(sid) → float` — max cosine sim to any other scenario<br>• `outlier_score(sid) → float` — distance to class centroid<br>• `label_confidence(sid) → float` — 1.0 if gt in options + unambiguous, else lower<br>• `ambiguity_score(sid) → float` — how close the second-best option is to the best<br>No thresholds, no decisions. Just numbers. A decides what to do with them. | D+3 |
| D6 | Extend to 16-dim | After Owner C lands C4, add {mainlobe_hit_count, tilt_angle_mean, overlap_ratio_max} to the feature vector. Rebuild index. | D+6 |

### 5.5 Success criteria

- On 50-sample F1 lift_tilt probe: ≥ 60% of top-6 RAG hits are also lift_tilt
- On 50-sample F1 insufficient_data probe: ≥ 60% of top-6 are also insufficient_data
- End-to-end RAG call ≤ 200ms per query
- `data_tools.py` scoring functions run over all 2000 scenarios in ≤ 30 s total
- Owner A reports val IOU lift of ≥ 0.02 after applying A's curation thresholds

### 5.6 Risks I own

| Risk | Mitigation |
|---|---|
| Stratification leaks val examples into train | Accept a val-set ID list from Owner A; exclude them from the RAG index |
| Feature-vector extension to 16-dim requires Owner A retrain | Coordinate D6 with Owner A's retrain schedule (A4) |
| `data_tools.py` scores are noisy → A's curation is suboptimal | Document expected score distributions; sanity-check on a 50-sample hand-labeled set |

---

## 6. Frozen Interfaces (§ change only with team approval)

These signatures must stay stable after **D+2 (2026-04-22)** so all four owners can work in parallel. Changes require a 24-hour heads-up in the team channel AND Owner A sign-off (because any change ripples through SFT data).

### 6.1 `ScenarioFeatures` (Owner B)

```python
@dataclass
class ScenarioFeatures:
    mobility: MobilityFeatures            # serving_rsrp_dbm, serving_sinr_db,
                                          # neighbor_advantage_db, a3_margin_db
    throughput: ThroughputFeatures        # worst_throughput_mbps, drop_magnitude,
                                          # drop_duration_s
    inter_freq: InterFreqFeatures         # max_dl_prb_util, max_ul_prb_util

    # 6 issue scores, all in [0, 1]
    score_late_handover: float
    score_missing_neighbor: float
    score_coverage_hole: float
    score_pdcch_issue: float
    score_backhaul_server: float
    score_inter_freq_load: float

    # Metadata
    top_issue: str                        # name of highest score
    serving_cell_id: str
    serving_cell_role: Optional[str]      # "primary" | "secondary" | "tertiary"

    # Extended by tool_caller (nullable)
    tool_evidence: Optional["ToolEvidence"] = None
```

### 6.2 `ToolEvidence` (Owner C)

```python
@dataclass
class ToolEvidence:
    mainlobe_flags:       dict[str, bool]    # cell_id → in main lobe?
    tilt_angles:          dict[str, float]   # cell_id → degrees
    overlap_ratio:        dict[str, float]   # cell_id → [0,1]
    kpi_data_raw:         dict               # raw KPI blob
    signaling_events:     list               # list of event dicts
    mr_sample_counts:     dict[str, int]     # cell_id → MR count
    tool_call_latency_ms: float              # for budget tracking

def augment_features(
    sf: ScenarioFeatures,
    ev: ToolEvidence,
) -> ScenarioFeatures: ...
```

### 6.3 `RAGHit` and `retrieve()` (Owner D)

```python
@dataclass
class RAGHit:
    scenario_id: str
    features: ScenarioFeatures
    gt_c_code: str
    gt_action_type: str                   # lift_tilt / insufficient_data / ...
    similarity: float                     # cosine, in [0,1]
    prompt_snippet: str                   # 2-line human-readable summary

def retrieve(
    sf: ScenarioFeatures,
    k: int = 6,
    stratify_by_action: bool = True,
    include_confusion_pairs: bool = True,
) -> list[RAGHit]: ...
```

### 6.4 `_build_f1_prompt()` (Owner A)

```python
def _build_f1_prompt(
    ps: ParsedScenario,
    sf: ScenarioFeatures,
    candidates: list[tuple[str, float, str]],   # (c_code, score, reason)
    rag_hits: list[RAGHit],
) -> str: ...
```

This function is shared between `build_sft_v3.py` and `main_v3.py` so train/inference distributions match.

---

## 7. Weekly Cadence

### 7.1 Sync schedule

| Day | Time | Ritual | Duration | Required |
|---|---|---|---|---|
| Mon | 10:00 | Planning — lock week's priorities | 30 min | All |
| Wed | 10:00 | Mid-week check-in — unblock cross-deps | 20 min | All |
| Fri | 16:00 | End-of-week eval run + demo | 60 min | All |
| Fri | 17:00 | Submission attempt (if gates pass) | — | A + observers |

### 7.2 Async daily update

Each owner posts in the team channel by 10:00 daily:

```
[Owner X — Day N]
- Done: <1 line>
- Doing: <1 line>
- Blocked on: <1 line, or "nothing">
```

### 7.3 Cross-owner communication

- **PR reviews**: cross-owner review is mandatory for any change to a frozen interface. Within one owner's files, self-merge OK.
- **Incident protocol**: if eval drops by > 0.05 after a merge, git revert first, discuss later.
- **Question routing**: tag the owner by name in the channel; response SLA 2 hours during work hours.

---

## 7.5 Integration Responsibilities (cross-cutting)

These don't fit cleanly in any single owner's scope but must be assigned. Default assignments below; each is A's unless otherwise noted, because A is the one who composes everyone's outputs at submission time.

### 7.5.1 End-to-end integration testing (Owner A)

- **Artifact**: `tests/test_pipeline_integration.py` — runs `parse → route → features → tools → rag → candidates → prompt → deterministic-fallback` on 5 fixture scenarios covering all 4 families.
- **Trigger**: runs on every PR touching any owned file. Fails the PR if any stage errors or output shape changes.
- **ETA**: D+3 (before any other retrains depend on integration being healthy).

### 7.5.2 `_build_f1_prompt()` stewardship (Owner A)

- The prompt template is the single source of truth for train/inference parity. Owner A is the steward.
- **Rule**: `build_sft_v3.py` and `main_v3.py` must import the *same* `_build_f1_prompt` symbol (enforced by a static check in the integration test).
- **Nullable-from-day-1 policy**: every field consumed from `ScenarioFeatures`, `ToolEvidence`, `RAGHit` must render cleanly with `None`/empty-list values. This way A3 (pre-tools) and A4 (with tools) use the same template and the earlier SFT signal transfers.

### 7.5.3 Infra — Qwen serving, GPU allocation, checkpoint storage (Owner A)

- Stand up `serve_model.py` with vLLM + LoRA adapter at `http://localhost:8001/v1`.
- Store checkpoints under `checkpoints/v3/ckpt_YYYYMMDD_stepN/`. Keep last 3; archive-delete older.
- **ETA**: D+2, before A3.

### 7.5.4 Wall-time budget ownership (Owner A, with C and D reporting)

Nobody owned the 5-minute total before. Now A does:

| Budget | Stage | Reports to A | Cap |
|---|---|---|---|
| 90 s | Tool calls (C) | Owner C via `tool_call_latency_ms` in `ToolEvidence` | C5 |
| 200 ms | RAG retrieval (D) | Owner D via retrieval timer | D success criterion |
| 180 s | Qwen inference (A) | Self-measured | A4 gate |
| 30 s | Parsing + scoring (B) | Negligible | — |
| **< 300 s total** | Pipeline end-to-end | A measures p50 and p95 on every eval run | A7 |

If p95 > 300 s, A triggers a wall-time review meeting — not a unilateral owner fix.

### 7.5.5 Schema-freeze-vs-bug-fix reconciliation (Owner B)

B's freeze at D+2 conflicts with B2/B3 fixes at D+3. Reconciliation:
- **Freeze covers the dataclass shape**, not field *values*. Adding logic that changes what gets written into existing fields is fine.
- **Adding new fields post-freeze** requires: (a) defaulting to `None`, (b) A's sign-off, (c) automatic retrain queued.
- `score_insufficient_data` (C4's new field) is pre-approved — it's the one known exception, must land by D+2.

### 7.5.6 RAG index rebuild → SFT rebuild (Owner A triggers on Owner D signal)

When D6 extends the feature vector from 13 → 16 dims:
1. D publishes a new `.rag_cache_v2/` under a versioned path.
2. D notifies A in channel with the new cache path.
3. A re-runs `build_sft_v3.py` against the new index (the extra 3 dims appear inside `RAGHit.features`, which flows into `_build_f1_prompt`).
4. A retrains (A4 or A5 — whichever is next). No separate retrain just for D6.

### 7.5.7 Prompt-tuning change process (Owner A proposes, all review)

`QWEN_V3_SYSTEM_PROMPT` is frozen by convention but not untouchable. If A wants to change it:
- Open a PR with before/after IOU on the 200-sample eval (both the baseline LoRA and the one retrained on the new prompt).
- All 4 owners review; 72h SLA.
- Approval requires no per-family regression > 0.02.

### 7.5.8 Documentation stewardship (shared)

- `README_V3.md` — Owner A, but anyone can PR.
- `TEAM_DIVISION.md` (this doc) — Owner A.
- `SCHEMA.md` — Owner B (frozen dataclasses reference).
- Each owner keeps a top-of-file docstring in their owned files current.

---

## 8. Risk Register

| Risk | Likelihood | Impact | Owner | Mitigation |
|---|---|---|---|---|
| Tool-call latency eats into time-discount | Med | High | C | Per-scenario budget (90s); fallback to features; response caching |
| Feature schema thrashes → A keeps retraining | Med | High | B | Hard freeze D+2; any change requires A sign-off |
| Fine-tune overfits to RAG examples | Med | Med | A | Stratified train/val; early stopping; prompt-randomize RAG order |
| F2 regression persists | Low | Med | B | Dedicated investigation session D+3; regression test suite |
| Server rate limits during eval | Med | Med | C | Response cache; exponential backoff; cap parallelism to 4 |
| Leaderboard submission bugs | Low | Critical | A | Dry-run against mock server D+7; actual submission D+8 |
| RAG dedup too aggressive | Low | Med | D | Tolerance > 0; manual review; track training set size delta |
| Owner A's LoRA inference too slow | Low | High | A | Benchmark at D+3; fallback to AWQ quantization |
| Qwen returns invalid C-codes | Low | High | A | Post-process: strip non-C-codes, retry once with stricter prompt |
| Inter-owner interface mismatch at integration | Med | High | All | Shared integration test in `tests/test_pipeline_integration.py` runs daily on CI |

---

## 9. Definition of Done (team-level)

All gates must pass before calling the work complete:

### 9.1 Quantitative

- ✅ Overall mean IOU ≥ **0.85** on offline 200-sample eval
- ✅ Per-family: F1 ≥ 0.78, F2 ≥ 0.98, F3 ≥ 0.98, F4 ≥ 0.98
- ✅ Per-action minimums (F1): lift_tilt ≥ 0.60, insufficient_data ≥ 0.60, add_neighbor ≥ 0.65
- ✅ Mean wall-clock ≤ 5 min/scenario (full time-discount)
- ✅ p95 wall-clock ≤ 8 min/scenario

### 9.2 Qualitative

- ✅ One-click reproducible from the `v3/` directory (`python main_v3.py ...`)
- ✅ All interfaces documented in `SCHEMA.md`
- ✅ Integration test covers every owner's handoff
- ✅ Ablation report explaining contribution of each owner's work
- ✅ Submission posted to leaderboard before 2026-05-04

### 9.3 Handoff artifacts

- `result_final.csv` (leaderboard submission)
- `ABLATION.md` (what each component added)
- `sft_v3.jsonl` (training data archive)
- LoRA adapter weights (checkpoint)
- `.rag_cache_v2/` index snapshot

---

## 9.5 Per-Owner Return Envelope (what each person ships back to A)

Everyone works in the same Git repo via feature branches. The table below specifies **exactly what files each owner touches and what they ship back to Owner A** for the final compilation + submission.

### 9.5.1 Owner A (me) — files I own end-to-end

| File | Status | What to do |
|---|---|---|
| `curate_sft.py` | **implemented** | A1a produced `curated_train_ids.txt`, `heldout_f1_val_ids.txt`, `curation_weights.json`, and `curation_report.md`; current run used fallback scoring because D's final scorer was unavailable |
| `build_sft_v3.py` | **implemented** | A1b produced `sft_v3.jsonl` and `sft_heldout_val.jsonl` using shared `_build_f1_prompt()` |
| `submit.py` | **dry-run implemented** | Runs `main_v3.py`, sanity-checks C-codes, writes `submission_receipt.json`; real upload waits for official portal/API details |
| `owner_a_verify.py` | **implemented** | One-command A integration guard for pytest, heldout tools eval, and submission dry-run after B/C/D handoffs |
| `main_v3.py` | **integrated** | `_build_f1_prompt()` is the train/inference parity source; deterministic/tools path writes `result.csv` and `result_final.csv` |
| `tests/test_pipeline_integration.py` | **partial** | Prompt parity runs; older placeholder fixture tests remain skipped |
| `tests/test_tools_regression.py` | **implemented** | Guards the current tools-baseline behavior for `insufficient_data`, `lift_tilt`, and `check_server` |
| LoRA training config | **implemented** | `configs/lora_v3.yaml` and `configs/lora_v3_zero3.yaml` |
| `checkpoints/` | **implemented** | Main LoRA references: `checkpoints/v3_zero3/checkpoint-477` and `checkpoints/v3_a3_corrected/checkpoint-246` |

**Final artifacts I need to assemble**:
- `sft_v3.jsonl` (from A1b)
- LoRA adapter weights (from A3/A4/A5)
- `result_final.csv` (from A6)
- `ABLATION.md` (from A7)
- `.rag_cache_v2/` tarball (from Owner D)

### 9.5.2 Owner B — modifies these files, ships via PR

| File | Action | Gate |
|---|---|---|
| `skills/feature_engine.py` | Modify: fix add_neighbor (B2), fix F2 regression (B3), add lift_tilt heuristic (B4), refactor (B6) | Must not change the `ScenarioFeatures` dataclass shape past D+2 (SCHEMA.md §1) |
| `skills/candidate_generator.py` | Modify: fix add_neighbor cell targeting (B2) | No interface changes |
| `skills/scenario_parser.py` | Light fixes as needed | `ParsedScenario` shape frozen |
| `skills/scenario_router.py` | Likely no changes | `Family` enum frozen |
| `skills/answer_parser.py` | Likely no changes | Scoring function frozen |
| `skills/rf_analysis.py` | Owner B scope, existing | No interface changes |
| `tests/test_features.py` | **new** | B5 — unit tests for all 6 F1 action classes |
| `SCHEMA.md` §1 | Only if dataclass adds pre-approved field | Requires A sign-off + retrain queue |

**Ship back to A**: just push PRs. No file transfer. CI must be green.

### 9.5.3 Owner C — modifies these files, ships via PR + fixture tarball

| File | Action | Gate |
|---|---|---|
| `skills/tool_caller.py` | Implement C1–C7: fill `ToolEvidence`, `call_targeted`, `augment_features`, budget, cache | `ToolEvidence` shape frozen D+2 (SCHEMA.md §2) |
| `tests/test_tools.py` | **new** | C6 — integration tests marked `@pytest.mark.integration` |

**Ship back to A**:
1. PR to main (the code).
2. `tool_evidence_fixtures.json` — a JSON file with 10 sample `ToolEvidence` outputs keyed by scenario_id. I need these to build `sft_v3.jsonl` before the real server is reachable.

### 9.5.4 Owner D — modifies these files, ships via PR + cache tarball

| File | Action | Gate |
|---|---|---|
| `rag_retriever_v2.py` | Implement D1–D4, D6: FAISS index, stratified retrieval, confusion pairs, 13→16 dim | `retrieve()` signature frozen D+2 (SCHEMA.md §3) |
| `skills/data_tools.py` | Implement D5: four scoring functions | Stub exists; fill in the NotImplementedErrors |

**Ship back to A**:
1. PR to main (the code).
2. `rag_cache_v2.tar.gz` — the built FAISS index. It's a binary artifact; don't commit it to Git. Place it at `/home/huangxiaolin/AITO/dataset/Track A/v3/artifacts/rag_cache_v2.tar.gz`.

### 9.5.5 Compilation order (Owner A runs these in sequence)

```bash
# 1. Pull everyone's PRs, run CI
pytest tests/ -v

# 1b. Owner A handoff guard after any B/C/D change
# Requires train-split tool server on 7861 and test-split server on 7860.
python owner_a_verify.py --train-server-url http://127.0.0.1:7861 --test-server-url http://127.0.0.1:7860

# 2. Unpack D's RAG cache
tar -xzf artifacts/rag_cache_v2.tar.gz -C .

# 3. Run D's scoring library → my curation decisions
python curate_sft.py --train_dir train_data --out_ids curated_train_ids.txt

# 4. Build SFT dataset
python build_sft_v3.py --ids curated_train_ids.txt --train train_data --out sft_v3.jsonl --upsample

# 5. Train LoRA
llamafactory-cli train configs/lora_v3.yaml

# 6. Offline eval with LoRA
python eval_v3.py --full --per-action --model http://localhost:8001/v1 --lora checkpoints/v3/ckpt_latest

# 7. Dry-run submission
python submit.py --server_url ... --model_url ... --lora checkpoints/v3/ckpt_latest --dry-run

# 8. Real submission (only after dry-run passes)
python submit.py --server_url ... --model_url ... --lora checkpoints/v3/ckpt_latest
```

---

## 10. Glossary

| Term | Definition |
|---|---|
| **C-code** | One of C1..C22, a coded option in the scenario's multiple choice |
| **F1/F2/F3/F4** | Scenario families; see README §2.2 |
| **ScenarioFeatures** | Owner B's output — the local deterministic analysis |
| **ToolEvidence** | Owner C's output — authoritative signals from server tools |
| **RAGHit** | Owner D's output — one retrieved similar training example |
| **SFT** | Supervised fine-tuning (Owner A's work) |
| **IOU** | Intersection-over-union on predicted vs ground-truth C-code sets |
| **Time discount** | Multiplier on IOU based on wall-time (<5min=1.0, <10min=0.5, ≥10min=0) |
| **Top issue** | Name of the highest-scoring field in `ScenarioFeatures` |
| **Main lobe** | The primary radiation direction of an antenna (served by `judge_mainlobe`) |
| **A3 event** | 3GPP-defined measurement report trigger for handover |
| **MR data** | Measurement Reports — UE-side radio measurements |
| **CCE utilization** | Downlink Control Channel Element usage — high = PDCCH congestion |
| **PRB utilization** | Physical Resource Block usage — high = radio congestion |

---

## 11. References

- Architecture spec: `v3/README_V3.md`
- Feature schema: `skills/feature_engine.py:ScenarioFeatures`
- Candidate logic: `skills/candidate_generator.py:generate_single_candidates`
- Scoring function: `skills/answer_parser.py:compute_iou`
- Server tools: `server.py` (do not modify)
- Historical V2: `../main_v2.py`
