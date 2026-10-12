# V4 Architecture

## High-level flow

```
                       ┌─────────────────────────────────────────┐
                       │  Raw scenario  (Phase 1: embedded JSON,  │
                       │                  Phase 2: stub + cloud)  │
                       └────────────────────┬────────────────────┘
                                            │
            ┌───────────────────────────────┴───────────────────────────────┐
            │                                                                │
   PHASE 1 (LLM-augmented)                                       PHASE 2 (agentic)
   main_v3.py                                                    phase2_run.py
            │                                                                │
            │                                                                ▼
            │                                                  ┌───────────────────────┐
            │                                                  │  api_data_gatherer    │
            │                                                  │  4 core API calls:    │
            │                                                  │   • config_data       │
            │                                                  │   • user_plane_data   │
            │                                                  │   • kpi_data          │
            │                                                  │   • mr_data           │
            │                                                  │  optional: signaling  │
            │                                                  │  → reconstructed raw  │
            │                                                  └───────────┬───────────┘
            │                                                              │
            └────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
                          ┌────────────────────────┐
                          │ scenario_parser        │
                          │ • parse 5 CSV fields   │
                          │ • parse 22+ option     │
                          │   labels → action_type │
                          │ • infer cell_roles     │
                          │ • build PCI↔cell_id    │
                          └───────────┬────────────┘
                                      │  ParsedScenario
                                      ▼
                          ┌────────────────────────┐
                          │ scenario_router        │
                          │ → Family (F1/F2/F3/F4) │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │ feature_engine         │
                          │ run_all_skills(ps)     │
                          │ • mobility skill       │
                          │ • inter-freq skill     │
                          │ • throughput skill     │
                          │ • interference skill   │
                          │ • PDCCH skill          │
                          │ • SINR-based grouping  │
                          │ → ScenarioFeatures     │
                          └───────────┬────────────┘
                                      │
                                      ▼  (Phase 2 only) optional disambiguation
                          ┌────────────────────────┐
                          │ judge_mainlobe?        │  ← API call if scores ambiguous
                          │ calculate_overlap?     │  ← API call for F2 overshoot
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │ candidate_generator    │
                          │ generate_single_       │
                          │   candidates(ps,sf,fm) │
                          │ • ISSUE_ACTION_MAP     │
                          │ • _score_option +      │
                          │   serving_cell_id fix  │
                          │ → List[RankedCandidate]│
                          └───────────┬────────────┘
                                      │
                       ┌──────────────┴──────────────┐
                       │                             │
        Phase 1: format prompt           Phase 2: public reasoning trace
        with candidates+RAG hits         + optional fine-tuned verifier
        → LLM → \boxed{C5}               → calibrated final answer
                       │                             │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                              result.csv
                              (ID, Track A, Track B)

```

## Family routing matrix

The router (`skills/scenario_router.py`) classifies each scenario into
one of four families.  V4's `phase2_run.py` adds a fallback override
because Phase 2 contains scenario shapes that V3's BS-count-only rules
don't cover.

| BS  | tag             | V3 router | V4 (Phase 2 override) | Purpose                             |
|-----|-----------------|-----------|-----------------------|-------------------------------------|
| 4   | single-answer   | F1        | F1                    | classic 4-BS troubleshoot           |
| 4   | multiple-answer | F2        | F2                    | composite 2-action fix              |
| 5   | single-answer   | F3¹       | **F1_5BS**²           | Phase 2 added these — use F1 logic  |
| 5   | multiple-answer | F3        | F3                    | canonical 4-action multi-select     |
| 13  | single-answer   | F4        | F4                    | inter-frequency / load focus        |
| 13  | multiple-answer | F4¹       | **F4M**²              | Phase 2 added these — use 13-BS inter-frequency pair logic |

¹ V3's router routes by BS count alone, so it would mis-classify these.
² Override happens in `phase2_run._solve_scenario` after `route()` returns.

## Score → action mapping

Computed by `feature_engine.run_all_skills()`, consumed by
`candidate_generator.generate_single_candidates()`:

| Score                       | Triggered by                                   | Recommended action          |
|-----------------------------|------------------------------------------------|-----------------------------|
| `score_late_handover`       | SINR < 0, neighbor advantage > 3, A3 too tight | `decrease_a3` (primary)     |
| `score_missing_neighbor`    | best neighbor not in config, advantage > 3     | `add_neighbor`              |
| `score_inter_freq_load`     | high PRB utilization at worst time             | `decrease_interfreq` thld   |
| `score_interference`        | SINR < 0 with RSRP > -110                      | (info — feeds into routing) |
| `score_coverage_hole`       | low SINR + low RSRP + outside mainlobe         | `lift_tilt`                 |
| `score_overshoot_overlap`   | high overlap ratio between serving + neighbor  | `decrease_power`+`tilt`     |
| `score_pdcch_issue`         | high CCE utilization                           | `pdcch_2sym`                |
| `score_backhaul_server`     | KPI anomaly                                    | `check_server`              |
| `score_insufficient_data`   | MR count < 7 OR inside mainlobe with low SINR  | `insufficient_data`         |

## Phase 2 API calls

The `APIDataGatherer.gather()` method makes 4 core calls by default, plus
optional signaling and 0–2 conditional disambiguation calls:

```
Round 1 (4 calls, always):
  GET /user-plane-data           ← serves user_plane_data CSV
       ← also derives worst_time / last_time
  GET /config-data               ← serves network_configuration_data CSV
  GET /get_kpi_data              ← serves traffic_data CSV
  GET /get_mr_data               ← serves mr_data CSV

Round 1b (0–1 calls, optional via --fetch_signaling):
  GET /signaling-plane-event-log?time={last_time}
       ← reconstructed back into pipe-CSV

Round 2 (0–2 calls, conditional):
  GET /judge_mainlobe?time={worst_time}&pci={serving_pci}
       ← for coverage/server/insufficient-data ambiguity
  GET /calculate_overlap_ratio?pci_serving=…&pci_neighbor=…
       ← for overlap/interference disambiguation
```

Default total: **4–6 API calls per scenario**.  With `--fetch_signaling`,
total becomes **5–7 calls per scenario**.  The audited full run averaged
about 11 s/scenario, comfortably under Phase 2's 5-minute time-discount cutoff.

## Phase 2 four-component agent contract

The Phase 2 runner intentionally keeps all four agent-building methods in the
workflow, while preventing a weak component from overwriting a stronger one:

| Component | Implementation | Output |
|---|---|---|
| Skills | `scenario_parser`, `scenario_router`, `feature_engine`, `candidate_generator` | Structured RF diagnosis and ranked options |
| Multi-turn tool use | `APIDataGatherer.gather()` plus conditional `judge_mainlobe_or_not`, `get_mr_data`, `calculate_overlap_ratio` | Live digital-twin evidence and tool-call audit |
| Reasoning | `agent_trace` in diagnostics JSONL | Public route/evidence/candidate/verifier decision trace |
| Fine-tuning | `Phase2FineTunedVerifier` in `phase2_run.py` using Qwen3.5-A3B LoRA over OpenAI-compatible API | Audited or gated boxed-answer verifier |

Verifier policies:

| Policy | Meaning | Recommended use |
|---|---|---|
| `off` | Do not call the LoRA verifier | Fast deterministic reproduction |
| `audit` | Call LoRA and log its answer, but keep deterministic final answer | Current safest Phase 2 mode |
| `gate` | Allow only low-risk gated overrides | Use after checkpoint passes heldout gates |
| `always` | Always use LoRA answer when valid | Debug only for current `v4_fast_balanced` |

## File responsibilities

```
skills/
  scenario_parser.py          ← raw JSON / CSVs → ParsedScenario
  scenario_router.py          ← (tag, num_bs) → Family
  feature_engine.py           ← ParsedScenario → ScenarioFeatures
  candidate_generator.py      ← (ps, sf, family) → List[RankedCandidate]
  api_data_gatherer.py        ← Environment + scenario_id → raw JSON dict
  tool_caller.py              ← Phase 1: server tool evidence → augmented sf
  answer_parser.py            ← string answer ⇄ set of C-codes; iou_score
  data_tools.py               ← CSV utilities
  rf_analysis.py              ← RF / pathloss helpers

main_v3.py                    ← Phase 1 LLM agent runner
phase2_run.py                 ← Phase 2 calibrated agent runner
                               writes platform columns: ID, Track A, Track B

rag_retriever_v2.py           ← Phase 1 stratified RAG over training set
build_sft_v3.py               ← Phase 1 SFT data builder (V4 weights)
eval_v3.py / eval_ckpt.py     ← Phase 1 local evaluation
submit.py                     ← Phase 1 submission helper

main.py                       ← Phase 2 LLM agent framework (Environment class)
_types.py / utils.py          ← Phase 2 Pydantic types + helpers
server.py                     ← Phase 2 reference local FastAPI server
```

## Why V4 uses calibrated gating instead of LLM-first for Phase 2

Phase 1 audited the LLM-vs-deterministic split:

| Configuration             | F1-heldout IOU |
|---------------------------|----------------|
| Base Qwen agent (no LoRA) | ≤ 0.80 (est.)  |
| Qwen + LoRA + RAG + tools | 0.84           |
| **Deterministic + tools** | **0.86**       |

Three structural reasons the skills/tools path remains the authority:

1. **Phase 1 GT scoring rules are non-stochastic** — feature → action
   mapping is mostly a decision tree the LLM has to *re-derive* from
   examples; a hard-coded version is at the ceiling by definition.
2. **Latency ceiling** — Phase 2 imposes a 5-min cutoff and time discount.
   LLM agents with 20-iteration tool loops blow this for ~10% of cases;
   bounded API/tool agents do not.
3. **No LoRA budget for Phase 2** — base Qwen alone is below the
   deterministic baseline.  Training a Phase 2-aware LoRA needs ≥ 13 h
   GPU + an SFT regen on Phase 2 data (which has no GT for direct
   supervision).

The Phase 2 `phase2_run.py` therefore uses the proven skills/tools
pipeline as the authority and adds an audited/gated fine-tuned verifier.
This satisfies the four-component agent design while avoiding unchecked
LoRA overrides on known failed heldout classes.
