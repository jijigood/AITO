# Track A Phase 2 Project Guide

This document explains how the V4/Track A Phase 2 project is organized: objective, active files, system components, current submission artifact, and next improvement priorities. The team competes in Track A only. The `Track B` CSV column remains empty only because the platform requires the column.

Chinese version: `PHASE2_PROJECT_CN.MD`

## 1. Phase Objective

Phase 2 contains 500 wireless troubleshooting scenarios. The local `test.json` contains scenario stubs, while the real sandbox data must be fetched through the organizer's Agent Tool Server. The system must output Track A option codes for every scenario.

The score uses answer-set IOU and has a time discount, so the project optimizes four properties at the same time:

- Stable answer format: `ID,Track A,Track B`; multiple answers are joined with `|`.
- Stable API access: live evidence is fetched through the bearer-token cloud tools.
- Stable reasoning: low-risk cases stay deterministic; high-risk cases keep evidence and diagnosis traces.
- Stable runtime: the current full run is about 10-12 seconds per scenario, well inside the practical latency budget.

## 2. Active Directory Structure

```text
V4/Track A/
├── phase2_run.py                         # Phase 2 main runner
├── phase2_audit.py                       # pre-submission audit
├── phase2_select_probe.py                # representative probe selector
├── main.py / server.py / utils.py         # official Phase 2 framework and local reference server
├── skills/
│   ├── api_data_gatherer.py              # cloud API data bridge
│   ├── agentic_overlay.py                # controlled high-risk agentic overlay
│   ├── scenario_parser.py                # CSV and option parsing
│   ├── feature_engine.py                 # RF/KPI/config feature engine
│   ├── candidate_generator.py            # C-code candidate generation
│   ├── tool_caller.py                    # tool evidence augmentation
│   └── ...
├── rag_retriever_v2.py                   # trajectory RAG / memory workstream
├── results/phase2/
│   ├── current_submission/               # regenerated run outputs only; local snapshots are archived externally
│   ├── agentic_overlay_tests/            # regenerated overlay probes
│   └── probe_selection/                  # representative probe IDs
└── archive/                               # empty placeholder; local archives are outside V4
```

Local-only artifacts were moved outside the shared tree to:

```text
/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track A/
```

That archive contains checkpoints, SFT JSONL files, run outputs, diagnostics, old experiments, and legacy V3/Phase 1 design documents. The active `V4/Track A` directory is now intended for source-code and documentation handoff.

## 3. System Components

### 3.1 Data Bridge

`skills/api_data_gatherer.py` reconstructs the raw scenario data by calling four core cloud APIs:

- `get_user_plane_data`
- `get_config_data`
- `get_kpi_data`
- `get_mr_data`

The user-plane data also provides `worst_time`, `last_time`, and serving PCI. `--fetch_signaling` can call `get_signaling_plane_event_log`, but live probes showed that signaling is usually empty, so full runs currently keep it disabled.

### 3.2 Parser and Feature Engine

`scenario_parser.py` converts raw JSON plus pipe-delimited CSV strings into `ParsedScenario`:

- Parses 22+ options into `action_type`, `target_cell`, and cell references.
- Builds `pci_to_cell_id` and `cell_id_to_pci` maps.
- Infers cell roles such as primary, secondary, and tertiary.

`feature_engine.py` produces `ScenarioFeatures`:

- Throughput features locate the poor-quality time window.
- Mobility features compare serving and neighboring cells.
- Inter-frequency, capacity, transport, geometry, and power skills produce issue scores.
- The combined scores cover `late_handover`, `missing_neighbor`, `coverage_hole`, `insufficient_data`, and related causes.

### 3.3 Candidate Generation

`candidate_generator.py` maps diagnosis scores to concrete option codes:

- Single-answer scenarios produce ranked `RankedCandidate` objects.
- Multiple-answer scenarios produce an `AnswerSet`.
- F3/F4/F4M keep strong family priors.
- F1/F1_5BS keep high-risk alternatives such as `insufficient_data`, `lift_tilt`, `decrease_a3`, and `add_neighbor`.

### 3.4 Tool Evidence

`phase2_run.py` adds conditional tool evidence after deterministic feature extraction:

- `judge_mainlobe_or_not` for coverage/mainlobe ambiguity.
- `calculate_overlap_ratio` for overlap/interference ambiguity.
- MR sample count for `insufficient_data`.

All evidence is written to diagnostics JSONL so later prompt, RAG, and submission decisions are auditable.

### 3.5 Controlled Agentic Overlay

`skills/agentic_overlay.py` is the high-risk agent loop. It is intentionally controlled rather than a fully open ReAct loop:

- It triggers only on high-risk branches such as `lift_tilt_vs_insufficient_data` and `add_neighbor_vs_decrease_a3`.
- Each scenario gets at most 2 LLM calls and at most 1 extra tool call.
- The LLM can request tools only from a branch-specific whitelist.
- Tool arguments are validated locally, and local evidence can override invalid copied PCI values.
- The LLM returns diagnosis JSON; the local mapper converts diagnosis to C-code.

Current status: the overlay is engineering-stable. In the 28-scenario high-risk probe, it completed without JSON/LLM/API errors. It currently behaves more like a confirmation layer than a correction layer.

### 3.6 LLM, RAG, and Fine-Tuning

Current model service:

- Endpoint: `http://127.0.0.1:8001/v1`
- Served model name: `qwen`
- Backing path: `/home/huangxiaolin/AITO/model`
- Verified model: `Qwen3.5-35B-A3B`

The `v4_fast_balanced` LoRA can be served, but heldout evaluation fails `lift_tilt` and `insufficient_data`, so it should not be used as an unconditional reranker. Current policy:

- Deterministic backbone remains the authority.
- LoRA is useful for audit/gated verification only until heldout gates pass.
- Trajectory RAG should retrieve evidence patterns, diagnosis actions, and target roles, not old scenario-local C-codes.

## 4. Main Phase 2 Flow

```text
test.json scenario
  -> APIDataGatherer.gather()
  -> parse_scenario()
  -> run_all_skills()
  -> Phase 2 MR/config fallback
  -> targeted tool checks
  -> generate_candidates()
  -> optional agentic_overlay(high_risk)
  -> optional verifier/arbitrator
  -> CSV row + diagnostics JSONL
```

## 5. Current Recommended Candidate

```text
/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track A/results/phase2/current_submission/result_full_uncertainfix.csv
```

Companion files:

```text
/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track A/results/phase2/current_submission/result_full_uncertainfix.audit.json
/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track A/results/phase2/current_submission/result_full_uncertainfix.diagnostics.jsonl
/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track A/results/phase2/current_submission/presubmission_risk_report.md
/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track A/results/phase2/current_submission/lora_vllm_eval_report.md
```

Current audit status before archival: 500/500 answered, 0 hard errors, 0 API errors, and 0 low-confidence rows. The `Track B` column is blank. The candidate snapshot now lives in the external local archive, not in the shared tree.

## 6. Agentic Overlay Test Artifacts

```text
results/phase2/agentic_overlay_tests/agentic_overlay_test_report.md
results/phase2/agentic_overlay_tests/high_risk_agentic_probe.csv
results/phase2/agentic_overlay_tests/high_risk_agentic_probe.diagnostics.jsonl
results/phase2/agentic_overlay_tests/probe_agentic_representative.csv
results/phase2/agentic_overlay_tests/probe_agentic_representative.diagnostics.jsonl
```

These artifacts prove that the overlay can run reliably. They do not yet prove a score gain. The next improvement target is making the overlay correct deterministic mistakes when tool evidence is strong.
