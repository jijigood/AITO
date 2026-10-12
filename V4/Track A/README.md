# Track A — V4 (Unified Phase 1 + Phase 2)

End-to-end agent for the Telco Troubleshooting & Optimization Challenge,
Track A (5G wireless drive-test).

V4 consolidates the optimized work from owners A/B/C/D into a single
codebase that supports both:

| Phase   | Entry point     | Mode                              | Result   |
|---------|-----------------|-----------------------------------|----------|
| Phase 1 | `main_v3.py`    | LLM (Qwen) + RAG + tool evidence  | **0.86** |
| Phase 2 | `phase2_run.py` | API agent + skills + tools + optional LLM/RAG arbitration | target ≥ 0.86 |

The two entry points share **the same `skills/` core** (feature_engine,
candidate_generator, scenario_parser, scenario_router, tool_caller).
V4 is one codebase; what changes is whether data is parsed from embedded
JSON (Phase 1) or fetched via REST API (Phase 2).

---

## Directory Layout

```
V4/Track A/
├── README.md                        ← this file
├── docs/
│   ├── README.md / README_CN.MD                 ← documentation index
│   ├── PHASE2_PROJECT.md / PHASE2_PROJECT_CN.MD ← Phase 2 project guide
│   ├── V4_PHASE2_ARCHITECTURE.md                ← Phase 2 architecture/workflow
│   ├── V4_PHASE2_ARCHITECTURE_CN.MD             ← Phase 2 架构/运行流程中文版
│   └── PHASE2_TEAM_DIVISION.md / _CN.MD         ← Phase 2 A/B/C/D 分工
│
├── skills/                          ← shared core (used by both phases)
│   ├── scenario_parser.py           [A]  raw JSON → ParsedScenario
│   ├── scenario_router.py           [A]  → F1/F2/F3/F4 family
│   ├── feature_engine.py            [B]  → ScenarioFeatures (SINR-based grouping)
│   ├── candidate_generator.py       [A]  → top-N C-codes (add_neighbor fix)
│   ├── tool_caller.py               [C]  → 3-group call_targeted + get_mr_data
│   ├── api_data_gatherer.py         [P2] → 4 core API calls → reconstructed raw JSON
│   ├── answer_parser.py             [A]  → IOU scoring helpers
│   ├── data_tools.py                [A]  → CSV utilities
│   └── rf_analysis.py               [A]  → RF/path-loss helpers
│
├── # ─── Phase 1 (LLM + RAG, scored 0.86) ───
├── main_v3.py                       LLM-based main runner
├── build_sft_v3.py                  SFT data builder (V4 weights)
├── eval_v3.py / eval_ckpt.py        Local evaluation
├── curate_sft.py                    Heldout / curation tooling
├── owner_a_verify.py                End-to-end smoke test
├── submit.py                        Submission helper
├── rag_retriever_v2.py        [D]   Stratified RAG retriever
│
├── # ─── Phase 2 (agentic cloud workflow) ───
├── phase2_run.py              [P2]  Skills + tools + reasoning trace + verifier/arbitrator
│
├── # ─── Phase 2 cloud-server framework (reference, mostly unchanged) ───
├── main.py                          Phase 2 LLM agent framework + Environment
├── _types.py                        Pydantic types (Scenario, ToolCall, …)
├── utils.py                         load_scenarios, extract_answer, compute_score
├── server.py                        Phase 2 FastAPI server (reference / local test)
├── logger.py                        logging setup
├── requirements.txt
│
└── data/                            symlinks to /home/huangxiaolin/AITO/5.4/Track A/data/
    ├── Phase_1/                     2000 train + 500 test
    └── Phase_2/                     500 test (cloud-API stubs)

Local-only artifacts are outside this shared tree at
`/home/huangxiaolin/AITO/V4_local_artifacts_archive_20260507/Track A/`.
That archive contains checkpoints, SFT JSONL files, run outputs, diagnostics,
old experiments, and legacy V3/Phase 1 docs.
```

---

## Phase 1 usage (already submitted, 0.86)

The Phase 1 pipeline is the LLM-based runner with a local FastAPI server
serving embedded scenarios.  It uses Qwen3.5-35B-A3B + tool evidence + RAG.

```bash
cd "/home/huangxiaolin/AITO/V4/Track A"

# Start local server (serves Phase_1/test.json scenarios on port 7860)
python server.py

# In another shell, run main_v3 against the local server
python main_v3.py \
    --server_url http://127.0.0.1:7860 \
    --model_url  http://localhost:8001/v1 \
    --save_dir   results/phase1 \
    --use-tools \
    --deterministic
```

Outputs `result.csv` and `result_final.csv` in `--save_dir`.
For dry-run validation: `python submit.py --dry-run …` (see `submit.py --help`).

---

## Phase 2 usage (the new V4 contribution)

Phase 2's `phase2_run.py` is a calibrated **agentic API workflow**. For
each scenario it uses all four competition-relevant components:

- **Skills**: parser/router/feature engine/candidate generator.
- **Multi-turn tool use**: cloud API data gathering plus conditional
  `judge_mainlobe_or_not` / overlap / MR evidence.
- **Reasoning trace**: diagnostics record the public route → evidence →
  candidate → verifier/arbitrator decision path for audit.
- **Fine-tuning**: optional Qwen3.5-A3B LoRA verifier via an
  OpenAI-compatible endpoint. The default policy is `audit`, so a weak
  checkpoint cannot silently overwrite a high-confidence rule answer.
- **LLM/RAG arbitration**: `--llm_policy arbitrate` restores a Phase-1-style
  final decision maker for high-risk Phase 2 cases, using structured skills,
  conditional tool evidence, all current options, and Phase 1 RAG examples.

```bash
cd "/home/huangxiaolin/AITO/V4/Track A"

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    AGENT_API_KEY=<YOUR_BEARER_TOKEN> \
    NO_PROXY=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
    no_proxy=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
python phase2_run.py \
    --server_url https://120.46.145.77/no \
    --test_json  data/Phase_2/test.json \
    --out        results/phase2/current_submission/result_next.csv \
    --save_freq  10
```

With the already-trained LoRA verifier running on `http://127.0.0.1:8001/v1`:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    AGENT_API_KEY=<YOUR_BEARER_TOKEN> \
    NO_PROXY=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
    no_proxy=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
python phase2_run.py \
    --server_url https://120.46.145.77/no \
    --test_json  data/Phase_2/test.json \
    --out        results/phase2/current_submission/result_agent_audit_next.csv \
    --model_url  http://127.0.0.1:8001/v1 \
    --model_name v4 \
    --llm_policy audit \
    --save_freq  10
```

Recommended repair mode for Phase 2 uses a base/strong Qwen-compatible model
as the final arbiter on high-risk cases:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    AGENT_API_KEY=<YOUR_BEARER_TOKEN> \
    NO_PROXY=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
    no_proxy=127.0.0.1,localhost,120.46.145.77,124.71.227.61 \
python phase2_run.py \
    --server_url https://120.46.145.77/no \
    --test_json  data/Phase_2/test.json \
    --out        results/phase2/current_submission/result_agent_arbitrate_next.csv \
    --model_url  http://127.0.0.1:8001/v1 \
    --model_name qwen \
    --llm_policy arbitrate \
    --rag_train_json data/Phase_1/train.json \
    --rag_top_k 6 \
    --save_freq  10
```

Use the old `v4_fast_balanced` LoRA in `audit` mode unless a new checkpoint
passes the heldout gates for `lift_tilt` and `insufficient_data`.

Outputs CSV with platform-ready columns `ID,Track A,Track B`.  This codebase
is Track A-only; `Track B` is kept as a required empty platform column because
we are not participating in Track B.

### Why calibrated instead of blindly LLM-first?

- Current deterministic skills after the F1 candidate repair score **0.8155**
  on Phase 1 train, with `lift_tilt` now present in Top2/Top3 but still 0%
  Top1. That is exactly the kind of boundary where an LLM arbiter helps.
- The currently trained `v4_fast_balanced` LoRA is serveable, but heldout
  validation shows `lift_tilt=0.0000` and `insufficient_data=0.0000`.
  Therefore it is included as an audited verifier, not as an unconditional
  answer source.
- Predictable timing: low-risk/template cases stay deterministic; only
  high-risk or close-score cases call the arbiter unless `--llm_arbitrate_all`
  is set.
- Reuses the entire V3 pipeline while satisfying the agent-design loop:
  skills, tool calls, RAG, reasoning trace, and model component.

### Phase 2 family routing

The router falls back to tag-based selection when V3's BS-count rules
don't fit Phase 2's distribution:

| num_bs | tag             | V3 expects | Phase 2 actual | V4 family |
|--------|-----------------|-----------|----------------|-----------|
| 4      | single-answer   | F1        | 241 scenarios  | F1        |
| 4      | multiple-answer | F2        | 145 scenarios  | F2        |
| 5      | single-answer   | (new)     | 28 scenarios   | **F1_5BS** (override) |
| 5      | multiple-answer | F3        | 29 scenarios   | F3        |
| 13     | single-answer   | F4        | 29 scenarios   | F4        |
| 13     | multiple-answer | (new)     | 28 scenarios   | **F4M** (override) |

---

## What's optimized vs base V3

These four optimizations are present in V4 (carried forward from
the merged ABCD work in v3/, plus the Phase 2 work):

1. **`feature_engine.py` — SINR-based grouping** (B's work)
   - Three SINR bands (< 0, 0–5, > 5) drive different score patterns
   - `_has_unconfigured_nb` flag distinguishes missing_neighbor from late_handover
   - 4-BS inter-freq clamping prevents F1 misclassification as load issue

2. **`candidate_generator.py` — add_neighbor cell-pair fix** (A's Phase 1 fix)
   - `_score_option` accepts `serving_cell_id` and selects the option
     whose `cells` list contains the serving cell.
   - Resolved 13/25 add_neighbor wrong-cell-pair failures on F1 heldout.

3. **`tool_caller.py` — 3-group call_targeted** (C's work + A's gap-fill)
   - Groups: coverage_hole, backhaul_server, ambiguous_boundary
   - `get_mr_data` added to Group 1 + Group 2 (C originally missed this)
   - `augment_features()` updates `score_insufficient_data` from
     mainlobe and MR-count signals.

4. **`rag_retriever_v2.py` — stratified retrieval** (D's work + A's quota fix)
   - F1 action-stratified retrieval guarantees ≥ 2 exemplars per class
   - `missing_nb_score` proxy from PCell Neighbor Cell config string
   - `_coerce_feature_dim()` for 13 ↔ 16-dim compat

### New in V4 (Phase 2)

5. **`skills/api_data_gatherer.py`** — Phase 2's CSV-via-API bridge
   - Core API calls: `get_user_plane_data`, `get_config_data`,
     `get_kpi_data`, `get_mr_data`
   - Optional `--fetch_signaling` path calls
     `get_signaling_plane_event_log`; live probes found signaling empty, so it
     is disabled by default.
   - Reconstructs the raw-JSON dict `parse_scenario()` already understands
   - Rejects Phase 2 placeholder strings so audit can catch missing cloud data
   - Records per-tool diagnostics for pre-submission review

6. **`phase2_run.py`** — Phase 2 agent runner
   - Uses V3 pipeline end-to-end, with optional audited/gated verifier or
     RAG-backed LLM arbitration
   - Family override for Phase 2's new (5-BS, single-answer) and
     (13-BS, multiple-answer) patterns; 13-BS multi uses an inter-frequency pair
   - Reuses C's optimized tool augmentation for mainlobe/MR/overlap evidence
   - Writes `agent_trace` with skills, tools, reasoning, and fine-tuning status
   - Periodic CSV save, exception-safe per-scenario solving

7. **`phase2_audit.py`** — pre-submission guard
   - Checks `ID,Track A,Track B` format, missing IDs, answer cardinality,
     API errors, low-confidence fallback answers, and missing parsed data.
     The `Track B` column is expected to stay blank for this Track A-only run.
   - Designed for Phase 2's three-submission limit: audit before submitting

Phase 2 cloud preflight, before burning a submission:

```bash
cd "/home/huangxiaolin/AITO/V4/Track A"

python phase2_select_probe.py \
  --test_json data/Phase_2/test.json \
  --per_group 3 \
  --out_ids results/phase2/probe_selection/probe_ids.txt \
  --out_summary results/phase2/probe_selection/probe_summary.json

NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python phase2_run.py \
  --server_url https://120.46.145.77/no \
  --token "<BEARER_TOKEN>" \
  --test_json data/Phase_2/test.json \
  --out results/phase2/preflight.csv \
  --diagnostics_jsonl results/phase2/preflight.diagnostics.jsonl \
  --scenario_ids_file results/phase2/probe_selection/probe_ids.txt \
  --save_freq 1

python phase2_audit.py \
  --test_json data/Phase_2/test.json \
  --result_csv results/phase2/preflight.csv \
  --diagnostics_jsonl results/phase2/preflight.diagnostics.jsonl \
  --allow_partial
```

Full result generation uses the same command without `--sample_per_group`, then
run `phase2_audit.py` again without `--allow_partial`.

---

## Quick smoke test

```bash
cd "/home/huangxiaolin/AITO/V4/Track A"

# Verify imports + run pipeline on 10 random Phase 1 training scenarios
python -c "
import sys, json, random
sys.path.insert(0, '.')
from skills.scenario_parser import parse_scenario
from skills.scenario_router import route
from skills.feature_engine import run_all_skills
from skills.candidate_generator import generate_single_candidates

random.seed(42)
data = json.load(open('data/Phase_1/train.json'))
sample = random.sample(data, 10)
ok = 0
for sc in sample:
    ps = parse_scenario(sc)
    family = route(sc['tag'], ps.num_base_stations)
    sf = run_all_skills(ps)
    cands = generate_single_candidates(ps, sf, family, top_n=6)
    if cands and cands[0].c_code == sc.get('answer', '').split('|')[0]:
        ok += 1
print(f'Smoke test: {ok}/10 top-1 correct')
"
```

Expected: ≥ 6/10 (V3 deterministic baseline).

---

## Source provenance

- All Phase 1 files (`main_v3.py`, `build_sft_v3.py`, `rag_retriever_v2.py`,
  curated SFT artefacts, etc.) come from `/home/huangxiaolin/AITO/dataset/Track A/v3/`
  — the merged-and-fixed working copy that submitted at 0.86 on 2026-04-26.
- Phase 2 files (`phase2_run.py`, `skills/api_data_gatherer.py`, `main.py`,
  `server.py`, `_types.py`, `utils.py`) come from `/home/huangxiaolin/AITO/5.4/Track A/`
  — the official Phase 2 framework + the calibrated agent built on top of it.
- `data/` is symlinked, not copied.
