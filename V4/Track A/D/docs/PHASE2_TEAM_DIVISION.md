# Phase 2 Team Division of Labor

**Competition**: Telco Troubleshooting Agentic Challenge — Track A  
**Phase 2 deadline**: 2026-05-18  
**Track**: Track A only. `Track B` is intentionally blank in all submissions.

**Chinese version**: `PHASE2_TEAM_DIVISION_CN.MD`

**AI task guide**: `docs/PHASE2_AI_TASK_GUIDE.md`

**Owner D task pack**: `docs/owner_d/README.md`

This version divides the work by **agent-system capability**, not by feedback/reporting. Each owner builds one relatively independent subsystem, and A integrates the three subsystem outputs.

---

## 1. Role Overview

| Owner | Agent subsystem | Core output |
|---|---|---|
| **A** | Integration + Fine-tuning + Submission | Working runner, SFT/LoRA, live probes, audits, submissions |
| **B** | Skills + Tools | RF skills, branch rules, tool menus, evidence schema |
| **C** | Prompt + CoT + Agentic Loop | System prompts, step prompts, JSON diagnosis, tool-use reasoning |
| **D** | Trajectory RAG + Memory | Trajectory dataset, retrieval index, few-shot/tool-use examples |

The integration contract is intentionally small:

```text
B -> A/C/D: branch_spec + evidence_schema + tool_menu
C -> A: prompt/CoT/loop package
D -> A/C: retrieve_trajectory_context() + trajectory examples
A -> all: probe/full-run diagnostics when needed for iteration
```

---

## 2. A — Integration, Fine-tuning & Submission

### Mission

A owns the full runnable system. A integrates B/C/D outputs, runs the live server, builds SFT datasets, trains/evaluates LoRA, audits CSVs, and manages the three submissions.

### Primary Files

| File/path | Purpose |
|---|---|
| `phase2_run.py` | Main Phase 2 runner and CLI |
| `skills/api_data_gatherer.py` | Cloud API data bridge |
| `phase2_audit.py` | Submission audit |
| `phase2_select_probe.py` | Probe ID selection |
| `build_sft_v3.py` | SFT data builder |
| `curate_sft.py` | SFT curation, balancing, hard negatives |
| `eval_ckpt.py` | Heldout checkpoint evaluation |
| external local archive `checkpoints/` | LoRA checkpoints, kept outside the shared tree |
| `results/phase2/current_submission/` | Current candidate CSV, diagnostics, audit |

### Responsibilities

- Maintain a runnable end-to-end pipeline.
- Integrate B's skills/tools specs into feature/tool/candidate logic.
- Integrate C's prompt/loop changes into `agentic_overlay.py`.
- Integrate D's trajectory retriever into the prompt/RAG context.
- Build final SFT JSONL datasets and train/evaluate LoRA checkpoints.
- Keep LoRA in `audit` until heldout gates pass.
- Run representative probes, high-risk probes, full runs, audits, and submissions.

### LoRA Gate

LoRA can move beyond `audit` only if heldout eval passes:

| Class | Current IOU | Gate |
|---|---:|---:|
| `lift_tilt` | 0.00 | >= 0.50 |
| `insufficient_data` | 0.00 | >= 0.50 |
| `add_neighbor` | 0.68 | >= 0.60 |
| `check_server` | 1.00 | >= 0.95 |
| `decrease_a3` | 0.94 | >= 0.90 |
| `pdcch_2sym` | 1.00 | >= 0.95 |

Both failing classes must pass simultaneously.

---

## 3. B — Skills + Tools

### Mission

B owns the telecom diagnostic capability of the agent: what to measure, which tool to call, how to interpret tool output, and how evidence maps to root cause and action.

### Primary Files

| File | Scope |
|---|---|
| `skills/feature_engine.py` | RF/KPI feature extraction and issue scores |
| `skills/candidate_generator.py` | Root-cause/action to candidate C-code ranking |
| `skills/tool_caller.py` | Tool evidence and feature augmentation |
| `skills/agentic_overlay.py` | `compute_risk_level()` and `BRANCH_TOOL_MENUS` spec, implemented by A |

### Responsibilities

- Define RF skills for each family and branch.
- Define branch triggers, tool menus, and evidence schemas.
- Define how tool results should affect issue scores and action selection.
- Improve F1/F1_5BS deterministic behavior, especially:
  - `F1_5BS` all collapsing to `press_down_tilt`
  - `coverage_hole` -> `increase_power` vs `lift_tilt`
  - `missing_neighbor` vs `decrease_a3`
  - `lift_tilt` vs `insufficient_data`
- Provide mapper semantics: root cause/action/target/relation -> option matching rules.

### Skill Development Priorities

The goal is not to add many loosely related rules. B should add or strengthen high-value skills that target Phase 2 ambiguity boundaries and expose clean evidence for C's prompts and D's trajectory memory.

| Priority | Skill | Target ambiguity |
|---|---|---|
| P0 | `coverage_vs_insufficient_data_skill` | Distinguish `lift_tilt`, `increase_power`, and `insufficient_data` |
| P0 | `neighbor_handover_skill` | Distinguish `add_neighbor` from `decrease_a3` |
| P0 | `f1_5bs_overlap_skill` | Reduce the risk of all `F1_5BS` cases collapsing to `press_down_tilt` |
| P0 | `multi_answer_pair_skill` | Improve F2/F3 pair selection so multi-answer cases do not get only one side correct |
| P1 | `inter_freq_load_skill_v2` | Strengthen F4/F4M load and inter-frequency threshold decisions |
| P1 | `evidence_confidence_skill` | Decide whether the available evidence is strong enough to override or should stay conservative |
| P1 | `target_cell_resolution_skill` | Normalize PCI, cell ID, option label, and target-role matching |

Each skill should output structured evidence fields, not only a final score. Those fields are the contract consumed by C and D.

### Deliverables

| Artifact | Contents |
|---|---|
| `branch_spec.md/json` | branch name, trigger condition, family/tag scope, risk level |
| `tool_menu_spec.md/json` | allowed tools per branch, required args, expected result fields |
| `evidence_schema.md/json` | evidence fields C/D can use in prompt/RAG |
| `skill_priority_spec.md/json` | P0/P1 skill list, input fields, output evidence fields, and expected score effect |
| code patches | focused edits to `feature_engine.py`, `candidate_generator.py`, `tool_caller.py` |

B does **not** own prompt wording or RAG retrieval. B provides the domain contract that C and D consume.

---

## 4. C — Prompt + CoT + Agentic Loop

### Mission

C owns the language-side intelligence of the agent: how the LLM is instructed, how it reasons with evidence, how it decides whether to call a tool, and how it returns a stable diagnosis JSON.

### Primary Files

| File | Scope |
|---|---|
| `skills/agentic_overlay.py` | `_STEP1_SYSTEM`, `_STEP2_SYSTEM`, message builders, diagnosis JSON expectations |
| prompt artifacts | prompt v2/v3, branch-specific prompt snippets, CoT templates |

### Responsibilities

- Write the telecom expert system prompt.
- Write branch-specific Step 1 and Step 2 prompts.
- Define the CoT/trajectory format shown to the model.
- Ensure the LLM outputs diagnosis JSON, not C-codes.
- Add `disagreement_reason` and evidence comparison fields where useful.
- Control JSON stability and avoid hallucinated tool calls.
- Decide how D's trajectory snippets are injected into prompts.

### Deliverables

| Artifact | Contents |
|---|---|
| `prompt_v2.md` | system prompt, Step 1 prompt, Step 2 prompt |
| `diagnosis_schema.json` | exact keys and allowed values for LLM output |
| `cot_template.md` | evidence comparison and diagnosis reasoning template |
| prompt patch | drop-in update for `agentic_overlay.py` message builders |

C does **not** define RF branch semantics; that comes from B. C does **not** build the trajectory index; that comes from D.

---

## 5. D — Trajectory RAG + Memory

### Mission

D owns the agent's memory layer. D turns Phase 1 training cases and Phase 2 probe traces into reusable diagnostic trajectories that C can inject and A can integrate.

### Primary Files

| File/artifact | Scope |
|---|---|
| `rag_retriever_v2.py` | retrieval logic |
| `artifacts/owner_d/data/trajectory_examples.jsonl` | normalized trajectory records |
| `artifacts/owner_d/index/trajectory_index.faiss` | vector index if FAISS is used |
| `artifacts/owner_d/docs/trajectory_schema.md/json` | stable fields for trajectory records |
| `artifacts/owner_d/docs/trajectory_prompt_snippets.md` | compact examples for C's prompt |

### Trajectory Record

Each record should describe diagnostic behavior, not copy scenario-local C-codes:

```python
{
    "scenario_id": str,
    "family": str,
    "tag": str,
    "branch": str,
    "serving_sinr": float,
    "serving_rsrp": float,
    "best_neighbor_advantage": float,
    "neighbor_in_config": bool,
    "mainlobe_outside": bool | None,
    "mr_count": int,
    "tool_sequence": list[str],
    "key_evidence": list[str],
    "root_cause": str,
    "action": str,
    "target_role": str | None,
    "confidence": "high|medium|low"
}
```

### Responsibilities

- Extract trajectory records from Phase 1 train cases.
- Build branch-filtered retrieval over RF/evidence features.
- Provide `retrieve_trajectory_context(sf, branch, k=3) -> str`.
- Provide compact tool-use few-shot examples for C's prompt.
- Ensure retrieved examples never tell the model to copy a training C-code.
- Keep retrieval deterministic and small enough for Phase 2 latency.

### Deliverables

| Artifact | Contents |
|---|---|
| `artifacts/owner_d/docs/trajectory_schema.md/json` | record fields and allowed values |
| `artifacts/owner_d/data/trajectory_examples.jsonl` | Phase 1 trajectory memory |
| `artifacts/owner_d/index/trajectory_index.faiss` or equivalent | retrieval index |
| `retrieve_trajectory_context()` | callable text injection function |
| `artifacts/owner_d/docs/trajectory_prompt_snippets.md` | few-shot tool-use examples grouped by branch |

D does **not** own prompt wording; C decides final prompt integration. D does **not** own model fine-tuning; A owns SFT/LoRA.

---

## 6. Timeline

| Date | A | B | C | D |
|---|---|---|---|---|
| 5/7-5/8 | Keep current candidate audited; prepare integration points | Draft branch/tool/evidence specs | Draft prompt/CoT schema | Draft trajectory schema |
| 5/9-5/10 | Integrate B/C/D minimal packages; run probes | Deliver F1_5BS and high-risk branch specs | Deliver prompt v2 and diagnosis schema | Deliver trajectory examples + retriever prototype |
| 5/11-5/12 | Full run; build/train/eval SFT if ready | Deliver code patches for high-impact skills/tools | Refine prompt from probe traces | Refine retrieval quality and snippets |
| 5/13 | Submission 1 | Review leaderboard-relevant symptoms | Review prompt/RAG behavior | Expand memory for failing branches |
| 5/13-5/15 | Targeted follow-up probes and training | Patch highest-impact skills/tools | Patch prompt/loop failures | Patch retrieval/few-shot coverage |
| 5/16-5/17 | Freeze, final full audit | Low-risk fixes only | Low-risk prompt fixes only | Low-risk retrieval fixes only |
| 5/18 | Final submission | - | - | - |

---

## 7. Three-Submission Budget

| Submission | Target date | Purpose |
|---|---|---|
| Submission 1 | 2026-05-13 | Stable candidate or clean small improvement; get leaderboard signal |
| Submission 2 | 2026-05-15 | Targeted fix based on signal |
| Submission 3 | 2026-05-18 | Conservative final candidate |

Do not spend a submission on an un-audited model-only experiment.
