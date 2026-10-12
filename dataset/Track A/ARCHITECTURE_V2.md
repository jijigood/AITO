# System Architecture V2 - Track A

## 1. Why A V2 Architecture

The first architecture version is a strong baseline, but it remains too LLM-centric for this benchmark. Track A is not an open-ended reasoning task. It is a structured wireless diagnosis task with:

- fixed question templates
- fixed answer space (`C1` to `C22`)
- repeated action families
- repeated topology patterns
- dense embedded telemetry inside each scenario

This means the highest-leverage system is not "let the model explore the problem from scratch." The better system is:

1. parse the scenario locally into structured data
2. route the case to the right solver family
3. extract diagnostic features
4. use deterministic or semi-deterministic skills to narrow the action space
5. use Qwen to make the final decision among a small candidate set

The V2 architecture is designed around that principle.

---

## 2. Design Goals

1. Maximize leaderboard score under the official IOU metric.
2. Keep per-question latency comfortably below the Phase 3 time threshold.
3. Generalize to hidden test questions from the same distribution.
4. Use Qwen3.5-35B-A3B as the final decision model while avoiding unnecessary free-form tool exploration.
5. Build a system that is ablatable, debuggable, and fine-tunable.

---

## 3. Core Design Principles

### 3.1 Structured diagnosis over free-form agent behavior

The agent should not "discover" the analysis path every time. The analysis path should already exist in the system as code and skills.

### 3.2 Local parsing over repeated tool round-trips

Each scenario already contains the relevant CSV data inside `scenario.data`. The hot path should parse this locally. Server tools remain useful, but mainly for:

- geometry validation
- overlap and antenna simulation
- optional sanity checks
- final low-confidence verification

### 3.3 Route first, solve second

Different topology families behave like different subproblems. The system should split cases before reasoning.

### 3.4 Candidate reduction before final model selection

Qwen should rarely choose from all 22 options directly. The system should first reduce the decision to a small candidate set.

### 3.5 Retrieval should be case-based, not prompt-based

Because the question text is mostly templated, retrieval should operate on structured case summaries and features, not only on raw question text.

### 3.6 Fine-tuning should teach decision behavior, not raw CSV memorization

The fine-tuned model should learn the mapping:

`structured evidence -> answer family -> final answer`

instead of memorizing a long tool trace over raw tabular strings.

---

## 4. High-Level Architecture

```text
                           +----------------------+
                           |   Scenario Source    |
                           |  server.py / input   |
                           +----------+-----------+
                                      |
                                      v
                         +------------+-------------+
                         |     Orchestrator V2      |
                         |        (main.py)         |
                         +------------+-------------+
                                      |
             +------------------------+------------------------+
             |                         |                        |
             v                         v                        v
   +---------+---------+   +-----------+-----------+   +--------+--------+
   |  Scenario Router   |   |   Local Scenario      |   |  Optional Tool  |
   | family assignment  |   |   Parser / Normalizer |   |    Validator    |
   +---------+---------+   +-----------+-----------+   +--------+--------+
             |                         |                        |
             +------------+------------+------------------------+
                          |
                          v
               +----------+-----------+
               |     Feature Engine    |
               | RF / mobility / load  |
               | geometry / signaling  |
               +----------+-----------+
                          |
          +---------------+------------------+
          |                                  |
          v                                  v
 +--------+---------+             +----------+-----------+
 |     Skill Layer   |             | Structured Case RAG |
 | deterministic or  |             | nearest solved      |
 | semi-deterministic|             | examples by family  |
 +--------+---------+             +----------+-----------+
          |                                  |
          +---------------+------------------+
                          |
                          v
               +----------+-----------+
               |   Candidate Generator |
               |  action family score  |
               |  2-5 final candidates |
               +----------+-----------+
                          |
                          v
               +----------+-----------+
               |  Qwen Final Selector  |
               | structured evidence + |
               | candidates + cases    |
               +----------+-----------+
                          |
                          v
               +----------+-----------+
               | Verifier / Fallback   |
               | only for low-confidence|
               +----------+-----------+
                          |
                          v
                     result.csv
```

---

## 5. Scenario Families

The system should explicitly route each question into one of the following families:

1. `4 BS + single-answer`
2. `4 BS + multiple-answer`
3. `5 BS + multiple-answer`
4. `13 BS + single-answer`

This routing should use:

- `task.tag`
- `context.wireless_network_information.num_base_stations`

These families should not share exactly the same solver policy.

### 5.1 Family-specific motivation

- `4 BS + single-answer` is the largest and most diverse family. It needs the richest solver.
- `4 BS + multiple-answer` has constrained answer combinations and should use a combination composer.
- `5 BS + multiple-answer` behaves like a highly regular composite-action subtask.
- `13 BS + single-answer` is much more concentrated around inter-frequency mobility and load balancing behavior.

---

## 6. Main Workflow

For each scenario:

1. Load the scenario.
2. Route the scenario to a family.
3. Parse all embedded CSV strings into normalized internal tables.
4. Extract family-agnostic and family-specific features.
5. Run the relevant skill bundle.
6. Retrieve similar structured cases from the training memory.
7. Score action families and shrink the decision space.
8. Ask Qwen to choose from the reduced candidate set.
9. If confidence is low, invoke a verifier policy and optional server tools.
10. Emit a normalized answer string in ascending order.

---

## 7. Component Detail

### 7.1 Orchestrator

The orchestrator coordinates the full pipeline. It should own:

- family routing
- parser invocation
- feature extraction
- skill execution
- structured RAG retrieval
- Qwen prompt building
- confidence handling
- answer normalization
- result persistence

The orchestrator should avoid placing the model in charge of deciding the entire reasoning sequence.

### 7.2 Local Scenario Parser

This module converts:

- `user_plane_data`
- `network_configuration_data`
- `traffic_data`
- `mr_data`
- `signaling_plane_data`

into normalized tables and helper objects.

The parser should also build:

- PCI to cell-info map
- timestamp-indexed user-plane lookup
- neighbor-list parser
- per-cell KPI summary
- per-scenario metadata summary

This parser should be the primary data interface for the hot path.

### 7.3 Feature Engine

The feature engine transforms raw tables into structured diagnostic signals.

#### Common features

- worst throughput timestamp
- throughput drop magnitude
- recovery behavior after degradation
- serving PCI at worst time
- serving PCI change before and after the worst time
- serving RSRP at worst time
- serving SINR at worst time
- RB allocation at worst time
- BLER / MCS / rank / grant patterns
- strongest visible neighbor and RSRP gap
- whether strongest neighbor is configured
- signaling events around degradation
- per-cell PRB utilization and CCE utilization
- MR summaries for weak coverage and neighbor dominance
- geometry features relative to serving and candidate neighbor cells

#### Derived decision features

- `late_handover_score`
- `missing_neighbor_score`
- `inter_freq_reselection_score`
- `interference_score`
- `coverage_hole_score`
- `overshoot_or_overlap_score`
- `pdcch_issue_score`
- `backhaul_or_server_score`

These should be numerical or categorical signals, not only prose.

### 7.4 Skill Layer

Each skill consumes structured features and emits structured conclusions.

#### Skill A: `throughput_localizer`

Purpose:

- find the most informative time window
- identify the degradation onset, trough, and recovery points

Outputs:

- `worst_time`
- `drop_start_time`
- `drop_end_time`
- `recovery_flag`

#### Skill B: `mobility_skill`

Purpose:

- detect late handover
- detect stuck-serving behavior
- compare neighbor advantage against configured A3 threshold
- detect missing neighbor relations

Outputs:

- `mobility_issue_type`
- `best_neighbor_pci`
- `neighbor_advantage_db`
- `a3_margin_db`
- `neighbor_in_config`
- `recommended_action_family`

#### Skill C: `inter_freq_skill`

Purpose:

- identify inter-frequency mobility or load balancing issues
- use KPI and MR evidence for high-load or poor inter-frequency trigger settings

Outputs:

- `inter_freq_issue_flag`
- `target_cell`
- `load_pressure_flag`
- `recommended_action_family=decrease_a5_a2`

#### Skill D: `coverage_geometry_skill`

Purpose:

- identify azimuth, tilt, mainlobe, and geometric mismatch issues

Outputs:

- `outside_mainlobe_flag`
- `horizontal_angle`
- `tilt_angle`
- `geometry_issue_type`
- `recommended_action_family`

#### Skill E: `power_interference_skill`

Purpose:

- distinguish weak coverage from overshoot/interference
- reason about whether power should increase or decrease

Outputs:

- `coverage_vs_interference`
- `power_direction`
- `overlap_risk_flag`
- `recommended_action_family`

#### Skill F: `capacity_control_skill`

Purpose:

- identify PDCCH, CCE, and resource-control issues

Outputs:

- `pdcch_issue_flag`
- `cce_issue_flag`
- `recommended_action_family=pdcch_2sym`

#### Skill G: `transport_server_skill`

Purpose:

- detect cases where radio looks healthy but throughput is still poor

Outputs:

- `radio_healthy_flag`
- `server_side_suspect_flag`
- `recommended_action_family=check_server`

#### Skill H: `multi_answer_composer`

Purpose:

- compose legal multi-answer sets under family-specific constraints

Outputs:

- ranked answer combinations
- legality checks
- expected answer count

### 7.5 Candidate Generator

This module translates skill outputs into a small final candidate set.

It should score:

- action family
- target cell
- directionality

Then it should map those scores to concrete options from `C1` to `C22`.

The candidate generator should produce:

- 2-5 candidates for single-answer questions
- 2-4 legal answer sets for multiple-answer questions

The generator is the bridge between telecom diagnosis and the competition's option space.

### 7.6 Structured Case RAG

The V2 retrieval layer should not use only raw question text.

Each training example should be converted into a compact case record:

- family label
- answer type
- answer family
- target cell role
- worst-time feature summary
- serving / neighbor relationship summary
- KPI / MR / signaling summary
- geometry summary
- final correct answer

Retrieval should happen in two stages:

1. hard filter by family and answer type
2. nearest-neighbor retrieval by structured case representation

Possible retrieval representations:

- textual case summaries for embedding search
- numeric feature vectors for approximate nearest neighbor
- hybrid retrieval combining both

RAG should provide:

- similar solved cases
- prior probabilities over answer families
- hard-negative awareness for confusing options

### 7.7 Qwen Final Selector

Qwen should not receive the raw scenario as its main reasoning substrate.

Instead, Qwen should receive:

- scenario family
- compact structured evidence
- top candidate options or answer sets
- 2-4 retrieved similar cases
- a short instruction to choose the final answer and output `\boxed{}`

This keeps the model focused on ranking and synthesis rather than search.

Recommended Qwen responsibilities:

- weigh conflicting evidence
- choose the best candidate among a reduced set
- handle residual ambiguity
- produce the final valid answer format

### 7.8 Verifier and Fallback Layer

This layer runs only when the final selector confidence is low.

It may invoke server tools such as:

- `judge_mainlobe_or_not`
- `calculate_horizontal_angle`
- `calculate_tilt_angle`
- `calculate_overlap_ratio`
- `optimize_antenna_gain`
- `get_kpi_data`
- `get_mr_data`
- `get_signaling_plane_event_log`

This keeps tool usage targeted and latency-efficient.

---

## 8. Family-Specific Solver Policies

### 8.1 `13 BS + single-answer`

Priority signals:

- inter-frequency thresholds
- load balancing pressure
- KPI and MR evidence

Primary action family:

- decrease `CovInterFreqA2RsrpThld` and `CovInterFreqA5RsrpThld1`

Solver policy:

1. inspect KPI load and MR context
2. identify whether the scenario indicates inter-frequency load balancing or mobility blocking
3. compare the two candidate inter-frequency actions
4. let Qwen choose the target cell

### 8.2 `5 BS + multiple-answer`

Priority signals:

- composite optimization
- mobility plus coverage shaping

Expected behavior:

- answer count is fixed and should be handled by a constrained composer
- do not let the LLM invent arbitrary multi-answer sets

Solver policy:

1. identify the dominant problematic cell
2. determine whether the composite fix is power + tilt + dual A3 changes
3. rank legal combinations only

### 8.3 `4 BS + multiple-answer`

Priority signals:

- two-action composite fixes
- geometry-plus-power or power-plus-tilt patterns

Solver policy:

1. determine whether the issue is interference or coverage shape
2. choose a legal pair template
3. rank pair candidates with Qwen

### 8.4 `4 BS + single-answer`

Priority signals:

- mobility, missing neighbors, interference, weak coverage, PDCCH, server-side issues

Solver policy:

1. run the full skill bundle
2. score answer families
3. retrieve similar cases
4. let Qwen resolve the top candidates

---

## 9. Tool Usage Policy

### 9.1 Tools should not be the default hot path

The system should first use the embedded scenario data directly.

### 9.2 Tools are for targeted verification

Server tools remain valuable when:

- geometry is ambiguous
- overlap estimation matters
- antenna adjustment impact needs validation
- KPI / MR / signaling evidence should be double-checked

### 9.3 Tool policies by module

`mobility_skill`

- mostly local parsing
- optional `get_signaling_plane_event_log` for handover evidence

`coverage_geometry_skill`

- local geometry first
- optional `judge_mainlobe_or_not`, `calculate_horizontal_angle`, `calculate_tilt_angle`

`power_interference_skill`

- local inference first
- optional `calculate_overlap_ratio` and `optimize_antenna_gain`

`inter_freq_skill`

- local parsing first
- optional `get_kpi_data` and `get_mr_data`

---

## 10. Fine-Tuning Strategy

Fine-tuning should follow a staged plan.

### Stage 0: Strong no-tuning baseline

Build the full structured pipeline first:

- router
- parser
- features
- skills
- structured RAG
- Qwen selector

This baseline should already be competitive and provides clean data for tuning.

### Stage 1: Structured SFT

Training format:

- input: family + structured evidence + candidate options + retrieved cases
- output: final answer with brief justification tags

This teaches Qwen how to convert evidence into the correct competition action.

### Stage 2: Ranking or preference tuning

Construct hard negatives from:

- same-family options
- near-miss answer families
- wrong but plausible target-cell choices
- wrong multi-answer combinations

This is especially valuable because the benchmark contains many semantically similar distractors.

### Stage 3: Optional family-specific adapters

If one unified LoRA underperforms on a specific family, consider:

- one global adapter plus family token
- or a small number of family-specific adapters

This step should be data-driven, not assumed from the start.

---

## 11. Prompting Strategy

The prompt should be short and structured.

Recommended prompt content:

- role: you are selecting the best telecom optimization action
- scenario family
- compact evidence block
- candidate options only
- top retrieved similar cases
- output contract: final `\boxed{}` only or brief reasoning plus `\boxed{}`

The prompt should not contain the entire raw CSV when structured features are already available.

---

## 12. Memory and RAG Design

The memory layer should store one entry per training scenario.

Each entry should include:

- family metadata
- structured feature summary
- issue-class tags
- answer-family tags
- exact final answer
- optional short rationale

Useful tags include:

- `late_handover`
- `missing_neighbor`
- `coverage_hole`
- `interference`
- `overshoot`
- `inter_freq_load_balance`
- `pdcch_issue`
- `test_server`

This memory can support:

- retrieval
- calibration
- error analysis
- synthetic fine-tuning data creation

---

## 13. Confidence Policy

The system should expose confidence at multiple stages:

- feature confidence
- skill confidence
- candidate confidence
- final selector confidence

Suggested policy:

- high confidence: answer directly
- medium confidence: use structured RAG plus Qwen selector
- low confidence: trigger verifier tools and re-rank

This avoids spending extra latency on easy questions.

---

## 14. Offline Evaluation Plan

Use a family-stratified validation split rather than only random splitting.

Recommended reporting:

- mean IOU
- exact match rate
- per-family IOU
- per-answer-family precision and recall
- average latency per scenario
- low-confidence escalation rate

Recommended ablations:

1. no router
2. no structured RAG
3. no skills, Qwen only
4. local parser only versus heavy tool loop
5. no fine-tune versus structured SFT
6. no verifier layer

---

## 15. Phase-by-Phase Deployment

### Phase 1

Goal:

- build the full structured pipeline
- measure which modules move validation IOU
- use public submissions to calibrate

Priority:

- router
- parser
- feature engine
- candidate generator
- structured RAG

### Phase 2

Goal:

- freeze the architecture
- tune thresholds, prompts, retrieval, and optional LoRA

Priority:

- stability
- confidence calibration
- reduced variance across submissions

### Phase 3

Goal:

- maximize accuracy under latency constraints

Priority:

- local parsing on hot path
- minimal tool use
- robust answer formatting
- deterministic fallback policy

---

## 16. Risks and Mitigations

### Risk: overfitting to public prompt behavior

Mitigation:

- optimize on structured validation, not only public leaderboard deltas

### Risk: RAG retrieves textually similar but diagnostically different cases

Mitigation:

- retrieve by family and structured features, not only prompt text

### Risk: excessive tool calls hurt latency

Mitigation:

- tools only for low-confidence verification

### Risk: fine-tune learns synthetic tool traces but not robust decisions

Mitigation:

- train on structured evidence-to-answer mappings, not only trace imitation

### Risk: multi-answer overprediction hurts IOU

Mitigation:

- constrained multi-answer composer with legal set templates and confidence-based ranking

---

## 17. Recommended Implementation Priority

1. Fix official-style local evaluation.
2. Build the router and local parser.
3. Implement the feature engine.
4. Implement family-specific skill bundles.
5. Add the candidate generator.
6. Add structured case RAG.
7. Put Qwen behind the reduced candidate interface.
8. Add verifier tools for low-confidence cases.
9. Fine-tune on structured evidence and hard negatives.

---

## 18. Final Summary

The V2 system is not a generic agent that happens to solve telecom tasks. It is a telecom diagnosis system with Qwen as the final decision model.

That distinction matters.

The winning direction for Track A is likely:

- family-aware routing
- local structured parsing
- feature-driven skills
- structured case retrieval
- candidate reduction
- Qwen final selection
- minimal and targeted tool verification

This design is more aligned with:

- the benchmark's repeated structure
- the fixed answer space
- the Phase 3 latency constraint
- the need for reliable, debuggable improvements
