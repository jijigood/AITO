# V3 Frozen Schema Reference

This file is the **authoritative reference** for the four frozen interfaces between owners. Any change to these signatures requires 24-hour notice in the team channel and Owner A sign-off (because A retrains on every schema change).

**Freeze date**: D+2 (2026-04-22).

If you're about to add/rename/remove a field in any dataclass below, stop and open a PR with the change clearly described. Don't just edit.

---

## 1. `ScenarioFeatures` (Owner B)

**Location**: `skills/feature_engine.py` line ~559. Import via:

```python
from skills.feature_engine import ScenarioFeatures, MobilityFeatures, ThroughputFeatures, InterFreqFeatures
```

### Frozen fields

```python
@dataclass
class ScenarioFeatures:
    # Sub-dataclasses (Owner B)
    mobility:    MobilityFeatures       # RSRP, SINR, neighbor_advantage, a3_margin
    throughput:  ThroughputFeatures     # worst_throughput_mbps, drop_magnitude, drop_duration_s
    inter_freq:  InterFreqFeatures      # max_dl_prb_util, max_ul_prb_util

    # 6 issue scores, all in [0, 1]
    score_late_handover:    float
    score_missing_neighbor: float
    score_coverage_hole:    float
    score_pdcch_issue:      float
    score_backhaul_server:  float
    score_inter_freq_load:  float

    # Metadata
    top_issue:          str                 # name of highest score
    serving_cell_id:    str
    serving_cell_role:  Optional[str]       # "primary" | "secondary" | "tertiary"

    # Extended by tool_caller (nullable — required by _build_f1_prompt nullable-from-day-1 rule)
    tool_evidence:           Optional[ToolEvidence] = None
    score_insufficient_data: Optional[float] = None   # pre-approved new field; Owner C writes this via augment_features
```

### Pre-approved additions

- `score_insufficient_data: Optional[float]` — set by Owner C's `augment_features()` when `mr_sample_count < 10`.

### Rules

- New numeric fields must default to `None`; never `0.0`.
- Never repurpose an existing field for a different meaning.
- Adding a new `score_*` field triggers Owner A retrain.

---

## 2. `ToolEvidence` (Owner C)

**Location**: `skills/tool_caller.py` line ~46. Import via:

```python
from skills.tool_caller import ToolEvidence, augment_features
```

### Frozen fields

```python
@dataclass
class ToolEvidence:
    mainlobe_flags:        dict[str, bool]    # cell_id → in main lobe of serving cell?
    tilt_angles:           dict[str, float]   # cell_id → degrees
    overlap_ratio:         dict[str, float]   # cell_id → [0,1]
    kpi_data_raw:          dict                # raw KPI blob from get_kpi_data
    signaling_events:      list                # list of event dicts
    mr_sample_counts:      dict[str, int]     # cell_id → MR sample count
    tool_call_latency_ms:  float              # for wall-time budget tracking
```

### Rules

- All collections default to empty (`{}`, `[]`), never `None`.
- `tool_call_latency_ms` is mandatory — Owner A reads it for §7.5.4 budget tracking.
- `augment_features(sf, ev) → ScenarioFeatures` must be pure (no side effects on its inputs).

---

## 3. `RAGHit` + `retrieve()` (Owner D)

**Location**: `rag_retriever_v2.py`. Import via:

```python
from rag_retriever_v2 import StructuredCaseRAG, RAGHit
```

### Frozen fields

```python
@dataclass
class RAGHit:
    scenario_id:     str
    features:        ScenarioFeatures
    gt_c_code:       str
    gt_action_type:  str                 # lift_tilt / insufficient_data / ...
    similarity:      float               # cosine in [0, 1]
    prompt_snippet:  str                 # 2-line human-readable summary
```

### Frozen signature

```python
def retrieve(
    sf: ScenarioFeatures,
    k: int = 6,
    stratify_by_action: bool = True,
    include_confusion_pairs: bool = True,
) -> list[RAGHit]: ...
```

### Rules

- Owner D may extend feature-vector dim (13 → 16) without breaking this signature.
- Returned list length ≤ k. Fewer is fine.
- `prompt_snippet` must be ≤ 120 chars so the prompt doesn't blow up.

---

## 4. `_build_f1_prompt()` (Owner A — shared template)

**Location**: `main_v3.py`. Imported by both `main_v3.py` and `build_sft_v3.py`.

### Frozen signature

```python
def _build_f1_prompt(
    ps:         ParsedScenario,
    sf:         ScenarioFeatures,
    candidates: list[tuple[str, float, str]],   # (c_code, score, reason)
    rag_hits:   list[RAGHit],
) -> str: ...
```

### Nullable-from-day-1 contract

The template **must render cleanly with any combination of**:

- `sf.tool_evidence is None` (before Owner C lands)
- `sf.score_insufficient_data is None`
- `rag_hits == []`
- Any `RAGHit.features.tool_evidence is None`

This is how A's A3 checkpoint (pre-tools) and A4/A5 (post-tools) share the same template. If you break this, A's SFT dataset becomes invalid overnight.

### Rules

- One function, one symbol, imported everywhere. Do not duplicate.
- Prompt text changes require §7.5.7 process (all-owner review).
- Template is checked by `tests/test_pipeline_integration.py` for render stability.

---

## 5. Ownership map for shared files

| File | Owner | Others may |
|---|---|---|
| `_types.py` | Shared (B/C/D additive only) | Read, import, add new dataclasses at end of file |
| `utils.py` | Shared infrastructure (A curates) | Read only — do not modify |
| `logger.py` | Shared infrastructure (A curates) | Read only — do not modify |
| `server.py` | Competition — DO NOT MODIFY | Read only |
| `rf_analysis.py` | Owner B | Read only |
| `main_v3.py` | Owner A | Read only — open an issue if you need a change |
| `eval_v3.py` | Owner A | Read only — open an issue if you need a change |

---

## 6. Change log

| Date | Change | Approved by |
|---|---|---|
| 2026-04-20 | Initial freeze | A, B, C, D |
