# CoT Template for Phase 2 Agentic Overlay

## 1. Evidence Comparison Framework

### 1.1 Comparison Matrix Template

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EVIDENCE COMPARISON MATRIX                       │
├─────────────────┬─────────────┬─────────────┬──────────────────────┤
│   Evidence Type │  Candidate 1 │  Candidate 2 │      Decision       │
│                 │ (Score: X)  │ (Score: Y)  │                      │
├─────────────────┼─────────────┼─────────────┼──────────────────────┤
│ SINR (dB)       │   -5.2      │   -4.8      │  Slightly favors C2  │
│ RSRP (dBm)      │   -118      │   -115      │  Favors C2           │
│ MR Count        │     3       │     12      │  Strongly favors C2  │
│ Mainlobe Flag   │    True     │    False    │  Favors C1           │
│ Neighbor Adv    │    5.3 dB   │    4.1 dB   │  Favors C1           │
│ In Config       │    False    │    True     │  Favors C2           │
├─────────────────┼─────────────┼─────────────┼──────────────────────┤
│   CONCLUSION    │   C1: 0.65  │   C2: 0.72  │  SELECT C2 (margin)  │
└─────────────────┴─────────────┴─────────────┴──────────────────────┘
```

### 1.2 Evidence Comparison Checklist

**Coverage Hole vs Insufficient Data:**
- [ ] Serving SINR < -3 dB (coverage indicator)
- [ ] Serving RSRP < -115 dBm (coverage indicator)
- [ ] MR count < 7 (insufficient indicator)
- [ ] Outside_Mainlobe_Flag = True (insufficient indicator)
- [ ] judge_mainlobe_or_not tool result available
- [ ] tilt_angle calculation supports coverage improvement

**Missing Neighbor vs Late Handover:**
- [ ] Best neighbor advantage > 3 dB
- [ ] Neighbor in config = False (missing neighbor indicator)
- [ ] Neighbor in config = True (late handover indicator)
- [ ] Serving SINR < 0 dB (late handover indicator)
- [ ] get_neighboring_cells_pci tool result available
- [ ] get_cell_info confirms PCI presence/absence

## 2. Step 1 CoT Template - Tool Decision Reasoning

```
### Step 1: Tool Request Decision

**Current Evidence Assessment:**
1. Evidence available: {{evidence_summary}}
2. Key missing information: {{missing_info}}
3. Ambiguity level: {{low|medium|high}}

**Decision Analysis:**
- [ ] Evidence is sufficient → skip tool call
- [ ] Evidence is insufficient → request tool

**Tool Selection (if needed):**
- Tool: {{tool_name}}
- Reason: {{reason_for_tool}}
- Expected outcome: {{expected_evidence}}

**Output:**
{"type": "{{tool_request|skip_to_diagnosis}}"[, "tool": "{{tool_name}}", "args": {...}]}
```

## 3. Step 2 CoT Template - Diagnosis Reasoning

### 3.1 Structured Reasoning Format

```
### Step 2: Final Diagnosis

**1. Evidence Summary:**
- Serving cell: PCI={{serving_pci}}, RSRP={{rsrp}}, SINR={{sinr}}
- Best neighbor: PCI={{nbr_pci}}, RSRP={{nbr_rsrp}}, advantage={{nbr_adv}}
- MR count: {{mr_count}}
- Additional tool results: {{tool_results}}

**2. Root Cause Analysis:**
{{root_cause_analysis}}

**3. Evidence Comparison:**
{{evidence_comparison}}

**4. Confidence Assessment:**
- Evidence quality: {{high|medium|low}}
- Evidence completeness: {{high|medium|low}}
- Overall confidence: {{high|medium|low}}

**5. Target Identification:**
- Target cell: PCI={{target_pci}}, CellID={{target_cell_id}}
- Relation cell: PCI={{relation_pci}}, CellID={{relation_cell_id}}

**6. Disagreement with Backbone (if any):**
- Backbone answer: {{backbone_answer}}
- Disagreement reason: {{disagreement_reason}}

**Output:**
{
  "root_cause": "{{root_cause}}",
  "action": "{{action}}",
  "target_pci": {{target_pci}},
  "target_cell_id": "{{target_cell_id}}",
  "relation_pci": {{relation_pci}},
  "relation_cell_id": "{{relation_cell_id}}",
  "evidence_confidence": "{{confidence}}",
  "disagreement_reason": {{disagreement_reason}},
  "evidence_comparison": "{{evidence_comparison}}"
}
```

### 3.2 Root Cause Analysis Templates

**Template A: Coverage Hole**
```
Root cause: coverage_hole
Reasoning:
- Serving SINR ({{sinr}} dB) and RSRP ({{rsrp}} dBm) indicate weak signal
- MR count ({{mr_count}}) is sufficient (> 7)
- Mainlobe flag is False or not available
- Feature scores show coverage_hole={{cov_score}} > insufficient_data={{insuf_score}}
- Action: lift_tilt on serving cell {{serving_cell_id}}
```

**Template B: Insufficient Data**
```
Root cause: insufficient_data
Reasoning:
- MR count ({{mr_count}}) is below threshold (< 7)
- OR Outside_Mainlobe_Flag is True
- Evidence quality is insufficient for confident diagnosis
- Action: insufficient_data (no action possible)
```

**Template C: Missing Neighbor**
```
Root cause: missing_neighbor
Reasoning:
- Best neighbor has significant advantage ({{nbr_adv}} dB > 3)
- Neighbor is NOT in configuration (neighbor_in_config=False)
- Feature scores show missing_neighbor={{miss_score}} > late_handover={{late_score}}
- Action: add_neighbor with target={{nbr_cell_id}}, relation={{serving_cell_id}}
```

**Template D: Late Handover**
```
Root cause: late_handover
Reasoning:
- Serving SINR ({{sinr}} dB) < 0
- Best neighbor has advantage ({{nbr_adv}} dB > 3)
- Neighbor IS in configuration (neighbor_in_config=True)
- Feature scores show late_handover={{late_score}} > missing_neighbor={{miss_score}}
- Action: decrease_a3 on serving cell {{serving_cell_id}}
```

## 4. Evidence Comparison Summary Template

```
**Evidence Comparison Summary:**

| Factor | Candidate A | Candidate B | Weight |
|--------|-------------|-------------|--------|
| SINR | {{sinr}} dB | - | {{weight}} |
| RSRP | {{rsrp}} dBm | - | {{weight}} |
| MR Count | {{mr_count}} | - | {{weight}} |
| Mainlobe | {{mainlobe}} | - | {{weight}} |
| Neighbor Advantage | - | {{nbr_adv}} dB | {{weight}} |
| In Config | - | {{in_config}} | {{weight}} |
| Tool Evidence | {{tool_info}} | - | {{weight}} |

**Scoring:**
- Candidate A: {{score_a}} ({{confidence_a}})
- Candidate B: {{score_b}} ({{confidence_b}})

**Conclusion:** {{winner}} is preferred with {{margin}} margin.
```

## 5. Confidence Assessment Rules

### 5.1 Evidence Quality Matrix

| Factor | High Quality | Medium Quality | Low Quality |
|--------|-------------|----------------|-------------|
| Signal Strength | Clear thresholds met | Borderline values | Missing or contradictory |
| Sample Count | MR > 20 | MR 7-20 | MR < 7 |
| Tool Confirmation | Multiple tools agree | Single tool confirms | No tool confirmation |
| Configuration Match | Exact match | Partial match | No match |

### 5.2 Confidence Calculation

```
Confidence = (Evidence Quality Score) * (Evidence Completeness Score)

Evidence Quality Score:
- All high quality: 1.0
- Mix of high/medium: 0.75
- All medium: 0.5
- Mix of medium/low: 0.25
- All low: 0.0

Evidence Completeness Score:
- All required evidence present: 1.0
- Missing one key piece: 0.75
- Missing two+ key pieces: 0.5

Final Confidence Mapping:
- >= 0.75 → high
- 0.5 - 0.74 → medium
- < 0.5 → low
```
