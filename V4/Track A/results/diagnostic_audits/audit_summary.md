# Diagnostic Audit Summary

Generated on 2026-05-14 from Phase 1 train labels and the current Phase 2
boundary-only diagnostics.

## Phase 1 candidate coverage

Top-8 audit:
- 2000/2000 labels are present in parsed options.
- 2000/2000 labels are present in deterministic top-8.
- 0 parsing failures.
- Coverage buckets: top1=1633, top3=362, top8=5.

Top-5 audit:
- 1999/2000 labels are present in deterministic top-5.
- 1/2000 is in parsed options but outside top-5.
- The one top-5 miss is an F1 add_neighbor case.

Conclusion: for Phase 1 training labels, the hard ceiling is not option parsing
or top-8 candidate omission. All-options prompt expansion is still useful for
train/inference consistency, but it is not the main explanation for the current
insufficient_data failure.

## Phase 1 insufficient_data labels

- insufficient_data GT rows: 252.
- All 252 are F1 single-answer cases.
- All 252 have deterministic top candidate:
  `lift_tilt`, reason `F1 high-PRB healthy-radio`, target role `primary`.
- Ground-truth insufficient_data rank:
  top3=248, top5/top8=4.
- Current subtype risk gate would invoke overlay for all 252.

Conclusion: the training signal is extremely clean: generic primary
high-PRB lift_tilt fallback should be treated as insufficient_data, while the
coverage-hypothesis tertiary subtype is the lift_tilt-positive subtype.

## Current Phase 2 overlay proxy

Existing diagnostics:
`results/phase2/sub2_lora_boundary/result_diag_lora_boundary_only.diagnostics.jsonl`

- 500 Phase 2 rows.
- 81 high-risk lift_tilt_vs_insufficient_data rows.
- 30 rows had deterministic top candidate as generic primary high-PRB lift_tilt.
- Of those 30:
  - completed LLM calls: 27
  - JSON parse failed: 3
  - model diagnosis among completed: lift_tilt=25, insufficient_data=2
  - final answer changed from deterministic top1: 2

Conclusion: the current submitted conservative overlay is safe, but it is not
yet exploiting the insufficient_data training pattern well.

## Structural issues to fix next

1. Runtime `_STEP2_SYSTEM` still says insufficient_data is mainly MR count 0/1
   or outside mainlobe, and explicitly says not to choose it for MR count 3-8.
   This conflicts with the subtype rule learned in the diagnostic SFT data.

2. Runtime Step 1/Step 2 prompts include all answer options, but not the
   deterministic candidate list with reason strings. The SFT prompt is trained
   on `Candidate Diagnoses`; the runtime prompt is therefore missing the exact
   subtype signal (`F1 high-PRB healthy-radio` vs `F1 coverage hypothesis`) that
   the model is supposed to arbitrate.

3. `evidence_confidence` is still overloaded as the override decision. The next
   SFT format should add `should_override`, with ground truth defined as:
   deterministic top1 action/code differs from the GT diagnosis/code.

Recommended next architecture:
- Compact all-options block.
- Scored deterministic candidate block.
- Tool evidence block.
- RAG/trajectory block.
- JSON output includes diagnostic fields plus `should_override`.
