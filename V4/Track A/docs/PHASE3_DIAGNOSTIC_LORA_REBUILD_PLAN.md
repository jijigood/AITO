# Phase 3 Diagnostic LoRA Rebuild Plan

Date: 2026-05-14

## Current State

Phase 2 should stay conservative. The runtime prompt bugs were real and the
runtime fix is correct, but the current checkpoint does not reliably convert
generic-primary high-PRB `lift_tilt` boundary cases into `insufficient_data`.

Key evidence:

- Phase 1 top-8 candidate coverage is not the ceiling: 2000/2000 GT answers are
  in parsed options and top-8 candidates.
- Phase 1 insufficient_data signal is clean: 252/252 F1 insufficient_data labels
  have deterministic top1 as generic-primary high-PRB `lift_tilt`.
- Runtime-fix targeted Phase 2 test on 10 generic-primary cases produced only
  1 insufficient_data override.

The next version needs a weight update, not another prompt-only patch.

## Highest-Leverage Change

Add `should_override: bool` to the SFT assistant JSON.

This decouples two different questions:

- `evidence_confidence`: how strong the diagnosis evidence is.
- `should_override`: whether the model recommends replacing deterministic top1.

Ground truth definition for single-answer training:

`should_override = deterministic_top1_code != gt_code`

For analysis, also store:

- `deterministic_top1_code`
- `deterministic_top1_action`
- `gt_code`
- `gt_action`
- `should_override_action = deterministic_top1_action != gt_action`

For F1 insufficient_data boundary rows, this gives direct positive supervision:
the deterministic top1 is always generic-primary high-PRB `lift_tilt`, while
the GT action is `insufficient_data`.

## Prompt Parity

Training and runtime Step 2 must use the same prompt structure.

Required blocks:

- Scenario metadata: family, answer cardinality, scenario id.
- Compact all-options block: every C-code/action/target in a concise format.
- Deterministic candidates: scored top candidates with action, target role,
  target cell, reason/subtype, and option code.
- Structured evidence: MR count, serving RSRP/SINR, neighbor advantage,
  A3 threshold/margin, PRB/CCE, feature scores.
- Tool evidence: mainlobe, tilt angle, user location, or explicit `not_available`.
- RAG/trajectory evidence: similar cases, branch-filtered where possible.
- Output contract: JSON only, no boxed answers.

Assistant JSON should include:

```json
{
  "root_cause": "insufficient_data",
  "action": "insufficient_data",
  "target_role": null,
  "target_cell_id": null,
  "relation_role": null,
  "relation_cell_id": null,
  "evidence_confidence": "medium",
  "should_override": true,
  "must_have_evidence": [],
  "blocking_evidence": []
}
```

The assistant still should not output boxed Track A answers. C-codes may appear
in the prompt as available options and deterministic candidates, but the final
assistant diagnosis remains structured JSON.

## Data Rebuild

Update `build_diagnostic_sft.py` or create a v2 builder that:

- Emits compact all-options and deterministic candidates in the same shape as
  `skills/agentic_overlay.py` runtime Step 2.
- Adds `should_override` labels.
- Keeps validation candidate ordering deployable; no label-aware promotion in
  validation.
- Weights `should_override=True` rows more heavily, especially F1
  lift_tilt-vs-insufficient_data boundary rows.
- Keeps hard negatives, but makes the subtype signal explicit:
  generic-primary high-PRB fallback -> insufficient_data;
  coverage hypothesis with isolated tertiary target -> lift_tilt.
- Includes tool evidence when possible. If Phase 1 server access is not
  available, include an explicit `tool_evidence_status: not_available` block
  instead of omitting the section.

## Training Strategy

Do not warm-start from the current checkpoint.

Recommended order:

1. Rebuild SFT JSONL with prompt parity and `should_override`.
2. Run a small smoke training job first.
3. Evaluate boundary heldout before any full run.
4. If smoke passes, train the rank-32 one-epoch run from the base model.

Smoke-test minimum signal:

- heldout insufficient_data should no longer collapse to lift_tilt.
- heldout lift_tilt should not flip wholesale to insufficient_data.
- `should_override=True` recall on deterministic-wrong boundary rows should
  improve clearly over the current checkpoint.
- invalid JSON should remain near zero.

## Gate Before Integration

Keep the original C-code IOU gate, but add override-specific checks.

Required:

- overall mapped C-code IOU >= deterministic baseline.
- lift_tilt IOU >= 0.70.
- insufficient_data IOU >= 0.70.
- add_neighbor IOU >= 0.65.
- invalid JSON <= 2%.
- wrong target role <= 5%.
- `should_override=True` precision must be high enough that broad runtime use
  does not create more regressions than recoveries.
- `should_override=True` recall on deterministic-wrong F1 boundary rows must be
  materially better than the current runtime-fix targeted result.

If lift_tilt or insufficient_data fails the gate, do not integrate the LoRA.

## Runtime Integration

Keep the runtime fixes already made in `skills/agentic_overlay.py`:

- subtype-based insufficient_data rule.
- deterministic candidate block in Step 2.

For Phase 3, change the override gate from confidence-driven to
`should_override`-driven:

- allow override only when `should_override is True`.
- mapped C-code must exist in current options.
- action must be in deterministic top candidates or directly supported by tool
  evidence.
- keep branch/subtype guards for the first integration pass.

Confidence should remain diagnostic metadata, not the primary override switch.
