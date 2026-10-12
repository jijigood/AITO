# Runtime-Fix Targeted Test

Date: 2026-05-14

## Code change tested

- Replaced the runtime Step 2 insufficient_data MR-count veto with the
  subtype arbitration rule.
- Injected a deterministic candidate block into Step 2 so the LoRA can see
  `F1 high-PRB healthy-radio -> lift_tilt/primary` vs coverage-hypothesis
  subtype cues.
- Increased the local LoRA HTTP timeout to 120s for the test run.

## Target set

Filtered from:
`results/phase2/sub2_lora_boundary/result_diag_lora_boundary_only.diagnostics.jsonl`

Selection rule:
- deterministic top candidate action is `lift_tilt`
- top reason contains `high-PRB healthy-radio`
- top reason contains `primary`
- top reason does not contain `coverage hypothesis`

Files:
- `generic_primary_30.test.json`
- `generic_primary_30.ids.txt`
- `generic_primary_30_run/result_generic_primary_30_runtimefix.csv`
- `generic_primary_30_run/result_generic_primary_30_runtimefix.diagnostics.jsonl`

## Result after first 10 targeted rows

- completed agentic calls: 10/10
- invalid/timeout/json failures: 0
- diagnosis actions: lift_tilt=9, insufficient_data=1
- final answer changed from deterministic top1: 1/10

The runtime fix does work mechanically: candidate prompts are present and the
longer timeout prevents the previous Step 2 timeout. It also produced one new
insufficient_data override:

- `da968423-724d-4d1d-a4b7-17541f21d116`: C8 -> C15

But it did not unlock the expected majority behavior. The model still chooses
lift_tilt on 9/10 generic-primary high-PRB cases. It also no longer reproduced
one previously useful override in the first 10:

- old run: `6060dca7-32c8-4cd8-adef-425aa1aec774`: C23 -> C22
- runtime-fix targeted run: stays C23

## Conclusion

Do not use the runtime-fix output for a Phase 2 submission as-is.

The confirmed runtime bugs were real, but the existing checkpoint still has a
strong lift_tilt prior on Phase 2 generic-primary cases. The remaining fix is
not just prompt wiring; it needs a Phase 3 SFT rebuild with explicit
`should_override` supervision and train/runtime prompt parity.

Recommended submission posture:
- keep the already-created conservative strict submission for Phase 2
- use these runtime fixes as part of the next architecture
- rebuild SFT with compact all-options, deterministic candidates, tool evidence,
  and `should_override`
