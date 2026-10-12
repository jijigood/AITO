# Phase 0.5 — Offline Harness Findings

Date: 2026-05-18

## Setup

- Offline Phase-1 harness reusing the exact phase2_run.py deterministic path
  (`infer_scenario_tag → parse_scenario → run_all_skills → _phase2_family →
  generate_candidates → _format_answer`), scored with
  `skills.answer_parser.iou_score` (the competition metric).
- No API dependency; ~0.02 s/scenario.
- F1 stratified by GT action, fixed seed 42, dev=872 / **test=584** (held-out gate).

## Phase-1 family availability (n=2000)

| Family | Phase-1 GT | Validatable offline |
|---|---|---|
| F1 | 1456 | yes (workhorse) |
| F4 | 245 | yes |
| F2 | 228 | yes |
| F3 | 71 | yes |
| F1_5BS | 0 | NO — transfer from F1 by assumption |
| F4M | 0 | NO — must stay frozen |

## F1 GT action distribution (n=1456, near-uniform)

lift_tilt 254 (17.4%) · insufficient_data 252 (17.3%) · pdcch_2sym 240 (16.5%) ·
add_neighbor 240 (16.5%) · check_server 239 (16.4%) · decrease_a3 231 (15.9%).

`adjust_azimuth` = **0/1456**, `press_down_tilt` = **0/1456** → whitelist proven.

## GATE REFERENCE — baseline deterministic, F1 TEST split (n=584)

**Mean IOU = 0.7671.** Confusion (rows=GT, cols=predicted):

| GT action | n | recall | leakage |
|---|---|---|---|
| insufficient_data | 101 | **0.00** | → lift_tilt ×101 |
| add_neighbor | 96 | 0.67 | → decrease_a3 ×31, press_down_tilt ×1 |
| check_server | 96 | 1.00 | — |
| decrease_a3 | 93 | 0.97 | → add_neighbor ×3 |
| lift_tilt | 102 | 1.00 | — (but precision 102/203 = 0.50) |
| pdcch_2sym | 96 | 1.00 | — |

Probes (deterministic, Phase-1): **F2 0.875, F3 1.000, F4 1.000**.

## Conclusions (measured, reprioritized)

1. **insufficient_data branch is 100% broken** → all 101 predicted lift_tilt;
   lift_tilt precision only 0.50. 17.3% of F1. Biggest single F1 lever
   (~+0.058 Track A if ~70% recovered). Was underrated as plan Phase 4.
2. **check_server is perfect (1.00) in pure deterministic**, but ≈0 in the
   submitted `diag_lora_boundary_resubmit` (0.26). The LoRA-boundary override
   destroyed it. Phase 3 → "remove the override", not "Qwen recovery".
3. **60 adjust_azimuth are an override artifact** — pure deterministic emits
   0 on Phase-1. Whitelist guard is a safety net, not the primary fix.
4. **Strategic reframe:** the 0.26 file is a corrupted-by-override version of a
   deterministic pipeline scoring 0.767/0.875/1.0/1.0 (F1/F2/F3/F4) on Phase-1.

## Caveats (held)

- Phase-1 ≠ Phase-2; the clean deterministic pipeline still carries the
  insufficient_data bug and Phase-2 input shift is real. Must be measured on a
  clean Phase-2 live run, not assumed.
- F1_5BS / F4M have no Phase-1 GT; treatment rests on transfer assumptions.

## Next

1. Clean pure-deterministic Phase-2 live run (fast, no submission slot);
   compare answer distribution vs the 0.26 file (expect 0 adjust_azimuth,
   ~16% check_server).
2. Build & gate the insufficient_data vs lift_tilt split on the F1 TEST split.
