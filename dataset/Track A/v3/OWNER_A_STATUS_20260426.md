# Owner A Status Audit — 2026-04-26

## Verdict

Owner A's SFT and LoRA training work has been executed, but the original LoRA
checkpoint target was not achieved. The submission dry-run path is now
implemented and tested. Real leaderboard upload remains not configured because
the official portal/API details are not present in this repository.

## Deliverable Checklist

| Item | Status | Evidence | Notes |
|---|---|---|---|
| A1a curation | Done with caveat | `curated_train_ids.txt`, `curation_report.md` | Report shows `Mode: fallback`, so Owner D's final `data_tools.py` scoring was not used. |
| A1b SFT build | Done | `sft_v3.jsonl`, `sft_heldout_val.jsonl` | `sft_v3.jsonl` has 1311 F1 records after balancing. |
| A2 LoRA config | Done | `configs/lora_v3.yaml`, `configs/lora_v3_zero3.yaml` | Configs match LoRA r=32/alpha=64/dropout=0.05 and target projection modules. |
| A3 first checkpoint | Done but target missed | `checkpoints/v3_zero3/checkpoint-477`, `logs/eval_ckpt477_fixed_t768.log` | F1 IOU `0.6000`, below the planned `>=0.70`. |
| A3 corrected checkpoint | Done but target missed | `checkpoints/v3_a3_corrected/checkpoint-246`, `logs/eval_corrected_fixed_t768.log` | F1 IOU `0.6050`, best LoRA-only accuracy so far. |
| A4 tool-evidence retrain | Not done | No tool-evidence LoRA checkpoint found | Current `0.8500` is `--deterministic --no-rag --use-tools`, not a LoRA retrain. |
| A5 full RAG/confusion retrain | Not done | No full-RAG LoRA checkpoint or ablation found | D should shift to RAG ablations against the tools baseline before A retrains. |
| A6 submission pipeline | Dry-run done, upload not configured | `submit.py`, `submissions/dry_run_det_tools_20260426_2030/submission_receipt.json`, `logs/owner_a_verify_summary_20260426_230608.json` | Dry-run produced and validated 500-row `result_final.csv`; `owner_a_verify.py` now wraps the repeatable A handoff check. |
| A7 ablation report | Done for current evidence | `ablation_report_20260426.csv`, `ABLATION.md` | Captures deterministic, LoRA-only, and tools-baseline comparisons. |

## Current Baselines

| Baseline | Overall F1 IOU | Latency | Source |
|---|---:|---:|---|
| Best LoRA-only accuracy | 0.6050 | 40.8s | `logs/eval_corrected_fixed_t768.log` |
| Best LoRA-only operational latency | 0.6000 | 19.4s | `logs/eval_ckpt477_fixed_t768.log` |
| Best auditable tools baseline | 0.8500 | 0.0s | `logs/eval_deterministic_tools_full_20260426_2021_rerun.log` |

## Remaining Owner A Work

1. Do not spend more time on longer LoRA training over the same dataset unless B/C/D change the input distribution.
2. After any B/C/D handoff, run `python owner_a_verify.py --train-server-url http://127.0.0.1:7861 --test-server-url http://127.0.0.1:7860`.
3. Wait for B's `add_neighbor` candidate-targeting fix, then rerun the 200-scenario heldout eval with `--strict-add-neighbor-gate`.
4. Ask D for RAG ablation numbers against `deterministic + tools`; only retrain LoRA if RAG helps or if B's fixes create a new prompt distribution.
5. Wire real leaderboard upload once the official portal/API details are available.
