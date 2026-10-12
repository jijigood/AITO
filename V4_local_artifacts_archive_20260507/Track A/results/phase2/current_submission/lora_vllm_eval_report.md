# LoRA / vLLM Evaluation Report

Date: 2026-05-06

## Serving

- Base model: `/home/huangxiaolin/AITO/model`
- Confirmed architecture:
  - `model_type`: `qwen3_5_moe`
  - `architectures`: `Qwen3_5MoeForConditionalGeneration`
  - experts: 256, experts per token: 8
- Old AITO vLLM env: `0.7.3`
  - Fails: no implementation for `Qwen3_5MoeForConditionalGeneration`
- Working vLLM env: `/home/huangxiaolin/dify_vllm_uv310`
  - vLLM `0.19.1`
  - Successfully starts text-only Qwen3.5 MoE with LoRA adapter `v4`
  - Models exposed by `/v1/models`: `qwen`, `v4`

Working command:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
/home/huangxiaolin/dify_vllm_uv310/bin/vllm serve /home/huangxiaolin/AITO/model \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name qwen \
  --tensor-parallel-size 2 \
  --enable-lora \
  --lora-modules v4=/home/huangxiaolin/AITO/V4/Track\ A/checkpoints/v4_fast_balanced \
  --max-lora-rank 32 \
  --gpu-memory-utilization 0.82 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --trust-remote-code \
  --language-model-only \
  --reasoning-parser qwen3
```

## Heldout Results

Command:

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
/home/huangxiaolin/AITO/bin/python eval_ckpt.py \
  --val-ids heldout_f1_val_ids.txt \
  --train data/Phase_1 \
  --model-url http://127.0.0.1:8001/v1 \
  --model-name v4 \
  --max-tokens 64
```

Result:

| Action | n | IOU |
|---|---:|---:|
| add_neighbor | 38 | 0.6842 |
| check_server | 38 | 1.0000 |
| decrease_a3 | 35 | 0.9429 |
| insufficient_data | 32 | 0.0000 |
| lift_tilt | 26 | 0.0000 |
| pdcch_2sym | 31 | 1.0000 |
| overall F1 | 200 | 0.6400 |

Focused hard-class eval:

- `lift_tilt + insufficient_data`: 58 scenarios, IOU `0.0000`

## Decision

Do not connect `v4_fast_balanced` to the Phase 2 submission agent as a general
reranker. Serving is fixed, but the checkpoint still fails both high-risk F1
classes that matter most for Phase 2 generalization.

Current best validated Phase 2 candidate remains:

```text
results/phase2/current_submission/result_full_uncertainfix.csv
```
