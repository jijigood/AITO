#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ADAPTER="/home/huangxiaolin/AITO/V4/Track A/checkpoints/diag_lora_r32_1epoch/checkpoint-50"
ADAPTER="${1:-${DIAG_LORA_ADAPTER:-${DEFAULT_ADAPTER}}}"
PORT="${PORT:-8001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
TOKENIZER="${TOKENIZER:-${ADAPTER}}"

if [[ ! -d "${ADAPTER}" ]]; then
  echo "Adapter checkpoint not found: ${ADAPTER}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec /home/huangxiaolin/dify_vllm_uv310/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/huangxiaolin/AITO/model \
  --tokenizer "${TOKENIZER}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --served-model-name qwen \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enforce-eager \
  --enable-lora \
  --lora-modules "diag=${ADAPTER}" \
  --max-lora-rank "${MAX_LORA_RANK}"
