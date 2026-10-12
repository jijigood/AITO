#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Lightweight OpenAI-compatible inference server for Qwen3.5-35B-A3B + LoRA adapter.
Exposes POST /v1/chat/completions — compatible with main_v2.py's OpenAI client.

Usage:
    CUDA_VISIBLE_DEVICES=0 python serve_model.py \
        --model_path /home/huangxiaolin/AITO/model \
        --adapter_path finetune/lora_output_v2 \
        --port 8000

Then point main_v2.py at:  --model_url http://localhost:8000/v1
"""

import argparse
import gc
import os
import re
import shutil
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────────────────────
# Request / response schemas (minimal OpenAI subset)
# ──────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 64
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False

# ──────────────────────────────────────────────────────────────────────────────
# Global model state
# ──────────────────────────────────────────────────────────────────────────────

MODEL = None
TOKENIZER = None
MAX_INPUT_TOKENS = int(os.environ.get("TRACKB_SERVER_MAX_INPUT_TOKENS", "8192"))

# Expected number of LoRA-wrapped Linear modules for our r=32 config:
#   40 layers × shared_expert.{down,gate,up}_proj  = 120
#   10 layers × self_attn.{q,k,v,o}_proj           =  40
_EXPECTED_LORA_MODULES = 160
_EXPECTED_LORA_TENSORS = 320  # 2 (A+B) per module


def _remap_adapter_dir(adapter_path: str) -> Optional[str]:
    """
    If adapter keys carry a .language_model. segment that the serving model
    (AutoModelForCausalLM, path: model.layers.*) doesn't have, strip that
    segment and write a corrected safetensors to a temp dir.

    Returns the temp dir path if remapping was needed, else None.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    st_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(st_file):
        return None

    with safe_open(st_file, framework="pt") as f:
        keys = list(f.keys())
        if not any(".language_model." in k for k in keys):
            print("[Server] Adapter key prefix OK — no remap needed.")
            return None
        print(f"[Server] Detected .language_model. prefix in {len(keys)} adapter keys — remapping...")
        tensors = {k: f.get_tensor(k) for k in keys}

    remapped = {k.replace(".language_model.", "."): v for k, v in tensors.items()}

    tmp = tempfile.mkdtemp(prefix="peft_remap_")
    for fname in os.listdir(adapter_path):
        src = os.path.join(adapter_path, fname)
        if fname == "adapter_model.safetensors" or not os.path.isfile(src):
            continue
        shutil.copy2(src, tmp)
    save_file(remapped, os.path.join(tmp, "adapter_model.safetensors"))
    print(f"[Server] Remapped adapter written to {tmp}")
    return tmp


def _lora_hard_check(model) -> None:
    """
    Abort if LoRA modules weren't actually wired into the model.
    Counts nn.Modules that have a non-empty lora_A attribute (PEFT LoRA Linear).
    """
    n = sum(
        1 for _, m in model.named_modules()
        if hasattr(m, "lora_A") and len(m.lora_A) > 0
    )
    if n != _EXPECTED_LORA_MODULES:
        raise RuntimeError(
            f"[Server] HARD CHECK FAILED: expected {_EXPECTED_LORA_MODULES} LoRA modules, "
            f"got {n}. Adapter keys did not map to model modules — check prefix mismatch."
        )
    print(f"[Server] LoRA hard check PASSED: {n}/{_EXPECTED_LORA_MODULES} modules loaded "
          f"({_EXPECTED_LORA_TENSORS} tensors).")


def load_model(model_path: str, adapter_path: Optional[str]):
    global MODEL, TOKENIZER
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok_path = adapter_path if adapter_path else model_path
    print(f"[Server] Loading tokenizer from {tok_path}...")
    TOKENIZER = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    print(f"[Server] Loading base model from {model_path}...")
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation=os.environ.get("TRACKB_ATTN_IMPLEMENTATION", "sdpa"),
    )

    if adapter_path:
        remap_dir = _remap_adapter_dir(adapter_path)
        load_from = remap_dir if remap_dir else adapter_path
        print(f"[Server] Loading LoRA adapter from {load_from}...")
        MODEL = PeftModel.from_pretrained(MODEL, load_from)
        if remap_dir:
            shutil.rmtree(remap_dir)
        _lora_hard_check(MODEL)

    MODEL.eval()
    print("[Server] Model ready. Starting HTTP server...")


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Qwen Inference Server")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if MODEL is None or TOKENIZER is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    max_new_tokens = req.max_tokens or 64

    # Apply chat template with thinking disabled
    text = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = TOKENIZER(text, return_tensors="pt").to(MODEL.device)
    if inputs["input_ids"].shape[1] > MAX_INPUT_TOKENS:
        inputs = {k: v[:, -MAX_INPUT_TOKENS:] for k, v in inputs.items()}

    t0 = time.time()
    try:
        with torch.inference_mode():
            out = MODEL.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=TOKENIZER.pad_token_id or TOKENIZER.eos_token_id,
            )
    except torch.OutOfMemoryError as exc:
        gc.collect()
        torch.cuda.empty_cache()
        return JSONResponse(
            status_code=503,
            content={"error": "CUDA out of memory", "detail": str(exc)[:500]},
        )
    latency = time.time() - t0

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    decoded = TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()
    decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()

    # OpenAI-compatible response envelope
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": decoded},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": len(new_tokens),
            "total_tokens": inputs["input_ids"].shape[1] + len(new_tokens),
        },
        "latency_s": round(latency, 3),
    }
    del inputs, out, new_tokens
    gc.collect()
    torch.cuda.empty_cache()
    return JSONResponse(content=response)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   default="/home/huangxiaolin/AITO/model")
    parser.add_argument("--adapter_path", default="finetune/lora_output_v2",
                        help="Path to LoRA adapter dir, or empty string to use base model")
    parser.add_argument("--host",         default="0.0.0.0")
    parser.add_argument("--port",         type=int, default=8000)
    args = parser.parse_args()

    adapter = args.adapter_path if args.adapter_path else None
    load_model(args.model_path, adapter)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
