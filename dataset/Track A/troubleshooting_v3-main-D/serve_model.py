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
import re
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
        attn_implementation="eager",
    )

    if adapter_path:
        print(f"[Server] Loading LoRA adapter from {adapter_path}...")
        MODEL = PeftModel.from_pretrained(MODEL, adapter_path)

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

    t0 = time.time()
    with torch.no_grad():
        out = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=TOKENIZER.pad_token_id or TOKENIZER.eos_token_id,
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
