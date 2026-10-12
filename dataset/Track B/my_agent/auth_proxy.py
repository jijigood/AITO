#!/usr/bin/env python
"""Authenticated OpenAI-compatible proxy for the local model server."""

import argparse
import os
import time
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse


app = FastAPI(title="Authenticated Model Proxy")

API_KEY = os.environ.get("MODEL_API_KEY", "")
UPSTREAM = os.environ.get("MODEL_UPSTREAM", "http://127.0.0.1:8001").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen")
TIMEOUT_S = float(os.environ.get("MODEL_PROXY_TIMEOUT", "600"))


def _check_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> None:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="MODEL_API_KEY is not configured")
    bearer = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()
    if bearer != API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
async def health():
    return {"status": "ok", "upstream": UPSTREAM}


@app.get("/v1/models")
async def models(authorization: Optional[str] = Header(default=None), x_api_key: Optional[str] = Header(default=None)):
    _check_key(authorization, x_api_key)
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_key(authorization, x_api_key)
    body = await request.body()
    headers = {"content-type": request.headers.get("content-type", "application/json")}
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        upstream_resp = await client.post(f"{UPSTREAM}/v1/chat/completions", content=body, headers=headers)
    return JSONResponse(status_code=upstream_resp.status_code, content=upstream_resp.json())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
