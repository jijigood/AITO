"""Execution-trace export helpers for Phase 2 submissions."""

from __future__ import annotations

import json
import os


TRACE_DIR_NAME = "exec_traces"


def init_exec_trace_dir(result_dir: str, clear: bool = False) -> str:
    trace_dir = os.path.join(result_dir, TRACE_DIR_NAME)
    os.makedirs(trace_dir, exist_ok=True)
    if clear:
        for name in os.listdir(trace_dir):
            if name.endswith(".json"):
                os.remove(os.path.join(trace_dir, name))
    return trace_dir


def export_exec_trace(result_dir: str, detail_row: dict) -> str:
    trace_dir = init_exec_trace_dir(result_dir)
    qid = int(detail_row["id"])
    payload = {
        "id": qid,
        "family": detail_row.get("family"),
        "answer": detail_row.get("answer", ""),
        "answer_valid": detail_row.get("answer_valid"),
        "validation_reasons": detail_row.get("validation_reasons", []),
        "call_count": detail_row.get("call_count", 0),
        "rewrite_count": detail_row.get("rewrite_count", 0),
        "duration_s": detail_row.get("duration_s"),
        "tool_trace": detail_row.get("trace", []),
        "llm_trace": detail_row.get("llm_trace", []),
    }
    path = os.path.join(trace_dir, f"q{qid:02d}.json")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)
    return path
