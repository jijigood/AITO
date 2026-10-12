#!/usr/bin/env python3
"""
Track B competition agent — main entry point.

Run from the trackb/ project root:

    # All 50 questions (max 2 concurrent):
    uv run agent/main.py \\
        --input data/Phase_1/test.json \\
        --server_url http://localhost:7860 \\
        --model_url  http://localhost:8001/v1 \\
        --model_name qwen

    # Specific questions only:
    uv run agent/main.py --questions 1,2,5

    # Remote server (requires auth token):
    uv run agent/main.py \\
        --server_url http://124.71.227.61 \\
        --token YOUR_BEARER_TOKEN

    # Resume an interrupted run:
    uv run agent/main.py --resume
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Optional

import httpx
from openai import OpenAI

# Allow importing sibling modules when invoked as `python my_agent/main.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools import NOCClient
from agent_loop import solve
from prompts import detect_family
from answer_validation import validate_answer
from trace_export import export_exec_trace, init_exec_trace_dir

# ---------------------------------------------------------------------------
# Paths (relative to this file so they work from any cwd)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(_HERE, "results")
RESULT_CSV = os.path.join(RESULT_DIR, "result.csv")
DETAIL_JSONL = os.path.join(RESULT_DIR, "detail.jsonl")
PROGRESS_JSON = os.path.join(RESULT_DIR, "progress.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_questions(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return sorted(
        [{"id": item["task"]["id"], "question": item["task"]["question"]} for item in data],
        key=lambda x: x["id"],
    )


def load_progress() -> set:
    result_ids: set[int] | None = None
    if os.path.exists(RESULT_CSV):
        with open(RESULT_CSV, encoding="utf-8-sig", newline="") as f:
            result_ids = {
                int(row["id"])
                for row in csv.DictReader(f)
                if row.get("id") and row.get("prediction", "").strip()
            }
    if os.path.exists(PROGRESS_JSON):
        with open(PROGRESS_JSON) as f:
            completed = set(json.load(f).get("completed", []))
        return completed if result_ids is None else completed & result_ids
    if result_ids is not None:
        return result_ids
    return set()


def save_progress(completed: set, lock: Lock):
    with lock:
        with open(PROGRESS_JSON, "w") as f:
            json.dump({"completed": sorted(completed)}, f)


def init_output_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(RESULT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(["id", "prediction"])
    open(DETAIL_JSONL, "w").close()  # clear detail log
    init_exec_trace_dir(RESULT_DIR, clear=True)


def append_csv(lock: Lock, qid: int, prediction: str):
    with lock:
        with open(RESULT_CSV, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([qid, prediction])


def append_detail(lock: Lock, row: dict):
    with lock:
        with open(DETAIL_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sort_result_csv():
    """Keep submission rows deterministic even when questions run concurrently."""
    if not os.path.exists(RESULT_CSV):
        return
    with open(RESULT_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    # A resumed/interrupted run should still leave a submission-shaped CSV:
    # one row per id, sorted by id. If duplicates exist, keep the latest row.
    unique_rows = {int(row["id"]): row for row in rows}
    rows = [unique_rows[qid] for qid in sorted(unique_rows)]
    with open(RESULT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prediction"])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------

def make_llm_client(model_url: str) -> OpenAI:
    return OpenAI(
        base_url=model_url,
        api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
        # trust_env=False disables ALL_PROXY / http_proxy env vars so localhost calls
        # are not routed through the system proxy
        http_client=httpx.Client(verify=False, trust_env=False, timeout=120.0),
    )


# ---------------------------------------------------------------------------
# Per-question worker
# ---------------------------------------------------------------------------

def process_one(
    q: dict,
    server_url: str,
    model_url: str,
    model_name: str,
    token: Optional[str],
    csv_lock: Lock,
    detail_lock: Lock,
    progress_lock: Lock,
    completed: set,
) -> tuple:
    qid = q["id"]
    question = q["question"]
    family = detect_family(question)

    llm_client = make_llm_client(model_url)
    noc_client = NOCClient(server_url, qid, token=token)

    t0 = time.time()
    try:
        prediction = solve(question, qid, llm_client, noc_client, model_name=model_name)
        prediction = prediction.strip()
    except Exception as exc:
        log.error("Q%d EXCEPTION: %s", qid, exc, exc_info=True)
        prediction = ""

    duration = time.time() - t0
    validation = validate_answer(prediction, family, question)

    detail_row = {
        "id": qid,
        "family": family,
        "call_count": noc_client.call_count,
        "rewrite_count": noc_client.rewrite_count,
        "answer_valid": validation.ok,
        "validation_reasons": list(validation.reasons),
        "trace": noc_client.trace,
        "llm_trace": noc_client.llm_trace,
        "answer": prediction,
        "duration_s": round(duration, 1),
    }

    append_csv(csv_lock, qid, prediction)
    append_detail(detail_lock, detail_row)
    export_exec_trace(RESULT_DIR, detail_row)

    completed.add(qid)
    save_progress(completed, progress_lock)

    preview = prediction.replace("\n", " | ")[:70]
    print(
        f"[Q{qid:02d}] {family:8s} | {duration:5.0f}s | {noc_client.call_count:3d} calls"
        f" | {preview!r}"
    )
    return qid, prediction


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Track B competition agent")
    parser.add_argument(
        "--input", default="data/Phase_1/test.json",
        help="Path to test.json (default: data/Phase_1/test.json)",
    )
    parser.add_argument(
        "--server_url", default="http://localhost:7860",
        help="Agent Tool Server URL (default: http://localhost:7860)",
    )
    parser.add_argument(
        "--model_url", default="http://localhost:8001/v1",
        help="OpenAI-compatible LLM server base URL",
    )
    parser.add_argument(
        "--model_name", default="qwen",
        help="Model name passed to the LLM API",
    )
    parser.add_argument(
        "--token", default=None,
        help="Bearer token for remote Agent Tool Server authentication",
    )
    parser.add_argument(
        "--questions", default=None,
        help="Comma-separated question IDs to run (e.g. --questions 1,2,5)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=2,
        help="Max concurrent questions (competition limit: 2)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip questions already present in progress.json",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory for result.csv/detail.jsonl/progress.json (default: my_agent/results)",
    )
    args = parser.parse_args()

    global RESULT_DIR, RESULT_CSV, DETAIL_JSONL, PROGRESS_JSON
    if args.output_dir:
        RESULT_DIR = os.path.abspath(args.output_dir)
        RESULT_CSV = os.path.join(RESULT_DIR, "result.csv")
        DETAIL_JSONL = os.path.join(RESULT_DIR, "detail.jsonl")
        PROGRESS_JSON = os.path.join(RESULT_DIR, "progress.json")

    questions = load_questions(args.input)
    log.info("Loaded %d questions from %s", len(questions), args.input)

    if args.questions:
        ids = {int(x.strip()) for x in args.questions.split(",") if x.strip()}
        questions = [q for q in questions if q["id"] in ids]
        log.info("Filtered to %d questions: %s", len(questions), sorted(ids))

    completed = load_progress() if args.resume else set()
    if completed:
        before = len(questions)
        questions = [q for q in questions if q["id"] not in completed]
        log.info("Resume: skipped %d completed, %d remaining", before - len(questions), len(questions))

    if not questions:
        print("No questions to process.")
        return

    if args.resume:
        os.makedirs(RESULT_DIR, exist_ok=True)  # don't clear existing CSV/JSONL
    else:
        init_output_dir()

    csv_lock = Lock()
    detail_lock = Lock()
    progress_lock = Lock()

    print(f"\n{'='*65}")
    print(f"Track B Agent  —  {len(questions)} questions  |  concurrency={args.concurrency}")
    print(f"Server : {args.server_url}")
    print(f"LLM    : {args.model_url}  [{args.model_name}]")
    print(f"Output : {RESULT_CSV}")
    print(f"{'='*65}\n")

    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                process_one,
                q,
                args.server_url,
                args.model_url,
                args.model_name,
                args.token,
                csv_lock,
                detail_lock,
                progress_lock,
                completed,
            ): q["id"]
            for q in questions
        }
        for future in as_completed(futures):
            qid = futures[future]
            try:
                future.result()
            except Exception as exc:
                log.error("Q%d thread raised: %s", qid, exc, exc_info=True)

    total = time.time() - t_start
    sort_result_csv()
    print(f"\n{'='*65}")
    print(f"All done in {total:.0f}s ({total/60:.1f} min)")
    print(f"Results : {RESULT_CSV}")
    print(f"Details : {DETAIL_JSONL}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
