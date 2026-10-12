#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Owner A integration verification runner.

This is the fixed "catcher's mitt" for B/C/D handoffs. After any feature,
tool, or RAG change lands, Owner A can run the same sequence and compare the
new result against the current audited baseline.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run(cmd: list[str], log_path: Path, env: Optional[dict] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("+ " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return proc.returncode


def _parse_metric(log_path: Path, pattern: str) -> Optional[float]:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def _parse_action_iou(log_path: Path, action: str) -> Optional[float]:
    return _parse_metric(log_path, rf"{re.escape(action)}\s+n=\s*\d+\s+iou=([0-9.]+)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Owner A handoff verification")
    parser.add_argument("--train-server-url", default="http://127.0.0.1:7861")
    parser.add_argument("--test-server-url", default="http://127.0.0.1:7860")
    parser.add_argument("--model-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--lora", default="checkpoints/v3_zero3/checkpoint-477")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-heldout", action="store_true")
    parser.add_argument("--skip-submission", action="store_true")
    parser.add_argument(
        "--strict-add-neighbor-gate",
        action="store_true",
        help="Fail if add_neighbor < 0.65. Off by default because this is the known open B item.",
    )
    args = parser.parse_args()

    stamp = _timestamp()
    log_dir = ROOT / "logs"
    submission_dir = ROOT / "submissions" / f"owner_a_verify_{stamp}"
    env = os.environ.copy()
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"

    summary = {
        "timestamp": stamp,
        "status": "passed",
        "logs": {},
        "metrics": {},
        "known_gaps": [],
    }

    if not args.skip_pytest:
        pytest_log = log_dir / f"owner_a_verify_pytest_{stamp}.log"
        rc = _run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_tools_regression.py",
                "tests/test_pipeline_integration.py",
                "-v",
            ],
            pytest_log,
            env,
        )
        summary["logs"]["pytest"] = str(pytest_log.relative_to(ROOT))
        if rc != 0:
            summary["status"] = "failed"
            summary["pytest_returncode"] = rc

    if not args.skip_heldout:
        heldout_log = log_dir / f"owner_a_verify_heldout_tools_{stamp}.log"
        rc = _run(
            [
                sys.executable,
                "eval_ckpt.py",
                "--val-ids",
                "heldout_f1_val_ids.txt",
                "--train",
                "data/Phase_1",
                "--server-url",
                args.train_server_url,
                "--use-tools",
                "--max-tokens",
                str(args.max_tokens),
                "--deterministic",
                "--no-rag",
            ],
            heldout_log,
            env,
        )
        summary["logs"]["heldout"] = str(heldout_log.relative_to(ROOT))
        if rc != 0:
            summary["status"] = "failed"
            summary["heldout_returncode"] = rc
        else:
            metrics = {
                "overall_iou": _parse_metric(heldout_log, r"Overall F1 IOU\s*:\s*([0-9.]+)"),
                "add_neighbor_iou": _parse_action_iou(heldout_log, "add_neighbor"),
                "insufficient_data_iou": _parse_action_iou(heldout_log, "insufficient_data"),
                "lift_tilt_iou": _parse_action_iou(heldout_log, "lift_tilt"),
            }
            summary["metrics"].update(metrics)
            if (metrics["overall_iou"] or 0.0) < 0.75:
                summary["status"] = "failed"
            if (metrics["insufficient_data_iou"] or 0.0) < 0.60:
                summary["status"] = "failed"
            if (metrics["lift_tilt_iou"] or 0.0) < 0.60:
                summary["status"] = "failed"
            if (metrics["add_neighbor_iou"] or 0.0) < 0.65:
                summary["known_gaps"].append("add_neighbor_iou_below_0.65")
                if args.strict_add_neighbor_gate:
                    summary["status"] = "failed"

    if not args.skip_submission:
        submit_log = log_dir / f"owner_a_verify_submission_{stamp}.log"
        rc = _run(
            [
                sys.executable,
                "submit.py",
                "--server_url",
                args.test_server_url,
                "--model_url",
                args.model_url,
                "--lora",
                args.lora,
                "--output_dir",
                str(submission_dir.relative_to(ROOT)),
                "--dry-run",
                "--use-tools",
                "--deterministic",
                "--no-rag",
                "--max-tokens",
                str(args.max_tokens),
            ],
            submit_log,
            env,
        )
        summary["logs"]["submission"] = str(submit_log.relative_to(ROOT))
        summary["submission_dir"] = str(submission_dir.relative_to(ROOT))
        if rc != 0:
            summary["status"] = "failed"
            summary["submission_returncode"] = rc

    summary_path = log_dir / f"owner_a_verify_summary_{stamp}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
