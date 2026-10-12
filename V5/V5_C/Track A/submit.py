#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
submit.py — Owner A's final submission pipeline.

Runs main_v3.py against the competition eval server, writes result_final.csv,
and uploads via the portal.

Deliverable: A6 in TEAM_DIVISION.md §2.3.

Safety:
  --dry-run mode runs everything locally but does not upload. MUST pass a
  full dry-run before any real submission.

Usage:
  python submit.py \\
      --server_url http://eval.server:7860 \\
      --model_url  http://localhost:8001/v1 \\
      --lora       checkpoints/v3/ckpt_latest \\
      --output_dir submissions/run_$(date +%Y%m%d_%H%M%S) \\
      --dry-run
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


VALID_CODE_RE = re.compile(r"^C(?:[1-9]|1[0-9]|2[0-4])(?:\|C(?:[1-9]|1[0-9]|2[0-4]))*$")


def _validate_submission(output_dir: Path) -> dict:
    result_csv = output_dir / "result.csv"
    result_final_csv = output_dir / "result_final.csv"
    if not result_csv.exists():
        raise FileNotFoundError(f"missing {result_csv}")
    if not result_final_csv.exists():
        raise FileNotFoundError(f"missing {result_final_csv}")

    df = pd.read_csv(result_final_csv, dtype=str).fillna("")
    required_cols = ["ID", "Track A", "Track B"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{result_final_csv} missing required columns: {missing}")

    bad_rows = []
    empty_rows = []
    for idx, row in df.iterrows():
        line_no = idx + 2
        track_a = str(row["Track A"]).strip()
        track_b = str(row["Track B"]).strip()
        if not track_a and not track_b:
            empty_rows.append(line_no)
            continue
        if track_a and not VALID_CODE_RE.fullmatch(track_a):
            bad_rows.append((line_no, track_a))
    if empty_rows:
        preview = ", ".join(f"line {line}" for line in empty_rows[:10])
        raise ValueError(f"rows with both Track A and Track B empty: {preview}")
    if bad_rows:
        preview = ", ".join(f"line {line}: {answer!r}" for line, answer in bad_rows[:10])
        raise ValueError(f"invalid Track A C-code(s): {preview}")

    duplicate_ids = int(df["ID"].duplicated().sum())
    if duplicate_ids:
        raise ValueError(f"duplicate scenario IDs in result_final.csv: {duplicate_ids}")

    return {
        "result_csv": str(result_csv),
        "result_final_csv": str(result_final_csv),
        "rows": int(len(df)),
        "duplicate_ids": duplicate_ids,
        "track_a_non_empty": int((df["Track A"].astype(str).str.strip() != "").sum()),
        "track_b_non_empty": int((df["Track B"].astype(str).str.strip() != "").sum()),
    }


def _write_receipt(output_dir: Path, payload: dict) -> None:
    receipt_path = output_dir / "submission_receipt.json"
    receipt_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_url", required=True)
    ap.add_argument("--model_url",  required=True)
    ap.add_argument("--lora",       required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--use-tools", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--save-freq", type=int, default=10)
    ap.add_argument("--model-name", type=str, default="qwen")
    args = ap.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_path = Path(args.lora)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA path does not exist: {lora_path}")

    main_py = Path(__file__).with_name("main_v3.py")
    run_log = output_dir / "run_v3.log"
    stdout_log = output_dir / "main_v3.stdout.log"
    cmd = [
        sys.executable,
        str(main_py),
        "--server_url",
        args.server_url,
        "--model_url",
        args.model_url,
        "--model_name",
        args.model_name,
        "--max_tokens",
        str(args.max_tokens),
        "--save_dir",
        str(output_dir),
        "--log_file",
        str(run_log),
        "--save_freq",
        str(args.save_freq),
    ]
    if args.max_samples is not None:
        cmd.extend(["--max_samples", str(args.max_samples)])
    if args.use_tools:
        cmd.append("--use-tools")
    if args.deterministic:
        cmd.append("--deterministic")
    if args.no_rag:
        cmd.append("--no_rag")

    env = os.environ.copy()
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"

    with stdout_log.open("w", encoding="utf-8") as stdout_fh:
        stdout_fh.write("+ " + " ".join(cmd) + "\n")
        stdout_fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=main_py.parent,
            env=env,
            stdout=stdout_fh,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "server_url": args.server_url,
        "model_url": args.model_url,
        "lora": str(lora_path),
        "output_dir": str(output_dir),
        "command": cmd,
        "main_returncode": proc.returncode,
        "stdout_log": str(stdout_log),
        "run_log": str(run_log),
    }

    if proc.returncode != 0:
        receipt["status"] = "main_v3_failed"
        _write_receipt(output_dir, receipt)
        raise SystemExit(proc.returncode)

    validation = _validate_submission(output_dir)
    receipt["validation"] = validation

    if args.dry_run:
        receipt["status"] = "dry_run_passed"
        receipt["upload"] = "skipped"
        _write_receipt(output_dir, receipt)
        print(f"Dry-run passed: {validation['rows']} rows validated.")
        print(f"Wrote {output_dir / 'result_final.csv'}")
        return

    # The competition portal endpoint/credential is intentionally not encoded
    # in repo. Real upload should be wired here once the official interface is
    # available, after a full dry-run has passed.
    receipt["status"] = "upload_not_configured"
    _write_receipt(output_dir, receipt)
    raise RuntimeError(
        "Real upload is not configured. Run with --dry-run, or wire the official "
        "competition upload endpoint before using non-dry-run mode."
    )


if __name__ == "__main__":
    main()
