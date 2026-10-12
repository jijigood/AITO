#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
phase2_run.py — Phase 2 deterministic agent for Track A.

Bypasses the LLM entirely.  For each Phase 2 test scenario:
  1. Parse option labels from JSON (no API calls needed).
  2. Route to family (F1/F2/F3/F4) by tag + num_base_stations.
  3. Fetch raw data from Phase 2 cloud server (6 API calls).
  4. Run the V3 deterministic pipeline (feature_engine → candidate_generator).
  5. Optionally call judge_mainlobe for lift_tilt/insufficient_data disambiguation.
  6. Write result.csv for submission.

Usage:
    python phase2_run.py \\
        --server_url https://120.46.145.77/no \\
        --token      <your-bearer-token> \\
        --test_json  data/Phase_2/test.json \\
        --out        result_phase2.csv \\
        --save_freq  10

The output CSV has columns: scenario_id, Track A
Merge with Track B answers separately to produce the final submission.csv.
"""

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure skills/ is importable even when run from a different cwd
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from main import Environment
from skills.scenario_parser import parse_scenario
from skills.scenario_router import route, FAMILIES
from skills.feature_engine import run_all_skills
from skills.candidate_generator import generate_single_candidates
from logger import init_logger


# ---------------------------------------------------------------------------
# Answer formatting
# ---------------------------------------------------------------------------

def _format_answer(candidates, family, tag: str) -> str:
    """Format candidate C-codes as a submission-ready string."""
    if not candidates:
        return ""

    is_multiple = "multiple" in tag.lower()

    if not is_multiple:
        return candidates[0].c_code

    # Multiple-answer: select top 2–4 candidates
    # F3 always returns exactly 4; F2 usually returns 2
    if family.name == "F3":
        selected = candidates[:4]
    else:
        # F2: default to top 2 (extend to 4 only if 3rd candidate has a strong score)
        if len(candidates) >= 4 and candidates[2].score >= 0.55:
            selected = candidates[:4]
        else:
            selected = candidates[:min(2, len(candidates))]

    c_codes = [c.c_code for c in selected]
    # Sort by C-code number for stable output
    c_codes.sort(key=lambda x: int(x[1:]))
    return "|".join(c_codes)


# ---------------------------------------------------------------------------
# Per-scenario solver
# ---------------------------------------------------------------------------

def _solve_scenario(
    scenario: Dict[str, Any],
    env: Environment,
    logger: logging.Logger,
) -> str:
    """
    Run the full deterministic pipeline for one scenario.
    Returns a C-code answer string (e.g. "C5" or "C3|C7").
    Returns empty string on unrecoverable error.
    """
    sid = scenario.get("scenario_id", "?")
    tag = scenario.get("tag", "single-answer")
    ctx = scenario.get("context", {})
    wi = ctx.get("wireless_network_information", {})
    num_bs = int(str(wi.get("num_base_stations", "4")).strip())

    # Route to family
    family = route(tag, num_bs)

    # Phase 2 safety override: respect the actual tag even if V3 expected otherwise.
    # (Phase 2 has 5BS single-answer scenarios that V3's F3 doesn't handle.)
    is_multiple_tag = "multiple" in tag.lower()
    if family.is_multiple != is_multiple_tag:
        if is_multiple_tag:
            family = FAMILIES["F2"]
        else:
            family = FAMILIES["F1"]
        logger.info(f"[{sid}] Family override: {family.name} (tag={tag}, bs={num_bs})")

    logger.info(f"[{sid}] tag={tag}, bs={num_bs}, family={family.name}")

    # ------------------------------------------------------------------
    # Step 1: gather data via API (6 calls)
    # ------------------------------------------------------------------
    from skills.api_data_gatherer import APIDataGatherer
    gatherer = APIDataGatherer(env, scenario, logger=logger)
    raw = gatherer.gather()

    # ------------------------------------------------------------------
    # Step 2: parse scenario (option labels + DataFrames)
    # ------------------------------------------------------------------
    ps = parse_scenario(raw)

    if not ps.parsed_options:
        logger.warning(f"[{sid}] No parsed options found — skipping")
        return ""

    # ------------------------------------------------------------------
    # Step 3: feature extraction
    # ------------------------------------------------------------------
    sf = run_all_skills(ps)

    # ------------------------------------------------------------------
    # Step 4: mainlobe disambiguation for lift_tilt / insufficient_data
    # ------------------------------------------------------------------
    cov = sf.score_coverage_hole or 0.0
    insuff = sf.score_insufficient_data or 0.0
    if abs(cov - insuff) < 0.10 and max(cov, insuff) >= 0.25:
        serving_pci = gatherer.serving_pci
        if serving_pci is None and ps.pci_to_cell_id:
            # Fall back: try to get serving PCI at worst_time
            if gatherer.worst_time:
                r = env._call_api(
                    "get_serving_cell_pci",
                    scenario_id=sid,
                    time=gatherer.worst_time,
                )
                if "error" not in r:
                    serving_pci = r.get("PCI")
        gatherer.serving_pci = serving_pci
        gatherer.disambiguate_mainlobe(sf, serving_pci=serving_pci)
        logger.info(
            f"[{sid}] Mainlobe disambiguation: cov={cov:.2f} insuff={insuff:.2f} "
            f"pci={serving_pci} → cov={sf.score_coverage_hole:.2f} insuff={sf.score_insufficient_data:.2f}"
        )

    # ------------------------------------------------------------------
    # Step 5: candidate generation
    # ------------------------------------------------------------------
    candidates = generate_single_candidates(ps, sf, family, top_n=6)
    if not candidates:
        logger.warning(f"[{sid}] No candidates produced")
        return ""

    answer = _format_answer(candidates, family, tag)
    logger.info(
        f"[{sid}] answer={answer!r} top_candidate={candidates[0].c_code} "
        f"score={candidates[0].score:.3f} action={candidates[0].action_type}"
    )
    return answer


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 2 deterministic Track A agent")
    ap.add_argument("--server_url", default="https://120.46.145.77/no",
                    help="Phase 2 competition server URL")
    ap.add_argument("--token", default=None,
                    help="Bearer token for Phase 2 server (set AGENT_API_KEY env var if preferred)")
    ap.add_argument("--test_json", type=Path,
                    default=Path(__file__).parent / "data/Phase_2/test.json",
                    help="Path to Phase 2 test.json")
    ap.add_argument("--out", type=Path, default=Path("result_phase2.csv"),
                    help="Output CSV path (columns: scenario_id, Track A)")
    ap.add_argument("--save_freq", type=int, default=10,
                    help="Save result.csv every N scenarios")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Limit number of scenarios (for testing)")
    ap.add_argument("--log_file", type=str, default=None,
                    help="Optional log file path")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logger = init_logger(log_file=args.log_file)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Apply bearer token if supplied via CLI (overrides env var)
    import os
    if args.token:
        os.environ["AGENT_API_KEY"] = args.token

    # ------------------------------------------------------------------
    # Load test scenarios
    # ------------------------------------------------------------------
    if not args.test_json.exists():
        logger.error(f"test.json not found: {args.test_json}")
        sys.exit(1)

    with open(args.test_json, encoding="utf-8") as f:
        scenarios: List[Dict[str, Any]] = json.load(f)

    if args.max_samples is not None:
        scenarios = scenarios[: args.max_samples]

    logger.info(f"Loaded {len(scenarios)} Phase 2 scenarios from {args.test_json}")

    # ------------------------------------------------------------------
    # Build Environment (reuse main.py's Environment class)
    # ------------------------------------------------------------------
    env = Environment(server_url=args.server_url, verbose=args.verbose, logger=logger)

    # ------------------------------------------------------------------
    # Solve each scenario
    # ------------------------------------------------------------------
    results: List[Dict[str, str]] = []
    start_total = time.time()

    for idx, scenario in enumerate(scenarios):
        sid = scenario.get("scenario_id", f"unknown_{idx}")
        t0 = time.time()

        try:
            answer = _solve_scenario(scenario, env, logger)
        except Exception:
            logger.error(f"[{sid}] EXCEPTION:\n{traceback.format_exc()}")
            answer = ""

        elapsed = time.time() - t0
        logger.info(f"[{sid}] done in {elapsed:.1f}s — answer: {answer!r}")

        results.append({"scenario_id": sid, "Track A": answer})

        # Periodic save
        if (idx + 1) % args.save_freq == 0 or (idx + 1) == len(scenarios):
            _save_results(results, args.out, logger)

    total_elapsed = time.time() - start_total
    answered = sum(1 for r in results if r["Track A"])
    logger.info(
        f"\nDone. {answered}/{len(results)} answered in {total_elapsed:.0f}s "
        f"(avg {total_elapsed / max(len(results), 1):.1f}s/scenario)"
    )


def _save_results(results: List[Dict[str, str]], out_path: Path, logger: logging.Logger) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario_id", "Track A"])
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Saved {len(results)} rows → {out_path}")


if __name__ == "__main__":
    main()
