"""
Phase-1 offline accuracy harness.

Reuses the EXACT phase2_run.py deterministic path (no reimplementation):
  infer_scenario_tag -> parse_scenario -> run_all_skills
  -> _phase2_family -> generate_candidates -> _format_answer
Scores with skills.answer_parser.iou_score (the competition metric).

Provides:
  - family assignment (faithful, via _phase2_family)
  - GT/predicted C-code -> action mapping (via ps.parsed_options)
  - stratified dev/test split for F1 (by GT action, fixed seed)
  - per-family mean IOU, per-action confusion matrix, illegal-action count
  - a pluggable post-processor hook so v6_1..v6_4 reuse cached features

This is the offline accuracy GATE: every v6 mechanism must clear it on the
held-out TEST split BEFORE touching the Phase-2 submission.
"""

import json
import random
import sys
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, "/home/huangxiaolin/AITO/V5/Track A")

from skills.scenario_parser import infer_scenario_tag, parse_scenario
from skills.feature_engine import run_all_skills
from skills.candidate_generator import generate_candidates
from skills.answer_parser import iou_score
from phase2_run import _phase2_family, _format_answer

PHASE1 = "/home/huangxiaolin/AITO/V5/Track A/data/Phase_1/train.json"
SEED = 42

# Phase-1 GT-proven F1 single-answer valid actions (0/1456 for the rest).
F1_VALID_ACTIONS = {
    "add_neighbor", "check_server", "decrease_a3",
    "insufficient_data", "lift_tilt", "pdcch_2sym",
}

# A post-processor takes the per-scenario context and returns a (possibly
# rewritten) answer string. Used to layer v6_1..v6_4 without re-running skills.
PostProc = Callable[[dict], str]


def code_action(ps, code: str) -> str:
    if not code:
        return "NONE"
    first = code.split("|")[0].strip()
    return ps.parsed_options.get(first, {}).get("action_type", "UNKNOWN")


def load_scenarios() -> List[dict]:
    with open(PHASE1) as f:
        return json.load(f)


def family_of(scenario: dict) -> str:
    tag = infer_scenario_tag(scenario)
    ctx = scenario.get("context", {})
    wi = ctx.get("wireless_network_information", {}) if isinstance(ctx, dict) else {}
    try:
        num_bs = int(str(wi.get("num_base_stations", "4")).strip())
    except (ValueError, TypeError):
        num_bs = 4
    _, family, _ = _phase2_family(tag, num_bs)
    return family.name


def evaluate(
    scenarios: List[dict],
    post: Optional[PostProc] = None,
) -> List[dict]:
    """Run the deterministic core on each scenario; return per-scenario records."""
    out = []
    for s in scenarios:
        rec = {"scenario_id": s.get("scenario_id", ""), "error": None}
        try:
            tag = infer_scenario_tag(s)
            ps = parse_scenario(s)
            if not ps.parsed_options:
                rec["error"] = "no_parsed_options"
                out.append(rec)
                continue
            sf = run_all_skills(ps)
            num_bs = ps.num_base_stations
            _, family, _ = _phase2_family(tag, num_bs)
            cand = generate_candidates(ps, sf, family)
            answer = _format_answer(cand, tag)

            gt = str(s.get("answer", "") or "")
            rec.update({
                "family": family.name,
                "tag": tag,
                "gt": gt,
                "gt_action": code_action(ps, gt),
                "det_answer": answer,
                "det_action": code_action(ps, answer),
            })
            if post is not None:
                ctx = {"ps": ps, "sf": sf, "family": family, "tag": tag,
                        "det_answer": answer, "scenario": s}
                answer = post(ctx)
            rec["answer"] = answer
            rec["pred_action"] = code_action(ps, answer)
            rec["iou"] = iou_score(answer, gt)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        out.append(rec)
    return out


def stratified_split(
    scenarios: List[dict], family: str = "F1",
    dev_frac: float = 0.6,
) -> Tuple[List[dict], List[dict]]:
    """Split one family by GT action, fixed seed, balanced per action."""
    rng = random.Random(SEED)
    by_action: Dict[str, List[dict]] = defaultdict(list)
    for s in scenarios:
        if family_of(s) != family:
            continue
        gt = str(s.get("answer", "") or "")
        if not gt or "|" in gt:
            continue
        ps = parse_scenario(s)
        by_action[code_action(ps, gt)].append(s)
    dev, test = [], []
    for action, group in sorted(by_action.items()):
        rng.shuffle(group)
        cut = int(len(group) * dev_frac)
        dev += group[:cut]
        test += group[cut:]
    return dev, test


def summarize(records: List[dict], label: str) -> dict:
    ok = [r for r in records if not r.get("error") and "iou" in r]
    errs = [r for r in records if r.get("error")]
    by_fam_iou: Dict[str, List[float]] = defaultdict(list)
    confusion: Dict[str, Counter] = defaultdict(Counter)
    illegal = []
    for r in ok:
        by_fam_iou[r["family"]].append(r["iou"])
        if r["family"] in ("F1", "F1_5BS"):
            confusion[r["gt_action"]][r["pred_action"]] += 1
            if r["pred_action"] not in F1_VALID_ACTIONS and r["pred_action"] != "NONE":
                illegal.append((r["scenario_id"][:8], r["pred_action"]))

    print(f"\n===== {label} =====")
    print(f"evaluated={len(records)} ok={len(ok)} errors={len(errs)}")
    if errs:
        ec = Counter(r["error"].split(":")[0] for r in errs)
        print(f"error types: {dict(ec)}")
    print("\nPer-family mean IOU:")
    fam_means = {}
    for fam in sorted(by_fam_iou):
        vals = by_fam_iou[fam]
        m = sum(vals) / len(vals)
        fam_means[fam] = m
        print(f"  {fam:8s} n={len(vals):4d}  IOU={m:.4f}")

    if confusion:
        actions = sorted(F1_VALID_ACTIONS | {"adjust_azimuth", "press_down_tilt",
                                              "decrease_power", "UNKNOWN", "NONE"})
        print("\nF1/F1_5BS confusion (rows=GT action, cols=predicted):")
        gtlabel = "GT\\PRED"
        hdr = "  " + f"{gtlabel:18s}" + "".join(f"{a[:9]:>10s}" for a in actions) + f"{'recall':>9s}"
        print(hdr)
        for gt_a in sorted(confusion):
            row = confusion[gt_a]
            tot = sum(row.values())
            hit = row.get(gt_a, 0)
            cells = "".join(f"{row.get(a,0):>10d}" for a in actions)
            print(f"  {gt_a:18s}{cells}{hit/tot if tot else 0:>9.2f}")
        print(f"\nF1 illegal-action predictions: {len(illegal)}")
        if illegal:
            ic = Counter(a for _, a in illegal)
            print(f"  by action: {dict(ic)}")
    return {"fam_means": fam_means, "n_ok": len(ok), "illegal": len(illegal)}


if __name__ == "__main__":
    scen = load_scenarios()
    print(f"Loaded {len(scen)} Phase-1 scenarios")

    # 0.5b: stratified F1 dev/test split
    dev, test = stratified_split(scen, "F1")
    print(f"F1 stratified split: dev={len(dev)} test={len(test)} (seed={SEED})")
    with open("/home/huangxiaolin/AITO/V5/Track A/results/phase1_harness_split.json", "w") as f:
        json.dump({"dev": [s["scenario_id"] for s in dev],
                   "test": [s["scenario_id"] for s in test]}, f)

    # 0.5d: BASELINE deterministic on the held-out TEST split (the gate reference)
    test_recs = evaluate(test)
    base = summarize(test_recs, "BASELINE deterministic — F1 TEST split (GATE REFERENCE)")

    # Small probes: F2/F3/F4 'untouched = safe' (sample to keep it fast)
    rng = random.Random(SEED)
    for fam in ("F2", "F3", "F4"):
        pool = [s for s in scen if family_of(s) == fam]
        rng.shuffle(pool)
        probe = pool[:80] if len(pool) > 80 else pool
        recs = evaluate(probe)
        summarize(recs, f"PROBE deterministic — {fam} (n={len(probe)})")
