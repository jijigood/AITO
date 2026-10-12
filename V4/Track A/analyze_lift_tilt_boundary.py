"""
Analyze the feature score distribution for lift_tilt vs insufficient_data
GT cases in Phase 1 train data to find the right discriminating threshold.
"""
import sys
import json
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from skills.scenario_parser import parse_scenario
from skills.feature_engine import run_all_skills

TRAIN_JSON = Path("/home/huangxiaolin/AITO/V4/Track A/D/data/Phase_1/train.json")

def get_mr_count(raw_case: dict) -> int:
    """Count MR rows from raw scenario data (mr_data field, pipe-delimited string)."""
    mr = raw_case.get("data", {}).get("mr_data", "")
    if not mr or not isinstance(mr, str):
        return 0
    lines = [l.strip() for l in mr.strip().split("\n") if l.strip()]
    # First line is header
    return max(0, len(lines) - 1)

def get_answer_action(raw_case: dict, options: list) -> str:
    """Extract the action type from the answer (Phase 1 free-text labels)."""
    answer_ids = raw_case.get("answer", [])
    if isinstance(answer_ids, str):
        answer_ids = [answer_ids]
    for opt in options:
        if opt.get("id") in answer_ids:
            label = opt.get("label", "").lower()
            if "insufficient" in label:
                return "insufficient_data"
            elif "lift the tilt" in label:
                return "lift_tilt"
            elif "press down the tilt" in label:
                return "press_down_tilt"
            elif "add neighbor" in label:
                return "add_neighbor"
            elif "decrease a3" in label:
                return "decrease_a3"
            elif "pdcchoccupied" in label or "2sym" in label:
                return "pdcch_2sym"
            elif "increase transmission power" in label or "increase power" in label:
                return "increase_power"
            elif "decrease transmission power" in label or "decrease power" in label:
                return "decrease_power"
            elif "check" in label and "server" in label:
                return "check_server"
            elif "azimuth" in label:
                return "adjust_azimuth"
            elif "inter" in label and "freq" in label:
                return "decrease_interfreq"
            else:
                return "other:" + label[:40]
    return "unknown"

def main():
    with open(TRAIN_JSON) as f:
        data = json.load(f)

    cases = data if isinstance(data, list) else data.get("data", [])
    print(f"Total cases: {len(cases)}")

    # Filter to F1 (4BS single)
    f1_cases = [c for c in cases
                if str(c.get("context", {}).get("wireless_network_information", {}).get("num_base_stations", "")) in ("4", "5")
                and "single" in c.get("tag", "").lower()]
    print(f"F1 cases (4BS single): {len(f1_cases)}")

    lift_tilt_data = []
    insuff_data = []
    other_data = []

    for raw in f1_cases:
        options = raw.get("task", {}).get("options", [])
        action = get_answer_action(raw, options)

        try:
            ps = parse_scenario(raw)
            sf = run_all_skills(ps)
        except Exception as e:
            continue

        mr_count = get_mr_count(raw)
        sinr = sf.mobility.serving_sinr_db
        rsrp = sf.mobility.serving_rsrp_dbm
        max_prb = sf.inter_freq.max_dl_prb_util
        worst_tp = sf.throughput.worst_throughput_mbps

        # PRE-grouping scores (before the SINR group block overrides them)
        # We need to capture state BEFORE lines 786-814 in feature_engine
        cov_pre = sf.power.coverage_hole_score  # from skill_e
        backhaul_pre = sf.transport.backhaul_or_server_score

        row = {
            "action": action,
            "mr_count": mr_count,
            "sinr": sinr,
            "rsrp": rsrp,
            "max_prb": max_prb,
            "worst_tp": worst_tp,
            "cov_pre": cov_pre,
            "backhaul_pre": backhaul_pre,
            "score_cov_final": sf.score_coverage_hole,
            "score_insuff_final": sf.score_insufficient_data,
            "top_issue": sf.top_issue()[0],
        }

        if action == "lift_tilt":
            lift_tilt_data.append(row)
        elif action == "insufficient_data":
            insuff_data.append(row)
        else:
            other_data.append(row)

    print(f"\nLift_tilt GT cases: {len(lift_tilt_data)}")
    print(f"Insufficient_data GT cases: {len(insuff_data)}")

    def summarize(label, data):
        if not data:
            print(f"  {label}: no data")
            return
        mr_counts = [d["mr_count"] for d in data]
        sinrs = [d["sinr"] for d in data if d["sinr"] is not None]
        rsrps = [d["rsrp"] for d in data if d["rsrp"] is not None]
        prbs = [d["max_prb"] for d in data if d["max_prb"] is not None]
        tps = [d["worst_tp"] for d in data if d["worst_tp"] is not None]
        covs = [d["score_cov_final"] for d in data]
        insuffs = [d["score_insuff_final"] for d in data if d["score_insuff_final"] is not None]

        print(f"\n=== {label} ({len(data)} cases) ===")
        if mr_counts:
            mr_sorted = sorted(mr_counts)
            print(f"  MR count: min={min(mr_counts)}, p25={mr_sorted[len(mr_sorted)//4]}, "
                  f"median={statistics.median(mr_counts):.1f}, p75={mr_sorted[3*len(mr_sorted)//4]}, "
                  f"max={max(mr_counts)}")
            # Distribution by threshold
            for thresh in [3, 5, 7, 10]:
                below = sum(1 for m in mr_counts if m < thresh)
                print(f"    mr < {thresh}: {below}/{len(mr_counts)} ({100*below/len(mr_counts):.1f}%)")
        if sinrs:
            print(f"  SINR: min={min(sinrs):.1f}, median={statistics.median(sinrs):.1f}, max={max(sinrs):.1f}")
            for thresh in [0, 5, 8, 10, 12]:
                above = sum(1 for s in sinrs if s > thresh)
                print(f"    sinr > {thresh}: {above}/{len(sinrs)} ({100*above/len(sinrs):.1f}%)")
        if rsrps:
            print(f"  RSRP: min={min(rsrps):.1f}, median={statistics.median(rsrps):.1f}, max={max(rsrps):.1f}")
        if prbs:
            above70 = sum(1 for p in prbs if p > 70)
            print(f"  PRB>70: {above70}/{len(prbs)} ({100*above70/len(prbs):.1f}%)")
        if tps:
            print(f"  Worst TP: min={min(tps):.1f}, median={statistics.median(tps):.1f}")
        if covs:
            print(f"  score_cov_final: min={min(covs):.2f}, median={statistics.median(covs):.2f}, max={max(covs):.2f}")
        if insuffs:
            print(f"  score_insuff_final: min={min(insuffs):.2f}, median={statistics.median(insuffs):.2f}, max={max(insuffs):.2f}")

        # Top issues
        top_issues = {}
        for d in data:
            ti = d["top_issue"]
            top_issues[ti] = top_issues.get(ti, 0) + 1
        print(f"  Top issues: {sorted(top_issues.items(), key=lambda x: -x[1])}")

    summarize("lift_tilt", lift_tilt_data)
    summarize("insufficient_data", insuff_data)

    # Cross-tabulation: for sinr>5 + prb>70, how are lift_tilt vs insuff distributed?
    print("\n=== Cross-tab: SINR>5 AND PRB>70 ===")
    def cross_tab(data, label):
        sinr5_prb70 = [d for d in data if (d["sinr"] or 0) > 5 and (d["max_prb"] or 0) > 70]
        print(f"  {label}: {len(sinr5_prb70)}/{len(data)} cases have sinr>5 AND prb>70")
        if sinr5_prb70:
            mr_counts = [d["mr_count"] for d in sinr5_prb70]
            mr_sorted = sorted(mr_counts)
            print(f"    MR count dist: {mr_sorted[:20]}...")
            for thresh in [3, 5, 7, 10]:
                below = sum(1 for m in mr_counts if m < thresh)
                print(f"    mr < {thresh}: {below}/{len(mr_counts)} ({100*below/len(mr_counts):.1f}%)")

    cross_tab(lift_tilt_data, "lift_tilt")
    cross_tab(insuff_data, "insufficient_data")

    # What MR threshold best separates lift_tilt from insufficient_data in sinr>5+prb>70?
    print("\n=== Threshold search (sinr>5 AND prb>70 group) ===")
    lt_in_group = [d["mr_count"] for d in lift_tilt_data if (d["sinr"] or 0) > 5 and (d["max_prb"] or 0) > 70]
    id_in_group = [d["mr_count"] for d in insuff_data if (d["sinr"] or 0) > 5 and (d["max_prb"] or 0) > 70]
    if lt_in_group and id_in_group:
        print(f"  lift_tilt in group: {len(lt_in_group)}, insuff in group: {len(id_in_group)}")
        print(f"  lift_tilt MR: {sorted(lt_in_group)}")
        print(f"  insuff MR:    {sorted(id_in_group)}")
        for thresh in range(1, 20):
            # Predict: if mr >= thresh → lift_tilt, if mr < thresh → insufficient_data
            lt_correct = sum(1 for m in lt_in_group if m >= thresh)
            id_correct = sum(1 for m in id_in_group if m < thresh)
            total = len(lt_in_group) + len(id_in_group)
            accuracy = (lt_correct + id_correct) / total if total > 0 else 0
            print(f"  thresh={thresh}: lift_tilt correct={lt_correct}/{len(lt_in_group)}, "
                  f"insuff correct={id_correct}/{len(id_in_group)}, "
                  f"accuracy={accuracy:.2%}")

if __name__ == "__main__":
    main()
