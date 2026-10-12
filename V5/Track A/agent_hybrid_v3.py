#!/usr/bin/env python3
"""
Hybrid Agent V3 for Track A — training-data-informed decision tree.

Key fixes from V2 based on Phase 1 training data analysis:
- Mainlobe logic INVERTED: outside_mainlobe → insufficient (not lift_tilt)
- In mainlobe + weak signal → lift_tilt (not pdcch)
- High SINR (>8) + in mainlobe → check_server (server issue despite good RF)
- add_neighbor vs decrease_a3: LLM-assisted with option label heuristic
- Better SINR/RSRP boundary routing for pdcch vs lift_tilt
"""

import argparse
import csv
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

try:
    import urllib3
    urllib3.disable_warnings()
except:
    pass

_HERE = Path(__file__).parent

# ──────────────────────────────────────────────────────────────────────────────
# API Client
# ──────────────────────────────────────────────────────────────────────────────

ENDPOINT_MAP = {
    "get_throughput_logs": "/throughput-logs",
    "get_cell_info": "/cell-info",
    "get_serving_cell_pci": "/serving-cell-pci",
    "get_serving_cell_rsrp": "/serving-cell-rsrp",
    "get_serving_cell_sinr": "/serving-cell-sinr",
    "get_neighboring_cells_pci": "/neighboring-cells-pci",
    "get_all_cells_pci": "/all-cells-pci",
    "get_mr_data": "/get_mr_data",
    "get_signaling_plane_event_log": "/signaling-plane-event-log",
    "get_rbs_allocated_to_user": "/rbs-allocated-to-user",
    "judge_mainlobe": "/judge_mainlobe",
}


class APIClient:
    def __init__(self, server_url: str, token: str, timeout: float = 15.0):
        self.server_url = server_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.verify = False
        self.call_count = 0

    def call(self, tool: str, sid: str, **params) -> Any:
        ep = ENDPOINT_MAP.get(tool)
        if not ep:
            return {"error": f"Unknown: {tool}"}
        headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Scenario-Id": sid,
            "X-API-Token": self.token,
        }
        try:
            r = self.session.get(
                f"{self.server_url}{ep}", params=params,
                headers=headers, timeout=self.timeout
            )
            self.call_count += 1
            if r.status_code == 200:
                try:
                    return r.json()
                except:
                    return r.text
            return {"error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"error": str(e)[:80]}


# ──────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ──────────────────────────────────────────────────────────────────────────────

def find_worst_time(tp_str: str) -> Optional[str]:
    if not tp_str:
        return None
    lines = tp_str.strip().split("\n")
    worst_t, worst_v = None, float("inf")
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) >= 2:
            try:
                v = float(parts[1].strip())
                if v < worst_v:
                    worst_v = v
                    worst_t = parts[0].strip()
            except:
                pass
    return worst_t


def extract_features(client: APIClient, sid: str) -> Dict[str, Any]:
    """Extract all diagnostic features from Phase 2 API."""
    feats = {"sid": sid}

    tp_resp = client.call("get_throughput_logs", sid)
    tp_str = tp_resp.get("Logs", "") if isinstance(tp_resp, dict) else ""
    feats["throughput_raw"] = tp_str
    feats["worst_time"] = find_worst_time(tp_str)

    if not feats["worst_time"]:
        return feats

    wt = feats["worst_time"]

    srv_resp = client.call("get_serving_cell_pci", sid, time=wt)
    feats["serving_pci"] = srv_resp.get("PCI") if isinstance(srv_resp, dict) else None

    rsrp_resp = client.call("get_serving_cell_rsrp", sid, time=wt)
    feats["serving_rsrp"] = rsrp_resp.get("SS-RSRP (dBm)") if isinstance(rsrp_resp, dict) else None

    sinr_resp = client.call("get_serving_cell_sinr", sid, time=wt)
    feats["serving_sinr"] = sinr_resp.get("SS-SINR (dB)") if isinstance(sinr_resp, dict) else None

    mr_resp = client.call("get_mr_data", sid)
    mr_str = mr_resp.get("MR Data", "") if isinstance(mr_resp, dict) else ""
    feats["mr_raw"] = mr_str

    srv_pci = feats["serving_pci"]
    nb_stronger = 0
    max_diff = 0.0
    strongest_nb_pci = None
    unique_strong_pcis = set()

    if isinstance(mr_str, str) and mr_str:
        for line in mr_str.replace('\r\n', '\n').split("\n")[1:]:
            parts = line.split("|")
            if len(parts) > 6:
                try:
                    srv_rsrp_mr = float(parts[3])
                    nb1_pci = int(parts[5])
                    nb1_rsrp = float(parts[6])
                    if srv_pci is not None and nb1_pci == int(srv_pci):
                        continue
                    if nb1_rsrp > srv_rsrp_mr:
                        nb_stronger += 1
                        unique_strong_pcis.add(nb1_pci)
                        diff = nb1_rsrp - srv_rsrp_mr
                        if diff > max_diff:
                            max_diff = diff
                            strongest_nb_pci = str(nb1_pci)
                except:
                    pass

    feats["nb_stronger_count"] = nb_stronger
    feats["nb_max_diff"] = max_diff
    feats["strongest_nb_pci"] = strongest_nb_pci
    feats["unique_strong_nb_count"] = len(unique_strong_pcis)

    # Mainlobe check for SERVING cell
    if srv_pci is not None:
        ml_resp = client.call("judge_mainlobe", sid, pci=int(srv_pci), time=wt)
        if isinstance(ml_resp, dict):
            feats["outside_mainlobe"] = ml_resp.get("Outside_Mainlobe_Flag")

    # Signaling: check for A2 events
    tp_lines = tp_str.strip().split("\n")
    last_t = tp_lines[-1].split("|")[0].strip() if len(tp_lines) > 1 else wt
    sig_resp = client.call("get_signaling_plane_event_log", sid, time=last_t)
    sig_str = sig_resp if isinstance(sig_resp, str) else ""
    feats["has_a2_event"] = "NREventA2|" in sig_str if sig_str else False

    return feats


# ──────────────────────────────────────────────────────────────────────────────
# F1 Decision Tree V3 (Training-informed)
# ──────────────────────────────────────────────────────────────────────────────

def find_option(options: List[Dict], keyword: str) -> Optional[str]:
    """Find option ID matching a keyword pattern."""
    kw = keyword.lower()
    for opt in options:
        lbl = opt["label"].lower()
        if kw == "check":
            if "check test server" in lbl or "transmission issues" in lbl:
                return opt["id"]
        elif kw == "insufficient":
            if "insufficient" in lbl:
                return opt["id"]
        elif kw == "lift":
            if "lift the tilt" in lbl or "lift_tilt" in lbl:
                return opt["id"]
        elif kw == "pdcch":
            if "pdcch" in lbl or "2sym" in lbl or "1sym" in lbl:
                return opt["id"]
        elif kw == "add_neighbor":
            if "add neighbor" in lbl:
                return opt["id"]
        elif kw == "decrease_a3":
            if "decrease" in lbl and "a3" in lbl:
                return opt["id"]
    return None


def pick_interference_action(feats: Dict, options: List[Dict]) -> Optional[str]:
    """
    Distinguish add_neighbor vs decrease_a3 for interference scenarios.
    
    Training insight: A3Offset=6 → add_neighbor, A3Offset=10 → decrease_a3
    Phase 2: cell_info returns 404, so use heuristic from option labels.
    
    Strategy: Look at "Decrease A3 Offset by N dB" value.
    - If N >= 4: implies current A3 is high → decrease_a3 is viable
    - If N <= 2: implies current A3 is already low → add_neighbor preferred
    - If multiple strong NB PCIs exist: more likely add_neighbor (missing config)
    """
    # Check A3 decrease values in options targeting the serving cell
    a3_decrease_values = []
    add_nb_options = []
    decrease_a3_options = []

    for opt in options:
        lbl = opt["label"]
        lbl_lower = lbl.lower()
        if "add neighbor" in lbl_lower:
            add_nb_options.append(opt)
        if "decrease" in lbl_lower and "a3" in lbl_lower:
            decrease_a3_options.append(opt)
            m = re.search(r'by\s+(\d+)\s*dB', lbl, re.IGNORECASE)
            if m:
                a3_decrease_values.append(int(m.group(1)))

    # Heuristic 1: If A3 decrease options suggest large decreases (>=4 dB),
    # current A3 offset is likely high → decrease_a3 is appropriate
    if a3_decrease_values:
        max_decrease = max(a3_decrease_values)
        if max_decrease >= 4:
            if decrease_a3_options:
                return decrease_a3_options[0]["id"]
        elif max_decrease <= 2:
            if add_nb_options:
                return add_nb_options[0]["id"]

    # Heuristic 2: If multiple unique strong NB PCIs, more likely missing neighbor
    if feats.get("unique_strong_nb_count", 0) >= 2:
        if add_nb_options:
            return add_nb_options[0]["id"]

    # Default: slight preference for decrease_a3 (more common in training)
    if decrease_a3_options:
        return decrease_a3_options[0]["id"]
    if add_nb_options:
        return add_nb_options[0]["id"]
    return None


def f1_decision_tree_v3(feats: Dict, scenario: Dict) -> Optional[str]:
    """
    Training-informed F1 diagnosis using Phase 2 signals.
    
    Key corrections from training data:
    - outside_mainlobe → insufficient (NOT lift_tilt)
    - in_mainlobe + weak signal → lift_tilt
    - in_mainlobe + SINR > 8 → check_server
    - Remaining in_mainlobe moderate → pdcch
    """
    options = scenario["task"]["options"]
    nb_str = feats.get("nb_stronger_count", 0)
    max_diff = feats.get("nb_max_diff", 0)
    outside_ml = feats.get("outside_mainlobe")
    rsrp = feats.get("serving_rsrp")
    sinr = feats.get("serving_sinr")

    rsrp_f = float(rsrp) if rsrp is not None else -100.0
    sinr_f = float(sinr) if sinr is not None else 0.0

    # ── Step 1: Interference detection ──
    # Training: SINR < 0, nb >= 2, max_diff > 3 → add_neighbor/decrease_a3
    if nb_str >= 2 and max_diff > 3:
        return pick_interference_action(feats, options)

    # ── Step 2: In mainlobe + high SINR + good RSRP → check_server ──
    # Training: check_server = good signal quality, issue is server-side
    if outside_ml is False and sinr_f > 8 and rsrp_f > -95:
        return find_option(options, "check")

    # ── Step 3: Outside mainlobe + weak signal → insufficient ──
    # Training: insufficient = UE outside beam coverage
    if outside_ml is True and (rsrp_f < -100 or sinr_f < 2):
        return find_option(options, "insufficient")

    # ── Step 4: Outside mainlobe + strong signal → check_server ──
    # UE outside beam but gets good signal = server/config issue
    if outside_ml is True and sinr_f > 8 and rsrp_f > -95:
        return find_option(options, "check")

    # ── Step 5: Outside mainlobe + moderate signal → lift_tilt ──
    # UE near beam edge, moderate quality = tilt could help
    if outside_ml is True and rsrp_f >= -100 and sinr_f >= 2:
        return find_option(options, "lift")

    # ── Step 6: In mainlobe + weak signal → lift_tilt ──
    # Training: lift_tilt = in mainlobe + needs tilt adjustment
    if outside_ml is False and rsrp_f < -100 and sinr_f < 5:
        return find_option(options, "lift")

    # ── Step 7: In mainlobe + moderate SINR → pdcch ──
    # Training: pdcch = in mainlobe, moderate conditions, control channel issue
    if outside_ml is False and sinr_f <= 8:
        return find_option(options, "pdcch")

    # ── Step 8: Fallback for remaining edge cases ──
    # In mainlobe, SINR > 8 but RSRP <= -95 → could be pdcch or check_server
    if outside_ml is False and sinr_f > 8:
        return find_option(options, "check")

    return find_option(options, "pdcch")


# ──────────────────────────────────────────────────────────────────────────────
# F2 Deterministic Handler (unchanged from V2)
# ──────────────────────────────────────────────────────────────────────────────

def handle_f2(feats: Dict, scenario: Dict) -> str:
    """
    F2 (4BS, 2 answers): deterministic Type A/B selection.
    Type B (interference/overshoot): nb_stronger >= 2 AND max_diff > 2 AND RSRP > -95
    Type A (coverage gap): everything else
    """
    options = scenario["task"]["options"]
    rsrp = feats.get("serving_rsrp")
    nb_str = feats.get("nb_stronger_count", 0)
    max_diff = feats.get("nb_max_diff", 0)

    is_type_b = (nb_str >= 2 and max_diff > 2 and
                 rsrp is not None and float(rsrp) > -95)

    if is_type_b:
        dp = next((o for o in options if "decrease transmission power" in o["label"].lower()), None)
        pt = next((o for o in options if "press down the tilt" in o["label"].lower()), None)
        if dp and pt:
            return "|".join(sorted([dp["id"], pt["id"]], key=lambda x: int(x[1:])))
    else:
        az = next((o for o in options if "adjust the azimuth" in o["label"].lower()), None)
        pw = next((o for o in options if "increase transmission power" in o["label"].lower()), None)
        if az and pw:
            return "|".join(sorted([az["id"], pw["id"]], key=lambda x: int(x[1:])))

    az = next((o for o in options if "azimuth" in o["label"].lower()), None)
    pw = next((o for o in options if "increase" in o["label"].lower() and "power" in o["label"].lower()), None)
    if az and pw:
        return "|".join(sorted([az["id"], pw["id"]], key=lambda x: int(x[1:])))
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# F3/F4 Deterministic Handlers (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def handle_f3(scenario: Dict) -> str:
    """5BS multiple (4 answers): decrease_power + press_down_tilt + 2x increase_a3."""
    options = scenario["task"]["options"]
    dp = [o for o in options if "decrease transmission power" in o["label"].lower()]
    pd = [o for o in options if "press down the tilt" in o["label"].lower()]
    ia3 = [o for o in options if re.search(r'increase.*a3', o["label"].lower())]

    if not dp or not pd or len(ia3) < 2:
        return ""

    dp_opt = dp[0]
    dp_cell = re.search(r'(\d{7}_\d+)', dp_opt["label"])
    dp_cell_id = dp_cell.group(1) if dp_cell else ""
    pd_opt = next((o for o in pd if dp_cell_id and dp_cell_id in o["label"]), pd[0])

    codes = sorted(set([dp_opt["id"], pd_opt["id"]] + [o["id"] for o in ia3[:2]]),
                   key=lambda x: int(x[1:]))
    return "|".join(codes)


def handle_f4m(scenario: Dict) -> str:
    """13BS multiple (2 answers): 2x decrease inter-frequency threshold."""
    options = scenario["task"]["options"]
    candidates = []
    for opt in options:
        lbl = opt["label"].lower()
        if ("interfreq" in lbl or "a2rsrp" in lbl or "a5rsrp" in lbl) and \
           ("decrease" in lbl or "threshold" in lbl):
            candidates.append(opt["id"])
    if len(candidates) >= 2:
        return "|".join(sorted(candidates[:2], key=lambda x: int(x[1:])))
    return ""


def handle_f4_single(scenario: Dict) -> str:
    """13BS single: decrease one inter-frequency threshold (prefer A5)."""
    options = scenario["task"]["options"]
    for opt in options:
        lbl = opt["label"].lower().replace(" ", "")
        if "a5rsrpthld1" in lbl or ("a5" in lbl and "decrease" in lbl):
            return opt["id"]
    for opt in options:
        lbl = opt["label"].lower().replace(" ", "")
        if "a2rsrp" in lbl and "decrease" in lbl:
            return opt["id"]
    for opt in options:
        lbl = opt["label"].lower()
        if "interfreq" in lbl:
            return opt["id"]
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Main Runner
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_url", default="https://120.46.145.77/no")
    ap.add_argument("--token", required=True)
    ap.add_argument("--test_json", type=Path, default=_HERE / "data/Phase_2/test.json")
    ap.add_argument("--out", type=Path, default=_HERE / "submission_hybrid_v3.csv")
    ap.add_argument("--save_freq", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--start_idx", type=int, default=0)
    args = ap.parse_args()

    with open(args.test_json) as f:
        scenarios = json.load(f)
    logger.info(f"Loaded {len(scenarios)} scenarios")

    if args.max_samples:
        scenarios = scenarios[args.start_idx:args.start_idx + args.max_samples]

    api = APIClient(args.server_url, args.token)

    existing = {}
    if args.resume and args.out.exists():
        with open(args.out, newline="") as f:
            for row in csv.DictReader(f):
                sid = row.get("ID", "").strip()
                if sid and row.get("Track A", "").strip():
                    existing[sid] = row["Track A"]
        logger.info(f"Resuming with {len(existing)} existing")

    results = []
    stats = {"F1": 0, "F2": 0, "F3": 0, "F4": 0, "F4M": 0, "F1_5BS": 0}
    category_dist = {}
    start = time.time()

    for idx, scenario in enumerate(scenarios):
        sid = scenario["scenario_id"]
        tag = scenario["tag"]
        num_bs = str(scenario["context"]["wireless_network_information"]["num_base_stations"])

        if sid in existing:
            results.append({"ID": sid, "Track A": existing[sid], "Track B": ""})
            continue

        t0 = time.time()
        answer = ""
        method = ""

        if tag == "multiple-answer" and num_bs == "13":
            answer = handle_f4m(scenario)
            method = "F4M-det"
            stats["F4M"] += 1

        elif tag == "multiple-answer" and num_bs == "5":
            answer = handle_f3(scenario)
            method = "F3-det"
            stats["F3"] += 1

        elif tag == "multiple-answer" and num_bs == "4":
            feats = extract_features(api, sid)
            answer = handle_f2(feats, scenario)
            method = "F2-det"
            stats["F2"] += 1

        elif tag == "single-answer" and num_bs == "4":
            feats = extract_features(api, sid)
            answer = f1_decision_tree_v3(feats, scenario) or ""
            if not answer:
                for opt in scenario["task"]["options"]:
                    if "insufficient" in opt["label"].lower():
                        answer = opt["id"]
                        break
            method = "F1-tree-v3"
            stats["F1"] += 1

            # Track category for distribution analysis
            if answer:
                for opt in scenario["task"]["options"]:
                    if opt["id"] == answer:
                        lbl = opt["label"].lower()
                        if "pdcch" in lbl or "sym" in lbl:
                            cat = "pdcch"
                        elif "add neighbor" in lbl:
                            cat = "add_neighbor"
                        elif "decrease" in lbl and "a3" in lbl:
                            cat = "decrease_a3"
                        elif "check" in lbl or "server" in lbl or "trans" in lbl:
                            cat = "check_server"
                        elif "lift" in lbl:
                            cat = "lift_tilt"
                        elif "insufficient" in lbl:
                            cat = "insufficient"
                        else:
                            cat = "other"
                        category_dist[cat] = category_dist.get(cat, 0) + 1
                        break

        elif tag == "single-answer" and num_bs == "5":
            feats = extract_features(api, sid)
            answer = f1_decision_tree_v3(feats, scenario) or ""
            if not answer:
                for opt in scenario["task"]["options"]:
                    if "insufficient" in opt["label"].lower():
                        answer = opt["id"]
                        break
            method = "F1_5BS-tree-v3"
            stats["F1_5BS"] += 1

        elif tag == "single-answer" and num_bs == "13":
            answer = handle_f4_single(scenario)
            method = "F4-det"
            stats["F4"] += 1

        else:
            method = f"unknown-{tag}/{num_bs}"

        valid_ids = {opt["id"] for opt in scenario["task"]["options"]}
        codes = [c.strip() for c in answer.split("|") if c.strip() in valid_ids] if answer else []

        if tag == "single-answer":
            final = codes[0] if codes else ""
        elif num_bs == "5":
            final = "|".join(sorted(set(codes[:4]), key=lambda x: int(x[1:])))
        else:
            final = "|".join(sorted(set(codes[:2]), key=lambda x: int(x[1:])))

        results.append({"ID": sid, "Track A": final, "Track B": ""})
        elapsed = time.time() - t0
        if (idx + 1) % 10 == 0 or elapsed > 5:
            logger.info(f"[{idx+1}/{len(scenarios)}] {sid[:8]} {method} → {final} ({elapsed:.1f}s)")

        if (idx + 1) % args.save_freq == 0:
            _save(results, args.out)

    _save(results, args.out)
    total_t = time.time() - start
    answered = sum(1 for r in results if r["Track A"])
    logger.info(f"\nDone: {answered}/{len(results)} answered in {total_t:.0f}s")
    logger.info(f"Stats: {stats}")
    logger.info(f"F1 category distribution: {category_dist}")
    logger.info(f"API calls: {api.call_count}")


def _save(results: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Track A", "Track B"])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
