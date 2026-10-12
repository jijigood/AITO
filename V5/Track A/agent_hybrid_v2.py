#!/usr/bin/env python3
"""
Hybrid Agent V2 for Track A — fixed decision tree for Phase 2 API.

Root causes fixed from V1:
- RBs always ~250+ in Phase 2 → removed RB-based check_server
- KPI/CCE endpoint returns 404 → removed CCE-based pdcch detection
- Signaling response is raw string not dict → fixed parsing
- MR parsing included serving PCI as "neighbor" → filter by serving PCI
- Mainlobe was checked on neighbor cells → now checks SERVING cell
- F2 LLM unreliable → replaced with deterministic RSRP+neighbor rule
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

    # Throughput
    tp_resp = client.call("get_throughput_logs", sid)
    tp_str = tp_resp.get("Logs", "") if isinstance(tp_resp, dict) else ""
    feats["throughput_raw"] = tp_str
    feats["worst_time"] = find_worst_time(tp_str)

    if not feats["worst_time"]:
        return feats

    wt = feats["worst_time"]

    # Serving cell at worst time
    srv_resp = client.call("get_serving_cell_pci", sid, time=wt)
    feats["serving_pci"] = srv_resp.get("PCI") if isinstance(srv_resp, dict) else None

    rsrp_resp = client.call("get_serving_cell_rsrp", sid, time=wt)
    feats["serving_rsrp"] = rsrp_resp.get("SS-RSRP (dBm)") if isinstance(rsrp_resp, dict) else None

    sinr_resp = client.call("get_serving_cell_sinr", sid, time=wt)
    feats["serving_sinr"] = sinr_resp.get("SS-SINR (dB)") if isinstance(sinr_resp, dict) else None

    # MR data — count neighbors STRONGER than serving, excluding serving PCI
    mr_resp = client.call("get_mr_data", sid)
    mr_str = mr_resp.get("MR Data", "") if isinstance(mr_resp, dict) else ""
    feats["mr_raw"] = mr_str

    srv_pci = feats["serving_pci"]
    nb_stronger = 0
    max_diff = 0.0
    strongest_nb_pci = None

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
                        diff = nb1_rsrp - srv_rsrp_mr
                        if diff > max_diff:
                            max_diff = diff
                            strongest_nb_pci = str(nb1_pci)
                except:
                    pass

    feats["nb_stronger_count"] = nb_stronger
    feats["nb_max_diff"] = max_diff
    feats["strongest_nb_pci"] = strongest_nb_pci

    # Neighbor list from serving cell config
    if srv_pci is not None:
        cfg_resp = client.call("get_cell_info", sid, pci=int(srv_pci))
        if isinstance(cfg_resp, dict) and "error" not in cfg_resp:
            feats["serving_config"] = cfg_resp
            feats["neighbor_list_str"] = cfg_resp.get("PCell Neighbor Cell (gNodeBID_ARFCN_PCI)", "")

    # Mainlobe check for SERVING cell
    if srv_pci is not None:
        ml_resp = client.call("judge_mainlobe", sid, pci=int(srv_pci), time=wt)
        if isinstance(ml_resp, dict):
            feats["outside_mainlobe"] = ml_resp.get("Outside_Mainlobe_Flag")

    # Signaling: check for A2 events (signal dropped below threshold)
    tp_lines = tp_str.strip().split("\n")
    last_t = tp_lines[-1].split("|")[0].strip() if len(tp_lines) > 1 else wt
    sig_resp = client.call("get_signaling_plane_event_log", sid, time=last_t)
    sig_str = sig_resp if isinstance(sig_resp, str) else ""
    feats["has_a2_event"] = "NREventA2|" in sig_str if sig_str else False

    return feats


def is_neighbor_in_list(pci_str: str, neighbor_list: str) -> bool:
    """Check if a PCI appears in the cell's configured neighbor list string."""
    if not pci_str or not neighbor_list:
        return False
    return (f"_{pci_str}]" in neighbor_list or
            f"_{pci_str}," in neighbor_list or
            f"_{pci_str})" in neighbor_list or
            f"_{pci_str} " in neighbor_list)


# ──────────────────────────────────────────────────────────────────────────────
# F1 Decision Tree (Phase 2 compatible)
# ──────────────────────────────────────────────────────────────────────────────

def f1_decision_tree(feats: Dict, scenario: Dict) -> Optional[str]:
    """
    Rule-based F1 diagnosis using Phase 2 available signals:
    1. Strong neighbor + NOT in list → add_neighbor
    2. Strong neighbor + IN list → decrease_a3
    3. Outside mainlobe (serving) → lift_tilt
    4. RSRP < -105 → insufficient
    5. RSRP > -95 AND SINR > 10 → check_server
    6. Else → pdcch
    """
    options = scenario["task"]["options"]

    # ── Step 1-2: Strong neighbor analysis ──
    if feats.get("nb_stronger_count", 0) >= 2 and feats.get("nb_max_diff", 0) > 3:
        nb_pci = feats.get("strongest_nb_pci")
        nb_list = feats.get("neighbor_list_str", "")

        if nb_pci and not is_neighbor_in_list(nb_pci, nb_list):
            for opt in options:
                if "add neighbor" in opt["label"].lower() or "add_neighbor" in opt["label"].lower():
                    return opt["id"]
        else:
            for opt in options:
                lbl = opt["label"].lower()
                if "decrease" in lbl and "a3" in lbl:
                    return opt["id"]
        return None

    # ── Step 3: Mainlobe check for serving cell ──
    if feats.get("outside_mainlobe") is True:
        for opt in options:
            if "lift the tilt" in opt["label"].lower() or "lift_tilt" in opt["label"].lower():
                return opt["id"]
        return None

    # ── Step 4-6: RF-quality-based classification ──
    rsrp = feats.get("serving_rsrp")
    sinr = feats.get("serving_sinr")

    if rsrp is not None:
        rsrp_f = float(rsrp)

        # Very weak signal → insufficient coverage
        if rsrp_f < -105:
            for opt in options:
                if "insufficient" in opt["label"].lower():
                    return opt["id"]
            return None

        # Good RF but low throughput → server/scheduling issue
        # Widened: SINR > 8 captures borderline check_server cases
        if sinr is not None and rsrp_f > -95 and float(sinr) > 8:
            for opt in options:
                lbl = opt["label"].lower()
                if "check test server" in lbl or "transmission issues" in lbl or "check_server" in lbl:
                    return opt["id"]
            return None

        # A2 event in signaling → signal dropped below threshold → insufficient
        if feats.get("has_a2_event") and rsrp_f < -100:
            for opt in options:
                if "insufficient" in opt["label"].lower():
                    return opt["id"]
            return None

    # Default: pdcch (moderate signal, moderate SINR)
    for opt in options:
        lbl = opt["label"].lower()
        if "pdcch" in lbl or "2sym" in lbl or "cce" in lbl:
            return opt["id"]

    return None


# ──────────────────────────────────────────────────────────────────────────────
# F2 Deterministic Handler
# ──────────────────────────────────────────────────────────────────────────────

def handle_f2(feats: Dict, scenario: Dict) -> str:
    """
    F2 (4BS, 2 answers): deterministic Type A/B selection.
    
    Type B (interference/overshoot): nb_stronger >= 2 AND max_diff > 2 AND RSRP > -95
    Type A (coverage gap): everything else (matches 62% training distribution)
    """
    options = scenario["task"]["options"]
    rsrp = feats.get("serving_rsrp")
    nb_str = feats.get("nb_stronger_count", 0)
    max_diff = feats.get("nb_max_diff", 0)

    is_type_b = (nb_str >= 2 and max_diff > 2 and
                 rsrp is not None and float(rsrp) > -95)

    if is_type_b:
        # Type B: decrease_power + press_down_tilt
        dp = next((o for o in options if "decrease transmission power" in o["label"].lower()), None)
        pt = next((o for o in options if "press down the tilt" in o["label"].lower()), None)
        if dp and pt:
            return "|".join(sorted([dp["id"], pt["id"]], key=lambda x: int(x[1:])))
    else:
        # Type A: adjust_azimuth + increase_power
        az = next((o for o in options if "adjust the azimuth" in o["label"].lower()), None)
        pw = next((o for o in options if "increase transmission power" in o["label"].lower()), None)
        if az and pw:
            return "|".join(sorted([az["id"], pw["id"]], key=lambda x: int(x[1:])))

    # Fallback: Type A
    az = next((o for o in options if "azimuth" in o["label"].lower()), None)
    pw = next((o for o in options if "increase" in o["label"].lower() and "power" in o["label"].lower()), None)
    if az and pw:
        return "|".join(sorted([az["id"], pw["id"]], key=lambda x: int(x[1:])))
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# F3/F4 Deterministic Handlers
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
    # Prefer A5 threshold
    for opt in options:
        lbl = opt["label"].lower().replace(" ", "")
        if "a5rsrpthld1" in lbl or ("a5" in lbl and "decrease" in lbl):
            return opt["id"]
    # Fallback: any A2 threshold
    for opt in options:
        lbl = opt["label"].lower().replace(" ", "")
        if "a2rsrp" in lbl and "decrease" in lbl:
            return opt["id"]
    # Last resort: any interfreq option
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
    ap.add_argument("--out", type=Path, default=_HERE / "submission_hybrid_v2.csv")
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

    # Resume support
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

        # ─── F4M (13BS multiple) ───
        if tag == "multiple-answer" and num_bs == "13":
            answer = handle_f4m(scenario)
            method = "F4M-det"
            stats["F4M"] += 1

        # ─── F3 (5BS multiple) ───
        elif tag == "multiple-answer" and num_bs == "5":
            answer = handle_f3(scenario)
            method = "F3-det"
            stats["F3"] += 1

        # ─── F2 (4BS multiple) ───
        elif tag == "multiple-answer" and num_bs == "4":
            feats = extract_features(api, sid)
            answer = handle_f2(feats, scenario)
            method = "F2-det"
            stats["F2"] += 1

        # ─── F1 (4BS single) ───
        elif tag == "single-answer" and num_bs == "4":
            feats = extract_features(api, sid)
            answer = f1_decision_tree(feats, scenario) or ""
            if not answer:
                # Last fallback: insufficient
                for opt in scenario["task"]["options"]:
                    if "insufficient" in opt["label"].lower():
                        answer = opt["id"]
                        break
            method = "F1-tree"
            stats["F1"] += 1

        # ─── F1_5BS (5BS single) ───
        elif tag == "single-answer" and num_bs == "5":
            feats = extract_features(api, sid)
            answer = f1_decision_tree(feats, scenario) or ""
            if not answer:
                for opt in scenario["task"]["options"]:
                    if "insufficient" in opt["label"].lower():
                        answer = opt["id"]
                        break
            method = "F1_5BS-tree"
            stats["F1_5BS"] += 1

        # ─── F4 (13BS single) ───
        elif tag == "single-answer" and num_bs == "13":
            answer = handle_f4_single(scenario)
            method = "F4-det"
            stats["F4"] += 1

        # ─── Unknown ───
        else:
            method = f"unknown-{tag}/{num_bs}"

        # Validate answer against legal options
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
    logger.info(f"API calls: {api.call_count}")


def _save(results: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Track A", "Track B"])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
