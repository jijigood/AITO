#!/usr/bin/env python3
"""
Maximum accuracy agent V3.
- F1/F1_5BS: LLM with full evidence + diagnostic rules (highest accuracy)
- F2: Deterministic rule (proven 62/38 distribution match)
- F3/F4M/F4: Deterministic templates
"""

import argparse
import csv
import json
import logging
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


class LLMClient:
    def __init__(self, base_url: str, model: str = "qwen"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = requests.Session()
        self.session.trust_env = False

    def complete(self, prompt: str, max_tokens: int = 10) -> str:
        try:
            resp = self.session.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stop": ["}"],
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""


# ──────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ──────────────────────────────────────────────────────────────────────────────

def find_worst_time(tp_str: str):
    if not tp_str:
        return None, None
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
    return worst_t, worst_v


def extract_features(client: APIClient, sid: str) -> Dict[str, Any]:
    feats = {"sid": sid}

    # Throughput
    tp_resp = client.call("get_throughput_logs", sid)
    tp_str = tp_resp.get("Logs", "") if isinstance(tp_resp, dict) else ""
    feats["throughput_raw"] = tp_str
    wt, wv = find_worst_time(tp_str)
    feats["worst_time"] = wt
    feats["worst_tp"] = wv

    if not wt:
        return feats

    # Serving cell
    srv_resp = client.call("get_serving_cell_pci", sid, time=wt)
    feats["serving_pci"] = srv_resp.get("PCI") if isinstance(srv_resp, dict) else None

    rsrp_resp = client.call("get_serving_cell_rsrp", sid, time=wt)
    feats["serving_rsrp"] = rsrp_resp.get("SS-RSRP (dBm)") if isinstance(rsrp_resp, dict) else None

    sinr_resp = client.call("get_serving_cell_sinr", sid, time=wt)
    feats["serving_sinr"] = sinr_resp.get("SS-SINR (dB)") if isinstance(sinr_resp, dict) else None

    # MR data
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

    # Neighbor list
    if srv_pci is not None:
        cfg_resp = client.call("get_cell_info", sid, pci=int(srv_pci))
        if isinstance(cfg_resp, dict) and "error" not in cfg_resp:
            feats["serving_config"] = cfg_resp
            feats["neighbor_list_str"] = cfg_resp.get("PCell Neighbor Cell (gNodeBID_ARFCN_PCI)", "")

    # Mainlobe
    if srv_pci is not None:
        ml_resp = client.call("judge_mainlobe", sid, pci=int(srv_pci), time=wt)
        if isinstance(ml_resp, dict):
            feats["outside_mainlobe"] = ml_resp.get("Outside_Mainlobe_Flag")

    # Signaling (get ALL events up to last timestamp)
    tp_lines = tp_str.strip().split("\n")
    last_t = tp_lines[-1].split("|")[0].strip() if len(tp_lines) > 1 else wt
    sig_resp = client.call("get_signaling_plane_event_log", sid, time=last_t)
    sig_str = sig_resp if isinstance(sig_resp, str) else (sig_resp.get("Logs", "") if isinstance(sig_resp, dict) else "")
    feats["signaling_raw"] = sig_str
    feats["has_a2_event"] = "NREventA2|" in sig_str if isinstance(sig_str, str) else False
    feats["has_a5_event"] = ("NREventA5|" in sig_str or "NREventA5Meas" in sig_str) if isinstance(sig_str, str) else False

    return feats


def is_neighbor_in_list(pci_str: str, neighbor_list: str) -> bool:
    if not pci_str or not neighbor_list:
        return False
    return (f"_{pci_str}]" in neighbor_list or
            f"_{pci_str}," in neighbor_list or
            f"_{pci_str})" in neighbor_list or
            f"_{pci_str} " in neighbor_list)


# ──────────────────────────────────────────────────────────────────────────────
# F1 LLM Diagnosis
# ──────────────────────────────────────────────────────────────────────────────

F1_SYSTEM = """You are a 5G network expert. Given drive test evidence, identify the root cause.

DIAGNOSTIC RULES (follow strictly in order):
1. If MR shows neighbor RSRP > serving RSRP by >3dB AND that neighbor PCI is NOT in serving cell's configured neighbor list → add_neighbor
2. If MR shows neighbor RSRP > serving RSRP by >3dB AND that neighbor PCI IS in serving cell's configured neighbor list → decrease_a3
3. If Outside_Mainlobe_Flag=True for serving cell → lift_tilt
4. If serving RSRP < -105 dBm OR signaling shows NREventA2 triggered → insufficient
5. If serving RSRP > -95 dBm AND SINR > 8 dB AND no strong neighbors → check_server
6. Otherwise (moderate RSRP/SINR, in mainlobe, no strong neighbors, no A2 events) → pdcch

Answer with EXACTLY one category name."""


def build_f1_evidence(feats: Dict) -> str:
    parts = []

    rsrp = feats.get("serving_rsrp")
    sinr = feats.get("serving_sinr")
    if rsrp is not None:
        qual = "EXCELLENT" if float(rsrp) > -90 else "GOOD" if float(rsrp) > -95 else "MODERATE" if float(rsrp) > -105 else "WEAK"
        parts.append(f"- Serving RSRP: {rsrp} dBm, SINR: {sinr} dB ({qual} RF)")

    nb = feats.get("nb_stronger_count", 0)
    diff = feats.get("nb_max_diff", 0)
    if nb >= 1 and diff > 2:
        nb_pci = feats.get("strongest_nb_pci", "?")
        nb_list = feats.get("neighbor_list_str", "")
        in_list = is_neighbor_in_list(nb_pci, nb_list)
        parts.append(f"- MR: {nb} neighbor(s) stronger than serving, max diff {diff:.1f} dB (strongest PCI: {nb_pci})")
        parts.append(f"- Neighbor PCI {nb_pci} {'IS' if in_list else 'is NOT'} in serving cell's configured neighbor list")
    else:
        parts.append(f"- MR: No neighbor significantly stronger than serving (max diff {diff:.1f} dB)")

    ml = feats.get("outside_mainlobe")
    if ml is True:
        parts.append("- Mainlobe: User is OUTSIDE serving cell's mainlobe (Outside_Mainlobe_Flag=True)")
    elif ml is False:
        parts.append("- Mainlobe: User IS in serving cell's mainlobe (Outside_Mainlobe_Flag=False)")

    if feats.get("has_a2_event"):
        parts.append("- Signaling: NREventA2 triggered (serving signal dropped below threshold)")
    elif feats.get("has_a5_event"):
        parts.append("- Signaling: NREventA5 measurement configured")
    else:
        parts.append("- Signaling: No A2/A3/A5 trigger events")

    # Throughput summary
    tp = feats.get("throughput_raw", "")
    if tp:
        lines = tp.strip().split("\n")
        vals = []
        for l in lines[1:]:
            p = l.split("|")
            if len(p) >= 2:
                try:
                    vals.append(float(p[1]))
                except:
                    pass
        if vals:
            parts.append(f"- Throughput: max={max(vals):.0f}, min={min(vals):.0f}, worst={feats.get('worst_tp', 0):.0f} Mbps")

    return "\n".join(parts)


CATEGORY_TO_KEYWORDS = {
    "add_neighbor": ["add neighbor"],
    "decrease_a3": ["decrease", "a3"],
    "lift_tilt": ["lift the tilt", "lift_tilt"],
    "insufficient": ["insufficient"],
    "check_server": ["check test server", "transmission issues", "check_server"],
    "pdcch": ["pdcch", "2sym", "pdcchocc"],
}


def match_option(category: str, options: list) -> Optional[str]:
    keywords = CATEGORY_TO_KEYWORDS.get(category, [])
    for opt in options:
        lbl = opt["label"].lower()
        if category == "decrease_a3":
            if "decrease" in lbl and "a3" in lbl:
                return opt["id"]
        elif category == "check_server":
            if "check" in lbl and ("server" in lbl or "transmission" in lbl):
                return opt["id"]
        else:
            for kw in keywords:
                if kw in lbl:
                    return opt["id"]
    return None


def f1_llm_solve(llm: LLMClient, feats: Dict, scenario: Dict) -> str:
    evidence = build_f1_evidence(feats)
    prompt = f"""<|im_start|>system
{F1_SYSTEM}<|im_end|>
<|im_start|>user
{evidence}

Category:<|im_end|>
<|im_start|>assistant
\\boxed{{"""

    raw = llm.complete(prompt, max_tokens=15)
    # Parse category
    raw_lower = raw.lower().strip()
    for cat in ["add_neighbor", "decrease_a3", "lift_tilt", "insufficient", "check_server", "pdcch"]:
        if cat in raw_lower:
            opt_id = match_option(cat, scenario["task"]["options"])
            if opt_id:
                return opt_id
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic Handlers (F2, F3, F4)
# ──────────────────────────────────────────────────────────────────────────────

def handle_f2(feats: Dict, scenario: Dict) -> str:
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
    
    az = next((o for o in options if "adjust the azimuth" in o["label"].lower()), None)
    pw = next((o for o in options if "increase transmission power" in o["label"].lower()), None)
    if az and pw:
        return "|".join(sorted([az["id"], pw["id"]], key=lambda x: int(x[1:])))
    return ""


def handle_f3(scenario: Dict) -> str:
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
    options = scenario["task"]["options"]
    for opt in options:
        lbl = opt["label"].lower().replace(" ", "")
        if "a5rsrpthld1" in lbl or ("a5" in lbl and "decrease" in lbl):
            return opt["id"]
    for opt in options:
        lbl = opt["label"].lower().replace(" ", "")
        if "a2rsrp" in lbl and "decrease" in lbl:
            return opt["id"]
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_url", default="https://120.46.145.77/no")
    ap.add_argument("--token", required=True)
    ap.add_argument("--model_url", default="http://127.0.0.1:8001")
    ap.add_argument("--test_json", type=Path, default=_HERE / "data/Phase_2/test.json")
    ap.add_argument("--out", type=Path, default=_HERE / "submission_max_v3.csv")
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
    llm = LLMClient(args.model_url)

    existing = {}
    if args.resume and args.out.exists():
        with open(args.out, newline="") as f:
            for row in csv.DictReader(f):
                sid = row.get("ID", "").strip()
                if sid and row.get("Track A", "").strip():
                    existing[sid] = row["Track A"]
        logger.info(f"Resuming with {len(existing)} existing")

    results = []
    stats = {"F1_llm": 0, "F2": 0, "F3": 0, "F4": 0, "F4M": 0, "F1_5BS": 0}
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
            method = "F4M"
            stats["F4M"] += 1

        elif tag == "multiple-answer" and num_bs == "5":
            answer = handle_f3(scenario)
            method = "F3"
            stats["F3"] += 1

        elif tag == "multiple-answer" and num_bs == "4":
            feats = extract_features(api, sid)
            answer = handle_f2(feats, scenario)
            method = "F2"
            stats["F2"] += 1

        elif tag == "single-answer" and num_bs in ("4", "5"):
            feats = extract_features(api, sid)
            answer = f1_llm_solve(llm, feats, scenario)
            if not answer:
                # Fallback: use deterministic
                if feats.get("outside_mainlobe") is True:
                    answer = match_option("lift_tilt", scenario["task"]["options"]) or ""
                elif feats.get("serving_rsrp") and float(feats["serving_rsrp"]) < -105:
                    answer = match_option("insufficient", scenario["task"]["options"]) or ""
                else:
                    answer = match_option("pdcch", scenario["task"]["options"]) or ""
            method = "F1-LLM" if num_bs == "4" else "F1_5BS-LLM"
            stats["F1_llm" if num_bs == "4" else "F1_5BS"] += 1

        elif tag == "single-answer" and num_bs == "13":
            answer = handle_f4_single(scenario)
            method = "F4"
            stats["F4"] += 1

        # Validate
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

        if (idx + 1) % 10 == 0 or elapsed > 3:
            logger.info(f"[{idx+1}/{len(scenarios)}] {sid[:8]} {method} → {final} ({elapsed:.1f}s)")

        if (idx + 1) % args.save_freq == 0:
            _save(results, args.out)

    _save(results, args.out)
    total_t = time.time() - start
    answered = sum(1 for r in results if r["Track A"])
    logger.info(f"\nDone: {answered}/{len(results)} in {total_t:.0f}s")
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
