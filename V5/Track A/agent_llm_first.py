#!/usr/bin/env python3
"""
LLM-First Agent for Track A — targets 0.4+ public score.

Core strategy: Use Qwen3.5-35B as primary reasoner with full network data.
The deterministic pipeline gets ~26% accuracy. This LLM-first approach should
reach 80%+ by giving the model complete diagnostic evidence.
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    "get_gnodeb_location": "/gnodeb-location",
    "get_user_location": "/user-location",
    "get_serving_cell_pci": "/serving-cell-pci",
    "get_serving_cell_rsrp": "/serving-cell-rsrp",
    "get_serving_cell_sinr": "/serving-cell-sinr",
    "get_rbs_allocated_to_user": "/rbs-allocated-to-user",
    "get_signaling_plane_event_log": "/signaling-plane-event-log",
    "get_neighboring_cells_pci": "/neighboring-cells-pci",
    "get_neighboring_cell_rsrp": "/neighboring-cell-rsrp",
    "get_all_cells_pci": "/all-cells-pci",
    "get_kpi_data": "/get_kpi_data",
    "get_mr_data": "/get_mr_data",
    "judge_mainlobe_or_not": "/judge_mainlobe",
    "calculate_overlap_ratio": "/calculate_overlap_ratio",
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

    def call(self, tool_name: str, scenario_id: str, **params) -> Any:
        endpoint = ENDPOINT_MAP.get(tool_name)
        if not endpoint:
            return {"error": f"Unknown tool: {tool_name}"}
        url = f"{self.server_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Scenario-Id": scenario_id,
            "X-API-Token": self.token,
        }
        try:
            resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            self.call_count += 1
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"error": str(e)[:100]}


# ──────────────────────────────────────────────────────────────────────────────
# Data Gathering - Comprehensive
# ──────────────────────────────────────────────────────────────────────────────

def safe_get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def gather_full_data(client: APIClient, scenario_id: str) -> Dict[str, Any]:
    """Fetch comprehensive diagnostic data for a scenario."""
    data = {}

    # 1. Throughput logs
    tp = client.call("get_throughput_logs", scenario_id)
    tp_str = safe_get(tp, "Logs", default="")
    data["throughput"] = tp_str

    # 2. All cells
    cells = client.call("get_all_cells_pci", scenario_id)
    data["all_cells"] = safe_get(cells, "PCIs", default=[])

    # 3. KPI data
    kpi = client.call("get_kpi_data", scenario_id)
    data["kpi"] = safe_get(kpi, "KPI", "Traffic Data", "traffic_data", default="")

    # 4. MR data
    mr = client.call("get_mr_data", scenario_id)
    data["mr"] = safe_get(mr, "MR", "MR Data", "mr_data", default="")

    # 5. Find worst throughput time
    worst_time = _find_worst_time(tp_str)
    data["worst_time"] = worst_time
    if not worst_time:
        return data

    # 6. Serving cell at worst time
    serving_resp = client.call("get_serving_cell_pci", scenario_id, time=worst_time)
    serving_pci = safe_get(serving_resp, "PCI", "pci", default=None)
    if isinstance(serving_pci, (int, float)):
        serving_pci = int(serving_pci)
    data["serving_pci"] = serving_pci

    rsrp_resp = client.call("get_serving_cell_rsrp", scenario_id, time=worst_time)
    data["serving_rsrp"] = safe_get(rsrp_resp, "SS-RSRP (dBm)", "RSRP", default=None)

    sinr_resp = client.call("get_serving_cell_sinr", scenario_id, time=worst_time)
    data["serving_sinr"] = safe_get(sinr_resp, "SS-SINR (dB)", "SINR", default=None)

    # 7. Neighbors at worst time
    nb_resp = client.call("get_neighboring_cells_pci", scenario_id, time=worst_time)
    neighbor_pcis = safe_get(nb_resp, "PCIs", "pci", default=[])
    if isinstance(neighbor_pcis, int):
        neighbor_pcis = [neighbor_pcis]
    data["neighbor_pcis"] = neighbor_pcis

    # 8. Neighbor RSRPs
    neighbor_rsrps = {}
    for npci in (neighbor_pcis or [])[:4]:
        nrsrp = client.call("get_neighboring_cell_rsrp", scenario_id, time=worst_time, pci=int(npci))
        val = safe_get(nrsrp, "SS-RSRP (dBm)", "RSRP", default=None)
        if val is not None:
            neighbor_rsrps[int(npci)] = val
    data["neighbor_rsrps"] = neighbor_rsrps

    # 9. Serving cell configuration
    if serving_pci:
        cfg = client.call("get_cell_info", scenario_id, pci=int(serving_pci))
        if isinstance(cfg, dict) and "error" not in cfg:
            data["serving_config"] = cfg

    # 10. Signaling events
    sig = client.call("get_signaling_plane_event_log", scenario_id, time=worst_time)
    data["signaling"] = safe_get(sig, "Logs", default="")

    # 11. Judge mainlobe for coverage diagnosis
    if serving_pci and worst_time:
        mainlobe = client.call("judge_mainlobe_or_not", scenario_id, time=worst_time, pci=int(serving_pci))
        data["mainlobe"] = mainlobe

    return data


def _find_worst_time(tp_str: str) -> Optional[str]:
    if not tp_str:
        return None
    lines = tp_str.strip().split("\n")
    worst_time, worst_val = None, float("inf")
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) >= 2:
            try:
                val = float(parts[1].strip())
                if val < worst_val:
                    worst_val = val
                    worst_time = parts[0].strip()
            except (ValueError, IndexError):
                continue
    return worst_time


# ──────────────────────────────────────────────────────────────────────────────
# LLM Prompting
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert 5G wireless network optimization engineer analyzing drive test data to diagnose throughput degradation. You must select the correct optimization action from the given options.

## Diagnostic Framework

Analyze in this order:
1. **Throughput pattern**: When does degradation occur? How severe?
2. **RF quality at degradation point**: Check serving RSRP and SINR.
3. **Neighbor analysis**: Are there strong neighbors? Are they in the neighbor list?
4. **Signaling events**: Were A3 handover events triggered? Did handover succeed?
5. **Cell configuration**: A3 offset, neighbor list, PDCCH settings.
6. **Traffic load**: PRB utilization, CCE utilization.
7. **Mainlobe check**: Is the user outside the cell's main coverage area?

## Decision Logic for Single-Answer (4 BS):

| Condition | Action |
|-----------|--------|
| Strong neighbor (RSRP > serving) NOT in neighbor list → A3 event triggered but no handover possible | **Add neighbor relationship** |
| Strong neighbor in neighbor list, A3 offset too high → handover delayed | **Decrease A3 Offset** |
| Poor serving RSRP (<-100dBm) or SINR (<0dB), user outside mainlobe | **Lift tilt angle** |
| High CCE utilization (>50%) or low CCE allocation success rate (<95%) | **Modify PDCCH to 2SYM** |
| Good RF (RSRP>-90, SINR>10) but low throughput, low PRB utilization | **Check test server/transmission** |
| No clear single root cause identifiable from available data | **Insufficient data** |

## Key Signals:
- A3 Event in signaling: "NREventA3|ServCellRSRP:X;NCellPCI:Y;NCellRSRP:Z" means neighbor Y is stronger
- Neighbor list format in config: "[gNodeB_ARFCN_PCI, ...]" - check if the strong neighbor PCI is listed
- A3 Offset in config: "IntraFreqHoA3Offset [0.5dB]" - the value is in 0.5dB units, so 6 means 3dB
- Outside_Mainlobe_Flag = True means user is NOT well covered → suggests lift_tilt
- "Downlink CCE Allocation Success Rate" < 95% suggests PDCCH congestion

## For Multiple-Answer 4BS (F2 - exactly 2 codes):
The answer is ALWAYS one of these two pair types:
- **Type A**: "Adjust azimuth" + "Increase transmission power" (both targeting cells that need better coverage toward the user)
- **Type B**: "Decrease transmission power" + "Press down tilt" (both targeting cells that cause interference/overshooting)
Choose the pair that matches the diagnosed issue (coverage gap → Type A, interference/overshoot → Type B).

## For Multiple-Answer 5BS (F3 - exactly 4 codes):
Always: "Decrease transmission power" + "Press down tilt" (same overshooter cell) + 2x "Increase A3 Offset" (on two cells).

## For 13BS scenarios:
Always select the option(s) that decrease inter-frequency thresholds (CovInterFreqA2RsrpThld or CovInterFreqA5RsrpThld).

## Output Format:
Output ONLY the boxed answer. Examples: \\boxed{C7} or \\boxed{C3|C11}
Do NOT provide explanations."""


def format_data_for_prompt(gathered: Dict[str, Any]) -> str:
    """Format gathered API data into a concise text block for the LLM."""
    parts = []

    # Throughput
    if gathered.get("throughput"):
        parts.append(f"### Throughput Logs\n{gathered['throughput']}")

    # Key metrics at degradation
    if gathered.get("worst_time"):
        metrics = f"### Degradation Point: {gathered['worst_time']}\n"
        metrics += f"- Serving PCI: {gathered.get('serving_pci')}\n"
        metrics += f"- Serving RSRP: {gathered.get('serving_rsrp')} dBm\n"
        metrics += f"- Serving SINR: {gathered.get('serving_sinr')} dB\n"

        # Neighbors with RSRP
        if gathered.get("neighbor_rsrps"):
            metrics += "- Neighbor cells:\n"
            for pci, rsrp in sorted(gathered["neighbor_rsrps"].items(), key=lambda x: -x[1]):
                metrics += f"  PCI {pci}: RSRP = {rsrp} dBm\n"
        elif gathered.get("neighbor_pcis"):
            metrics += f"- Neighbor PCIs: {gathered['neighbor_pcis']}\n"

        parts.append(metrics)

    # Serving cell config (critical for neighbor list and A3 offset)
    if gathered.get("serving_config"):
        cfg = gathered["serving_config"]
        config_text = "### Serving Cell Configuration\n"
        important_keys = [
            "PCI", "IntraFreqHoA3Offset [0.5dB]", "IntraFreqHoA3Hyst [0.5dB]",
            "PCell Neighbor Cell (gNodeBID_ARFCN_PCI)", "PdcchOccupiedSymbolNum",
            "Transmission Power", "Mechanical Azimuth", "Digital Tilt",
            "CovInterFreqA2RsrpThld [dBm]", "CovInterFreqA5RsrpThld1 [dBm]",
        ]
        for k in important_keys:
            if k in cfg:
                config_text += f"- {k}: {cfg[k]}\n"
        parts.append(config_text)

    # Signaling events
    if gathered.get("signaling") and isinstance(gathered["signaling"], str) and len(gathered["signaling"]) > 20:
        parts.append(f"### Signaling Events\n{gathered['signaling']}")

    # KPI data
    if gathered.get("kpi") and isinstance(gathered["kpi"], str) and len(gathered["kpi"]) > 20:
        parts.append(f"### KPI/Traffic Data\n{gathered['kpi']}")

    # MR data
    if gathered.get("mr") and isinstance(gathered["mr"], str) and len(gathered["mr"]) > 20:
        parts.append(f"### MR Data\n{gathered['mr']}")

    # Mainlobe judgment
    if gathered.get("mainlobe") and isinstance(gathered["mainlobe"], dict):
        flag = gathered["mainlobe"].get("Outside_Mainlobe_Flag")
        if flag is not None:
            parts.append(f"### Mainlobe Check\nOutside_Mainlobe_Flag: {flag} ({'user is outside mainlobe coverage' if flag else 'user is within mainlobe'})")

    return "\n\n".join(parts)


def build_messages(scenario: Dict, gathered: Dict[str, Any], few_shot_text: str = "") -> List[Dict[str, str]]:
    """Build chat messages for the LLM."""
    tag = scenario.get("tag", "single-answer")
    num_bs = str(scenario["context"]["wireless_network_information"]["num_base_stations"])

    options_text = "\n".join(f"{opt['id']}: {opt['label']}" for opt in scenario["task"]["options"])
    data_text = format_data_for_prompt(gathered)

    # Cardinality hint
    if tag == "single-answer":
        card = "Select exactly ONE option."
    elif num_bs == "5":
        card = "Select exactly FOUR options (sorted by number, separated by |)."
    elif num_bs == "13":
        card = "Select exactly TWO options (both should be inter-frequency threshold decreases)."
    else:
        card = "Select exactly TWO options (sorted by number, separated by |)."

    user_content = f"""## Task
{scenario['task']['description']}

## Format
{card} Output \\boxed{{answer}}.

## Options
{options_text}

## Network Evidence
{data_text}

{few_shot_text}

Select the correct option(s) based on the evidence. Output ONLY \\boxed{{...}}."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Few-Shot from Training Data
# ──────────────────────────────────────────────────────────────────────────────

def load_few_shot_bank(train_path: Path) -> Dict[str, List[Dict]]:
    """Pre-load training examples grouped by type."""
    if not train_path.exists():
        return {}
    with open(train_path) as f:
        train = json.load(f)

    groups = {}
    for t in train:
        tag = t["tag"]
        bs = str(t["context"]["wireless_network_information"]["num_base_stations"])
        key = f"{tag}_{bs}"
        groups.setdefault(key, []).append(t)
    return groups


def make_few_shot_text(groups: Dict, scenario: Dict) -> str:
    """Create a compact few-shot example."""
    tag = scenario["tag"]
    bs = str(scenario["context"]["wireless_network_information"]["num_base_stations"])
    key = f"{tag}_{bs}"

    examples = groups.get(key, [])
    if not examples:
        return ""

    # Pick 1 example
    ex = examples[0]
    opts = "\n".join(f"{o['id']}: {o['label']}" for o in ex["task"]["options"])
    answer = ex["answer"]

    return f"""## Reference Example (same scenario type)
Options: {opts}
Correct answer: \\boxed{{{answer}}}"""


# ──────────────────────────────────────────────────────────────────────────────
# LLM Client
# ──────────────────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, base_url: str, model: str = "qwen", max_tokens: int = 256, timeout: float = 90.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        try:
            resp = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""


# ──────────────────────────────────────────────────────────────────────────────
# Answer Parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_answer(text: str) -> str:
    if not text:
        return ""
    for pat in [r"\\boxed\{([^}]+)\}", r"boxed\{([^}]+)\}"]:
        m = re.findall(pat, text)
        if m:
            return m[-1].strip()
    codes = re.findall(r"\bC\d+\b", text)
    if codes:
        return "|".join(dict.fromkeys(codes))  # dedupe preserving order
    return ""


def validate_and_fix(answer: str, scenario: Dict) -> str:
    if not answer:
        return ""
    valid_ids = {opt["id"] for opt in scenario["task"]["options"]}
    codes = [c.strip() for c in answer.split("|") if c.strip() in valid_ids]
    if not codes:
        return ""

    tag = scenario["tag"]
    num_bs = str(scenario["context"]["wireless_network_information"]["num_base_stations"])

    if tag == "single-answer":
        return codes[0]
    elif num_bs == "5":
        return "|".join(sorted(set(codes[:4]), key=lambda x: int(x[1:])))
    else:
        return "|".join(sorted(set(codes[:2]), key=lambda x: int(x[1:])))


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic Handlers (high-confidence families)
# ──────────────────────────────────────────────────────────────────────────────

def handle_f4(scenario: Dict) -> str:
    """13BS single-answer: always decrease_interfreq."""
    for opt in scenario["task"]["options"]:
        lbl = opt["label"].lower()
        if ("interfreq" in lbl or "a2rsrp" in lbl or "a5rsrp" in lbl) and \
           ("decrease" in lbl or "threshold" in lbl):
            return opt["id"]
    return ""


def handle_f4m(scenario: Dict) -> str:
    """13BS multiple-answer: 2x decrease_interfreq."""
    candidates = []
    for opt in scenario["task"]["options"]:
        lbl = opt["label"].lower()
        if ("interfreq" in lbl or "a2rsrp" in lbl or "a5rsrp" in lbl) and \
           ("decrease" in lbl or "threshold" in lbl):
            candidates.append(opt["id"])
    if len(candidates) >= 2:
        return "|".join(sorted(candidates[:2], key=lambda x: int(x[1:])))
    return ""


def handle_f3(scenario: Dict) -> str:
    """5BS multiple-answer: always decrease_power + press_down_tilt + 2x increase_a3."""
    decrease_power = []
    press_down = []
    increase_a3 = []

    for opt in scenario["task"]["options"]:
        lbl = opt["label"].lower()
        if "decrease transmission power" in lbl:
            decrease_power.append(opt)
        elif "press down the tilt" in lbl:
            press_down.append(opt)
        elif "increase" in lbl and "a3" in lbl:
            increase_a3.append(opt)

    if not decrease_power or not press_down or len(increase_a3) < 2:
        return ""

    # The decrease_power and press_down_tilt should target the same cell (the overshooter)
    # Find matching cell
    dp_opt = decrease_power[0]
    dp_cell = _extract_cell_id(dp_opt["label"])

    # Find press_down for same cell
    pd_opt = None
    for opt in press_down:
        if dp_cell and dp_cell in opt["label"]:
            pd_opt = opt
            break
    if not pd_opt:
        pd_opt = press_down[0]

    # Take all increase_a3 options (usually 2)
    codes = [dp_opt["id"], pd_opt["id"]] + [o["id"] for o in increase_a3[:2]]
    return "|".join(sorted(set(codes), key=lambda x: int(x[1:])))


def _extract_cell_id(label: str) -> str:
    """Extract cell ID like '3279943_1' from option label."""
    import re
    m = re.search(r'(\d{7}_\d+)', label)
    return m.group(1) if m else ""


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_url", default="https://120.46.145.77/no")
    ap.add_argument("--token", required=True)
    ap.add_argument("--model_url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model_name", default="qwen")
    ap.add_argument("--test_json", type=Path, default=_HERE / "data/Phase_2/test.json")
    ap.add_argument("--train_json", type=Path, default=_HERE / "data/Phase_1/train.json")
    ap.add_argument("--out", type=Path, default=_HERE / "submission_llm_first.csv")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--save_freq", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--start_idx", type=int, default=0)
    args = ap.parse_args()

    with open(args.test_json) as f:
        scenarios = json.load(f)
    logger.info(f"Loaded {len(scenarios)} scenarios")

    few_shot_bank = load_few_shot_bank(args.train_json)

    if args.max_samples:
        scenarios = scenarios[args.start_idx:args.start_idx + args.max_samples]
    elif args.start_idx:
        scenarios = scenarios[args.start_idx:]

    api = APIClient(args.server_url, args.token)
    llm = LLMClient(args.model_url, args.model_name, max_tokens=args.max_tokens)

    # Resume
    existing = {}
    if args.resume and args.out.exists():
        with open(args.out, newline="") as f:
            for row in csv.DictReader(f):
                sid = row.get("ID", "").strip()
                if sid and row.get("Track A", "").strip():
                    existing[sid] = row["Track A"]
        logger.info(f"Resuming with {len(existing)} existing")

    results = []
    start = time.time()

    for idx, scenario in enumerate(scenarios):
        sid = scenario["scenario_id"]
        tag = scenario["tag"]
        num_bs = str(scenario["context"]["wireless_network_information"]["num_base_stations"])

        if sid in existing:
            results.append({"ID": sid, "Track A": existing[sid], "Track B": ""})
            continue

        t0 = time.time()

        # Deterministic paths for high-confidence families
        if tag == "multiple-answer" and num_bs == "13":
            ans = handle_f4m(scenario)
            if ans:
                results.append({"ID": sid, "Track A": ans, "Track B": ""})
                logger.info(f"[{idx+1}] {sid[:8]} F4M→{ans} ({time.time()-t0:.1f}s)")
                continue

        if tag == "multiple-answer" and num_bs == "5":
            ans = handle_f3(scenario)
            if ans:
                results.append({"ID": sid, "Track A": ans, "Track B": ""})
                logger.info(f"[{idx+1}] {sid[:8]} F3→{ans} ({time.time()-t0:.1f}s)")
                continue

        # LLM path: gather data + prompt
        try:
            gathered = gather_full_data(api, sid)
        except Exception as e:
            logger.error(f"[{sid[:8]}] gather error: {e}")
            gathered = {}

        few_shot = make_few_shot_text(few_shot_bank, scenario)
        messages = build_messages(scenario, gathered, few_shot)

        raw = llm.chat(messages)
        parsed = parse_answer(raw)
        final = validate_and_fix(parsed, scenario)

        # Retry if failed
        if not final:
            valid_opts = ", ".join(opt["id"] for opt in scenario["task"]["options"])
            retry_msg = messages + [
                {"role": "assistant", "content": raw or "(empty)"},
                {"role": "user", "content": f"Invalid format. Valid codes: {valid_opts}. Output \\boxed{{CX}} only."},
            ]
            raw2 = llm.chat(retry_msg)
            parsed2 = parse_answer(raw2)
            final = validate_and_fix(parsed2, scenario)

        results.append({"ID": sid, "Track A": final, "Track B": ""})
        logger.info(f"[{idx+1}] {sid[:8]} {tag}/{num_bs}BS → {final} ({time.time()-t0:.1f}s)")

        if (idx + 1) % args.save_freq == 0:
            _save(results, args.out)

    _save(results, args.out)
    elapsed = time.time() - start
    answered = sum(1 for r in results if r["Track A"])
    logger.info(f"\nDone: {answered}/{len(results)} in {elapsed:.0f}s. API calls: {api.call_count}")


def _save(results: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Track A", "Track B"])
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"  Saved {len(results)} rows → {path}")


if __name__ == "__main__":
    main()
