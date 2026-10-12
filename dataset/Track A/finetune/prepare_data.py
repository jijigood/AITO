#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Fine-tuning dataset generator for Track A.

Reads the 2000 training examples from train.json and produces a JSONL file
in the OpenAI multi-turn chat format suitable for supervised fine-tuning (SFT)
with LLaMA-Factory.

Each output record is a full agent conversation:
  [system] diagnostic protocol
  [user]   question + 22 options
  [assistant → tool] get_throughput_logs call
  [tool]   throughput data
  [assistant → tool] get_serving_cell_pci call
  [tool]   PCI response
  [assistant → tool] get_cell_info + get_rsrp + get_sinr + get_neighbors (parallel)
  [tool × 4] responses
  [assistant → tool] get_neighboring_cell_rsrp × N neighbors (parallel)
  [tool × N] responses
  [assistant] Evidence → Findings → Recommendations → \\boxed{answer}

The tool call results are synthesised directly from the embedded CSV data in
train.json — no HTTP server needed during dataset generation.

Output:
    finetune/sft_train.jsonl  (1800 examples)
    finetune/sft_val.jsonl    (200 examples)

Usage:
    cd "Track A"
    python finetune/prepare_data.py
"""

import json
import io
import os
import re
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add parent directory to path so we can import from Track A root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import SYSTEM_PROMPT

# ──────────────────────────────────────────────────────────────────────────────
# Data parsing helpers (mirror server.py logic without HTTP)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_csv(raw: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(raw), sep="|")


def _get_throughput_logs(data: Dict) -> str:
    """Simulate get_throughput_logs response."""
    raw = data.get("user_plane_data", "")
    if not raw:
        return json.dumps({"error": "No user-plane data"})
    df = _parse_csv(raw)
    result = df[["Timestamp", "5G KPI PCell Layer2 MAC DL Throughput [Mbps]"]].to_csv(index=False, sep="|")
    return json.dumps({"Logs": result})


def _find_worst_timestamp(data: Dict) -> Optional[str]:
    """Find the timestamp with the lowest throughput."""
    raw = data.get("user_plane_data", "")
    if not raw:
        return None
    df = _parse_csv(raw)
    col = "5G KPI PCell Layer2 MAC DL Throughput [Mbps]"
    if col not in df.columns:
        return None
    df[col] = pd.to_numeric(df[col], errors="coerce")
    idx = df[col].idxmin()
    return str(df.loc[idx, "Timestamp"]).strip()


def _get_serving_cell_pci(data: Dict, time: str) -> str:
    """Simulate get_serving_cell_pci response."""
    raw = data.get("user_plane_data", "")
    if not raw:
        return json.dumps({"error": "No user-plane data"})
    df = _parse_csv(raw)
    col_pci = "5G KPI PCell RF Serving PCI"
    col_ts = "Timestamp"
    if col_pci not in df.columns:
        return json.dumps({"error": "No PCI column"})
    row = df[df[col_ts].str.strip() == time.strip()]
    if row.empty:
        # Fall back to first row
        row = df.iloc[[0]]
    pci = int(float(row[col_pci].iloc[0]))
    return json.dumps({"PCI": pci})


def _get_serving_pci_value(data: Dict, time: str) -> Optional[int]:
    result = json.loads(_get_serving_cell_pci(data, time))
    return result.get("PCI")


def _get_cell_info(data: Dict, pci: int) -> str:
    """Simulate get_cell_info response."""
    raw = data.get("network_configuration_data", "")
    if not raw:
        return json.dumps({"error": "No config data"})
    df = _parse_csv(raw)
    df["PCI"] = pd.to_numeric(df["PCI"], errors="coerce")
    row = df[df["PCI"] == pci]
    if row.empty:
        return json.dumps({"error": f"No cell found for PCI {pci}"})
    result = row.iloc[0].to_dict()
    result["notes"] = data.get("notes", "")
    return json.dumps({k: (None if pd.isna(v) else v) for k, v in result.items()})


def _get_serving_cell_rsrp(data: Dict, time: str) -> str:
    raw = data.get("user_plane_data", "")
    if not raw:
        return json.dumps({"error": "No user-plane data"})
    df = _parse_csv(raw)
    col = "5G KPI PCell RF Serving SS-RSRP [dBm]"
    row = df[df["Timestamp"].str.strip() == time.strip()]
    if row.empty or col not in df.columns:
        return json.dumps({"error": "No RSRP data"})
    return json.dumps({"SS-RSRP (dBm)": float(row[col].iloc[0])})


def _get_serving_cell_sinr(data: Dict, time: str) -> str:
    raw = data.get("user_plane_data", "")
    if not raw:
        return json.dumps({"error": "No user-plane data"})
    df = _parse_csv(raw)
    col = "5G KPI PCell RF Serving SS-SINR [dB]"
    row = df[df["Timestamp"].str.strip() == time.strip()]
    if row.empty or col not in df.columns:
        return json.dumps({"error": "No SINR data"})
    return json.dumps({"SS-SINR (dB)": float(row[col].iloc[0])})


def _get_neighboring_cells_pci(data: Dict, time: str) -> str:
    """Simulate get_neighboring_cells_pci — reads neighbour PCI columns from user_plane_data."""
    raw = data.get("user_plane_data", "")
    if not raw:
        return json.dumps({"error": "No user-plane data"})
    df = _parse_csv(raw)
    row = df[df["Timestamp"].str.strip() == time.strip()]
    if row.empty:
        row = df.iloc[[0]]

    # Neighbour PCI columns follow pattern: "5G KPI SCell* RF Neighbor PCI*" or similar
    neighbor_cols = [c for c in df.columns if "Neighbor" in c and "PCI" in c and "Serving" not in c]
    pcis = []
    for col in neighbor_cols:
        val = row[col].iloc[0]
        try:
            pci = int(float(val))
            if pci > 0:
                pcis.append(pci)
        except (ValueError, TypeError):
            pass
    return json.dumps({"PCIs": list(set(pcis))})


def _get_neighbor_pci_list(data: Dict, time: str) -> List[int]:
    result = json.loads(_get_neighboring_cells_pci(data, time))
    return result.get("PCIs", [])


def _get_neighboring_cell_rsrp(data: Dict, time: str, pci: int) -> str:
    """Simulate get_neighboring_cell_rsrp — reads from mr_data or user_plane_data."""
    # Try mr_data first
    raw_mr = data.get("mr_data", "")
    if raw_mr:
        df = _parse_csv(raw_mr)
        # Look for columns like "Neighbor N PCI" and "Neighbor N RSRP(dBm)"
        for n in range(1, 7):
            pci_col = f"Neighbor {n} PCI"
            rsrp_col = f"Neighbor {n} RSRP(dBm)"
            if pci_col in df.columns and rsrp_col in df.columns:
                mask = df[pci_col].apply(lambda x: _safe_int(x) == pci)
                if mask.any():
                    rsrp_val = df.loc[mask, rsrp_col].iloc[0]
                    return json.dumps({"Filtered Tx BRSRP [dBm]": float(rsrp_val)})

    return json.dumps({"error": f"No RSRP data for neighbor PCI {pci}"})


def _safe_int(v) -> Optional[int]:
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Conversation builder
# ──────────────────────────────────────────────────────────────────────────────

def _make_tool_call_block(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)}
    }


def _make_assistant_tool_turn(tool_calls: List[Tuple[str, str, dict]]) -> dict:
    """Build an assistant message that requests tool calls."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            _make_tool_call_block(call_id, name, args)
            for call_id, name, args in tool_calls
        ]
    }


def _make_tool_result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def build_conversation(example: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Build a full multi-turn conversation for one training example.
    Returns None if the data is insufficient to construct a valid conversation.
    """
    data = example.get("data", {})
    task = example.get("task", {})
    answer = example.get("answer", "")
    tag = example.get("tag", "single-answer")
    options = task.get("options", [])

    if not answer or not data.get("user_plane_data"):
        return None

    # ── Question text ──
    root_causes = "".join([f"{opt['id']}:{opt['label']}\n" for opt in options])
    question = task.get("description", "") + f"\nOptions:\n{root_causes}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # ── Turn 1: get_throughput_logs ──
    t1_id = "call_001"
    messages.append(_make_assistant_tool_turn([(t1_id, "get_throughput_logs", {})]))
    tp_result = _get_throughput_logs(data)
    messages.append(_make_tool_result(t1_id, tp_result))

    # ── Find worst timestamp ──
    worst_t = _find_worst_timestamp(data)
    if not worst_t:
        return None

    # ── Turn 2: get_serving_cell_pci ──
    t2_id = "call_002"
    messages.append(_make_assistant_tool_turn([(t2_id, "get_serving_cell_pci", {"time": worst_t})]))
    pci_result = _get_serving_cell_pci(data, worst_t)
    messages.append(_make_tool_result(t2_id, pci_result))

    serving_pci = _get_serving_pci_value(data, worst_t)
    if serving_pci is None:
        return None

    # ── Turn 3: parallel — get_cell_info + rsrp + sinr + neighbors ──
    t3_calls = [
        ("call_003", "get_cell_info", {"pci": serving_pci}),
        ("call_004", "get_serving_cell_rsrp", {"time": worst_t}),
        ("call_005", "get_serving_cell_sinr", {"time": worst_t}),
        ("call_006", "get_neighboring_cells_pci", {"time": worst_t}),
    ]
    messages.append(_make_assistant_tool_turn(t3_calls))
    # Compute once — reuse for both message building and final reasoning
    cell_info_str = _get_cell_info(data, serving_pci)
    rsrp_result_str = _get_serving_cell_rsrp(data, worst_t)
    sinr_result_str = _get_serving_cell_sinr(data, worst_t)
    neighbor_result = _get_neighboring_cells_pci(data, worst_t)
    messages.append(_make_tool_result("call_003", cell_info_str))
    messages.append(_make_tool_result("call_004", rsrp_result_str))
    messages.append(_make_tool_result("call_005", sinr_result_str))
    messages.append(_make_tool_result("call_006", neighbor_result))

    # Extract scalar values for final reasoning block
    rsrp_val: Optional[float] = None
    sinr_val: Optional[float] = None
    a3_threshold: Optional[float] = None
    try:
        rsrp_val = float(json.loads(rsrp_result_str).get("SS-RSRP (dBm)", ""))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    try:
        sinr_val = float(json.loads(sinr_result_str).get("SS-SINR (dB)", ""))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    try:
        ci = json.loads(cell_info_str)
        a3_off = ci.get("IntraFreqHoA3Offset")
        a3_hyst = ci.get("IntraFreqHoA3Hyst")
        if a3_off is not None and a3_hyst is not None:
            # Values stored in 0.5 dB units (e.g., 10 → 5.0 dB)
            a3_threshold = float(a3_off) * 0.5 + float(a3_hyst) * 0.5
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    neighbor_pcis = json.loads(neighbor_result).get("PCIs", [])

    # ── Turn 4: parallel — get_neighboring_cell_rsrp for each neighbor ──
    neighbor_rsrp_map: Dict[int, float] = {}
    if neighbor_pcis:
        t4_calls = [
            (f"call_{7 + j:03d}", "get_neighboring_cell_rsrp", {"time": worst_t, "pci": n_pci})
            for j, n_pci in enumerate(neighbor_pcis[:5])  # cap at 5 neighbors
        ]
        messages.append(_make_assistant_tool_turn(t4_calls))
        for t4_id, _, t4_args in t4_calls:
            n_pci = t4_args["pci"]
            n_rsrp_str = _get_neighboring_cell_rsrp(data, worst_t, n_pci)
            messages.append(_make_tool_result(t4_id, n_rsrp_str))
            try:
                n_rsrp = json.loads(n_rsrp_str).get("Filtered Tx BRSRP [dBm]")
                if n_rsrp is not None:
                    neighbor_rsrp_map[n_pci] = float(n_rsrp)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

    # ── Final assistant turn: data-driven reasoning + \boxed{answer} ──
    answer_labels = []
    for code in answer.split("|"):
        lbl = next((o["label"] for o in options if o["id"] == code), code)
        answer_labels.append(f"{code} ({lbl})")

    # Build evidence block from computed RF values
    evidence_lines = [f"- Worst throughput at {worst_t}"]
    if rsrp_val is not None and sinr_val is not None:
        evidence_lines.append(
            f"- Serving PCI {serving_pci}: RSRP={rsrp_val:.1f} dBm, SINR={sinr_val:.1f} dB"
        )
    else:
        evidence_lines.append(f"- Serving PCI {serving_pci}: signal data unavailable")

    if neighbor_rsrp_map:
        best_n = max(neighbor_rsrp_map, key=neighbor_rsrp_map.get)
        best_rsrp = neighbor_rsrp_map[best_n]
        if rsrp_val is not None:
            advantage = best_rsrp - rsrp_val
            evidence_lines.append(
                f"- Strongest neighbor PCI {best_n}: RSRP={best_rsrp:.1f} dBm "
                f"(advantage={advantage:+.1f} dB over serving)"
            )
            if a3_threshold is not None:
                margin = advantage - a3_threshold
                evidence_lines.append(
                    f"- A3 HO threshold={a3_threshold:.1f} dB; "
                    f"margin={margin:+.1f} dB "
                    f"({'HO should have triggered — late HO' if margin > 0 else 'HO condition not met'})"
                )
        else:
            evidence_lines.append(
                f"- Strongest neighbor PCI {best_n}: RSRP={best_rsrp:.1f} dBm"
            )

    if rsrp_val is not None and rsrp_val < -110:
        evidence_lines.append("- RSRP < -110 dBm → coverage hole condition")
    if rsrp_val is not None and sinr_val is not None and sinr_val < 0 and rsrp_val > -110:
        evidence_lines.append("- SINR < 0 dB with adequate RSRP → co-channel interference")

    final_content = (
        "Based on the data collected:\n\n"
        "**Evidence:**\n"
        + "\n".join(evidence_lines)
        + f"\n\n**Findings:** The data points to the following optimization action(s): "
        f"{'; '.join(answer_labels)}.\n\n"
        f"**Recommendations:** Apply the identified optimization(s) to resolve the fault.\n\n"
        f"\\boxed{{{answer}}}"
    )

    messages.append({"role": "assistant", "content": final_content})
    return messages


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    track_a_dir = os.path.dirname(script_dir)
    train_json = os.path.join(track_a_dir, "data", "Phase_1", "train.json")
    out_dir = script_dir

    print(f"Loading training data from {train_json}...")
    with open(train_json, "r", encoding="utf-8") as f:
        examples = json.load(f)

    print(f"Processing {len(examples)} examples...")
    conversations = []
    skipped = 0

    for i, ex in enumerate(examples):
        conv = build_conversation(ex)
        if conv is None:
            skipped += 1
            continue
        conversations.append({
            "messages": conv,
            "scenario_id": ex.get("scenario_id"),
            "answer": ex.get("answer"),
            "tag": ex.get("tag"),
        })
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(examples)} ({skipped} skipped)")

    print(f"\nTotal usable: {len(conversations)} | Skipped: {skipped}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(conversations)
    val_size = 200
    val_set = conversations[:val_size]
    train_set = conversations[val_size:]

    def write_jsonl(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} records to {path}")

    write_jsonl(os.path.join(out_dir, "sft_train.jsonl"), train_set)
    write_jsonl(os.path.join(out_dir, "sft_val.jsonl"), val_set)
    print("\nDone. Dataset ready for LLaMA-Factory fine-tuning.")


if __name__ == "__main__":
    main()
