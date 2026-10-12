#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
RF analysis utilities for 5G network fault diagnosis.

Two core checks used by the Diagnostic Agent:
  - check_a3_handover_condition: determines whether A3 HO should have fired
  - check_neighbor_in_config:    determines if a neighbor PCI is in the serving cell's config list
"""

import re
from typing import Optional


def check_a3_handover_condition(
    neighbor_rsrp: float,
    serving_rsrp: float,
    a3_offset_half_db: int,
    a3_hyst_half_db: int,
) -> dict:
    """
    Evaluate the 3GPP A3 intra-frequency handover condition:

        Neighbor_RSRP > Serving_RSRP + Offset + Hyst

    where Offset and Hyst are stored in 0.5 dB units in the network config.

    Args:
        neighbor_rsrp:      Filtered BRSRP of the candidate neighbor cell (dBm)
        serving_rsrp:       SS-RSRP of the current serving cell (dBm)
        a3_offset_half_db:  IntraFreqHoA3Offset value (in 0.5 dB units, e.g. 10 → 5.0 dB)
        a3_hyst_half_db:    IntraFreqHoA3Hyst value   (in 0.5 dB units, e.g. 2  → 1.0 dB)

    Returns:
        dict with keys:
          - condition_met (bool): True if HO should have triggered
          - delta_db (float):     neighbor_rsrp - serving_rsrp
          - threshold_db (float): offset_db + hyst_db (total HO threshold)
          - offset_db (float):    offset in real dB
          - hyst_db (float):      hysteresis in real dB
          - margin_db (float):    delta - threshold (positive → HO should fire, negative → stuck)
    """
    offset_db = a3_offset_half_db * 0.5
    hyst_db = a3_hyst_half_db * 0.5
    threshold_db = offset_db + hyst_db
    delta_db = neighbor_rsrp - serving_rsrp
    condition_met = delta_db > threshold_db
    margin_db = round(delta_db - threshold_db, 2)

    return {
        "condition_met": condition_met,
        "delta_db": round(delta_db, 2),
        "threshold_db": round(threshold_db, 2),
        "offset_db": offset_db,
        "hyst_db": hyst_db,
        "margin_db": margin_db,
    }


def check_neighbor_in_config(serving_cell_info: dict, neighbor_pci: int) -> bool:
    """
    Check whether a neighbor PCI appears in the serving cell's configured neighbor list.

    The neighbor list is stored as a string like:
        "[3276821_504990_761,3265067_504990_848,3253526_504990_443]"
    where each entry is  gNodeBID_ARFCN_PCI.

    Args:
        serving_cell_info: dict returned by get_cell_info (contains 'PCell Neighbor Cell' key)
        neighbor_pci:       PCI of the candidate neighbor to look up

    Returns:
        True if the PCI is present in the neighbor list, False otherwise.
    """
    neighbor_list_str = serving_cell_info.get("PCell Neighbor Cell (gNodeBID_ARFCN_PCI)", "") or ""
    # Extract all PCI values from entries like "gNodeBID_ARFCN_PCI"
    pcis_in_config = re.findall(r"_(\d+)(?:\]|,)", neighbor_list_str)
    return str(neighbor_pci) in pcis_in_config


def classify_rf_issue(
    serving_rsrp: float,
    serving_sinr: float,
    neighbor_rsrp_list: list,
    serving_cell_info: dict,
    neighbor_pci_list: list,
) -> dict:
    """
    Heuristic classifier that maps measured RF conditions to a root-cause category.
    Used as a reasoning aid for the LLM — it does NOT replace the model's judgment.

    Categories (returned as 'issue_type'):
      - 'late_handover'         : strong neighbor exists, A3 threshold not met
      - 'missing_neighbor'      : strong neighbor not in config list
      - 'interference'          : poor SINR despite acceptable RSRP
      - 'coverage_hole'         : very weak RSRP
      - 'overloaded_cell'       : (detected via KPI, not RF — flag for caller)
      - 'test_server'           : RF fine, issue likely backhaul/server
      - 'unknown'               : no clear pattern

    Args:
        serving_rsrp:        SS-RSRP of serving cell (dBm)
        serving_sinr:        SS-SINR of serving cell (dB)
        neighbor_rsrp_list:  list of (pci, rsrp_dBm) tuples for visible neighbors
        serving_cell_info:   dict from get_cell_info for serving cell
        neighbor_pci_list:   list of neighbor PCIs visible at degradation time

    Returns:
        dict with keys:
          - issue_type (str)
          - strongest_neighbor_pci (int | None)
          - strongest_neighbor_rsrp (float | None)
          - a3_check (dict | None)   — result of check_a3_handover_condition if applicable
          - neighbor_in_config (bool | None)
          - reasoning (str)          — human-readable explanation
    """
    a3_offset = serving_cell_info.get("IntraFreqHoA3Offset [0.5dB]", 10)
    a3_hyst = serving_cell_info.get("IntraFreqHoA3Hyst [0.5dB]", 2)

    # Find strongest neighbor
    strongest_pci = None
    strongest_rsrp = None
    if neighbor_rsrp_list:
        strongest_pci, strongest_rsrp = max(neighbor_rsrp_list, key=lambda x: x[1])

    result = {
        "strongest_neighbor_pci": strongest_pci,
        "strongest_neighbor_rsrp": strongest_rsrp,
        "a3_check": None,
        "neighbor_in_config": None,
        "issue_type": "unknown",
        "reasoning": "",
    }

    # 1. Coverage hole — RSRP very weak
    if serving_rsrp < -110:
        result["issue_type"] = "coverage_hole"
        result["reasoning"] = (
            f"Serving RSRP={serving_rsrp} dBm is below -110 dBm. "
            "Coverage hole — consider increasing power or lifting tilt."
        )
        return result

    # 2. Interference — poor SINR despite acceptable RSRP
    if serving_sinr < 0 and serving_rsrp > -110:
        result["issue_type"] = "interference"
        result["reasoning"] = (
            f"Serving SINR={serving_sinr} dB (< 0) with RSRP={serving_rsrp} dBm. "
            "Strong interference — consider tilting down or reducing power."
        )
        # Still check if a neighbor is causing the interference
        if strongest_pci is not None and strongest_rsrp is not None:
            a3 = check_a3_handover_condition(strongest_rsrp, serving_rsrp, a3_offset, a3_hyst)
            result["a3_check"] = a3
            in_config = check_neighbor_in_config(serving_cell_info, strongest_pci)
            result["neighbor_in_config"] = in_config
        return result

    # 3. Handover-related issue — check A3 condition
    if strongest_pci is not None and strongest_rsrp is not None:
        a3 = check_a3_handover_condition(strongest_rsrp, serving_rsrp, a3_offset, a3_hyst)
        result["a3_check"] = a3
        in_config = check_neighbor_in_config(serving_cell_info, strongest_pci)
        result["neighbor_in_config"] = in_config

        if not in_config:
            result["issue_type"] = "missing_neighbor"
            result["reasoning"] = (
                f"Strongest neighbor PCI={strongest_pci} (RSRP={strongest_rsrp} dBm) "
                "is NOT in the serving cell's neighbor list. Add neighbor relationship."
            )
            return result

        if not a3["condition_met"]:
            result["issue_type"] = "late_handover"
            result["reasoning"] = (
                f"Neighbor PCI={strongest_pci} RSRP={strongest_rsrp} dBm vs serving "
                f"RSRP={serving_rsrp} dBm. Delta={a3['delta_db']} dB, threshold="
                f"{a3['threshold_db']} dB (margin={a3['margin_db']} dB). "
                "A3 condition NOT met — UE stuck on weaker cell. Decrease A3 Offset."
            )
            return result

    # 4. RF looks fine — possibly test server or other issue
    if serving_rsrp > -105 and serving_sinr > 5:
        result["issue_type"] = "test_server"
        result["reasoning"] = (
            f"RF metrics look healthy (RSRP={serving_rsrp} dBm, SINR={serving_sinr} dB) "
            "but throughput is low. Check test server and transmission path."
        )
        return result

    result["reasoning"] = (
        f"RSRP={serving_rsrp} dBm, SINR={serving_sinr} dB. "
        "No clear pattern — check KPI data for cell overload or consult MR data."
    )
    return result
