#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Local Scenario Parser — V2 Architecture

Converts raw scenario JSON (with embedded pipe-delimited CSV strings) into
normalised Python objects that every downstream module can consume without
any HTTP round-trips.

Key responsibilities:
  - parse all five CSV data fields into DataFrames
  - parse the 22 option labels into structured dicts (action_type + target cells)
  - infer which cells are "primary" / "secondary" in the scenario
  - build a PCI → cell_id mapping from network_configuration_data
  - provide fast accessor helpers for common lookups
"""

import io
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Option action-type parsing
# ──────────────────────────────────────────────────────────────────────────────

# Regex to extract cell IDs in "gNodeBID_CellID" format (e.g. "3279943_1")
_CELL_ID_RE = re.compile(r'\b(\d{6,8}_\d+)\b')

# Action type detection patterns (checked in order — first match wins)
_ACTION_PATTERNS: List[Tuple[str, str]] = [
    ("check_server",       r"[Cc]heck test server"),
    ("insufficient_data",  r"[Ii]nsufficient data"),
    ("add_neighbor",       r"[Aa]dd neighbor"),
    ("decrease_interfreq", r"[Dd]ecrease CovInterFreq"),
    ("press_down_tilt",    r"[Pp]ress down the? tilt"),
    ("lift_tilt",          r"[Ll]ift the? tilt"),
    ("pdcch_2sym",         r"[Mm]odify Pdcch"),
    ("adjust_azimuth",     r"[Aa]djust the? azimuth"),
    ("increase_a3",        r"[Ii]ncrease A3 [Oo]ffset"),
    ("decrease_a3",        r"[Dd]ecrease A3 [Oo]ffset"),
    ("decrease_power",     r"[Dd]ecrease transmission power"),
    ("increase_power",     r"[Ii]ncrease transmission power"),
]
_ACTION_RES = [(name, re.compile(pat)) for name, pat in _ACTION_PATTERNS]

_DEGREE_RE = re.compile(r'(\d+)\s*degrees?')


def parse_option_label(label: str) -> Dict[str, Any]:
    """
    Parse one option label string into a structured dict.

    Returns:
        {
          "action_type": str,          # one of 12 action types above
          "target_cell": str | None,   # primary cell being acted on (last cell mentioned)
          "cells": List[str],          # all cell IDs found in the label
          "degrees": int | None,       # tilt/azimuth adjustment magnitude
          "raw": str,                  # original label
        }
    """
    cells = _CELL_ID_RE.findall(label)
    degrees = None
    m = _DEGREE_RE.search(label)
    if m:
        degrees = int(m.group(1))

    for action_type, pattern in _ACTION_RES:
        if pattern.search(label):
            # For add_neighbor, the "target" cell (the one getting the new neighbour)
            # is the LAST cell mentioned in the label.
            target_cell = cells[-1] if cells else None
            return {
                "action_type": action_type,
                "target_cell": target_cell,
                "cells": cells,
                "degrees": degrees,
                "raw": label,
            }

    return {
        "action_type": "unknown",
        "target_cell": cells[-1] if cells else None,
        "cells": cells,
        "degrees": degrees,
        "raw": label,
    }


def parse_all_options(options: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    """
    Parse all 22 option dicts into a mapping  { "C1": parsed_dict, ... }.
    """
    result = {}
    for opt in options:
        cid = opt["id"]
        parsed = parse_option_label(opt["label"])
        parsed["id"] = cid
        result[cid] = parsed
    return result


def infer_cell_roles(parsed_options: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    Determine cell roles by counting how many options target each cell ID.

    Returns:
        { cell_id: role }  where role ∈ {"primary", "secondary", "tertiary"}

    The cell with the most targeted options is "primary" (usually the serving cell
    or the most problematic one).  The second-most is "secondary".
    """
    freq: Dict[str, int] = {}
    for opt in parsed_options.values():
        for cid in opt.get("cells", []):
            freq[cid] = freq.get(cid, 0) + 1

    if not freq:
        return {}

    sorted_cells = sorted(freq.items(), key=lambda x: -x[1])
    roles: Dict[str, str] = {}
    role_names = ["primary", "secondary", "tertiary"]
    for i, (cell_id, _) in enumerate(sorted_cells):
        roles[cell_id] = role_names[i] if i < len(role_names) else "other"
    return roles


# ──────────────────────────────────────────────────────────────────────────────
# CSV column name aliases (handle slight naming variations)
# ──────────────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    # Partial match fallback
    for c in candidates:
        matches = [col for col in df.columns if c.lower() in col.lower()]
        if matches:
            return matches[0]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Parsed scenario container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedScenario:
    """All normalised data for one scenario, ready for the feature engine."""

    # ── identifiers ──
    scenario_id: str
    tag: str                          # "single-answer" | "multiple-answer"
    num_base_stations: int            # from context field
    ground_truth: Optional[str]       # available in training set, None for test

    # ── parsed DataFrames ──
    user_plane: pd.DataFrame = field(default_factory=pd.DataFrame)
    net_config: pd.DataFrame = field(default_factory=pd.DataFrame)
    signaling: pd.DataFrame = field(default_factory=pd.DataFrame)
    traffic: pd.DataFrame = field(default_factory=pd.DataFrame)
    mr: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ── option structure ──
    parsed_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cell_roles: Dict[str, str] = field(default_factory=dict)   # cell_id → role
    role_to_cells: Dict[str, List[str]] = field(default_factory=dict)  # role → [cell_ids]

    # ── PCI → cell_id mapping (built from net_config) ──
    pci_to_cell_id: Dict[int, str] = field(default_factory=dict)
    cell_id_to_pci: Dict[str, int] = field(default_factory=dict)

    # ── column name cache ──
    _col_cache: Dict[str, Optional[str]] = field(default_factory=dict, repr=False)

    # ── helpers ──

    def col(self, df_name: str, candidates: List[str]) -> Optional[str]:
        key = f"{df_name}:{'|'.join(candidates)}"
        if key not in self._col_cache:
            df = getattr(self, df_name, pd.DataFrame())
            self._col_cache[key] = _find_col(df, candidates)
        return self._col_cache[key]

    def up_col(self, *candidates) -> Optional[str]:
        return self.col("user_plane", list(candidates))

    def nc_col(self, *candidates) -> Optional[str]:
        return self.col("net_config", list(candidates))

    # ── user-plane accessors ──

    def get_timestamps(self) -> List[str]:
        col = self.up_col("Timestamp")
        if col is None or self.user_plane.empty:
            return []
        return self.user_plane[col].tolist()

    def get_throughput_series(self) -> pd.Series:
        col = self.up_col(
            "5G KPI PCell Layer2 MAC DL Throughput [Mbps]",
            "DL Throughput",
            "Throughput [Mbps]",
            "Throughput",
        )
        if col is None or self.user_plane.empty:
            return pd.Series(dtype=float)
        return pd.to_numeric(self.user_plane[col], errors="coerce")

    def get_rsrp_series(self) -> pd.Series:
        col = self.up_col(
            "5G KPI PCell RF Serving SS-RSRP [dBm]",
            "Serving SS-RSRP",
            "SS-RSRP",
            "Serving RSRP",
            "PCell RSRP",
            "Filtered Tx BRSRP",
            "BRSRP",
            "RSRP",
        )
        if col is None or self.user_plane.empty:
            return pd.Series(dtype=float)
        return pd.to_numeric(self.user_plane[col], errors="coerce")

    def get_sinr_series(self) -> pd.Series:
        col = self.up_col(
            "5G KPI PCell RF Serving SS-SINR [dB]",
            "Serving SS-SINR",
            "SS-SINR",
            "Serving SINR",
            "PCell SINR",
            "SINR",
        )
        if col is None or self.user_plane.empty:
            return pd.Series(dtype=float)
        return pd.to_numeric(self.user_plane[col], errors="coerce")

    def get_serving_pci_series(self) -> pd.Series:
        col = self.up_col(
            "5G KPI PCell RF Serving PCI",
            "Serving Cell PCI",
            "Serving PCI",
            "PCell PCI",
        )
        if col is None or self.user_plane.empty:
            return pd.Series(dtype=float)
        return pd.to_numeric(self.user_plane[col], errors="coerce")

    def get_row_at_time(self, timestamp: str) -> Optional[pd.Series]:
        col = self.up_col("Timestamp")
        if col is None or self.user_plane.empty:
            return None
        mask = self.user_plane[col].str.strip() == timestamp.strip()
        rows = self.user_plane[mask]
        return rows.iloc[0] if not rows.empty else None

    # ── net-config accessors ──

    def get_cell_config(self, pci: int) -> Optional[Dict[str, Any]]:
        if self.net_config.empty:
            return None
        col = self.nc_col("PCI")
        if col is None:
            return None
        self.net_config[col] = pd.to_numeric(self.net_config[col], errors="coerce")
        rows = self.net_config[self.net_config[col] == pci]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    def get_neighbor_pcis_from_config(self, pci: int) -> List[int]:
        """Extract neighbour PCIs from the 'PCell Neighbor Cell' string."""
        cfg = self.get_cell_config(pci)
        if cfg is None:
            return []
        neighbor_str = str(cfg.get("PCell Neighbor Cell (gNodeBID_ARFCN_PCI)", "") or "")
        found = re.findall(r'_(\d+)(?:\]|,)', neighbor_str)
        return [int(p) for p in found]

    def is_neighbor_configured(self, serving_pci: int, neighbor_pci: int) -> bool:
        return neighbor_pci in self.get_neighbor_pcis_from_config(serving_pci)

    # ── traffic accessors ──

    def get_cell_traffic(self, gnodeb_id: int, cell_id_num: int) -> Optional[Dict[str, Any]]:
        if self.traffic.empty:
            return None
        gb_col = _find_col(self.traffic, ["gNodeB_ID", "gNodeB ID"])
        ci_col = _find_col(self.traffic, ["Cell_ID", "Cell ID"])
        if gb_col is None or ci_col is None:
            return None
        mask = (
            (pd.to_numeric(self.traffic[gb_col], errors="coerce") == gnodeb_id) &
            (pd.to_numeric(self.traffic[ci_col], errors="coerce") == cell_id_num)
        )
        rows = self.traffic[mask]
        if rows.empty:
            return None
        # Return average across time slots
        numeric = rows.select_dtypes(include="number")
        return numeric.mean().to_dict() if not numeric.empty else None

    # ── MR accessors ──

    def get_mr_neighbor_rsrp(self, neighbor_pci: int) -> Optional[float]:
        """Average RSRP of a specific neighbor from measurement report data."""
        if self.mr.empty:
            return None
        rsrps = []
        for n in range(1, 7):
            pci_col = _find_col(self.mr, [f"Neighbor {n} PCI"])
            rsrp_col = _find_col(self.mr, [f"Neighbor {n} RSRP(dBm)"])
            if pci_col and rsrp_col:
                mask = pd.to_numeric(self.mr[pci_col], errors="coerce") == neighbor_pci
                vals = pd.to_numeric(self.mr.loc[mask, rsrp_col], errors="coerce").dropna()
                rsrps.extend(vals.tolist())
        return float(pd.Series(rsrps).mean()) if rsrps else None

    # ── option helpers ──

    def find_options_by(
        self,
        action_type: str,
        target_role: Optional[str] = None,
        target_cell_id: Optional[str] = None,
    ) -> List[str]:
        """
        Find C-codes that match a given action_type and optionally a cell role or cell_id.

        Args:
            action_type:    one of the 12 action type strings
            target_role:    "primary" | "secondary" | "tertiary" | None (any)
            target_cell_id: exact cell_id string, or None (any)

        Returns:
            List of matching C-code strings (e.g. ["C14", "C22"])
        """
        results = []
        for cid, opt in self.parsed_options.items():
            if opt["action_type"] != action_type:
                continue
            tc = opt.get("target_cell")
            if target_cell_id is not None and tc != target_cell_id:
                continue
            if target_role is not None:
                if tc is None or self.cell_roles.get(tc) != target_role:
                    continue
            results.append(cid)
        return results

    def get_primary_cell_id(self) -> Optional[str]:
        return next(
            (cid for cid, role in self.cell_roles.items() if role == "primary"), None
        )

    def get_secondary_cell_id(self) -> Optional[str]:
        return next(
            (cid for cid, role in self.cell_roles.items() if role == "secondary"), None
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main parser entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_scenario(raw: Dict[str, Any]) -> ParsedScenario:
    """
    Convert a raw scenario dict (from train.json or server /scenario/all) into
    a ParsedScenario.  Works for both training examples (have 'answer') and
    test examples (no 'answer').
    """

    def _read_csv(raw_str: Optional[str]) -> pd.DataFrame:
        if not raw_str:
            return pd.DataFrame()
        try:
            return pd.read_csv(io.StringIO(raw_str), sep="|")
        except Exception:
            return pd.DataFrame()

    data = raw.get("data", {})
    task = raw.get("task", {})
    ctx = raw.get("context", {})
    wi = ctx.get("wireless_network_information", {})

    num_bs = int(str(wi.get("num_base_stations", "4")).strip())

    ps = ParsedScenario(
        scenario_id=raw.get("scenario_id", ""),
        tag=raw.get("tag", "single-answer"),
        num_base_stations=num_bs,
        ground_truth=raw.get("answer", None),
        user_plane=_read_csv(data.get("user_plane_data")),
        net_config=_read_csv(data.get("network_configuration_data")),
        signaling=_read_csv(data.get("signaling_plane_data")),
        traffic=_read_csv(data.get("traffic_data")),
        mr=_read_csv(data.get("mr_data")),
    )

    # ── Parse options ──
    options = task.get("options", [])
    ps.parsed_options = parse_all_options(options)
    ps.cell_roles = infer_cell_roles(ps.parsed_options)

    # Build reverse mapping role → [cell_ids]
    role_to_cells: Dict[str, List[str]] = {}
    for cell_id, role in ps.cell_roles.items():
        role_to_cells.setdefault(role, []).append(cell_id)
    ps.role_to_cells = role_to_cells

    # ── Build PCI → cell_id mapping ──
    if not ps.net_config.empty:
        pci_col = _find_col(ps.net_config, ["PCI"])
        gnb_col = _find_col(ps.net_config, ["gNodeB ID", "gNodeB_ID"])
        ci_col = _find_col(ps.net_config, ["Cell ID", "Cell_ID"])
        if pci_col and gnb_col and ci_col:
            for _, row in ps.net_config.iterrows():
                try:
                    pci = int(float(row[pci_col]))
                    cell_id = f"{int(float(row[gnb_col]))}_{int(float(row[ci_col]))}"
                    ps.pci_to_cell_id[pci] = cell_id
                    ps.cell_id_to_pci[cell_id] = pci
                except (ValueError, TypeError):
                    pass

    return ps
