#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
api_data_gatherer.py — Phase 2 data bridge.

Calls the Phase 2 cloud server API tools to fetch the same CSV data that was
embedded in Phase 1 JSON.  Returns a reconstructed raw-dict that is 100%
compatible with parse_scenario() from scenario_parser.py.

API calls made per F1/F2 scenario (in order):
  1. get_throughput_logs()          → find worst_time and last_time
  2. get_config_data()              → network_configuration_data CSV (no time param)
  3. get_user_plane_data()          → user_plane_data CSV (no time param)
  4. get_signaling_plane_event_log(last_time) → signaling events, reconstructed as CSV
  5. get_kpi_data()                 → traffic_data CSV (no time param)
  6. get_mr_data()                  → mr_data CSV (no time param)

Total: 6 calls per scenario (fast, deterministic, no LLM needed).

Optional disambiguation calls (added to ScenarioFeatures after run_all_skills):
  judge_mainlobe_or_not(worst_time, serving_pci) → lift_tilt vs insufficient_data
"""

import io
import logging
from typing import Any, Dict, List, Optional

import pandas as pd


class APIDataGatherer:
    """
    Fetches raw CSV data for a single scenario via Phase 2 API tools,
    and reconstructs a raw-JSON dict compatible with parse_scenario().
    """

    def __init__(
        self,
        env,                        # main.Environment instance
        scenario: Dict[str, Any],  # the scenario dict from test.json
        logger: Optional[logging.Logger] = None,
    ):
        self.env = env
        self.scenario = scenario
        self.scenario_id: str = scenario.get("scenario_id", "")
        self.logger = logger or logging.getLogger(__name__)

        # Populated by gather()
        self.worst_time: Optional[str] = None
        self.last_time: Optional[str] = None
        self.serving_pci: Optional[int] = None   # at worst_time (for mainlobe check)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def gather(self) -> Dict[str, Any]:
        """
        Run all API calls and return a raw-dict compatible with parse_scenario().

        On any call failure the corresponding CSV is left as None; parse_scenario()
        handles missing DataFrames gracefully (returns empty DF, scores 0.0).
        """
        sid = self.scenario_id

        # Step 1 — find worst_time and last_time from throughput logs
        user_plane_csv = self._get_user_plane_csv()
        self._extract_timing_from_user_plane(user_plane_csv)

        # Step 2 — network config (no time param)
        config_csv = self._call_csv("get_config_data", "Network Configuration Data", sid)

        # Step 3 — signaling events (needs last_time)
        signaling_csv = self._get_signaling_csv(sid)

        # Step 4 — KPI / traffic data (no time param)
        kpi_csv = self._call_csv("get_kpi_data", "Traffic Data", sid)

        # Step 5 — MR data (no time param)
        mr_csv = self._call_csv("get_mr_data", "MR Data", sid)

        # Reconstruct raw-JSON so parse_scenario() works unchanged
        raw = dict(self.scenario)
        raw["data"] = {
            "user_plane_data": user_plane_csv,
            "network_configuration_data": config_csv,
            "signaling_plane_data": signaling_csv,
            "traffic_data": kpi_csv,
            "mr_data": mr_csv,
            "notes": self.scenario.get("data", {}).get("notes", ""),
            "collection_method": self.scenario.get("data", {}).get("collection_method", ""),
        }
        return raw

    def disambiguate_mainlobe(self, sf, serving_pci: Optional[int] = None) -> None:
        """
        Call judge_mainlobe_or_not() for scenarios where lift_tilt and
        insufficient_data are within 0.10 of each other.  Updates sf in-place.

        Interpretation of Outside_Mainlobe_Flag:
          True  → user is OUTSIDE the cell's beam → coverage hole → lift_tilt
          False → user is INSIDE the cell's beam  → signal is OK  → insufficient_data
        """
        pci = serving_pci or self.serving_pci
        t = self.worst_time
        if pci is None or t is None:
            return

        result = self.env._call_api(
            "judge_mainlobe_or_not",
            scenario_id=self.scenario_id,
            time=t,
            pci=pci,
        )
        if "error" in result:
            return

        outside = result.get("Outside_Mainlobe_Flag")
        if outside is True:
            # Outside main beam → coverage hole → boost lift_tilt
            sf.score_coverage_hole = max(sf.score_coverage_hole or 0.0, 0.72)
            insuff = sf.score_insufficient_data or 0.0
            sf.score_insufficient_data = min(insuff, 0.25)
        elif outside is False:
            # Inside main beam → data is poor despite good geometry → insufficient_data
            insuff = sf.score_insufficient_data or 0.0
            sf.score_insufficient_data = max(insuff, 0.78)
            sf.score_coverage_hole = min(sf.score_coverage_hole or 1.0, 0.25)

    def get_overlap_ratio(self, pci_serving: int, pci_neighbor: int) -> Optional[float]:
        """Call calculate_overlap_ratio for F2 overshoot discrimination."""
        result = self.env._call_api(
            "calculate_overlap_ratio",
            scenario_id=self.scenario_id,
            pci_serving=pci_serving,
            pci_neighbor=pci_neighbor,
        )
        if "error" in result:
            return None
        # Key is dynamic: "The overlap coverage ratio of the cell X and the cell Y is "
        for val in result.values():
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_user_plane_csv(self) -> Optional[str]:
        result = self.env._call_api("get_user_plane_data", scenario_id=self.scenario_id)
        if "error" in result:
            return None
        csv_str = result.get("User Plane Data")
        return csv_str if isinstance(csv_str, str) and csv_str.strip() else None

    def _extract_timing_from_user_plane(self, user_plane_csv: Optional[str]) -> None:
        """Parse user_plane CSV to find worst_time, last_time, and serving_pci."""
        if not user_plane_csv:
            return
        try:
            df = pd.read_csv(io.StringIO(user_plane_csv), sep="|")
        except Exception:
            return

        # Find timestamp column
        ts_col = next((c for c in df.columns if "Timestamp" in c or "timestamp" in c), None)
        if ts_col is None:
            return

        timestamps = df[ts_col].dropna().astype(str).str.strip()
        if timestamps.empty:
            return

        self.last_time = timestamps.iloc[-1]

        # Find worst throughput timestamp
        tp_col = next(
            (c for c in df.columns if "Throughput" in c or "throughput" in c),
            None,
        )
        if tp_col is not None:
            tp_series = pd.to_numeric(df[tp_col], errors="coerce")
            valid = tp_series.dropna()
            if not valid.empty:
                worst_idx = valid.idxmin()
                self.worst_time = str(df.loc[worst_idx, ts_col]).strip()
                # Grab serving PCI at worst point
                pci_col = next(
                    (c for c in df.columns if "Serving PCI" in c or "Serving_PCI" in c),
                    None,
                )
                if pci_col is not None:
                    pci_val = df.loc[worst_idx, pci_col]
                    try:
                        self.serving_pci = int(float(pci_val))
                    except (ValueError, TypeError):
                        pass

        if self.worst_time is None:
            self.worst_time = self.last_time

    def _call_csv(self, tool: str, key: str, scenario_id: str) -> Optional[str]:
        """Call a no-param tool that returns {key: csv_string}."""
        result = self.env._call_api(tool, scenario_id=scenario_id)
        if "error" in result:
            return None
        csv_str = result.get(key)
        return csv_str if isinstance(csv_str, str) and csv_str.strip() else None

    def _get_signaling_csv(self, scenario_id: str) -> Optional[str]:
        """
        Fetch signaling events as JSON list and reconstruct pipe-delimited CSV.
        Uses last_time so all events are included.
        """
        t = self.last_time
        if t is None:
            return None

        result = self.env._call_api(
            "get_signaling_plane_event_log",
            scenario_id=scenario_id,
            time=t,
        )
        if "error" in result or not isinstance(result, list):
            return None
        if not result:
            return None

        try:
            df = pd.DataFrame(result)
            return df.to_csv(index=False, sep="|")
        except Exception:
            return None
