#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Server Tool Caller — V3 Architecture

Provides targeted HTTP calls to the competition FastAPI server (server.py)
for scenarios where local feature extraction alone is insufficient.

Primarily used for F1 (4BS single-answer) scenarios where the top-2 candidates
are ambiguous (e.g., lift_tilt vs increase_power vs insufficient_data).

Tool selection strategy:
  - judge_mainlobe:          distinguishes tilt/azimuth issues from power issues
  - calculate_tilt_angle:    quantifies optimal tilt adjustment
  - calculate_overlap_ratio: confirms overshoot/overlap between cells
  - get_kpi_data:            KPI details (CCE, PRB, throughput per cell)
  - signaling-plane-event-log: handover events for mobility diagnosis

────────────────────────────────────────────────────────────────────────────
TODO [Teammate 2]:
  1. Implement _select_tools_for_group() to choose the right tools for each
     SINR group (handover / high-PRB / low-PRB).
  2. Implement parse_tool_results() to convert raw JSON responses into
     structured evidence that augments ScenarioFeatures.
  3. Test with the server running: python server.py --port 7860
  4. Measure the accuracy uplift, especially for lift_tilt vs increase_power
     and insufficient_data vs check_server.
────────────────────────────────────────────────────────────────────────────
"""

import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from skills.feature_engine import ScenarioFeatures


# ──────────────────────────────────────────────────────────────────────────────
# Tool evidence container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolEvidence:
    """Structured results from server tool calls."""
    mainlobe_flags: Dict[int, bool] = field(default_factory=dict)  # pci → bool
    tilt_angles: Dict[int, float] = field(default_factory=dict)     # pci → degrees
    mr_sample_count: Optional[int] = None
    mr_sample_counts: Dict[str, int] = field(default_factory=dict)  # aggregate MR counts
    overlap_ratio: Optional[float] = None
    kpi_data_raw: Optional[str] = None
    signaling_events: List[str] = field(default_factory=list)
    tool_call_count: int = 0

    def as_prompt_text(self) -> str:
        """Format tool evidence for injection into the Qwen prompt."""
        lines = []
        if self.mainlobe_flags:
            for pci, outside in self.mainlobe_flags.items():
                status = "OUTSIDE mainlobe" if outside else "inside mainlobe"
                lines.append(f"PCI {pci}: {status}")
        if self.tilt_angles:
            for pci, angle in self.tilt_angles.items():
                lines.append(f"PCI {pci}: optimal tilt angle = {angle:.1f}°")
        if self.mr_sample_count is not None:
            lines.append(f"MR sample count: {self.mr_sample_count}")
        if self.overlap_ratio is not None:
            lines.append(f"Cell overlap ratio: {self.overlap_ratio:.2f}")
        if self.signaling_events:
            lines.append("Key signaling events: " + "; ".join(self.signaling_events[:5]))
        if not lines:
            return ""
        return "## Server Tool Evidence\n" + "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Server caller
# ──────────────────────────────────────────────────────────────────────────────

class ToolCaller:
    """
    Wraps HTTP calls to the competition FastAPI server.

    Usage:
        caller = ToolCaller("http://localhost:7860")
        evidence = caller.call_targeted(scenario_id, sf, worst_time)
    """

    def __init__(
        self,
        server_url: str,
        timeout: float = 15.0,
        mr_sample_threshold: int = 7,
        logger: Optional[logging.Logger] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.mr_sample_threshold = mr_sample_threshold
        self.logger = logger or logging.getLogger(__name__)

    def _headers(self, scenario_id: str) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-Scenario-Id": scenario_id,
        }

    def _get(
        self,
        endpoint: str,
        scenario_id: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Safe HTTP GET with error handling."""
        url = f"{self.server_url}{endpoint}"
        try:
            resp = requests.get(
                url,
                params=params or {},
                headers=self._headers(scenario_id),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            self.logger.warning("Tool call %s failed: %s", endpoint, exc)
            return None

    # ── Individual tool wrappers ──────────────────────────────────────────────

    def judge_mainlobe(
        self, scenario_id: str, pci: int, time: str
    ) -> Optional[bool]:
        """Returns True if user is outside mainlobe of the given cell."""
        result = self._get(
            "/judge_mainlobe",
            scenario_id,
            {"pci": pci, "time": time},
        )
        if result and "Outside_Mainlobe_Flag" in result:
            return bool(result["Outside_Mainlobe_Flag"])
        return None

    def calculate_tilt_angle(
        self, scenario_id: str, pci: int, time: str
    ) -> Optional[float]:
        """Returns the calculated tilt angle for a cell at a given time."""
        result = self._get(
            "/calculate_tilt_angle",
            scenario_id,
            {"pci": pci, "time": time},
        )
        if result and "Tilt_Angle" in result:
            return float(result["Tilt_Angle"])
        return None

    def calculate_overlap_ratio(
        self, scenario_id: str, pci_serving: int, pci_neighbor: int
    ) -> Optional[float]:
        """Returns overlap coverage ratio between two cells."""
        result = self._get(
            "/calculate_overlap_ratio",
            scenario_id,
            {"pci_serving": pci_serving, "pci_neighbor": pci_neighbor},
        )
        if result and "Overlap_Ratio" in result:
            return float(result["Overlap_Ratio"])
        return None

    def get_kpi_data(self, scenario_id: str) -> Optional[str]:
        """Returns raw KPI/traffic data string."""
        result = self._get("/get_kpi_data", scenario_id)
        if result and "Traffic Data" in result:
            return result["Traffic Data"]
        return None

    def get_mr_data(self, scenario_id: str) -> Optional[str]:
        """Returns raw MR data string."""
        result = self._get("/get_mr_data", scenario_id)
        if result and "MR Data" in result:
            return result["MR Data"]
        return None

    def get_signaling_events(
        self, scenario_id: str, time: str = ""
    ) -> Optional[str]:
        """Returns signaling plane event log."""
        result = self._get(
            "/signaling-plane-event-log",
            scenario_id,
            {"time": time} if time else {},
        )
        if result:
            return json.dumps(result, ensure_ascii=False)[:1000]
        return None

    @staticmethod
    def _summarize_mr_data(mr_raw: str) -> Tuple[Optional[int], Dict[str, int]]:
        """
        Convert raw MR CSV into compact aggregate counts.

        `mr_sample_count` is the scenario-level row count. The companion dict
        keeps lightweight aggregates for future prompting / debugging.
        """
        if not mr_raw or not mr_raw.strip():
            return None, {}

        try:
            df = pd.read_csv(io.StringIO(mr_raw), sep="|")
        except Exception:
            return None, {}

        row_count = int(len(df))
        summary = {"rows": row_count}

        pci_cols = [c for c in df.columns if "PCI" in c]
        if pci_cols:
            visible = set()
            for col in pci_cols:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                visible.update(int(v) for v in series.tolist() if int(v) > 0)
            summary["unique_pcis"] = len(visible)

        return row_count, summary

    # ── Targeted calling strategy ─────────────────────────────────────────────
    # TODO [Teammate 2]: expand this method with more sophisticated strategies

    def call_targeted(
        self,
        scenario_id: str,
        sf: ScenarioFeatures,
        worst_time: Optional[str] = None,
    ) -> ToolEvidence:
        """
        Select and execute the most informative tool calls based on the
        current feature assessment.

        This is the main entry point. The strategy:
        1. Determine which "group" the scenario falls into
        2. Call 1-3 tools that can disambiguate the top-2 candidates
        3. Return structured evidence

        Returns:
            ToolEvidence with results from 0-3 tool calls.
        """
        evidence = ToolEvidence()
        top_issue, top_score = sf.top_issue()
        serving_pci = sf.mobility.serving_pci_at_worst
        wt = worst_time or sf.throughput.worst_time or ""

        if not serving_pci or not wt:
            return evidence

        # ── Group 1: ambiguous coverage issue ──
        # lift_tilt vs increase_power vs insufficient_data
        # → judge_mainlobe can tell if antenna coverage shape is the problem
        if top_issue in ("coverage_hole",) and top_score >= 0.3:
            flag = self.judge_mainlobe(scenario_id, serving_pci, wt)
            if flag is not None:
                evidence.mainlobe_flags[serving_pci] = flag
                evidence.tool_call_count += 1

            # Also check the best neighbor if available
            nb_pci = sf.mobility.best_neighbor_pci
            if nb_pci:
                flag_nb = self.judge_mainlobe(scenario_id, nb_pci, wt)
                if flag_nb is not None:
                    evidence.mainlobe_flags[nb_pci] = flag_nb
                    evidence.tool_call_count += 1

            # Coverage-hole vs insufficient-data is the core F1 ambiguity.
            mr_raw = self.get_mr_data(scenario_id)
            if mr_raw:
                count, counts = self._summarize_mr_data(mr_raw)
                evidence.mr_sample_count = count
                evidence.mr_sample_counts.update(counts)
                evidence.tool_call_count += 1

        # ── Group 2: ambiguous pdcch/server issue ──
        # pdcch_2sym vs check_server
        # → KPI data might reveal more about CCE utilization details
        elif top_issue in ("pdcch_issue", "backhaul_server") and top_score >= 0.3:
            kpi = self.get_kpi_data(scenario_id)
            if kpi:
                evidence.kpi_data_raw = kpi[:600]
                evidence.tool_call_count += 1

            if top_issue == "backhaul_server":
                mr_raw = self.get_mr_data(scenario_id)
                if mr_raw:
                    count, counts = self._summarize_mr_data(mr_raw)
                    evidence.mr_sample_count = count
                    evidence.mr_sample_counts.update(counts)
                    evidence.tool_call_count += 1

        # ── Group 3: handover issues ──
        # add_neighbor vs decrease_a3
        # → signaling events can confirm handover attempts/failures
        elif top_issue in ("late_handover", "missing_neighbor") and top_score >= 0.3:
            sig = self.get_signaling_events(scenario_id, wt)
            if sig:
                evidence.signaling_events = [sig[:400]]
                evidence.tool_call_count += 1

            # Check overlap if strong neighbor exists
            nb_pci = sf.mobility.best_neighbor_pci
            if nb_pci:
                ratio = self.calculate_overlap_ratio(
                    scenario_id, serving_pci, nb_pci
                )
                if ratio is not None:
                    evidence.overlap_ratio = ratio
                    evidence.tool_call_count += 1

        return evidence

    def augment_features(
        self,
        sf: ScenarioFeatures,
        evidence: ToolEvidence,
    ) -> ScenarioFeatures:
        """
        Update ScenarioFeatures scores based on tool evidence.

        TODO [Teammate 2]: implement score adjustments based on:
          - mainlobe flag → if outside mainlobe, boost lift_tilt / adjust_azimuth
          - overlap ratio → if high overlap, boost press_down_tilt
          - KPI details → if CCE pressure in detail, boost pdcch_2sym
        """
        if not evidence.tool_call_count:
            return sf

        original_top_issue, _ = sf.top_issue()
        serving_pci = sf.mobility.serving_pci_at_worst
        current_insufficient = getattr(sf, "score_insufficient_data", 0.0) or 0.0
        serving_flag = (
            evidence.mainlobe_flags.get(serving_pci)
            if serving_pci is not None
            else None
        )
        any_false_mainlobe = any(flag is False for flag in evidence.mainlobe_flags.values())

        # Phase 2 note: judge_mainlobe_or_not returns Outside_Mainlobe_Flag=False
        # for ALL Phase 2 coverage_hole scenarios (constant output), so False can
        # no longer be trusted as a lift_tilt signal. Only True (UE outside mainlobe)
        # is a reliable non-constant signal. Phase 1 coverage_hole rarely triggered
        # (RSRP too strong), so this change has no Phase 1 regression risk.
        if serving_flag is True:
            # Outside mainlobe → antenna geometry is fine → insufficient_data likely
            sf.score_coverage_hole = min(sf.score_coverage_hole, 0.25)
            sf.score_backhaul_server = min(max(sf.score_backhaul_server, 0.25), 0.35)
            sf.score_insufficient_data = max(current_insufficient, 0.78)
            sf.geometry.recommended_action_family = "none"
            sf.geometry.recommended_target_role = None
            sf.power.recommended_action_family = "none"
            sf.power.recommended_target_role = None
        # serving_flag is False (inside mainlobe): no-op — let raw feature scores
        # decide.  The Phase 1 empirical False→lift_tilt correlation cannot be used
        # when the API returns the same constant for every scenario.

        if evidence.mr_sample_count is not None:
            if (
                evidence.mr_sample_count < self.mr_sample_threshold
                and serving_flag is not True
                and original_top_issue != "backhaul_server"
            ):
                # Sparse MR + not explicitly outside mainlobe → insufficient_data.
                # Changed guard from "not False" to "not True" so that the constant-
                # False API response no longer blocks this branch from firing.
                sf.score_insufficient_data = max(
                    getattr(sf, "score_insufficient_data", 0.0) or 0.0,
                    0.82,
                )
                sf.score_backhaul_server = min(max(sf.score_backhaul_server, 0.25), 0.35)
                sf.score_coverage_hole = min(sf.score_coverage_hole, 0.25)

        if evidence.overlap_ratio is not None and evidence.overlap_ratio > 0.3:
            sf.score_overshoot_overlap = max(
                sf.score_overshoot_overlap, 0.55
            )

        return sf
