import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
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

    mainlobe_flags: Dict[int, bool] = field(default_factory=dict)
    tilt_angles: Dict[int, float] = field(default_factory=dict)

    mr_sample_count: Optional[int] = None
    mr_sample_counts: Dict[str, int] = field(default_factory=dict)

    overlap_ratio: Optional[float] = None

    kpi_data_raw: Optional[str] = None

    signaling_events: List[str] = field(default_factory=list)

    tool_call_count: int = 0

    def as_prompt_text(self) -> str:
        lines = []

        if self.mainlobe_flags:
            for pci, outside in self.mainlobe_flags.items():
                status = (
                    "OUTSIDE mainlobe"
                    if outside
                    else "inside mainlobe"
                )
                lines.append(f"PCI {pci}: {status}")

        if self.tilt_angles:
            for pci, angle in self.tilt_angles.items():
                lines.append(
                    f"PCI {pci}: optimal tilt angle = {angle:.1f}°"
                )

        if self.mr_sample_count is not None:
            lines.append(
                f"MR sample count: {self.mr_sample_count}"
            )

        if self.overlap_ratio is not None:
            lines.append(
                f"Cell overlap ratio: {self.overlap_ratio:.2f}"
            )

        if self.signaling_events:
            lines.append(
                "Key signaling events: "
                + "; ".join(self.signaling_events[:5])
            )

        if not lines:
            return ""

        return "## Server Tool Evidence\n" + "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Scenario grouping
# ──────────────────────────────────────────────────────────────────────────────

class ScenarioGroup(str, Enum):
    COVERAGE = "coverage"
    CAPACITY = "capacity"
    HANDOVER = "handover"
    SERVER = "server"
    UNKNOWN = "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Tool execution plan
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolPlan:
    tools: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)

    priority: int = 1

    max_calls: int = 3

    def add(self, tool_name: str, **kwargs):
        self.tools.append((tool_name, kwargs))


# ──────────────────────────────────────────────────────────────────────────────
# Server caller
# ──────────────────────────────────────────────────────────────────────────────

class ToolCaller:
    """
    Wraps HTTP calls to the competition FastAPI server.
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

        # unified tool registry
        self.tool_registry = {
            "judge_mainlobe": self.judge_mainlobe,
            "calculate_tilt_angle": self.calculate_tilt_angle,
            "calculate_overlap_ratio": self.calculate_overlap_ratio,
            "get_kpi_data": self.get_kpi_data,
            "get_mr_data": self.get_mr_data,
            "get_signaling_events": self.get_signaling_events,
        }

    # ──────────────────────────────────────────────────────────────────────
    # HTTP helpers
    # ──────────────────────────────────────────────────────────────────────

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
            self.logger.warning(
                "Tool call %s failed: %s",
                endpoint,
                exc,
            )
            return None

    # ──────────────────────────────────────────────────────────────────────
    # Tool wrappers
    # ──────────────────────────────────────────────────────────────────────

    def judge_mainlobe(
        self,
        scenario_id: str,
        pci: int,
        time: str,
    ) -> Optional[bool]:

        result = self._get(
            "/judge_mainlobe",
            scenario_id,
            {
                "pci": pci,
                "time": time,
            },
        )

        if result and "Outside_Mainlobe_Flag" in result:
            return bool(result["Outside_Mainlobe_Flag"])

        return None

    def calculate_tilt_angle(
        self,
        scenario_id: str,
        pci: int,
        time: str,
    ) -> Optional[float]:

        result = self._get(
            "/calculate_tilt_angle",
            scenario_id,
            {
                "pci": pci,
                "time": time,
            },
        )

        if result and "Tilt_Angle" in result:
            return float(result["Tilt_Angle"])

        return None

    def calculate_overlap_ratio(
        self,
        scenario_id: str,
        pci_serving: int,
        pci_neighbor: int,
    ) -> Optional[float]:

        result = self._get(
            "/calculate_overlap_ratio",
            scenario_id,
            {
                "pci_serving": pci_serving,
                "pci_neighbor": pci_neighbor,
            },
        )

        if result and "Overlap_Ratio" in result:
            return float(result["Overlap_Ratio"])

        return None

    def get_kpi_data(
        self,
        scenario_id: str,
    ) -> Optional[str]:

        result = self._get(
            "/get_kpi_data",
            scenario_id,
        )

        if result and "Traffic Data" in result:
            return result["Traffic Data"]

        return None

    def get_mr_data(
        self,
        scenario_id: str,
    ) -> Optional[str]:

        result = self._get(
            "/get_mr_data",
            scenario_id,
        )

        if result and "MR Data" in result:
            return result["MR Data"]

        return None

    def get_signaling_events(
        self,
        scenario_id: str,
        time: str = "",
    ) -> Optional[str]:

        result = self._get(
            "/signaling-plane-event-log",
            scenario_id,
            {"time": time} if time else {},
        )

        if result:
            return json.dumps(
                result,
                ensure_ascii=False,
            )[:1000]

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Parsing helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _summarize_mr_data(
        mr_raw: str,
    ) -> Tuple[Optional[int], Dict[str, int]]:

        if not mr_raw or not mr_raw.strip():
            return None, {}

        try:
            df = pd.read_csv(
                io.StringIO(mr_raw),
                sep="|",
            )
        except Exception:
            return None, {}

        row_count = int(len(df))

        summary = {
            "rows": row_count,
        }

        pci_cols = [
            c for c in df.columns
            if "PCI" in c
        ]

        if pci_cols:
            visible = set()

            for col in pci_cols:
                series = (
                    pd.to_numeric(
                        df[col],
                        errors="coerce",
                    )
                    .dropna()
                )

                visible.update(
                    int(v)
                    for v in series.tolist()
                    if int(v) > 0
                )

            summary["unique_pcis"] = len(visible)

        return row_count, summary

    def _extract_kpi_signals(
        self,
        raw_text: str,
    ) -> Dict[str, float]:

        signals = {}

        if not raw_text:
            return signals

        try:
            df = pd.read_csv(
                io.StringIO(raw_text),
                sep="|",
            )
        except Exception:
            df = pd.DataFrame()

        if not df.empty:
            cce_cols = [
                c for c in df.columns
                if "cce" in c.lower()
            ]
            prb_cols = [
                c for c in df.columns
                if "prb" in c.lower()
            ]
            tp_cols = [
                c for c in df.columns
                if "throughput" in c.lower()
            ]

            for col in cce_cols:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if not vals.empty and float(vals.max()) > 0.5:
                    signals["cce_pressure"] = 1.0
                    break

            for col in prb_cols:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if not vals.empty and float(vals.max()) > 70.0:
                    signals["high_prb"] = 1.0
                    break

            for col in tp_cols:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if not vals.empty and float(vals.min()) < 50.0:
                    signals["low_tp"] = 1.0
                    break

            return signals

        text = raw_text.lower()

        # Text fallback for non-CSV tool responses. Avoid treating headers alone
        # as pressure evidence; require an explicit abnormal wording.
        if "high cce" in text or "cce pressure" in text or "cce failure" in text:
            signals["cce_pressure"] = 1.0

        if "high prb" in text or "prb pressure" in text or "overloaded" in text:
            signals["high_prb"] = 1.0

        if "low throughput" in text or "throughput drop" in text:
            signals["low_tp"] = 1.0

        return signals

    # ──────────────────────────────────────────────────────────────────────
    # Scenario classification
    # ──────────────────────────────────────────────────────────────────────

    def _classify_group(
        self,
        sf: ScenarioFeatures,
    ) -> ScenarioGroup:

        top_issue, score = sf.top_issue()

        if score < 0.25:
            return ScenarioGroup.UNKNOWN

        if top_issue in (
            "coverage_hole",
            "overshoot_overlap",
        ):
            return ScenarioGroup.COVERAGE

        if top_issue in (
            "pdcch_issue",
            "high_prb",
        ):
            return ScenarioGroup.CAPACITY

        if top_issue in (
            "late_handover",
            "missing_neighbor",
        ):
            return ScenarioGroup.HANDOVER

        if top_issue in (
            "backhaul_server",
            "check_server",
        ):
            return ScenarioGroup.SERVER

        return ScenarioGroup.UNKNOWN

    # ──────────────────────────────────────────────────────────────────────
    # Tool selection
    # ──────────────────────────────────────────────────────────────────────

    def _select_tools_for_group(
        self,
        scenario_id: str,
        sf: ScenarioFeatures,
        worst_time: str,
    ) -> ToolPlan:

        group = self._classify_group(sf)

        plan = ToolPlan()

        serving_pci = sf.mobility.serving_pci_at_worst
        neighbor_pci = sf.mobility.best_neighbor_pci

        # coverage
        if group == ScenarioGroup.COVERAGE:

            if serving_pci:
                plan.add(
                    "judge_mainlobe",
                    scenario_id=scenario_id,
                    pci=serving_pci,
                    time=worst_time,
                )

                plan.add(
                    "calculate_tilt_angle",
                    scenario_id=scenario_id,
                    pci=serving_pci,
                    time=worst_time,
                )

            if neighbor_pci:
                plan.add(
                    "calculate_overlap_ratio",
                    scenario_id=scenario_id,
                    pci_serving=serving_pci,
                    pci_neighbor=neighbor_pci,
                )

            plan.add(
                "get_mr_data",
                scenario_id=scenario_id,
            )

            return plan

        # capacity
        if group == ScenarioGroup.CAPACITY:

            plan.add(
                "get_kpi_data",
                scenario_id=scenario_id,
            )

            return plan

        # handover
        if group == ScenarioGroup.HANDOVER:

            plan.add(
                "get_signaling_events",
                scenario_id=scenario_id,
                time=worst_time,
            )

            if neighbor_pci:
                plan.add(
                    "calculate_overlap_ratio",
                    scenario_id=scenario_id,
                    pci_serving=serving_pci,
                    pci_neighbor=neighbor_pci,
                )

            return plan

        # server
        if group == ScenarioGroup.SERVER:

            plan.add(
                "get_kpi_data",
                scenario_id=scenario_id,
            )

            plan.add(
                "get_mr_data",
                scenario_id=scenario_id,
            )

            return plan

        return plan

    # ──────────────────────────────────────────────────────────────────────
    # Result normalization
    # ──────────────────────────────────────────────────────────────────────

    def parse_tool_results(
        self,
        raw_results: Dict[str, Any],
    ) -> ToolEvidence:

        evidence = ToolEvidence()

        for tool_name, result in raw_results.items():

            if result is None:
                continue

            if tool_name.startswith("judge_mainlobe"):

                pci = result.get("_pci")
                flag = result.get("value")

                if pci is not None and flag is not None:
                    evidence.mainlobe_flags[pci] = bool(flag)

            elif tool_name.startswith("calculate_tilt_angle"):

                pci = result.get("_pci")
                angle = result.get("value")

                if pci is not None and angle is not None:
                    evidence.tilt_angles[pci] = float(angle)

            elif tool_name.startswith("calculate_overlap_ratio"):

                ratio = result.get("value")

                if ratio is not None:
                    evidence.overlap_ratio = float(ratio)

            elif tool_name.startswith("get_kpi_data"):

                text = str(result.get("value", ""))

                evidence.kpi_data_raw = text[:1000]

            elif tool_name.startswith("get_mr_data"):

                mr_raw = result.get("value")

                if mr_raw:
                    count, counts = self._summarize_mr_data(
                        mr_raw
                    )

                    evidence.mr_sample_count = count
                    evidence.mr_sample_counts.update(counts)

            elif tool_name.startswith("get_signaling_events"):

                sig = result.get("value")

                if sig:
                    evidence.signaling_events.append(sig[:500])

        evidence.tool_call_count = len(raw_results)

        return evidence

    # ──────────────────────────────────────────────────────────────────────
    # Main execution pipeline
    # ──────────────────────────────────────────────────────────────────────

    def call_targeted(
        self,
        scenario_id: str,
        sf: ScenarioFeatures,
        worst_time: Optional[str] = None,
    ) -> ToolEvidence:

        wt = worst_time or sf.throughput.worst_time or ""

        if not wt:
            return ToolEvidence()

        plan = self._select_tools_for_group(
            scenario_id,
            sf,
            wt,
        )

        if not plan.tools:
            return ToolEvidence()

        raw_results = {}

        with ThreadPoolExecutor(max_workers=4) as executor:

            future_map = {}

            for idx, (tool_name, kwargs) in enumerate(plan.tools):

                tool_fn = self.tool_registry.get(tool_name)

                if not tool_fn:
                    continue

                future = executor.submit(
                    tool_fn,
                    **kwargs,
                )

                future_map[future] = (
                    idx,
                    tool_name,
                    kwargs,
                )

            for future in as_completed(future_map):

                idx, tool_name, kwargs = future_map[future]

                try:
                    result = future.result()

                    raw_results[f"{tool_name}_{idx}"] = {
                        "_pci": kwargs.get("pci"),
                        "value": result,
                    }

                except Exception as exc:
                    self.logger.warning(
                        "Tool execution failed: %s %s",
                        tool_name,
                        exc,
                    )

        return self.parse_tool_results(raw_results)

    # ──────────────────────────────────────────────────────────────────────
    # Feature augmentation
    # ──────────────────────────────────────────────────────────────────────

    def augment_features(
            self,
            sf: ScenarioFeatures,
            evidence: ToolEvidence,
    ) -> ScenarioFeatures:

        if not evidence.tool_call_count:
            return sf

        serving_pci = sf.mobility.serving_pci_at_worst

        serving_flag = (
            evidence.mainlobe_flags.get(serving_pci)
            if serving_pci is not None
            else None
        )

        # IMPORTANT:
        # Tool evidence should only weakly rerank.
        # Never overwrite feature-engine priors.

        # ─────────────────────────────
        # mainlobe
        # ─────────────────────────────

        if serving_flag is True:
            # Outside mainlobe:
            # weakly suppress coverage-hole confidence
            sf.score_coverage_hole -= 0.05

            # weakly favor transport/server ambiguity
            sf.score_backhaul_server += 0.04

        # ─────────────────────────────
        # sparse MR
        # ─────────────────────────────

        if evidence.mr_sample_count is not None:

            if evidence.mr_sample_count < self.mr_sample_threshold:
                # sparse MR reduces confidence
                # of geometry-based diagnosis
                sf.score_coverage_hole -= 0.04

        # ─────────────────────────────
        # overlap
        # ─────────────────────────────

        if (
                evidence.overlap_ratio is not None
                and evidence.overlap_ratio > 0.3
        ):
            sf.score_overshoot_overlap += 0.08

        # ─────────────────────────────
        # KPI
        # ─────────────────────────────

        if evidence.kpi_data_raw:

            kpi_signals = self._extract_kpi_signals(
                evidence.kpi_data_raw
            )

            if kpi_signals.get("cce_pressure"):
                sf.score_pdcch_issue += 0.08

            if kpi_signals.get("high_prb"):
                sf.score_pdcch_issue += 0.05

            if (
                    kpi_signals.get("low_tp")
                    and not kpi_signals.get("cce_pressure")
            ):
                sf.score_backhaul_server += 0.05

        # ─────────────────────────────
        # clamp
        # ─────────────────────────────

        score_fields = [
            "score_late_handover",
            "score_missing_neighbor",
            "score_inter_freq_load",
            "score_interference",
            "score_coverage_hole",
            "score_overshoot_overlap",
            "score_pdcch_issue",
            "score_backhaul_server",
        ]

        for field in score_fields:
            value = getattr(sf, field)

            setattr(
                sf,
                field,
                max(0.0, min(1.0, value)),
            )

        return sf
