"""NOC API HTTP client for the Agent Tool Server."""

import re
import time
from typing import Optional
import requests


KNOWN_HUAWEI_DEVICES = {
    "Alpha-Center-01", "Alpha-Center-02",
    "Beta-Portal-01", "Beta-Portal-02", "Beta-Aegis-01", "Beta-Aegis-02",
    "Beta-Axis-01", "Beta-Axis-02", "Beta-Node-01", "Beta-Node-02",
    "Beta-Node-03", "Beta-Node-04",
    "Gamma-Portal-01", "Gamma-Portal-02", "Gamma-Aegis-01", "Gamma-Aegis-02",
    "Gamma-Axis-01", "Gamma-Axis-02", "Gamma-Node-01", "Gamma-Node-02",
    "Gamma-Node-03", "Gamma-Node-04",
    "Delta-Portal-01", "Delta-Portal-02", "Delta-Aegis-01", "Delta-Aegis-02",
    "Delta-Axis-01", "Delta-Axis-02", "Delta-Node-01", "Delta-Node-02",
    "Delta-Node-03", "Delta-Node-04",
    "Demeter-Prime-01", "Demeter-Prime-02", "Atlas-Prime-01", "Atlas-Prime-02",
    "Janus-Prime-01", "Janus-Prime-02", "Aegis-Prime-01", "Aegis-Prime-02",
    "Hermes-Prime-01", "Hermes-Prime-02", "PX1",
    "Eon-Node-01", "Gaia-Node-01", "Aegis-Node-01", "Janus-Node-01",
    "Janus-Node-02", "Atlas-Node-01", "Atlas-Node-02", "Demeter-Node-01",
    "Demeter-Node-02", "Hermes-Node-01", "Hermes-Node-02",
}


def normalize_command(command: str, device_name: str | None = None) -> str:
    """Rewrite common LLM-generated invalid CLI variants to whitelisted commands."""
    cmd = " ".join(command.strip().split())
    lower = cmd.lower()

    # Huawei tables in this environment do not support pipe filters.
    if " | " in lower:
        cmd = re.split(r"\s+\|\s+", cmd, maxsplit=1)[0].strip()
        lower = cmd.lower()

    if lower == "display arp all" and (device_name in KNOWN_HUAWEI_DEVICES or device_name is None):
        return "display arp"

    # The whitelist only allows the full routing table, not destination lookups.
    if re.match(r"^display ip routing-table\s+\d{1,3}(?:\.\d{1,3}){3}(?:\s|$)", lower):
        return "display ip routing-table"
    if re.match(r"^show ip route\s+\d{1,3}(?:\.\d{1,3}){3}(?:\s|$)", lower):
        return "show ip route"

    if lower.startswith("display current-configuration interface "):
        return "display current-configuration"

    # Interface detail commands are not available; use summary tables instead.
    if re.match(r"^display interface\s+(?:ge|xge|gigabitethernet|eth|ethernet)\S*(?:\s+\S+)?$", lower):
        return "display interface brief"
    if re.match(r"^show interface\s+(?:gi|gigabitethernet|eth|ethernet)\S*(?:\s+\S+)?$", lower):
        return "show interface brief"

    return cmd


class NOCClient:
    def __init__(self, server_url: str, question_number: int, token: Optional[str] = None):
        self.url = f"{server_url}/api/agent/execute"
        self.question_number = question_number
        self.call_count = 0
        self.rewrite_count = 0
        self.trace: list[dict] = []
        self.llm_trace: list[dict] = []
        self._session = requests.Session()
        self._session.trust_env = False  # disable system proxy interference
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"

    def execute(self, device_name: str, command: str) -> str:
        """
        Execute a CLI command on a network device.

        Returns the CLI output string on success, or a tagged error string on failure
        so the LLM can understand what went wrong and adapt.
        """
        requested_command = command.strip()
        command = normalize_command(requested_command, device_name)
        was_rewritten = command != requested_command
        if was_rewritten:
            self.rewrite_count += 1

        counted = False
        try:
            started = time.time()
            r = self._session.post(
                self.url,
                json={
                    "device_name": device_name,
                    "command": command,
                    "question_number": self.question_number,
                },
                timeout=30,
            )
            elapsed_ms = round((time.time() - started) * 1000, 1)
            self.call_count += 1
            counted = True
            body = r.json()
            record = {
                "question_number": self.question_number,
                "started_at": round(started, 3),
                "elapsed_ms": elapsed_ms,
                "device": device_name,
                "requested_command": requested_command,
                "executed_command": command,
                "rewritten": was_rewritten,
                "status_code": r.status_code,
                "vendor": body.get("vendor"),
                "result_preview": (body.get("result") or body.get("error") or r.text)[:2000],
            }
            self.trace.append(record)

            if r.status_code == 200:
                result = body.get("result", "")
                if was_rewritten:
                    return (
                        f"[COMMAND NORMALIZED] requested: {requested_command}; "
                        f"executed: {command}\n{result}"
                    )
                return result

            # 403: no_permission restriction from question_limits_config.json
            if r.status_code == 403:
                err = body.get("result") or body.get("error", "No permission")
                return f"[PERMISSION DENIED] '{command}' on '{device_name}': {err}"

            # 422: bad command syntax — server also returns CLI-style error text
            if r.status_code == 422:
                err = body.get("result") or body.get("error", "Command error")
                return f"[COMMAND ERROR] {err}"

            # 404: device not in DEVICE_VENDOR_MAP
            if r.status_code == 404:
                err = body.get("error", "Device not found")
                return f"[DEVICE NOT FOUND] '{device_name}': {err}"

            return f"[HTTP {r.status_code}] {r.text[:300]}"

        except Exception as exc:
            if not counted:
                self.call_count += 1
            self.trace.append({
                "question_number": self.question_number,
                "started_at": round(time.time(), 3),
                "elapsed_ms": None,
                "device": device_name,
                "requested_command": requested_command,
                "executed_command": command,
                "rewritten": was_rewritten,
                "status_code": None,
                "vendor": None,
                "result_preview": str(exc)[:2000],
            })
            return f"[REQUEST FAILED] {exc}"
