"""NOC API HTTP client for the Agent Tool Server."""

from typing import Optional
import requests


class NOCClient:
    def __init__(self, server_url: str, question_number: int, token: Optional[str] = None):
        self.url = f"{server_url}/api/agent/execute"
        self.question_number = question_number
        self.call_count = 0
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
        try:
            r = self._session.post(
                self.url,
                json={
                    "device_name": device_name,
                    "command": command.strip(),
                    "question_number": self.question_number,
                },
                timeout=30,
            )
            self.call_count += 1
            body = r.json()

            if r.status_code == 200:
                return body.get("result", "")

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
            self.call_count += 1
            return f"[REQUEST FAILED] {exc}"
