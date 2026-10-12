"""Physical-interface path playbook."""

from deterministic import solve_path
from tools import NOCClient


def should_run(question_text: str, family: str) -> bool:
    return family == "path"


def run(question_text: str, noc_client: NOCClient) -> str | None:
    return solve_path(question_text, noc_client)
