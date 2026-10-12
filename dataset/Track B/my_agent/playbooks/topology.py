"""Topology reconstruction playbook."""

from deterministic import solve_topology
from tools import NOCClient


def should_run(question_text: str, family: str) -> bool:
    return family == "topology"


def run(question_text: str, noc_client: NOCClient) -> str | None:
    return solve_topology(question_text, noc_client)
