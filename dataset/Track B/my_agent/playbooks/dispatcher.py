"""Dispatch rule-based playbooks before the general ReAct loop."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from answer_validation import validate_answer
from prompts import detect_family
from tools import NOCClient

from playbooks import fault_evpn_srv6, fault_port, fault_routing, path_physical, path_pj, topology


log = logging.getLogger(__name__)

PLAYBOOKS = (
    topology,
    path_pj,
    path_physical,
    fault_port,
    fault_evpn_srv6,
    fault_routing,
)


def run_playbooks(
    question_text: str,
    noc_client: NOCClient,
    family: str | None = None,
    playbooks: Iterable = PLAYBOOKS,
) -> str | None:
    family = family or detect_family(question_text)
    for playbook in playbooks:
        if not playbook.should_run(question_text, family):
            continue
        before = noc_client.call_count
        try:
            answer = playbook.run(question_text, noc_client)
        except Exception as exc:
            log.warning("%s failed: %s", playbook.__name__, exc, exc_info=True)
            continue
        if not answer:
            continue
        validation = validate_answer(answer, family, question_text)
        if validation.ok:
            log.info(
                "%s produced valid answer using %d tool calls",
                playbook.__name__,
                noc_client.call_count - before,
            )
            return answer
        log.warning(
            "%s produced invalid answer (%s)",
            playbook.__name__,
            "; ".join(validation.reasons),
        )
    return None
