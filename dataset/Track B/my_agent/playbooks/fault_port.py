"""High-confidence port-fault playbook."""

from tools import NOCClient

from playbooks.fault_common import (
    find_ip_owner,
    get_interface_info,
    is_shutdown,
    parse_fault_question,
    validate_fault_answer,
)


def should_run(question_text: str, family: str) -> bool:
    lower = question_text.lower()
    return family == "fault" and "port fault" in lower


def run(question_text: str, noc_client: NOCClient) -> str | None:
    parsed = parse_fault_question(question_text, noc_client)

    if parsed.destination_device and parsed.destination_port:
        info = get_interface_info(noc_client, parsed.destination_device, parsed.destination_port)
        if info and is_shutdown(info):
            return validate_fault_answer(
                f"{parsed.destination_device};{parsed.destination_port};shutdown",
                question_text,
            )

    if (
        parsed.destination_ip is not None
        and not (parsed.destination_device and parsed.destination_port)
        and "pj area" not in parsed.body.lower()
        and "pjgfa area" not in parsed.body.lower()
    ):
        owner = find_ip_owner(noc_client, parsed.destination_ip)
        if owner and is_shutdown(owner):
            return validate_fault_answer(
                f"{owner.device};{owner.port};shutdown",
                question_text,
            )

    return None
