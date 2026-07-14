"""
Human Escalation (SRS Section: Functional Requirements -> 8. Human Escalation).

STEP 4 of the build: stubbed email (prints instead of sending via AWS SES,
since no AWS account/keys exist yet). Swap `_send_escalation_email`'s body
for a real boto3 SES call later -- the node and graph wiring won't change.

Per the SRS: "Escalation only fires from the no-context path, never from
the off-topic/reject path" -- this node is only reachable when the
relevance gate fails, never from scope_reject.
"""
from app.graph_state import ChatState


def _send_escalation_email(session_id: str, message: str, relevance_score: float):
    """
    STUB: replace with real AWS SES send_email call once AWS creds exist.
    """
    print(
        f"[ESCALATION EMAIL - STUB] session_id={session_id} "
        f"relevance_score={relevance_score:.3f} query={message!r}"
    )


def escalation_node(state: ChatState) -> ChatState:
    _send_escalation_email(
        session_id=state["session_id"],
        message=state["message"],
        relevance_score=state.get("relevance_score", 0.0),
    )
    state["response_text"] = (
        "I wasn't able to find a confident answer to that in our documentation, "
        "so I've forwarded your question to our support team -- they'll follow up "
        "with you shortly."
    )
    state["escalated"] = True
    state["citations"] = []
    return state