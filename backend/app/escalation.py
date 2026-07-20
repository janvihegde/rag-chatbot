# backend/app/escalation.py
from datetime import datetime, timezone
from app.graph_state import ChatState
from app.db import db

def _log_escalation(session_id: str, message: str, relevance_score: float):
    """
    Saves the escalation to the database (e.g. for a future admin review
    tool). NOTE: nothing currently reads/actions this beyond storing it --
    see escalation_node's response_text, which is honest about that: it
    tells the user to contact Truelift directly, rather than implying a
    support team is automatically notified when no such handoff exists.
    """
    db.escalations.insert_one({
        "session_id": session_id,
        "user_message": message,
        "relevance_score": relevance_score,
        "status": "pending",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    print(f"[ESCALATION LOGGED] session_id={session_id}, query={message!r}")

def escalation_node(state: ChatState) -> ChatState:
    _log_escalation(
        session_id=state["session_id"],
        message=state["message"],
        relevance_score=state.get("relevance_score", 0.0),
    )
    state["response_text"] = (
        "I wasn't able to find a confident answer to that in our documentation. "
        "Please email pratham@truelift.ai with your question and our team will "
        "get back to you directly."
    )
    state["escalated"] = True
    state["citations"] = []
    return state