# backend/app/escalation.py
from datetime import datetime, timezone
from app.graph_state import ChatState
from app.db import db

def _log_escalation(session_id: str, message: str, relevance_score: float):
    """
    Saves the escalation to the database for the Admin Dashboard.
    """
    db.escalations.insert_one({
        "session_id": session_id,
        "user_message": message,
        "relevance_score": relevance_score,
        "status": "pending",  # Can be updated to 'resolved' by an admin later
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    # You can keep the print statement as a placeholder for AWS SES
    print(f"[ESCALATION LOGGED] session_id={session_id}, query={message!r}")

def escalation_node(state: ChatState) -> ChatState:
    _log_escalation(
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