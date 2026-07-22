# backend/app/escalation.py
from app.graph_state import ChatState

def escalation_node(state: ChatState) -> ChatState:
    """
    Runs when the relevance gate doesn't have a confident answer. No
    ticket/queue is created here -- deliberately simplified away (see git
    history for the earlier `db.escalations` version): the user just gets
    a direct, honest message pointing them at a real person, and that's
    the end of it. Anyone reviewing what a user asked can read their chat
    history directly via the admin panel; there's no separate escalation
    log to maintain.
    """
    state["response_text"] = (
        "I wasn't able to find a confident answer to that in our documentation. "
        "Please email pratham@truelift.ai with your question and our team will "
        "get back to you directly."
    )
    state["escalated"] = True
    state["citations"] = []
    return state