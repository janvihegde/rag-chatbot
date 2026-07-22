"""
Tests for app/escalation.py.

Covers:
  - escalation_node sets escalated=True, empty citations, and a
    user-facing message pointing to a real contact email.

NOTE: escalation logging (a `db.escalations` queue) was intentionally
removed -- there is no ticket/notification side effect to test here
anymore. escalation_node is now a pure state transformation.
"""
from app.escalation import escalation_node


def test_escalation_node_sets_expected_state():
    state = {
        "session_id": "sess_123",
        "message": "what is quantum entanglement in our product",
        "relevance_score": 0.12,
    }
    result = escalation_node(state)

    assert result["escalated"] is True
    assert result["citations"] == []
    assert "pratham@truelift.ai" in result["response_text"]


def test_escalation_node_works_without_relevance_score():
    # relevance_score deliberately absent from state -- escalation_node
    # doesn't depend on it now that there's no log entry to write.
    state = {"session_id": "sess_1", "message": "q"}
    result = escalation_node(state)
    assert result["escalated"] is True
    assert "pratham@truelift.ai" in result["response_text"]