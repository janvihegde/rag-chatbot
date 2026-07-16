"""
Tests for app/escalation.py.

Covers:
  - escalation_node sets escalated=True, empty citations, and a
    user-facing fallback message.
  - The (stubbed) email side-effect fires with the right session_id,
    message, and relevance_score.
"""
from app.escalation import escalation_node


def test_escalation_node_sets_expected_state(monkeypatch):
    sent = {}

    def _fake_send(session_id, message, relevance_score):
        sent["session_id"] = session_id
        sent["message"] = message
        sent["relevance_score"] = relevance_score

    monkeypatch.setattr("app.escalation._send_escalation_email", _fake_send)

    state = {
        "session_id": "sess_123",
        "message": "what is quantum entanglement in our product",
        "relevance_score": 0.12,
    }
    result = escalation_node(state)

    assert result["escalated"] is True
    assert result["citations"] == []
    assert "forwarded your question" in result["response_text"]

    assert sent["session_id"] == "sess_123"
    assert sent["message"] == state["message"]
    assert sent["relevance_score"] == 0.12


def test_escalation_node_defaults_missing_relevance_score(monkeypatch):
    sent = {}
    monkeypatch.setattr(
        "app.escalation._send_escalation_email",
        lambda session_id, message, relevance_score: sent.update(
            relevance_score=relevance_score
        ),
    )
    # relevance_score deliberately absent from state
    state = {"session_id": "sess_1", "message": "q"}
    escalation_node(state)
    assert sent["relevance_score"] == 0.0
