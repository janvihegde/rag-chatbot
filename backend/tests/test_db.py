# backend/tests/test_db.py
import pytest
import mongomock
from datetime import datetime, timezone

# 1. Patch the global db object BEFORE importing the app modules
@pytest.fixture(autouse=True)
def mock_mongo(monkeypatch):
    mock_client = mongomock.MongoClient()
    mock_db = mock_client.rag_chatbot
    
    monkeypatch.setattr("app.session_store.db", mock_db, raising=False)
    monkeypatch.setattr("app.escalation.db", mock_db, raising=False)
    monkeypatch.setattr("app.db.db", mock_db, raising=False)
    return mock_db

from app.session_store import append_message, get_history, get_recent_history, clear_session
from app.escalation import _log_escalation

class TestSessionStore:
    def test_append_and_get_history(self, mock_mongo):
        append_message("sess_1", "user", "Hello")
        append_message("sess_1", "assistant", "Hi there")
        
        history = get_history("sess_1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert "timestamp" in history[0]

        # Verify it actually hit the mocked DB collection
        assert mock_mongo.messages.count_documents({}) == 2

    def test_get_recent_history_truncates(self):
        for i in range(10):
            append_message("sess_2", "user", f"msg {i}")
            
        recent = get_recent_history("sess_2", max_turns=6)
        assert len(recent) == 6
        assert recent[-1]["content"] == "msg 9"
        
    def test_clear_session(self, mock_mongo):
        append_message("sess_3", "user", "drop me")
        assert mock_mongo.messages.count_documents({"session_id": "sess_3"}) == 1
        
        clear_session("sess_3")
        assert mock_mongo.messages.count_documents({"session_id": "sess_3"}) == 0


class TestEscalationLog:
    def test_log_escalation_writes_to_db(self, mock_mongo):
        _log_escalation("sess_99", "I need a human", 0.15)
        
        doc = mock_mongo.escalations.find_one({"session_id": "sess_99"})
        assert doc is not None
        assert doc["user_message"] == "I need a human"
        assert doc["relevance_score"] == 0.15
        assert doc["status"] == "pending"