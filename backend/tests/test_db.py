# backend/tests/test_db.py
import pytest
import mongomock

# 1. Patch the global db object BEFORE importing the app modules
@pytest.fixture(autouse=True)
def mock_mongo(monkeypatch):
    mock_client = mongomock.MongoClient()
    mock_db = mock_client.rag_chatbot

    monkeypatch.setattr("app.session_store.db", mock_db, raising=False)
    monkeypatch.setattr("app.db.db", mock_db, raising=False)
    return mock_db

from app.session_store import (
    append_message,
    get_history,
    get_recent_history,
    clear_session,
    create_session,
    list_sessions_for_user,
    list_users,
    ensure_session,
)


class TestSessionStore:
    def test_append_and_get_history(self, mock_mongo):
        append_message("sess_1", "user", "Hello", user_id="user_a")
        append_message("sess_1", "assistant", "Hi there", user_id="user_a")

        history = get_history("sess_1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert "timestamp" in history[0]

        # Verify it actually hit the mocked DB collection
        assert mock_mongo.messages.count_documents({}) == 2

    def test_append_message_tags_user_id(self, mock_mongo):
        append_message("sess_tag", "user", "hi", user_id="user_xyz")
        doc = mock_mongo.messages.find_one({"session_id": "sess_tag"})
        assert doc["user_id"] == "user_xyz"

    def test_get_recent_history_truncates(self):
        for i in range(10):
            append_message("sess_2", "user", f"msg {i}", user_id="user_b")

        recent = get_recent_history("sess_2", max_turns=6)
        assert len(recent) == 6
        assert recent[-1]["content"] == "msg 9"

    def test_clear_session(self, mock_mongo):
        append_message("sess_3", "user", "drop me", user_id="user_c")
        assert mock_mongo.messages.count_documents({"session_id": "sess_3"}) == 1

        clear_session("sess_3")
        assert mock_mongo.messages.count_documents({"session_id": "sess_3"}) == 0


class TestSessions:
    def test_create_session_returns_new_id(self, mock_mongo):
        session_id = create_session("user_1")
        assert session_id
        doc = mock_mongo.sessions.find_one({"session_id": session_id})
        assert doc["user_id"] == "user_1"
        assert doc["preview"] is None

    def test_list_sessions_for_user_sorted_most_recent_first(self, mock_mongo):
        s1 = create_session("user_2")
        s2 = create_session("user_2")

        # Simulate activity so last_active_at differs meaningfully.
        append_message(s1, "user", "first chat", user_id="user_2")
        append_message(s2, "user", "second chat", user_id="user_2")

        sessions = list_sessions_for_user("user_2")
        assert len(sessions) == 2
        # Most recently active (s2, touched last) should come first.
        assert sessions[0]["session_id"] == s2

    def test_first_user_message_becomes_preview(self, mock_mongo):
        session_id = create_session("user_3")
        append_message(session_id, "user", "What is your pricing?", user_id="user_3")
        append_message(session_id, "assistant", "We don't disclose...", user_id="user_3")

        sessions = list_sessions_for_user("user_3")
        assert sessions[0]["preview"] == "What is your pricing?"

    def test_list_sessions_for_unknown_user_is_empty(self, mock_mongo):
        assert list_sessions_for_user("nobody") == []

    def test_ensure_session_creates_if_missing(self, mock_mongo):
        ensure_session("stray_session", "user_4")
        doc = mock_mongo.sessions.find_one({"session_id": "stray_session"})
        assert doc is not None
        assert doc["user_id"] == "user_4"

    def test_ensure_session_is_noop_if_already_exists(self, mock_mongo):
        session_id = create_session("user_5")
        append_message(session_id, "user", "hi", user_id="user_5")
        before = mock_mongo.sessions.find_one({"session_id": session_id})

        ensure_session(session_id, "user_5")
        after = mock_mongo.sessions.find_one({"session_id": session_id})
        # Preview set by the real message shouldn't be clobbered back to None.
        assert after["preview"] == before["preview"]


class TestListUsers:
    def test_list_users_aggregates_session_counts(self, mock_mongo):
        create_session("user_a")
        create_session("user_a")
        create_session("user_b")

        users = list_users()
        by_id = {u["user_id"]: u for u in users}
        assert by_id["user_a"]["session_count"] == 2
        assert by_id["user_b"]["session_count"] == 1

    def test_list_users_empty_when_no_sessions(self, mock_mongo):
        assert list_users() == []