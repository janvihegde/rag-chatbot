# backend/tests/test_admin_api.py
import pytest
import mongomock
from fastapi.testclient import TestClient
from app.main import app
from app.auth import verify_admin

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_mongo(monkeypatch):
    mock_client = mongomock.MongoClient()
    mock_db = mock_client.rag_chatbot

    # Each module holds its own reference to `db` (from app.db import db),
    # so each needs patching individually -- patching app.main.db alone
    # would miss the actual reads/writes that happen inside
    # session_store.py and ingest.py.
    monkeypatch.setattr("app.main.db", mock_db, raising=False)
    monkeypatch.setattr("app.db.db", mock_db, raising=False)
    monkeypatch.setattr("app.session_store.db", mock_db, raising=False)
    monkeypatch.setattr("app.ingest.db", mock_db, raising=False)
    return mock_db


@pytest.fixture(autouse=True)
def admin_override():
    app.dependency_overrides[verify_admin] = lambda: "super-secret-test-key"
    yield
    app.dependency_overrides = {}


AUTH_HEADERS = {"Authorization": "Bearer super-secret-test-key"}


class TestAuthRBAC:
    def test_ingest_rejects_without_token(self):
        # Bypass the auto-applied override for just this one request, to
        # exercise the real "no token" rejection path.
        app.dependency_overrides = {}
        res = client.post("/api/admin/ingest")
        assert res.status_code == 401

    def test_ingest_rejects_with_invalid_token(self):
        app.dependency_overrides = {}
        res = client.post(
            "/api/admin/ingest",
            headers={"Authorization": "Bearer wrong-key"}
        )
        assert res.status_code == 403

    def test_admin_routes_accept_valid_token(self, monkeypatch):
        monkeypatch.setattr(
            "app.main.ingest_files", lambda files: {"documents": len(files), "chunks": 1, "sources": []}
        )
        res = client.post(
            "/api/admin/ingest",
            headers=AUTH_HEADERS,
            files={"files": ("test.html", b"<p>hello</p>", "text/html")},
        )
        assert res.status_code == 200


class TestAdminDocuments:
    def test_list_documents(self, mock_mongo):
        mock_mongo.documents.insert_many([
            {"source": "a.pdf", "chunk_count": 5, "ingested_at": "2026-01-01T00:00:00"},
            {"source": "b.html", "chunk_count": 2, "ingested_at": "2026-01-02T00:00:00"},
        ])
        res = client.get("/api/admin/documents", headers=AUTH_HEADERS)
        assert res.status_code == 200
        sources = {d["source"] for d in res.json()["documents"]}
        assert sources == {"a.pdf", "b.html"}

    def test_delete_document(self, monkeypatch, mock_mongo):
        mock_mongo.documents.insert_one({"source": "a.pdf", "chunk_count": 5, "ingested_at": "x"})
        monkeypatch.setattr("app.main.delete_document", lambda source: source == "a.pdf")

        res = client.delete("/api/admin/documents/a.pdf", headers=AUTH_HEADERS)
        assert res.status_code == 200
        assert res.json()["deleted"] == "a.pdf"

    def test_delete_unknown_document_returns_404(self, monkeypatch):
        monkeypatch.setattr("app.main.delete_document", lambda source: False)
        res = client.delete("/api/admin/documents/nonexistent.pdf", headers=AUTH_HEADERS)
        assert res.status_code == 404


class TestAdminUsersAndSessions:
    def test_list_users(self, monkeypatch):
        monkeypatch.setattr(
            "app.main.list_users",
            lambda: [{"user_id": "u1", "session_count": 2, "last_active_at": "2026-01-02"}],
        )
        res = client.get("/api/admin/users", headers=AUTH_HEADERS)
        assert res.status_code == 200
        assert res.json()["users"][0]["user_id"] == "u1"

    def test_get_user_sessions(self, monkeypatch):
        monkeypatch.setattr(
            "app.main.list_sessions_for_user",
            lambda user_id: [{"session_id": "s1", "preview": "hi", "last_active_at": "x", "created_at": "x"}],
        )
        res = client.get("/api/admin/users/u1/sessions", headers=AUTH_HEADERS)
        assert res.status_code == 200
        assert res.json()["sessions"][0]["session_id"] == "s1"

    def test_get_session_messages(self, monkeypatch):
        monkeypatch.setattr(
            "app.main.get_history",
            lambda session_id: [{"role": "user", "content": "hi", "timestamp": "x"}],
        )
        res = client.get("/api/admin/sessions/s1/messages", headers=AUTH_HEADERS)
        assert res.status_code == 200
        assert res.json()["messages"][0]["content"] == "hi"

    def test_admin_routes_require_auth(self):
        app.dependency_overrides = {}
        res = client.get("/api/admin/users")
        assert res.status_code == 401