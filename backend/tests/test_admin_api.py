# backend/tests/test_admin_api.py
import pytest
import mongomock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_mongo(monkeypatch):
    mock_client = mongomock.MongoClient()
    mock_db = mock_client.rag_chatbot
    
    # FIX: Patch the db object exactly where it is being used!
    monkeypatch.setattr("app.main.db", mock_db)
    monkeypatch.setattr("app.db.db", mock_db)
    return mock_db

@pytest.fixture(autouse=True)
def mock_mongo(monkeypatch):
    mock_client = mongomock.MongoClient()
    mock_db = mock_client.rag_chatbot
    
    # Add raising=False to prevent crashes if imports differ
    monkeypatch.setattr("app.main.db", mock_db, raising=False)
    monkeypatch.setattr("app.db.db", mock_db, raising=False)
    return mock_db

from app.auth import verify_admin

class TestAuthRBAC:
    def test_ingest_rejects_without_token(self):
        res = client.post("/api/admin/ingest")
        assert res.status_code == 401

    def test_ingest_rejects_with_invalid_token(self):
        res = client.post(
            "/api/admin/ingest", 
            headers={"Authorization": "Bearer wrong-key"}
        )
        assert res.status_code == 403

    def test_admin_routes_accept_valid_token(self, monkeypatch):
        monkeypatch.setattr("app.main.ingest_from_s3", lambda s: {"documents": 1})
        
        # Override the auth dependency just for this test
        app.dependency_overrides[verify_admin] = lambda: "super-secret-test-key"
        
        res = client.post(
            "/api/admin/ingest",
            headers={"Authorization": "Bearer super-secret-test-key"},
            data={"source": "s3://test-bucket/docs"}
        )
        assert res.status_code == 200
        
        # Clean up the override
        app.dependency_overrides = {}


class TestAdminEndpoints:
    def test_get_analytics(self, mock_mongo):
        mock_mongo.messages.insert_many([
            {"role": "user"}, {"role": "assistant"}, {"role": "user"}
        ])
        mock_mongo.escalations.insert_one({"status": "pending"})
        
        # Override auth dependency
        app.dependency_overrides[verify_admin] = lambda: "super-secret-test-key"
        
        res = client.get(
            "/api/admin/analytics", 
            headers={"Authorization": "Bearer super-secret-test-key"}
        )
        assert res.status_code == 200
        
        data = res.json()["metrics"]
        assert data["total_user_queries"] == 2
        assert data["total_escalations"] == 1
        
        app.dependency_overrides = {}

    def test_get_escalations(self, mock_mongo):
        mock_mongo.escalations.insert_many([
            {"session_id": "1", "status": "pending", "timestamp": "2026-01-01"},
            {"session_id": "2", "status": "resolved", "timestamp": "2026-01-02"}
        ])
        
        # Override auth dependency
        app.dependency_overrides[verify_admin] = lambda: "super-secret-test-key"
        
        res_pending = client.get(
            "/api/admin/escalations",
            headers={"Authorization": "Bearer super-secret-test-key"}
        )
        
        assert res_pending.status_code == 200
        assert len(res_pending.json()["escalations"]) == 1
        assert res_pending.json()["escalations"][0]["session_id"] == "1"
        
        app.dependency_overrides = {}