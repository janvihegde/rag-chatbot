"""
Tests for the document-tracking side of app/ingest.py: list_documents()
and delete_document(). These operate on the `db.documents` Mongo
collection, which conftest's stubbed QdrantClient doesn't touch, so we
only need to mock Mongo here, not the vector store itself.
"""
import pytest
import mongomock

from app import ingest


@pytest.fixture(autouse=True)
def mock_mongo(monkeypatch):
    mock_client = mongomock.MongoClient()
    mock_db = mock_client.rag_chatbot
    monkeypatch.setattr("app.ingest.db", mock_db, raising=False)
    return mock_db


class TestDocumentTracking:
    def test_add_documents_tracks_source_and_chunk_count(self, mock_mongo):
        ingest._add_documents([{"id": "a.pdf", "text": "one two three four five."}])
        doc = mock_mongo.documents.find_one({"source": "a.pdf"})
        assert doc is not None
        assert doc["chunk_count"] >= 1
        assert "ingested_at" in doc

    def test_reingesting_same_source_updates_not_duplicates(self, mock_mongo):
        ingest._add_documents([{"id": "a.pdf", "text": "short text"}])
        ingest._add_documents([{"id": "a.pdf", "text": "a much longer text with more words in it"}])
        assert mock_mongo.documents.count_documents({"source": "a.pdf"}) == 1

    def test_list_documents_sorted_most_recent_first(self, mock_mongo):
        mock_mongo.documents.insert_many([
            {"source": "old.pdf", "chunk_count": 1, "ingested_at": "2026-01-01T00:00:00"},
            {"source": "new.pdf", "chunk_count": 1, "ingested_at": "2026-01-02T00:00:00"},
        ])
        docs = ingest.list_documents()
        assert docs[0]["source"] == "new.pdf"

    def test_delete_document_removes_tracking_record(self, monkeypatch, mock_mongo):
        mock_mongo.documents.insert_one({"source": "a.pdf", "chunk_count": 3, "ingested_at": "x"})
        deleted_filters = []
        monkeypatch.setattr(
            "app.ingest._client.delete",
            lambda **kwargs: deleted_filters.append(kwargs),
        )

        result = ingest.delete_document("a.pdf")

        assert result is True
        assert mock_mongo.documents.find_one({"source": "a.pdf"}) is None
        assert len(deleted_filters) == 1  # Qdrant delete was actually called

    def test_delete_unknown_document_returns_false(self, mock_mongo):
        assert ingest.delete_document("nonexistent.pdf") is False