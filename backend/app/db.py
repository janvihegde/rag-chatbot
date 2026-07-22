# backend/app/db.py
import os
from pymongo import MongoClient, ASCENDING, DESCENDING

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)

# Use a database named 'rag_chatbot'
db = client.rag_chatbot

def ensure_indexes():
    """
    Create indexes matching the app's actual query patterns. Without
    these, message history and session-listing queries do a full
    collection scan -- confirmed live via a Mongo "Slow query" / COLLSCAN
    log entry before these were added.

    Deliberately NOT run at module import time: this module is imported
    almost everywhere (including test modules, which monkeypatch `db`
    AFTER import, not before), and create_index() needs a real live
    connection. Running it at import time would mean every test import
    attempts a real MongoDB connection even in environments with no
    Mongo running at all. Call this explicitly from main.py's FastAPI
    startup event instead, where a live connection is an actual
    requirement of the running app.

    create_index() is idempotent (a no-op if the index already exists),
    so this is safe to call on every process start.
    """
    db.messages.create_index([("session_id", ASCENDING), ("timestamp_ns", ASCENDING)])
    db.sessions.create_index([("user_id", ASCENDING), ("last_active_at_ns", DESCENDING)])
    db.sessions.create_index([("session_id", ASCENDING)], unique=True)
    db.documents.create_index([("source", ASCENDING)], unique=True)