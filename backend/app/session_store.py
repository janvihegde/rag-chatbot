# backend/app/session_store.py
from datetime import datetime, timezone
from app.db import db

MAX_HISTORY_TURNS = 6

def get_history(session_id: str) -> list[dict]:
    """Fetch full history for a session from MongoDB."""
    # Find all messages for this session, sorted by timestamp ascending
    cursor = db.messages.find({"session_id": session_id}).sort("timestamp", 1)
    
    # Strip the MongoDB _id and session_id before returning to the graph
    history = []
    for doc in cursor:
        history.append({
            "role": doc["role"],
            "content": doc["content"],
            "timestamp": doc["timestamp"]
        })
    return history

def get_recent_history(session_id: str, max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """Most recent `max_turns` messages."""
    return get_history(session_id)[-max_turns:]

def append_message(session_id: str, role: str, content: str) -> None:
    """Insert a new message into MongoDB."""
    db.messages.insert_one({
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

def clear_session(session_id: str) -> None:
    """Delete a session's history."""
    db.messages.delete_many({"session_id": session_id})