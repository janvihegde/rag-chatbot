# backend/app/session_store.py
import uuid
from datetime import datetime, timezone
from app.db import db

MAX_HISTORY_TURNS = 6
PREVIEW_LENGTH = 80


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_session(user_id: str) -> str:
    """Start a new chat for this user. Returns the new session_id."""
    session_id = str(uuid.uuid4())
    db.sessions.insert_one({
        "session_id": session_id,
        "user_id": user_id,
        "created_at": _now(),
        "last_active_at": _now(),
        "preview": None,  # filled in on the first user message, see touch_session
    })
    return session_id


def list_sessions_for_user(user_id: str) -> list[dict]:
    """
    Past chats for this user, most recently active first -- used both for
    the "continue a previous chat" picker on the frontend and for the
    admin panel's per-user chat history browsing.
    """
    cursor = db.sessions.find({"user_id": user_id}).sort("last_active_at", -1)
    return [
        {
            "session_id": doc["session_id"],
            "created_at": doc["created_at"],
            "last_active_at": doc["last_active_at"],
            "preview": doc.get("preview"),
        }
        for doc in cursor
    ]


def ensure_session(session_id: str, user_id: str | None) -> None:
    """
    Creates the session's tracking record if it doesn't already exist.
    Defensive fallback for /api/chat -- the expected flow is the frontend
    calls create_session() first (via POST /api/users/{user_id}/sessions),
    but this keeps a stray/unknown session_id from silently having no
    entry in the sessions collection, which would otherwise break
    list_sessions_for_user() and the admin panel's per-user browsing.
    """
    db.sessions.update_one(
        {"session_id": session_id},
        {"$setOnInsert": {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": _now(),
            "last_active_at": _now(),
            "preview": None,
        }},
        upsert=True,
    )


def session_exists(session_id: str) -> bool:
    return db.sessions.find_one({"session_id": session_id}) is not None


def touch_session(session_id: str, role: str, content: str) -> None:
    """
    Updates last_active_at on every message, and captures a short preview
    of the conversation (the first user message) the first time one comes
    in -- shown in the continue-chat picker and the admin panel's session
    list so a person doesn't have to open every chat just to see what it
    was about.
    """
    update = {"$set": {"last_active_at": _now()}}
    session_doc = db.sessions.find_one({"session_id": session_id})
    if session_doc is not None and not session_doc.get("preview") and role == "user":
        update["$set"]["preview"] = content[:PREVIEW_LENGTH]
    db.sessions.update_one({"session_id": session_id}, update)


def get_history(session_id: str) -> list[dict]:
    """Fetch full history for a session from MongoDB."""
    # Find all messages for this session, sorted by timestamp ascending
    cursor = db.messages.find({"session_id": session_id}).sort("timestamp", 1)

    # Strip the MongoDB _id, session_id, and user_id before returning to
    # the graph / API response -- callers only need role/content/timestamp.
    history = []
    for doc in cursor:
        history.append({
            "role": doc["role"],
            "content": doc["content"],
            "timestamp": doc["timestamp"]
        })
    return history

def get_recent_history(session_id: str, max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """
    Most recent `max_turns` messages, fetched directly from MongoDB with a
    query-level limit -- NOT get_history()[-max_turns:], which used to
    pull the entire session's history into memory on every single chat
    turn just to slice off the last few messages. For a long-running
    session under concurrent load, that meant O(full history length)
    document transfer per request instead of O(max_turns); the query-level
    limit below is bounded regardless of how long the conversation gets.
    """
    cursor = (
        db.messages.find({"session_id": session_id})
        .sort("timestamp", -1)
        .limit(max_turns)
    )
    recent_desc = [
        {"role": doc["role"], "content": doc["content"], "timestamp": doc["timestamp"]}
        for doc in cursor
    ]
    return list(reversed(recent_desc))

def append_message(session_id: str, role: str, content: str, user_id: str | None = None) -> None:
    """
    Insert a new message into MongoDB, tagged with user_id (denormalized
    onto every message) so the admin panel can query a user's full message
    history directly without joining through the sessions collection.
    """
    db.messages.insert_one({
        "session_id": session_id,
        "user_id": user_id,
        "role": role,
        "content": content,
        "timestamp": _now(),
    })
    touch_session(session_id, role, content)

def list_users() -> list[dict]:
    """
    Every user who has started at least one chat, with a session count and
    their most recent activity -- powers the admin panel's user list.
    """
    pipeline = [
        {"$group": {
            "_id": "$user_id",
            "session_count": {"$sum": 1},
            "last_active_at": {"$max": "$last_active_at"},
        }},
        {"$sort": {"last_active_at": -1}},
    ]
    return [
        {
            "user_id": doc["_id"],
            "session_count": doc["session_count"],
            "last_active_at": doc["last_active_at"],
        }
        for doc in db.sessions.aggregate(pipeline)
    ]


def clear_session(session_id: str) -> None:
    """Delete a session's history."""
    db.messages.delete_many({"session_id": session_id})
    db.sessions.delete_one({"session_id": session_id})