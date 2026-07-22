# backend/app/session_store.py
import time
import uuid
from datetime import datetime, timezone
from app.db import db

MAX_HISTORY_TURNS = 6
PREVIEW_LENGTH = 80


def _now() -> str:
    """Human-readable timestamp, for display only -- not for ordering.
    See _now_ns() for why ordering needs a separate field."""
    return datetime.now(timezone.utc).isoformat()


def _now_ns() -> int:
    """
    Nanosecond-resolution monotonic-ish sort key, stored ALONGSIDE the
    human-readable ISO timestamp from _now(). This exists specifically to
    fix a real ordering bug: datetime.now().isoformat() looks like it has
    microsecond precision, but the underlying OS clock (notably on
    Windows) often has much coarser actual resolution -- rapid successive
    inserts (e.g. a fast test loop, or a burst of messages) can get the
    EXACT SAME timestamp string, making sort order on that field
    unstable/arbitrary for ties. time.time_ns() has far higher effective
    resolution, making collisions between distinct calls astronomically
    unlikely, so it's used for all `.sort()` calls; the ISO string field
    stays purely for human display.
    """
    return time.time_ns()


def create_session(user_id: str) -> str:
    """Start a new chat for this user. Returns the new session_id."""
    session_id = str(uuid.uuid4())
    db.sessions.insert_one({
        "session_id": session_id,
        "user_id": user_id,
        "created_at": _now(),
        "last_active_at": _now(),
        "last_active_at_ns": _now_ns(),
        "preview": None,  # filled in on the first user message, see touch_session
    })
    return session_id


def list_sessions_for_user(user_id: str) -> list[dict]:
    """
    Past chats for this user, most recently active first -- used both for
    the "continue a previous chat" picker on the frontend and for the
    admin panel's per-user chat history browsing.

    Re-sorted explicitly in Python (see get_history()'s comment) rather
    than trusting the driver's cursor.sort() alone across every backend.
    """
    docs = sorted(
        db.sessions.find({"user_id": user_id}),
        key=lambda d: d["last_active_at_ns"],
        reverse=True,
    )
    return [
        {
            "session_id": doc["session_id"],
            "created_at": doc["created_at"],
            "last_active_at": doc["last_active_at"],
            "preview": doc.get("preview"),
        }
        for doc in docs
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
            "last_active_at_ns": _now_ns(),
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
    update = {"$set": {"last_active_at": _now(), "last_active_at_ns": _now_ns()}}
    session_doc = db.sessions.find_one({"session_id": session_id})
    if session_doc is not None and not session_doc.get("preview") and role == "user":
        update["$set"]["preview"] = content[:PREVIEW_LENGTH]
    db.sessions.update_one({"session_id": session_id}, update)


def get_history(session_id: str) -> list[dict]:
    """Fetch full history for a session from MongoDB."""
    # Sort explicitly in Python rather than trusting cursor.sort() alone --
    # some mongomock versions have had bugs applying sort/limit in the
    # wrong order (limit truncating BEFORE sort), which real MongoDB never
    # does, but which broke test reliability. Full history is already
    # small enough per-session that this costs nothing.
    docs = sorted(db.messages.find({"session_id": session_id}), key=lambda d: d["timestamp_ns"])
    return [
        {"role": doc["role"], "content": doc["content"], "timestamp": doc["timestamp"]}
        for doc in docs
    ]

def get_recent_history(session_id: str, max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """
    Most recent `max_turns` messages, fetched directly from MongoDB with a
    query-level limit -- NOT get_history()[-max_turns:], which used to
    pull the entire session's history into memory on every single chat
    turn just to slice off the last few messages. For a long-running
    session under concurrent load, that meant O(full history length)
    document transfer per request instead of O(max_turns); the query-level
    limit below is bounded regardless of how long the conversation gets.

    The small result set is explicitly re-sorted in Python (see
    get_history()'s comment) rather than trusting the DB driver's
    sort()+limit() chaining order across every possible backend/mock
    implementation.
    """
    cursor = db.messages.find({"session_id": session_id}).sort("timestamp_ns", -1).limit(max_turns)
    docs = sorted(cursor, key=lambda d: d["timestamp_ns"])
    return [
        {"role": doc["role"], "content": doc["content"], "timestamp": doc["timestamp"]}
        for doc in docs
    ]

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
        "timestamp_ns": _now_ns(),
    })
    touch_session(session_id, role, content)

def list_users() -> list[dict]:
    """
    Every user who has started at least one chat, with a session count and
    their most recent activity -- powers the admin panel's user list.

    Sorted explicitly in Python after the aggregation rather than relying
    on the pipeline's own $sort stage (see get_history()'s comment on why
    driver/mock sort behavior isn't trusted blindly here).
    """
    pipeline = [
        {"$group": {
            "_id": "$user_id",
            "session_count": {"$sum": 1},
            "last_active_at": {"$max": "$last_active_at"},
            "last_active_at_ns": {"$max": "$last_active_at_ns"},
        }},
    ]
    grouped = sorted(
        db.sessions.aggregate(pipeline),
        key=lambda d: d["last_active_at_ns"],
        reverse=True,
    )
    return [
        {
            "user_id": doc["_id"],
            "session_count": doc["session_count"],
            "last_active_at": doc["last_active_at"],
        }
        for doc in grouped
    ]


def clear_session(session_id: str) -> None:
    """Delete a session's history."""
    db.messages.delete_many({"session_id": session_id})
    db.sessions.delete_one({"session_id": session_id})