"""
Session / chat history store
(SRS Section: Functional Requirements -> 1. Chat Interface,
 Non-Functional -> Data persistence for conversation history).

STEP 7 of the build: in-memory store (a plain dict), since there's no
database configured yet. This means history is lost on server restart --
acceptable for local development. Swap `_sessions` for a real DB
(Postgres, Redis, etc.) later; every other file only calls the four
functions below, so nothing else needs to change when that happens.
"""
from datetime import datetime, timezone

# session_id -> list of {"role": "user"|"assistant", "content": str, "timestamp": str}
_sessions: dict[str, list[dict]] = {}

# How many most-recent turns to feed back into generation as context.
# Kept small on purpose: more history = more prompt tokens = slower/costlier
# calls, and most support questions don't need deep history anyway.
MAX_HISTORY_TURNS = 6


def get_history(session_id: str) -> list[dict]:
    """Full history for a session, oldest first. Empty list if unseen."""
    return _sessions.get(session_id, [])


def get_recent_history(session_id: str, max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """Most recent `max_turns` messages (user+assistant combined), oldest first."""
    return get_history(session_id)[-max_turns:]


def append_message(session_id: str, role: str, content: str) -> None:
    _sessions.setdefault(session_id, []).append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)