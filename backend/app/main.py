"""
FastAPI entrypoint.

Implements API endpoints:

    POST   /api/chat                       -- Chat (streaming via SSE)
    GET    /api/chat/{session_id}/history  -- Full message history for a session
    POST   /api/users/{user_id}/sessions   -- Start a new chat for this user
    GET    /api/users/{user_id}/sessions   -- List a user's past chats (for the
                                               "continue a previous chat or start
                                               new" picker)
    POST   /api/admin/ingest               -- Document ingestion (admin)
    GET    /api/admin/documents            -- List ingested documents (admin)
    DELETE /api/admin/documents/{source}   -- Remove an ingested document (admin)
    GET    /api/admin/users                -- List all users (admin)
    GET    /api/admin/users/{user_id}/sessions -- A user's chats (admin)
    GET    /api/admin/sessions/{session_id}/messages -- Full transcript (admin)

NOTE on identity: user_id is a persistent anonymous identifier the
frontend generates once and stores in localStorage -- NOT a real login.
This is intentional (per product decision): it's enough to recognize a
returning browser/device for a nicer UX (continue a previous chat vs
start fresh), but it is not a security boundary. Anyone who guessed or
was given a user_id/session_id string could read that history through
these APIs. Don't put anything sensitive in this chatbot's conversations
that wouldn't be fine under that assumption, and revisit this if real
user accounts become a requirement.

NOTE on escalation: there is no ticket/queue system. When the bot can't
help (off-topic, or a low-confidence retrieval), it just tells the user
to email pratham@truelift.ai directly -- see app/escalation.py and
app/scope_check.py. Admins wanting to know what a user asked can read
their chat history directly via the admin endpoints above.

NOTE on auth: all /api/admin/* routes require a bearer token matching
ADMIN_API_KEY (see app/auth.py's verify_admin dependency).

NOTE on concurrency: the chat endpoint bridges LangGraph's synchronous
`.stream()` (which does real model inference -- embeddings, reranking --
plus a network call to Mistral) onto a worker thread rather than running
it directly on the asyncio event loop. Without this, one user's request
would block every other concurrent user's request for the full duration
of that inference, regardless of how many people are chatting at once.
Blocking MongoDB calls are similarly moved off the event loop via
run_in_threadpool. See _stream_graph_events() below.
"""
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
load_dotenv()  # reads .env in the project root and sets env vars from it

from typing import List, Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.graph import compiled_graph
from app.session_store import (
    get_history,
    get_recent_history,
    append_message,
    create_session,
    list_sessions_for_user,
    list_users,
    ensure_session,
)
from app.ingest import ingest_files, list_documents, delete_document
from app.db import db, ensure_indexes
from app.auth import verify_admin


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Best-effort: don't crash app startup if Mongo is briefly unreachable
    # (e.g. container ordering during a fresh `docker compose up`).
    try:
        ensure_indexes()
    except Exception as e:
        print(f"[startup] Could not ensure MongoDB indexes: {e}")
    yield


app = FastAPI(title="RAG Customer Support Chatbot", version="0.1.0", lifespan=lifespan)

# CORS: without this, no browser-based frontend (dev or prod, any port/origin
# other than this API's own) can call these endpoints at all -- the browser
# blocks the request before it even reaches FastAPI. FRONTEND_ORIGINS is a
# comma-separated list (e.g. "http://localhost:5173,https://chat.truelift.ai");
# defaults to allowing any origin, which is fine for local dev but should be
# locked down to your real frontend's origin(s) before this is public.
_frontend_origins = os.environ.get("FRONTEND_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _frontend_origins == "*" else _frontend_origins.split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_id: Optional[str] = None


# Bridges LangGraph's synchronous, blocking .stream() (real model
# inference + a network call to Mistral) onto a worker thread, so the
# FastAPI event loop stays free to serve OTHER concurrent requests while
# this one's inference runs -- without this, every chat request would be
# fully serialized regardless of how many users are active at once.
_graph_executor = ThreadPoolExecutor(
    max_workers=int(os.environ.get("GRAPH_WORKER_THREADS", "4"))
)


async def _stream_graph_events(graph_input: dict):
    """Runs compiled_graph.stream() on a worker thread and yields its
    updates back into the async world as they arrive, preserving the
    incremental "node_update" events the frontend uses to show live
    pipeline stages."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    _SENTINEL = object()

    def _run():
        try:
            for update in compiled_graph.stream(graph_input, stream_mode="updates"):
                loop.call_soon_threadsafe(queue.put_nowait, update)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    loop.run_in_executor(_graph_executor, _run)

    while True:
        item = await queue.get()
        if item is _SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE). Yields
    intermediate node updates, followed by the final result.
    """
    history = await run_in_threadpool(get_recent_history, req.session_id)
    await run_in_threadpool(ensure_session, req.session_id, req.user_id)

    async def event_generator():
        final_state = {}

        graph_input = {
            "session_id": req.session_id,
            "message": req.message,
            "history": history,
        }
        async for update in _stream_graph_events(graph_input):
            for node_name, node_state in update.items():
                final_state.update(node_state)
                yield f"data: {json.dumps({'event': 'node_update', 'node': node_name})}\n\n"

        response_payload = {
            "event": "final_result",
            "session_id": req.session_id,
            "response": final_state.get("response_text", ""),
            "citations": final_state.get("citations", []),
            "escalated": final_state.get("escalated", False),
            "debug_scope_label": final_state.get("scope_label"),
            "debug_relevance_score": final_state.get("relevance_score"),
        }

        # Persist this turn AFTER the graph runs.
        await run_in_threadpool(append_message, req.session_id, "user", req.message, req.user_id)
        await run_in_threadpool(
            append_message, req.session_id, "assistant", final_state.get("response_text", ""), req.user_id
        )

        yield f"data: {json.dumps(response_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/chat/{session_id}/history")
def chat_history(session_id: str):
    return {"session_id": session_id, "history": get_history(session_id)}


@app.post("/api/users/{user_id}/sessions")
def start_new_session(user_id: str):
    """Start a new chat for this user -- the "New chat" action."""
    session_id = create_session(user_id)
    return {"session_id": session_id}


@app.get("/api/users/{user_id}/sessions")
def get_user_sessions(user_id: str):
    """
    A user's past chats, most recent first. An empty list means this is
    a brand-new user (the frontend should skip straight to a new chat);
    a non-empty list means the frontend should offer "continue a previous
    chat" (with this list) or "start new".
    """
    return {"sessions": list_sessions_for_user(user_id)}


@app.post("/api/admin/ingest", dependencies=[Depends(verify_admin)])
async def admin_ingest(files: List[UploadFile] = File(...)):
    """
    Document Ingestion. Secured via admin token authentication. Accepts
    one or more files directly (PDF, HTML, or DOCX -- see
    app/document_parsers.py).
    """
    try:
        file_bytes = [(f.filename, await f.read()) for f in files]
        result = await run_in_threadpool(ingest_files, file_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {"status": "ok", **result}


@app.get("/api/admin/documents", dependencies=[Depends(verify_admin)])
def admin_list_documents():
    return {"documents": list_documents()}


@app.delete("/api/admin/documents/{source}", dependencies=[Depends(verify_admin)])
def admin_delete_document(source: str):
    deleted = delete_document(source)
    if not deleted:
        raise HTTPException(404, f"No ingested document found with source '{source}'.")
    return {"status": "ok", "deleted": source}


@app.get("/api/admin/users", dependencies=[Depends(verify_admin)])
def admin_list_users():
    return {"users": list_users()}


@app.get("/api/admin/users/{user_id}/sessions", dependencies=[Depends(verify_admin)])
def admin_get_user_sessions(user_id: str):
    return {"sessions": list_sessions_for_user(user_id)}


@app.get("/api/admin/sessions/{session_id}/messages", dependencies=[Depends(verify_admin)])
def admin_get_session_messages(session_id: str):
    return {"session_id": session_id, "messages": get_history(session_id)}