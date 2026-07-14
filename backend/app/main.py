"""
FastAPI entrypoint.

Implements the API contract from the SRS (Section: Functional Requirements
-> 1. Chat Interface):

    POST /api/chat
    { "session_id": "sess_12345", "message": "What is our refund policy?" }

Only this one endpoint exists so far. Later steps add:
    GET  /api/chat/{session_id}/history
    POST /api/admin/ingest
    GET  /api/admin/escalations
    GET  /api/admin/analytics
"""
from dotenv import load_dotenv
load_dotenv()  # reads .env in the project root and sets env vars from it

from fastapi import FastAPI
from pydantic import BaseModel

from app.graph import compiled_graph
from app.session_store import get_history, get_recent_history, append_message

app = FastAPI(title="RAG Customer Support Chatbot", version="0.1.0")


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    citations: list = []
    escalated: bool = False
    debug_scope_label: str | None = None
    debug_relevance_score: float | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = get_recent_history(req.session_id)

    result = compiled_graph.invoke(
        {"session_id": req.session_id, "message": req.message, "history": history}
    )

    # Persist this turn AFTER the graph runs, so the message the user just
    # sent doesn't leak into its own "prior history" during this same call.
    append_message(req.session_id, "user", req.message)
    append_message(req.session_id, "assistant", result.get("response_text", ""))

    return ChatResponse(
        session_id=req.session_id,
        response=result.get("response_text", ""),
        citations=result.get("citations", []),
        escalated=result.get("escalated", False),
        debug_scope_label=result.get("scope_label"),
        debug_relevance_score=result.get("relevance_score"),
    )


@app.get("/api/chat/{session_id}/history")
def chat_history(session_id: str):
    return {"session_id": session_id, "history": get_history(session_id)}