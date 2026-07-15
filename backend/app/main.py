"""
FastAPI entrypoint.

Implements API endpoints from the SRS (Section: Functional Requirements):

    POST /api/chat                     -- 1. Chat Interface
    GET  /api/chat/{session_id}/history -- 10. Conversation History
    POST /api/admin/ingest             -- 11. Document Ingestion (Admin)

Still to add:
    GET  /api/admin/escalations
    GET  /api/admin/analytics

NOTE on /api/admin/ingest: the SRS shows a pure-JSON body
({"source": "s3://..."}). This implementation instead uses multipart form
fields (`files` and/or `source`) so the *same* endpoint can also accept
direct file uploads for local testing without needing an S3 bucket. If you
want to match the SRS body shape exactly for the S3-only case, add a
second JSON-only route that just calls ingest_from_s3().

NOTE on auth: this endpoint has NO auth/role check yet. SRS Section:
Security -> Permissions -> Admin requires this to be admin-only. Do not
expose this route publicly until that's added -- tracked as a follow-up.
"""
from dotenv import load_dotenv
load_dotenv()  # reads .env in the project root and sets env vars from it

from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from app.graph import compiled_graph
from app.session_store import get_history, get_recent_history, append_message
from app.ingest import ingest_files, ingest_from_s3

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


@app.post("/api/admin/ingest")
async def admin_ingest(
    files: Optional[List[UploadFile]] = File(default=None),
    source: Optional[str] = Form(default=None),
):
    """
    Document Ingestion (SRS Section: Functional Requirements -> 11).

    Provide exactly one of:
      - `files`: one or more uploaded PDF/HTML/DOCX files (multipart)
      - `source`: an S3 path, e.g. "s3://company-docs/policies/"

    Returns a summary of what was ingested: document count, chunk count,
    and the list of source filenames/keys.
    """
    if not files and not source:
        raise HTTPException(400, "Provide either 'files' or a 'source' S3 path.")
    if files and source:
        raise HTTPException(400, "Provide either 'files' or 'source', not both.")

    try:
        if source:
            if not source.startswith("s3://"):
                raise HTTPException(400, "'source' must be an s3:// path.")
            result = ingest_from_s3(source)
        else:
            file_bytes = [(f.filename, await f.read()) for f in files]
            result = ingest_files(file_bytes)
    except ValueError as e:
        # Unsupported file type, empty S3 prefix, malformed path, etc. --
        # these are client errors (400), not server failures (500).
        raise HTTPException(400, str(e))

    return {"status": "ok", **result}