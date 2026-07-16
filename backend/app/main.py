"""
FastAPI entrypoint.

Implements API endpoints from the SRS (Section: Functional Requirements):

    POST /api/chat                     -- 1. Chat Interface (Streaming via SSE)
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
import json
from dotenv import load_dotenv
load_dotenv()  # reads .env in the project root and sets env vars from it

from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.graph import compiled_graph
from app.session_store import get_history, get_recent_history, append_message
from app.ingest import ingest_files, ingest_from_s3
from fastapi import Depends
from app.auth import verify_admin

app = FastAPI(title="RAG Customer Support Chatbot", version="0.1.0")


class ChatRequest(BaseModel):
    session_id: str
    message: str


# Note: ChatResponse is kept for reference or if you need a standard fallback endpoint,
# but the streaming endpoint yields dicts directly instead of a Pydantic model.
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


@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    Yields intermediate node updates, followed by the final result.
    """
    history = get_recent_history(req.session_id)

    async def event_generator():
        final_state = {}
        
        # compiled_graph.stream yields state updates as each node finishes
        for update in compiled_graph.stream(
            {"session_id": req.session_id, "message": req.message, "history": history},
            stream_mode="updates"
        ):
            for node_name, node_state in update.items():
                final_state.update(node_state)
                
                # Yield an event notifying the frontend which node just completed
                yield f"data: {json.dumps({'event': 'node_update', 'node': node_name})}\n\n"

        # Once the graph has fully traversed, prepare the final payload
        response_payload = {
            "event": "final_result",
            "session_id": req.session_id,
            "response": final_state.get("response_text", ""),
            "citations": final_state.get("citations", []),
            "escalated": final_state.get("escalated", False),
            "debug_scope_label": final_state.get("scope_label"),
            "debug_relevance_score": final_state.get("relevance_score"),
        }

        # Persist this turn AFTER the graph runs
        append_message(req.session_id, "user", req.message)
        append_message(req.session_id, "assistant", final_state.get("response_text", ""))

        yield f"data: {json.dumps(response_payload)}\n\n"

    # Return the generator wrapped in a StreamingResponse with the standard SSE media type
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/chat/{session_id}/history")
def chat_history(session_id: str):
    return {"session_id": session_id, "history": get_history(session_id)}


@app.post("/api/admin/ingest", dependencies=[Depends(verify_admin)])
async def admin_ingest(
    files: Optional[List[UploadFile]] = File(default=None),
    source: Optional[str] = Form(default=None),
):
    """
    Document Ingestion (SRS Section: Functional Requirements -> 11).
    Secured via admin token authentication.
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
        raise HTTPException(400, str(e))

    return {"status": "ok", **result}


# Add these endpoints to your backend/app/main.py

@app.get("/api/admin/escalations", dependencies=[Depends(verify_admin)])
def get_escalations(status: Optional[str] = "pending"):
    """
    Returns a list of escalations. 
    Can be filtered by status (pending vs resolved).
    """
    query = {"status": status} if status else {}
    # Sort newest first
    cursor = db.escalations.find(query).sort("timestamp", -1).limit(100)
    
    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"]) # Convert ObjectId to string for JSON serialization
        results.append(doc)
        
    return {"escalations": results}

@app.get("/api/admin/analytics", dependencies=[Depends(verify_admin)])
def get_analytics():
    """
    Aggregates database metrics for the React Admin Dashboard (e.g., Recharts).
    """
    total_messages = db.messages.count_documents({})
    total_escalations = db.escalations.count_documents({})
    pending_escalations = db.escalations.count_documents({"status": "pending"})
    
    # Calculate global escalation rate
    user_queries = db.messages.count_documents({"role": "user"})
    escalation_rate = (total_escalations / user_queries * 100) if user_queries > 0 else 0

    return {
        "metrics": {
            "total_user_queries": user_queries,
            "total_escalations": total_escalations,
            "pending_escalations": pending_escalations,
            "escalation_rate_percent": round(escalation_rate, 2)
        }
    }