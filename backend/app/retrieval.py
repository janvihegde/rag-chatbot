# backend/app/retrieval.py (No changes needed, but verifying structure)
from app.graph_state import ChatState
from app.ingest import retrieve

def retrieval_node(state: ChatState) -> ChatState:
    # This now utilizes LlamaIndex under the hood
    chunks = retrieve(state["message"], top_k=20)
    state["retrieved_chunks"] = chunks
    return state