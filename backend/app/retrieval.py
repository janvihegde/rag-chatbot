"""
Retrieval LangGraph node (SRS Section: Functional Requirements ->
3. Document Retrieval).
"""
from app.graph_state import ChatState
from app.ingest import retrieve


def retrieval_node(state: ChatState) -> ChatState:
    chunks = retrieve(state["message"], top_k=20)
    state["retrieved_chunks"] = chunks
    return state