"""
Shared state object that flows through every node of the LangGraph pipeline.

Per the SRS (Section: System Architecture), the query pipeline is a single
stateful graph: scope check -> retrieve -> relevance gate -> rerank ->
generate OR escalate -> respond.

We define the full state now (even fields unused by today's single node)
so later steps only add nodes, not restructure the schema.
"""
from typing import Optional, Literal
from typing_extensions import TypedDict


class ChatState(TypedDict, total=False):
    # ---- input ----
    session_id: str
    message: str
    history: list  
    # ---- scope check (Step 3) ----
    scope_label: Optional[Literal["in_scope", "off_topic", "injection"]]

    # ---- retrieval (Step 4) ----
    retrieved_chunks: list  # raw candidates from Qdrant, with similarity scores

    # ---- relevance gating (Step 5) ----
    relevance_gate_passed: Optional[bool]
    relevance_score: Optional[float]

    # ---- reranking (Step 5) ----
    reranked_chunks: list  # top 3-8 chunks post cross-encoder

    # ---- generation / escalation (Step 6) ----
    answer: Optional[str]
    citations: list
    escalated: bool

    # ---- final response shape returned to the client ----
    response_text: Optional[str]
