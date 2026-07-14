"""
LangGraph orchestration graph.

STEP 5 of the build:
    scope_check -> (in_scope) -> retrieval -> relevance_gate
                       -> (gate passed)  -> generation   (REAL, Step 5)
                       -> (gate failed)  -> escalation
                 -> (off_topic/injection) -> scope_reject

Generation uses Mistral if MISTRAL_API_KEY is set, otherwise an
offline extractive fallback -- see app/generation.py.
"""
from langgraph.graph import StateGraph, END
from app.graph_state import ChatState
from app.scope_check import scope_check_node, scope_reject_node
from app.retrieval import retrieval_node
from app.relevance_gate import relevance_gate_node
from app.escalation import escalation_node
from app.generation import generation_node


def route_after_scope_check(state: ChatState) -> str:
    return "in_scope" if state.get("scope_label") == "in_scope" else "reject"


def route_after_relevance_gate(state: ChatState) -> str:
    return "answer" if state.get("relevance_gate_passed") else "escalate"


def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("scope_check", scope_check_node)
    graph.add_node("scope_reject", scope_reject_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("relevance_gate", relevance_gate_node)
    graph.add_node("generation", generation_node)
    graph.add_node("escalation", escalation_node)

    graph.set_entry_point("scope_check")

    graph.add_conditional_edges(
        "scope_check",
        route_after_scope_check,
        {
            "in_scope": "retrieval",
            "reject": "scope_reject",
        },
    )

    graph.add_edge("retrieval", "relevance_gate")

    graph.add_conditional_edges(
        "relevance_gate",
        route_after_relevance_gate,
        {
            "answer": "generation",
            "escalate": "escalation",
        },
    )

    graph.add_edge("generation", END)
    graph.add_edge("escalation", END)
    graph.add_edge("scope_reject", END)

    return graph.compile()


# Compiled once at import time; reused across requests.
compiled_graph = build_graph()