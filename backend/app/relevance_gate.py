"""
Relevance Gating + Reranking
(SRS Section: Functional Requirements -> 4. Relevance Gating, 5. Reranking).

STEP 4 of the build.

Two-stage check, per the SRS table:
    Cheap gate  -> raw Qdrant similarity score -> skip reranking when
                   there's clearly no match
    Real gate   -> cross-encoder score         -> final decision: generate
                   or escalate

Local model: cross-encoder/ms-marco-MiniLM-L-6-v2 (via sentence-transformers).
Downloaded once from HuggingFace on first run, then cached locally --
no API key needed. Swap for BGE-reranker-v2 later by changing only
_MODEL_NAME below; everything downstream (rerank(), the graph nodes)
stays the same.
"""
from sentence_transformers import CrossEncoder
from app.graph_state import ChatState
import math

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cheap-gate threshold: below this raw retrieval score, don't even bother
# calling the (more expensive) cross-encoder -- there's clearly no match.
# TF-IDF cosine scores are typically small (see Step 3 test: 0.13-0.38 for
# genuine matches), so this is set conservatively low.
CHEAP_GATE_MIN_SCORE = 0.05

# Real-gate threshold on the cross-encoder score. Matches the SRS default
# ("Relevance score threshold (reranker): Tunable, default 0.30").
RERANK_GATE_THRESHOLD = 0.30

# Reranker output size, per SRS Section 5 ("top 3-8 chunks").
RERANK_TOP_N = 5

_model = None  # lazy-loaded so import doesn't trigger a download by itself


def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(query: str, chunks: list, top_n: int = RERANK_TOP_N):
    """
    Re-score chunks against the query with a cross-encoder.
    Returns chunks (list of dicts, same shape as retrieval output) sorted
    by cross-encoder score desc, each with an added "rerank_score" field.
    Truncated to top_n.

    NOTE: ms-marco-MiniLM-L-6-v2 outputs a raw, unbounded logit (roughly
    -10 to +10), not a 0-1 probability. We apply a sigmoid so
    RERANK_GATE_THRESHOLD (a 0-1 value) actually means what it says.
    Without this, a mildly-relevant-but-not-great match (e.g. a raw score
    of -0.3, which sigmoid maps to ~0.43) would incorrectly look like a
    strong negative signal instead of a borderline-positive one.
    """
    if not chunks:
        return []

    model = _get_model()
    pairs = [(query, c["text"]) for c in chunks]
    raw_scores = model.predict(pairs)

    scored = [
        {**chunk, "rerank_score": 1.0 / (1.0 + math.exp(-float(score)))}
        for chunk, score in zip(chunks, raw_scores)
    ]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_n]


def relevance_gate_node(state: ChatState) -> ChatState:
    """
    Runs both gates and stores the outcome on state:
        relevance_gate_passed: bool
        relevance_score:       float (best available score, for logging/escalation)
        reranked_chunks:       list  (only populated if the real gate ran)
    """
    chunks = state.get("retrieved_chunks", [])

    # --- Cheap gate ---
    top_raw_score = chunks[0]["score"] if chunks else 0.0
    if not chunks or top_raw_score < CHEAP_GATE_MIN_SCORE:
        state["relevance_gate_passed"] = False
        state["relevance_score"] = top_raw_score
        state["reranked_chunks"] = []
        return state

    # --- Real gate: cross-encoder rerank ---
    reranked = rerank(state["message"], chunks, top_n=RERANK_TOP_N)
    state["reranked_chunks"] = reranked

    top_rerank_score = reranked[0]["rerank_score"] if reranked else -1.0
    state["relevance_score"] = top_rerank_score
    state["relevance_gate_passed"] = top_rerank_score >= RERANK_GATE_THRESHOLD

    return state