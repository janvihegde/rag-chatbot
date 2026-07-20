"""
Relevance Gating + Reranking
(SRS Section: Functional Requirements -> 4. Relevance Gating, 5. Reranking).

STEP 4 of the build. Redesigned after live calibration -- see "WHY THIS
CHANGED" below.

Three-tier check:
    1. Cheap gate    -> raw Qdrant/BGE-M3 similarity score -> reject
                        immediately if there's clearly no match at all
    2. Confident tier -> raw score is high enough to trust retrieval
                        directly -> pass without requiring the
                        cross-encoder's approval
    3. Ambiguous tier -> raw score passed the cheap gate but isn't high
                        enough to trust on its own -> defer to the
                        cross-encoder as the tiebreaker

Local reranker model: cross-encoder/ms-marco-MiniLM-L-6-v2 (via
sentence-transformers). Downloaded once from HuggingFace on first run,
then cached locally -- no API key needed. Swap for BGE-reranker-v2 later
by changing only _MODEL_NAME below; everything downstream (rerank(), the
graph nodes) stays the same.

WHY THIS CHANGED (from a pure "cross-encoder is the only real gate"
design): live testing against the real Truelift doc showed the raw
BGE-M3 retrieval score was a consistently reliable signal --
genuinely-correct chunks scored 0.42-0.55 raw, every time, across both
broad and specific queries. The cross-encoder, by contrast, scored those
SAME correct chunks anywhere from -11.3 to -3.6 raw logit (sigmoid
~0.0000-0.18) -- i.e. it was NOT reliably discriminating good matches
from bad ones on this corpus at all. ms-marco-MiniLM-L-6-v2 is trained on
short, keyword-overlapping web-search passages; it under-scores relevant
content when the query and the source's formal/paraphrased prose don't
share much vocabulary, even when the content is exactly right. Requiring
it to independently clear a positive threshold was rejecting correct
answers by design, not by bug.

The fix: trust a strongly confident raw retrieval score directly, and
only ask the cross-encoder to adjudicate the genuinely ambiguous middle
ground where retrieval itself isn't sure. This uses retrieval for what
it's proven reliable at, and the cross-encoder for what it's actually
suited to -- breaking near-ties -- rather than treating it as the final
word on content it wasn't trained to judge well.
"""
from sentence_transformers import CrossEncoder
from app.graph_state import ChatState
import math

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Tier 1: cheap gate ---
# Below this raw retrieval score, there's clearly no match at all -- reject
# immediately, no reranking needed. Confirmed via live testing: true
# no-match/garbage queries scored well under 0.001 raw, consistently.
CHEAP_GATE_MIN_SCORE = 0.001

# --- Tier 2: confident-retrieval threshold ---
# At or above this raw retrieval score, trust retrieval directly and pass
# the gate without requiring cross-encoder approval. Set from live
# evidence: genuinely correct chunks against the real Truelift doc
# consistently scored 0.42-0.55 raw; 0.35 sits comfortably below that
# observed floor while staying well above near-zero garbage scores.
#
# NOT YET VALIDATED: we have real examples of "genuinely correct" and
# "genuinely no match at all" raw scores, but no confirmed example yet of
# a topically-similar-but-actually-wrong chunk landing in the 0.35-0.55
# band (a false positive this tier would incorrectly wave through). If
# that starts happening in practice, tighten this number and/or lean on
# NEGATIVE_LOGIT_VETO below.
RAW_SCORE_CONFIDENT_THRESHOLD = 0.35

# --- Tier 2 safety valve ---
# Even a confidently-high raw score gets overridden if the cross-encoder's
# RAW logit (pre-sigmoid) is catastrophically negative -- a defensive
# floor, not a calibrated one. Live data showed even correct matches
# reaching -11.3, so this is set well below that to avoid vetoing real
# answers; it exists only to catch a genuinely bizarre mismatch, not to
# do real discrimination (that's the cheap gate + Tier 2 threshold's job).
NEGATIVE_LOGIT_VETO = -15.0

# --- Tier 3: ambiguous-zone threshold ---
# Below RAW_SCORE_CONFIDENT_THRESHOLD but above CHEAP_GATE_MIN_SCORE:
# retrieval isn't confident enough to trust alone, so the cross-encoder's
# sigmoid score becomes the deciding vote. Matches the SRS default
# ("Relevance score threshold (reranker): Tunable, default 0.30") but
# note per the finding above, this tier is now a secondary path, not the
# primary gate -- most confidently-correct answers should pass via Tier 2
# without ever needing to clear this bar.
RERANK_GATE_THRESHOLD = 0.20

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
    by cross-encoder score desc, each with two added fields:
        rerank_score:     sigmoid(raw_logit), in 0-1, for threshold comparisons
        rerank_raw_logit: the model's raw unbounded output, for the Tier 2
                           safety-valve check (which needs the true logit,
                           not the sigmoid-compressed version)
    Truncated to top_n.
    """
    if not chunks:
        return []

    model = _get_model()
    pairs = [(query, c["text"]) for c in chunks]
    raw_scores = model.predict(pairs)

    scored = [
        {
            **chunk,
            "rerank_score": 1.0 / (1.0 + math.exp(-float(score))),
            "rerank_raw_logit": float(score),
        }
        for chunk, score in zip(chunks, raw_scores)
    ]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_n]


def relevance_gate_node(state: ChatState) -> ChatState:
    """
    Runs the three-tier check and stores the outcome on state:
        relevance_gate_passed: bool
        relevance_score:       float (best available score, for logging/escalation)
        reranked_chunks:       list  (populated whenever the cross-encoder ran)
        relevance_gate_tier:   str   ("cheap_reject" | "confident_retrieval" |
                                       "reranker_pass" | "reranker_reject")
                                      -- for debugging/analytics, so you can
                                      see WHICH tier decided each response.
    """
    chunks = state.get("retrieved_chunks", [])

    # --- Tier 1: cheap gate ---
    top_raw_score = chunks[0]["score"] if chunks else 0.0
    if not chunks or top_raw_score < CHEAP_GATE_MIN_SCORE:
        state["relevance_gate_passed"] = False
        state["relevance_score"] = top_raw_score
        state["reranked_chunks"] = []
        state["relevance_gate_tier"] = "cheap_reject"
        return state

    # Always rerank once we're past the cheap gate -- reranked_chunks feed
    # generation.py's citation filtering regardless of which tier decides
    # pass/fail, and Tier 2's safety valve needs the cross-encoder's logit
    # on the top retrieval match.
    reranked = rerank(state["message"], chunks, top_n=RERANK_TOP_N)
    state["reranked_chunks"] = reranked

    # --- Tier 2: confident retrieval ---
    if top_raw_score >= RAW_SCORE_CONFIDENT_THRESHOLD:
        # Find this specific top-retrieved chunk's cross-encoder logit for
        # the safety-valve check (reranked list is sorted by rerank score,
        # not retrieval score, so it may not be reranked[0]).
        top_chunk_source = chunks[0].get("source")
        top_chunk_text = chunks[0].get("text")
        matching = next(
            (
                c for c in reranked
                if c.get("source") == top_chunk_source and c.get("text") == top_chunk_text
            ),
            None,
        )
        top_chunk_logit = matching["rerank_raw_logit"] if matching else 0.0

        if top_chunk_logit >= NEGATIVE_LOGIT_VETO:
            state["relevance_gate_passed"] = True
            state["relevance_score"] = top_raw_score
            state["relevance_gate_tier"] = "confident_retrieval"
            return state
        # else: catastrophic cross-encoder disagreement -- fall through to
        # Tier 3 and let the reranker's own scoring decide instead.

    # --- Tier 3: ambiguous zone -- cross-encoder is the deciding vote ---
    top_rerank_score = reranked[0]["rerank_score"] if reranked else -1.0
    state["relevance_score"] = top_rerank_score
    state["relevance_gate_passed"] = top_rerank_score >= RERANK_GATE_THRESHOLD
    state["relevance_gate_tier"] = (
        "reranker_pass" if state["relevance_gate_passed"] else "reranker_reject"
    )

    return state