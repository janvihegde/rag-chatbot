"""
Prompt Assembly + Answer Generation
(SRS Section: Functional Requirements -> 6. Answer Generation, 7. Citation).

STEP 5 of the build.

Two code paths, chosen automatically based on whether MISTRAL_API_KEY is
set in the environment:

  - REAL PATH (key present): assembles a grounded prompt from the
    reranked chunks and calls Mistral's chat completion API. This is
    what the SRS actually specifies.
  - FALLBACK PATH (no key, e.g. today): produces an extractive answer
    -- the most relevant sentence(s) straight from the top chunks --
    with the same citation structure. Not as fluent as an LLM answer,
    but fully offline, real (not a stub string), and grounded only in
    retrieved text, matching the SRS's core anti-hallucination goal:
    "the model must answer using ONLY the provided context."

Set the key once you have one:
    PowerShell:  $env:MISTRAL_API_KEY = "your-key-here"
No other code changes needed -- generate_answer() picks the path itself.
"""
import os
import re
from app.graph_state import ChatState
from app.relevance_gate import RERANK_GATE_THRESHOLD

_SYSTEM_PROMPT = (
    "You are a customer support assistant. Answer the user's question using "
    "ONLY the provided context below. If the context does not fully answer "
    "the question, say so -- do not invent information. Do not add your own "
    "source citations or a 'Source:' line -- citations are appended "
    "automatically after your answer, so just answer the question directly."
)


def _assemble_context(chunks: list) -> str:
    """Format reranked chunks into a numbered context block for the prompt."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] Source: {c['source']}\n{c['text']}")
    return "\n\n".join(lines)


def _call_mistral(query: str, chunks: list, history: list = None) -> str:
    from mistralai.client import Mistral

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    context = _assemble_context(chunks)

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    # Prior turns give the model the conversational context it needs for
    # follow-ups like "what about the other one?" -- without this, every
    # message would be answered as if it were the first one in the chat.
    for turn in (history or []):
        # Mistral's roles are "user"/"assistant", matching our stored roles.
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append(
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    )

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=messages,
    )
    return response.choices[0].message.content


def _extractive_fallback(query: str, chunks: list) -> str:
    """
    No LLM available: pick the most relevant sentence(s) from the top
    chunk(s) via simple keyword overlap, and present them directly.
    Grounded-by-construction since we're literally quoting the source doc,
    not generating free text.
    """
    query_words = set(re.findall(r"\w+", query.lower()))

    best_sentences = []
    for c in chunks[:2]:  # top 2 reranked chunks
        sentences = re.split(r"(?<=[.!?])\s+", c["text"])
        scored = []
        for s in sentences:
            s_words = set(re.findall(r"\w+", s.lower()))
            overlap = len(query_words & s_words)
            if overlap > 0:
                scored.append((overlap, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored:
            best_sentences.append(scored[0][1])
        elif sentences:
            best_sentences.append(sentences[0])  # fall back to first sentence

    if not best_sentences:
        return "The documentation doesn't seem to directly address that."

    return " ".join(best_sentences)


def generate_answer(query: str, chunks: list, history: list = None):
    """
    Returns (answer_text, citations) where citations is a list of source
    filenames actually used, per SRS Section 7 ("every factual claim
    references its source").

    Only chunks that individually clear RERANK_GATE_THRESHOLD are passed
    to the model / cited. `chunks` may include lower-scoring runner-up
    matches (the reranker returns up to RERANK_TOP_N regardless of
    quality) -- without this filter, a barely-relevant chunk that merely
    helped the *overall* gate pass would still get blindly cited, even
    though the answer doesn't actually rely on it. Example: a 2FA
    question correctly answered from account-security.pdf was previously
    also citing refund-policy.pdf just because it was chunk #2.

    `history`: prior turns for this session (Step 7), used only by the
    Mistral path so follow-up questions have conversational context. The
    offline extractive fallback ignores it -- it isn't a conversational
    model, so stuffing history in wouldn't help it answer better.
    """
    relevant_chunks = [c for c in chunks if c.get("rerank_score", 0) >= RERANK_GATE_THRESHOLD]
    # Guard against the edge case where the gate passed on chunks[0] but
    # every chunk somehow filters out here -- fall back to just the top one
    # rather than generating with an empty context.
    if not relevant_chunks and chunks:
        relevant_chunks = chunks[:1]

    if os.environ.get("MISTRAL_API_KEY"):
        answer = _call_mistral(query, relevant_chunks, history=history)
    else:
        answer = _extractive_fallback(query, relevant_chunks)

    citations = [c["source"] for c in relevant_chunks]
    return answer, citations


def generation_node(state: ChatState) -> ChatState:
    chunks = state.get("reranked_chunks", [])
    answer, citations = generate_answer(
        state["message"], chunks, history=state.get("history", [])
    )

    source_list = ", ".join(citations)
    state["response_text"] = f"{answer}\n\n(Source: {source_list})"
    state["citations"] = citations
    state["escalated"] = False
    return state