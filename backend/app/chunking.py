"""
Text chunking for document ingestion (SRS Section: Functional Requirements ->
11. Document Ingestion (Admin)).

Sentence-boundary-aware chunking, sized to match the reranker's training
distribution.

WHY THIS CHANGED FROM FIXED-SIZE WORD WINDOWS:
The original implementation split on raw whitespace into 220-word windows
(~800-1000 chars), with no regard for sentence boundaries. That produced
chunks spanning multiple unrelated topics (e.g. half of "Core Features"
plus half of "Onboarding") and starting/ending mid-sentence. The
cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2) is trained
on MS MARCO passages -- short, single-topic, 1-3 sentence spans -- and
degrades sharply (often scoring genuinely relevant passages as strongly
NEGATIVE) when given long, multi-topic, mid-sentence-cut input far outside
that distribution. Real-world symptom: a chunk that was a perfect textual
match for a user's question still scored ~0.0001 after the sigmoid,
because the surrounding 300 words of unrelated content confused the model,
not because the content didn't match.

Upgrade path: swap for a smarter/semantic splitter later (e.g. LangChain's
RecursiveCharacterTextSplitter, or splitting on the document's own
"SECTION" headers first, then sentences within each section) without
touching any caller -- every caller only depends on this function's
signature (str -> list[str]).
"""
import re

# Match sentence-ending punctuation followed by whitespace, but avoid
# splitting on common abbreviations/decimals where possible. Simple and
# good enough for well-formed prose docs like policy/product PDFs; swap
# for a proper sentence tokenizer (e.g. nltk.sent_tokenize) if the corpus
# has heavier abbreviation use.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, collapsing internal whitespace/newlines
    within each sentence so PDF line-wrap artifacts don't leave stray
    linebreaks embedded in the middle of a sentence."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size: int = 70, overlap: int = 1) -> list[str]:
    """
    Split `text` into chunks of up to roughly `chunk_size` words each,
    built by accumulating whole sentences (never cutting mid-sentence),
    with `overlap` sentences carried over into the next chunk for context
    continuity across boundaries.

    Defaults (70 words) land around 350-450 characters per chunk -- close
    to the short, single-topic passage length the cross-encoder reranker
    was trained on (MS MARCO passages are typically 1-3 sentences). A
    single sentence longer than `chunk_size` on its own is kept whole
    rather than being truncated, since we never want to cut mid-sentence.

    `overlap` is a SENTENCE count here (not words, unlike the previous
    implementation) -- 1 sentence of carry-over is usually enough context
    continuity for short chunks; raise it if retrieval quality suggests
    boundary context is still being lost.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current: list[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        # Flush the current chunk before adding this sentence if doing so
        # would exceed chunk_size AND we already have content -- a lone
        # oversized sentence is still allowed to form its own chunk.
        if current and current_word_count + sentence_word_count > chunk_size:
            chunks.append(" ".join(current))
            # Carry over the last `overlap` sentences for continuity.
            current = current[-overlap:] if overlap > 0 else []
            current_word_count = sum(len(s.split()) for s in current)

        current.append(sentence)
        current_word_count += sentence_word_count

    if current:
        chunks.append(" ".join(current))

    return chunks