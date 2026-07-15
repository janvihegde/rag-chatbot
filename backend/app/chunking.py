"""
Text chunking for document ingestion (SRS Section: Functional Requirements ->
11. Document Ingestion (Admin)).

Fixed-size chunking, chosen for simplicity to get real ingestion working
end-to-end first. Splits on whitespace into overlapping word windows so a
sentence isn't left with zero surrounding context right at a chunk boundary.

Upgrade path: swap the body of chunk_text() for a semantic/recursive
splitter later (e.g. LangChain's RecursiveCharacterTextSplitter, or
sentence-boundary-aware chunking) without touching any caller -- every
caller only depends on this function's signature (str -> list[str]).
"""


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> list[str]:
    """
    Split `text` into overlapping chunks of up to `chunk_size` words each,
    sliding forward by (chunk_size - overlap) words per step.

    Defaults (220 words, 40 overlap) land around 800-1000 characters per
    chunk with ~18% overlap -- a reasonable starting point for short
    policy/FAQ-style documents. Tune per corpus if chunks come back too
    broad (dilutes the embedding) or too narrow (loses context).
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks