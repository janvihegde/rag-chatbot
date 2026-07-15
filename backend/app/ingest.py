"""
Document ingestion & retrieval (SRS Section: Functional Requirements ->
3. Document Retrieval, and 11. Document Ingestion (Admin)).

STEP 8 upgrade: real document ingestion. Docs can now come from:
  - direct file upload (PDF / HTML / DOCX) via POST /api/admin/ingest
  - an S3 path (s3://bucket/prefix), matching the SRS API example

Each document is parsed to plain text (app/document_parsers.py), split into
overlapping chunks (app/chunking.py), embedded (app/embeddings.py), and
upserted into Qdrant -- one vector per CHUNK, not per whole document, so
retrieval can return the specific paragraph that answers a question rather
than an entire policy PDF.

IMPORTANT -- why we re-embed everything on every ingest call:
When MISTRAL_API_KEY is not set, app/embeddings.py falls back to TF-IDF.
TF-IDF's vector space is defined by whatever corpus it was last fit on. If
we naively called embed_documents() on only the *new* batch each time,
every additional ingest call would silently invalidate every previously
ingested vector (different vocabulary -> different vector space -> old
stored vectors no longer comparable to a fresh query). To keep retrieval
correct regardless of which embedder is active, we keep every chunk ever
ingested in memory (_all_chunks) and do a full re-embed + re-upsert of the
whole corpus on every ingest call. This is O(total chunks) per ingest,
which is fine for a support KB (hundreds to low thousands of chunks) but
would need a smarter incremental strategy at much larger scale. This
becomes a non-issue once this fully switches to Mistral embeddings, since
those don't require refitting.
"""
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.embeddings import embed_documents, embed_query
from app.document_parsers import parse_file
from app.chunking import chunk_text

COLLECTION_NAME = "company_docs"

# Bootstrap/demo content -- used only if retrieve() is called before any
# real ingestion has happened (e.g. fresh local dev environment).
SAMPLE_DOCS = [
    {
        "id": "refund-policy.pdf",
        "text": (
            "Annual subscriptions can be refunded within 30 days of purchase. "
            "Monthly subscriptions are non-refundable once the billing cycle "
            "has started. Refund requests must be submitted through the "
            "account billing page."
        ),
    },
    {
        "id": "shipping-policy.pdf",
        "text": (
            "Standard shipping takes 5-7 business days within the country. "
            "Express shipping takes 1-2 business days and is available at "
            "checkout for an additional fee. We do not currently ship "
            "internationally."
        ),
    },
    {
        "id": "account-security.pdf",
        "text": (
            "Users can reset their password from the login page by clicking "
            "'Forgot password'. Two-factor authentication can be enabled in "
            "Account Settings > Security. We never ask for your password by "
            "email or chat."
        ),
    },
    {
        "id": "product-faq.pdf",
        "text": (
            "Our product supports export to CSV and PDF from the reports "
            "page. Team plans allow up to 10 seats; enterprise plans support "
            "unlimited seats with SSO. All plans include email support; "
            "priority chat support is available on Team and Enterprise."
        ),
    },
]

# Every chunk ever ingested, so the TF-IDF fallback can be consistently
# re-fit on the full corpus each time (see module docstring).
# Each item: {"source": str, "text": str}
_all_chunks: list[dict] = []

_vector_size = None  # set once the first embed happens

# ":memory:" -> pure in-memory Qdrant, no server/Docker required.
_client = QdrantClient(":memory:")


def _recreate_collection(vector_size: int):
    if _client.collection_exists(COLLECTION_NAME):
        _client.delete_collection(COLLECTION_NAME)
    _client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def _reindex_all():
    """Re-embed and re-upsert every chunk ever ingested. See module docstring."""
    global _vector_size
    if not _all_chunks:
        return

    texts = [c["text"] for c in _all_chunks]
    vectors = embed_documents(texts)
    _vector_size = len(vectors[0])

    _recreate_collection(_vector_size)

    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "source": _all_chunks[i]["source"],
                "text": _all_chunks[i]["text"],
            },
        )
        for i in range(len(_all_chunks))
    ]
    _client.upsert(collection_name=COLLECTION_NAME, points=points)


def _add_documents(docs: list[dict]) -> int:
    """
    Chunk + register a batch of {"id": source_name, "text": full_text} docs,
    then reindex the whole corpus. Returns the number of chunks added.
    """
    added = 0
    for doc in docs:
        for chunk in chunk_text(doc["text"]):
            _all_chunks.append({"source": doc["id"], "text": chunk})
            added += 1
    _reindex_all()
    return added


def ingest_documents(docs=None) -> int:
    """Bootstrap/demo path -- ingests SAMPLE_DOCS unless other docs given."""
    docs = docs if docs is not None else SAMPLE_DOCS
    return _add_documents(docs)


def ingest_files(files: list[tuple[str, bytes]]) -> dict:
    """
    Ingest uploaded files.

    `files` is a list of (filename, raw_bytes) pairs -- kept as plain tuples
    rather than FastAPI's UploadFile so this function has no web-framework
    dependency and is easy to unit test / call from a script.

    Returns {"documents": N, "chunks": M, "sources": [filenames]}.
    Files with unsupported extensions raise ValueError from parse_file();
    the caller (the API route) is responsible for turning that into an
    HTTP error response.
    """
    docs = []
    for filename, content in files:
        text = parse_file(filename, content)
        if text.strip():
            docs.append({"id": filename, "text": text})

    chunks_added = _add_documents(docs)
    return {
        "documents": len(docs),
        "chunks": chunks_added,
        "sources": [d["id"] for d in docs],
    }


def ingest_from_s3(s3_path: str) -> dict:
    """
    Ingest every object under an s3://bucket/prefix path, per the SRS
    example (POST /api/admin/ingest {"source": "s3://company-docs/policies/"}).

    Uses boto3's default credential chain (env vars / instance role /
    ~/.aws/config) -- no credentials are handled in this code. boto3 is
    imported lazily so it isn't a hard dependency for deployments that only
    ever use direct file-upload ingestion.
    """
    import boto3

    parsed = urlparse(s3_path)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected an s3:// path, got: {s3_path}")
    bucket, prefix = parsed.netloc, parsed.path.lstrip("/")

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):  # skip "folder" placeholder objects
                continue
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            files.append((key, body))

    if not files:
        raise ValueError(f"No objects found under {s3_path}")

    return ingest_files(files)


def retrieve(query: str, top_k: int = 20):
    """Returns list of {"source", "text", "score"} ordered by score desc."""
    if _vector_size is None:
        ingest_documents()  # bootstrap with SAMPLE_DOCS if nothing ingested yet

    query_vector = embed_query(query)

    hits = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    return [
        {"source": hit.payload["source"], "text": hit.payload["text"], "score": hit.score}
        for hit in hits
    ]