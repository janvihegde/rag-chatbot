"""
Document ingestion & retrieval (SRS Section: Functional Requirements ->
3. Document Retrieval, and 11. Document Ingestion (Admin)).

STEP 7 upgrade: uses app/embeddings.py (Mistral real embeddings, or
TF-IDF fallback offline) instead of raw TF-IDF directly. Qdrant stays
in-memory (no Docker/cloud needed) regardless of which embedder is active.

For now, `SAMPLE_DOCS` stands in for real company docs. Step 8
(Document Ingestion Admin API) will let you POST real docs instead.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.embeddings import embed_documents, embed_query

COLLECTION_NAME = "company_docs"

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

_vector_size = None  # set once ingest_documents() runs

# ":memory:" -> pure in-memory Qdrant, no server/Docker required.
_client = QdrantClient(":memory:")


def _ensure_collection(vector_size: int):
    if not _client.collection_exists(COLLECTION_NAME):
        _client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def ingest_documents(docs=None):
    """Embed (Mistral or TF-IDF fallback) and upsert docs into Qdrant."""
    global _vector_size
    docs = docs if docs is not None else SAMPLE_DOCS

    texts = [d["text"] for d in docs]
    vectors = embed_documents(texts)
    _vector_size = len(vectors[0])

    _ensure_collection(_vector_size)

    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload={"source": docs[i]["id"], "text": docs[i]["text"]},
        )
        for i in range(len(docs))
    ]
    _client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def retrieve(query: str, top_k: int = 20):
    """Returns list of {"source", "text", "score"} ordered by score desc."""
    if _vector_size is None:
        ingest_documents()

    query_vector = embed_query(query)

    hits = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    return [
        {
            "source": hit.payload["source"],
            "text": hit.payload["text"],
            "score": hit.score,
        }
        for hit in hits
    ]


ingest_documents()