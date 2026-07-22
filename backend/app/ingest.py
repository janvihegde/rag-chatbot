"""
Document ingestion & retrieval using LlamaIndex and BGE-M3.
(SRS Section: Functional Requirements -> 3 & 11)

Storage: QDRANT_URL env var picked up if set (point this at a real Qdrant
server/cluster for production); otherwise falls back to an on-disk local
Qdrant store at QDRANT_PATH (default ./qdrant_storage) so ingested docs
survive server restarts without needing a Qdrant server running at all.
Previously this used QdrantClient(":memory:"), which silently threw away
every ingested document on restart -- that's fixed by this change.

NOTE: on-disk local mode (no QDRANT_URL) holds an exclusive file lock, so
only one process can have it open at a time -- fine for a single API
process, but don't point two uvicorn workers at the same QDRANT_PATH.
"""
import os
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.document_parsers import parse_file
from app.chunking import chunk_text
from app.db import db

COLLECTION_NAME = "company_docs"

EMBED_MODEL_NAME = "BAAI/bge-m3"
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrant_storage")

# BGE-M3 (1024-dim), per the SRS. Loaded once at import time and cached
# locally by HuggingFace after the first run (~2GB download).
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
# NOTE: the generic e-commerce SAMPLE_DOCS placeholder set (refund policy,
# shipping policy, etc.) has been removed. This bot is scoped to Truelift's
# real documentation only -- ingest the real doc(s) via ingest_files(),
# never a synthetic bootstrap set, so retrieval never competes against or
# gets diluted by unrelated placeholder content.

# Initialize Qdrant and LlamaIndex Vector Store.
# Remote server (production) if QDRANT_URL is set; otherwise persistent
# on-disk local storage so data survives restarts in dev/single-instance
# deployments without requiring a Qdrant server at all.
if QDRANT_URL:
    _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    _client = QdrantClient(path=QDRANT_PATH)
_vector_store = QdrantVectorStore(client=_client, collection_name=COLLECTION_NAME)
_storage_context = StorageContext.from_defaults(vector_store=_vector_store)

# Create an empty index (LlamaIndex handles embedding the nodes dynamically)
_index = VectorStoreIndex(
    nodes=[], 
    storage_context=_storage_context
)

def _add_documents(docs: list[dict]) -> int:
    """
    Chunk + register a batch of docs directly into the index. Also
    records/updates each doc's tracking entry in Mongo (db.documents) so
    the admin panel can list what's been ingested and delete it later --
    Qdrant itself has no natural "list distinct source documents" query,
    so this tracking collection is the source of truth for that.

    NOTE: re-ingesting a filename that's already been ingested ADDS new
    chunks alongside the old ones rather than replacing them -- this
    upserts the Mongo tracking record's chunk_count to reflect only the
    latest ingest call, which would then under-count the real number of
    chunks for that source sitting in Qdrant. Use delete_document() first
    if you want to fully replace a previously-ingested file.
    """
    added = 0
    nodes = []

    for doc in docs:
        doc_chunks = chunk_text(doc["text"])
        for chunk in doc_chunks:
            nodes.append(TextNode(text=chunk, metadata={"source": doc["id"]}))
        added += len(doc_chunks)

        db.documents.update_one(
            {"source": doc["id"]},
            {"$set": {
                "source": doc["id"],
                "chunk_count": len(doc_chunks),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }},
            upsert=True,
        )

    if nodes:
        _index.insert_nodes(nodes)

    return added

def ingest_documents(docs: list[dict]) -> int:
    """Ingest a batch of docs (list of {"id": ..., "text": ...}). No default
    sample-doc fallback -- caller must always provide the real docs to
    ingest, so an empty/misconfigured index never silently gets populated
    with placeholder content."""
    return _add_documents(docs)

def ingest_files(files: list[tuple[str, bytes]]) -> dict:
    """Ingest uploaded files."""
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

def retrieve(query: str, top_k: int = 20):
    """Retrieves chunks using LlamaIndex, mapping back to the expected dict format."""
    retriever = _index.as_retriever(similarity_top_k=top_k)
    query_result = retriever.retrieve(query)

    results = []
    for node in query_result:
        results.append({
            "source": node.metadata["source"],
            "text": node.text,
            "score": node.score
        })
    return results


def list_documents() -> list[dict]:
    """All ingested documents, most recently ingested first."""
    cursor = db.documents.find({}).sort("ingested_at", -1)
    return [
        {
            "source": d["source"],
            "chunk_count": d["chunk_count"],
            "ingested_at": d["ingested_at"],
        }
        for d in cursor
    ]


def delete_document(source: str) -> bool:
    """
    Removes every chunk belonging to `source` from Qdrant, and drops its
    tracking record from Mongo. Returns False if `source` wasn't a known
    ingested document (nothing to delete); True otherwise.

    NOTE: this filters Qdrant points by the "source" field, the same
    field already read back via node.metadata["source"] in retrieve()
    above -- so it's filtering on a field confirmed to actually exist on
    stored points. Still worth a live sanity check after deploying (e.g.
    delete a test document and confirm retrieve() no longer returns any
    of its chunks) since this hasn't been exercised against a real
    running Qdrant server, only against the stubbed client used in tests.
    """
    if db.documents.find_one({"source": source}) is None:
        return False

    _client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        ),
    )
    db.documents.delete_one({"source": source})
    return True