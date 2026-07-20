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
from urllib.parse import urlparse
   
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.document_parsers import parse_file 
from app.chunking import chunk_text 

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
# real documentation only -- ingest the real doc(s) via ingest_files() or
# ingest_from_s3(), never a synthetic bootstrap set, so retrieval never
# competes against or gets diluted by unrelated placeholder content.

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
    """Chunk + register a batch of docs directly into the index."""
    added = 0
    nodes = []
    
    for doc in docs:
        for chunk in chunk_text(doc["text"]):
            node = TextNode(
                text=chunk,
                metadata={"source": doc["id"]}
            )
            nodes.append(node)
            added += 1
            
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

def ingest_from_s3(s3_path: str) -> dict:
    """Ingest every object under an s3://bucket/prefix path."""
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
            if key.endswith("/"): 
                continue
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            files.append((key, body))

    if not files:
        raise ValueError(f"No objects found under {s3_path}")

    return ingest_files(files)

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