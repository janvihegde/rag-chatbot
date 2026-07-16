"""
Root pytest conftest.

Two jobs:

1. Put `backend/` on sys.path so `import app...` resolves regardless of
   where pytest is invoked from.

2. Stub out the heavy external dependencies that app/ingest.py and
   app/relevance_gate.py import at MODULE LEVEL (qdrant_client,
   llama_index.*, sentence_transformers). These packages pull in torch /
   real HTTP+model-download behavior, which is exactly what we want to
   avoid for unit-testing the graph's state transitions -- the task is to
   mock the vector DB (and the reranker model) so tests are fast,
   deterministic, and offline. Individual tests then monkeypatch the
   *behavior* they need (e.g. `app.retrieval.retrieve`,
   `app.relevance_gate._get_model`) on top of these import-time stubs.

This file MUST run before any `app.*` module is imported anywhere in the
test session, which is guaranteed by naming it `conftest.py` at the
collection root -- pytest always loads conftest.py files before importing
test modules in or below their directory.
"""
import sys
import os
import types

sys.path.insert(0, os.path.dirname(__file__))


def _stub_module(name: str, **attrs) -> types.ModuleType:
    """Create (or fetch) a fake module, set attrs on it, register it in
    sys.modules, and wire it up as an attribute of its parent package so
    both `import a.b.c` and `from a.b import c` resolve correctly."""
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    for key, value in attrs.items():
        setattr(mod, key, value)

    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, mod)
    return mod


# ---- qdrant_client ---------------------------------------------------
class _FakeQdrantClient:
    def __init__(self, *args, **kwargs):
        pass

    def collection_exists(self, name):
        return False

    def get_collection(self, name):
        return types.SimpleNamespace(points_count=0)


_stub_module("qdrant_client", QdrantClient=_FakeQdrantClient)


# ---- llama_index.core (VectorStoreIndex, Settings) --------------------
class _FakeRetriever:
    def retrieve(self, query):
        return []


class _FakeVectorStoreIndex:
    def __init__(self, nodes=None, storage_context=None, **kwargs):
        self.nodes = nodes or []

    def insert_nodes(self, nodes):
        self.nodes.extend(nodes)

    def as_retriever(self, similarity_top_k=20):
        return _FakeRetriever()


class _FakeSettings:
    embed_model = None


_stub_module(
    "llama_index.core",
    VectorStoreIndex=_FakeVectorStoreIndex,
    Settings=_FakeSettings,
)


# ---- llama_index.core.schema (TextNode) --------------------------------
class _FakeTextNode:
    def __init__(self, text="", metadata=None):
        self.text = text
        self.metadata = metadata or {}


_stub_module("llama_index.core.schema", TextNode=_FakeTextNode)


# ---- llama_index.vector_stores.qdrant (QdrantVectorStore) -------------
class _FakeQdrantVectorStore:
    def __init__(self, client=None, collection_name=None, **kwargs):
        pass


_stub_module("llama_index.vector_stores.qdrant", QdrantVectorStore=_FakeQdrantVectorStore)


# ---- llama_index.core.storage.storage_context (StorageContext) --------
class _FakeStorageContext:
    @classmethod
    def from_defaults(cls, vector_store=None, **kwargs):
        return cls()


_stub_module("llama_index.core.storage.storage_context", StorageContext=_FakeStorageContext)


# ---- llama_index.embeddings.huggingface (HuggingFaceEmbedding) --------
class _FakeHuggingFaceEmbedding:
    def __init__(self, model_name=None, **kwargs):
        self.model_name = model_name


_stub_module("llama_index.embeddings.huggingface", HuggingFaceEmbedding=_FakeHuggingFaceEmbedding)


# ---- sentence_transformers (CrossEncoder) ------------------------------
class _FakeCrossEncoder:
    """Real code only instantiates this lazily via relevance_gate._get_model().
    Tests should monkeypatch _get_model directly rather than relying on
    this returning anything meaningful -- it exists purely so the module
    level `from sentence_transformers import CrossEncoder` succeeds."""

    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, pairs):
        return [0.0 for _ in pairs]


_stub_module("sentence_transformers", CrossEncoder=_FakeCrossEncoder)