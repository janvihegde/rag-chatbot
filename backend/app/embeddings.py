"""
Embeddings for document retrieval
(SRS Section: Functional Requirements -> 3. Document Retrieval).

STEP 7 upgrade: replaces plain TF-IDF with real semantic embeddings
(Mistral's "mistral-embed" model) when MISTRAL_API_KEY is available.

Why this matters: TF-IDF only matches exact vocabulary. A doc saying
"Two-factor authentication can be enabled" scores essentially ZERO
against a query asking "How do I turn on 2FA?" -- zero shared words,
even though they mean the same thing. Real embeddings understand that
semantic equivalence. This was discovered as a live bug during Step 7
testing (a 2FA follow-up question got wrongly escalated).

Two functions, matching how the fit/transform lifecycle differs between
the two approaches:
    embed_documents(texts) -> fit (TF-IDF only) + embed the corpus once
                               at ingest time
    embed_query(text)      -> embed a single query using whatever was
                               set up by embed_documents()

Both return plain Python lists of floats, so callers (app/ingest.py)
don't need to know or care which path is active.

IMPORTANT: because TF-IDF's vector space is defined by whatever corpus
it was fit on, documents and queries MUST use the same embedding method.
Which method is active is decided once, at startup (based on whether
MISTRAL_API_KEY is set when this module is first imported) -- not
re-checked per request. If you add a key later, restart the server so
the corpus gets re-ingested with the matching method.
"""
import os

USE_MISTRAL = bool(os.environ.get("MISTRAL_API_KEY"))

if USE_MISTRAL:
    from mistralai.client import Mistral

    _client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    _EMBED_MODEL = "mistral-embed"

    def embed_documents(texts: list) -> list:
        response = _client.embeddings.create(model=_EMBED_MODEL, inputs=texts)
        return [d.embedding for d in response.data]

    def embed_query(text: str) -> list:
        return embed_documents([text])[0]

else:
    from sklearn.feature_extraction.text import TfidfVectorizer

    _vectorizer = TfidfVectorizer()

    def embed_documents(texts: list) -> list:
        matrix = _vectorizer.fit_transform(texts)  # fit here -- defines the vector space
        return [row.toarray()[0].tolist() for row in matrix]

    def embed_query(text: str) -> list:
        return _vectorizer.transform([text]).toarray()[0].tolist()