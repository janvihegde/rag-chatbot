# RAG Chatbot (Truelift Support Assistant)

A retrieval-augmented generation (RAG) customer support chatbot with a FastAPI backend and a React (Vite) admin/chat frontend. It answers questions **only** from a company's own ingested documents (PDF / HTML / DOCX), refuses to fabricate contact details, detects out-of-scope or prompt-injection messages, and escalates low-confidence queries to a human instead of guessing.

The backend is built around a [LangGraph](https://github.com/langchain-ai/langgraph) state graph:

```
scope_check ──off_topic/injection──▶ scope_reject
     │
   in_scope
     ▼
 retrieval ──▶ relevance_gate ──gate passed──▶ generation
                     │
                gate failed
                     ▼
                escalation
```

## Features

- **Streaming chat** over Server-Sent Events (`POST /api/chat`)
- **Scope checking** — a regex-based injection filter (always on, model-independent) plus an LLM- or keyword-based in-scope/off-topic classifier
- **Retrieval** via [LlamaIndex](https://www.llamaindex.ai/) + [Qdrant](https://qdrant.tech/) using the `BAAI/bge-m3` embedding model
- **Relevance gating & reranking** — a two-tier check (raw similarity score, with `cross-encoder/ms-marco-MiniLM-L-6-v2` as a tiebreaker for ambiguous cases) so the bot escalates instead of hallucinating when nothing relevant is found
- **Grounded answer generation** — uses the Mistral API when `MISTRAL_API_KEY` is set, otherwise falls back to an offline extractive answer built directly from retrieved chunks (no network/API key required)
- **Escalation logging** to MongoDB when the bot can't confidently answer
- **Conversation history** per session (MongoDB-backed)
- **Document ingestion** (PDF, HTML, DOCX) via file upload or an `s3://` path, admin-token protected
- **Admin endpoints** for viewing escalations and basic usage analytics
- **React admin/chat frontend** (Vite + Recharts) for chatting and reviewing escalations/analytics

## Project structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI app & routes
│   │   ├── graph.py             # LangGraph pipeline wiring
│   │   ├── graph_state.py       # Shared state schema for the graph
│   │   ├── scope_check.py       # Injection detection + in-scope/off-topic classification
│   │   ├── retrieval.py         # Retrieval node
│   │   ├── ingest.py            # Document ingestion + vector retrieval (LlamaIndex/Qdrant)
│   │   ├── chunking.py          # Sentence-aware text chunking
│   │   ├── document_parsers.py  # PDF/HTML/DOCX text extraction
│   │   ├── relevance_gate.py    # Similarity gate + cross-encoder reranking
│   │   ├── generation.py        # Prompt assembly + Mistral / extractive answer generation
│   │   ├── escalation.py        # Escalation logging
│   │   ├── session_store.py     # Conversation history (MongoDB)
│   │   ├── db.py                # MongoDB client
│   │   └── auth.py              # Admin bearer-token auth
│   ├── tests/                   # pytest suite
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Chat UI
│   │   ├── Adminpanel.jsx       # Admin dashboard (escalations, analytics)
│   │   ├── Adminapi.js
│   │   └── useChatStream.js     # SSE client hook
│   └── package.json
├── docker-compose.yml            # qdrant + mongo + backend
└── *.html / Truelift_Doc.pdf     # Sample source documents used for ingestion/testing
```

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn |
| Orchestration | LangGraph |
| Vector store | Qdrant |
| Embeddings | `BAAI/bge-m3` (via LlamaIndex + HuggingFace, sentence-transformers) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Mistral API (optional — falls back to an offline extractive mode) |
| Conversation/escalation storage | MongoDB |
| Document parsing | pypdf, BeautifulSoup4, python-docx |
| Optional ingestion source | AWS S3 (boto3) |
| Frontend | React 19, Vite, Axios, Recharts, lucide-react |

## Getting started

### Option A: Docker Compose (recommended)

This spins up Qdrant, MongoDB, and the backend together.

```bash
# from the repo root
cp .env.example .env   # create this if it doesn't exist yet — see "Environment variables" below
docker compose up --build
```

The API will be available at `http://localhost:8000` (health check: `GET /health`). Qdrant's REST API is exposed on `:6333` and MongoDB on `:27017`.

### Option B: Run the backend manually

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Requires a running MongoDB instance (MONGO_URI) and either a Qdrant
# server (QDRANT_URL) or it will fall back to an on-disk local store.
uvicorn app.main:app --reload
```

> Note: the first run downloads the `BAAI/bge-m3` embedding model (~2GB) from HuggingFace and caches it locally.

### Run the frontend

```bash
cd frontend
npm install
npm run dev
```

By default the frontend talks to `http://localhost:8000`. Override with a `frontend/.env` file:

```
VITE_API_BASE_URL=http://localhost:8000
```

### Run tests

```bash
cd backend
pytest
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `MONGO_URI` | Yes | MongoDB connection string (default `mongodb://localhost:27017`) |
| `QDRANT_URL` | No | Qdrant server URL. If unset, falls back to an on-disk store at `QDRANT_PATH` |
| `QDRANT_PATH` | No | Local on-disk Qdrant storage path when `QDRANT_URL` isn't set (default `./qdrant_storage`) |
| `QDRANT_API_KEY` | No | API key for a hosted/secured Qdrant instance |
| `MISTRAL_API_KEY` | No | Enables LLM-based generation and scope classification. Without it, the bot uses an offline extractive fallback |
| `ADMIN_API_KEY` | Yes (for admin routes) | Bearer token required to call `/api/admin/*` endpoints |
| `FRONTEND_ORIGINS` | No | Comma-separated list of allowed CORS origins (defaults to `*`) |
| `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | No | Needed only if ingesting documents from an `s3://` source |
| `SES_SENDER_EMAIL`, `SES_RECIPIENT_EMAIL` | No | Reserved for email notifications on escalation (not yet wired up — see Known limitations) |

## API reference

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | — | Health check |
| `POST` | `/api/chat` | — | Streams a chat response via SSE. Body: `{ "session_id": str, "message": str }` |
| `GET` | `/api/chat/{session_id}/history` | — | Returns stored conversation history for a session |
| `POST` | `/api/admin/ingest` | Admin | Ingests documents — either multipart `files`, or a JSON/form `source` of the form `s3://...` |
| `GET` | `/api/admin/escalations` | Admin | Lists escalations, optionally filtered by `status` (default `pending`) |
| `GET` | `/api/admin/analytics` | Admin | Aggregate metrics: query volume, escalation counts/rate |

Admin routes require an `Authorization: Bearer <ADMIN_API_KEY>` header.

## Known limitations

- `/api/admin/ingest` has no dedicated request-size/type validation beyond PDF/HTML/DOCX support.
- Escalations are logged to MongoDB but not yet actively routed anywhere (no email/Slack notification is sent yet, despite the `SES_*` variables being present in `docker-compose.yml`).
- The on-disk Qdrant fallback (no `QDRANT_URL`) holds an exclusive file lock, so only one backend process can use it at a time — fine for local dev, not for multiple Uvicorn workers.
- The cross-encoder reranker underperforms on this corpus's formal/paraphrased prose; see the comments in `app/relevance_gate.py` for the tuning rationale.

## License

No license file is currently included in this repository.
