#!/usr/bin/env bash
# backend/run_local.sh
#
# Runs the backend natively (not in Docker) against Mongo + Qdrant, which
# are expected to already be running via `docker compose up -d` from the
# project root (that command starts only qdrant + mongo by default -- see
# docker-compose.yml's comments on the 'full' profile).
#
# Usage (from the backend/ directory):
#     ./run_local.sh
set -euo pipefail

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "Installing/updating dependencies..."
pip install -r requirements.txt --quiet

# Load .env from the project root if present.
ENV_FILE="$(dirname "$0")/../.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# Point at the Dockerized data stores via their host-exposed ports (NOT
# the "qdrant"/"mongo" hostnames used inside docker-compose's own network
# -- those only resolve for other containers, not a process on the host).
export QDRANT_URL="http://localhost:6333"
export MONGO_URI="mongodb://localhost:27017"

if [ -z "${ADMIN_API_KEY:-}" ]; then
    echo "WARNING: ADMIN_API_KEY is not set -- /api/admin/* routes will reject every request." >&2
fi
if [ -z "${MISTRAL_API_KEY:-}" ]; then
    echo "WARNING: MISTRAL_API_KEY is not set -- falling back to offline scope classification and extractive answer generation." >&2
fi

echo "Starting backend on http://localhost:8000 (auto-reload enabled)..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000