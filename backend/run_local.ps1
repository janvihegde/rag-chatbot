# backend/run_local.ps1
#
# Runs the backend natively (not in Docker) against Mongo + Qdrant, which
# are expected to already be running via `docker compose up -d` from the
# project root (that command starts only qdrant + mongo by default -- see
# docker-compose.yml's comments on the 'full' profile).
#
# Benefit over the containerized backend: --reload picks up code changes
# instantly, no `docker compose up -d --build` needed per edit.
#
# Usage (from the backend/ directory):
#     .\run_local.ps1

$ErrorActionPreference = "Stop"

# Create the venv on first run only.
if (-not (Test-Path ".\venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

.\venv\Scripts\Activate.ps1

Write-Host "Installing/updating dependencies..."
pip install -r requirements.txt --quiet

# Load .env from the project root if present (same file docker compose
# itself reads for variable substitution), without requiring the person
# to re-export these every session.
$envFile = Join-Path (Split-Path $PSScriptRoot -Parent) ".env"
if (Test-Path $envFile) {
    Write-Host "Loading environment from $envFile"
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#=][^=]*)\s*=\s*(.*)\s*$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim().Trim('"').Trim("'")
            [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

# Point at the Dockerized data stores via their host-exposed ports (NOT
# the "qdrant"/"mongo" hostnames used inside docker-compose's own network
# -- those only resolve for other containers, not for a process running
# directly on the host).
$env:QDRANT_URL = "http://localhost:6333"
$env:MONGO_URI = "mongodb://localhost:27017"

# Redirect the HuggingFace model cache (BGE-M3 + the reranker, ~2-4GB
# combined) off the C: drive -- this has repeatedly caused download
# failures/crashes when C: ran low on space. Override HF_HOME_DRIVE below
# if D: isn't the right target on your machine.
if (-not $env:HF_HOME) {
    $hfHomeDrive = if ($env:HF_HOME_DRIVE) { $env:HF_HOME_DRIVE } else { "D:\hf_cache" }
    if (-not (Test-Path $hfHomeDrive)) {
        New-Item -ItemType Directory -Path $hfHomeDrive -Force | Out-Null
    }
    $env:HF_HOME = $hfHomeDrive
    Write-Host "HF_HOME set to $hfHomeDrive (model downloads will land here, not C:)"
}

if (-not $env:ADMIN_API_KEY) {
    Write-Warning "ADMIN_API_KEY is not set -- /api/admin/* routes will reject every request. Set it in the project root .env file."
}
if (-not $env:MISTRAL_API_KEY) {
    Write-Warning "MISTRAL_API_KEY is not set -- falling back to offline scope classification and extractive answer generation."
}

Write-Host "Starting backend on http://localhost:8000 (auto-reload enabled)..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000# backend/run_local.ps1
#
# Runs the backend natively (not in Docker) against Mongo + Qdrant, which
# are expected to already be running via `docker compose up -d` from the
# project root (that command starts only qdrant + mongo by default -- see
# docker-compose.yml's comments on the 'full' profile).
#
# Benefit over the containerized backend: --reload picks up code changes
# instantly, no `docker compose up -d --build` needed per edit.
#
# Usage (from the backend/ directory):
#     .\run_local.ps1

$ErrorActionPreference = "Stop"

# Create the venv on first run only.
if (-not (Test-Path ".\venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

.\venv\Scripts\Activate.ps1

Write-Host "Installing/updating dependencies..."
pip install -r requirements.txt --quiet

# Load .env from the project root if present (same file docker compose
# itself reads for variable substitution), without requiring the person
# to re-export these every session.
$envFile = Join-Path (Split-Path $PSScriptRoot -Parent) ".env"
if (Test-Path $envFile) {
    Write-Host "Loading environment from $envFile"
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#=][^=]*)\s*=\s*(.*)\s*$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim().Trim('"').Trim("'")
            [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

# Point at the Dockerized data stores via their host-exposed ports (NOT
# the "qdrant"/"mongo" hostnames used inside docker-compose's own network
# -- those only resolve for other containers, not for a process running
# directly on the host).
$env:QDRANT_URL = "http://localhost:6333"
$env:MONGO_URI = "mongodb://localhost:27017"

# Redirect the HuggingFace model cache (BGE-M3 + the reranker, ~2-4GB
# combined) off the C: drive -- this has repeatedly caused download
# failures/crashes when C: ran low on space. Override HF_HOME_DRIVE below
# if D: isn't the right target on your machine.
if (-not $env:HF_HOME) {
    $hfHomeDrive = if ($env:HF_HOME_DRIVE) { $env:HF_HOME_DRIVE } else { "D:\hf_cache" }
    if (-not (Test-Path $hfHomeDrive)) {
        New-Item -ItemType Directory -Path $hfHomeDrive -Force | Out-Null
    }
    $env:HF_HOME = $hfHomeDrive
    Write-Host "HF_HOME set to $hfHomeDrive (model downloads will land here, not C:)"
}

if (-not $env:ADMIN_API_KEY) {
    Write-Warning "ADMIN_API_KEY is not set -- /api/admin/* routes will reject every request. Set it in the project root .env file."
}
if (-not $env:MISTRAL_API_KEY) {
    Write-Warning "MISTRAL_API_KEY is not set -- falling back to offline scope classification and extractive answer generation."
}

Write-Host "Starting backend on http://localhost:8000 (auto-reload enabled)..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000