#!/bin/bash
# =============================================================================
# CIRISBench AgentBeats Entrypoint
# =============================================================================

set -e

echo "=============================================="
echo "CIRISBench for AgentBeats"
echo "=============================================="
echo "Mode: ${AGENTBEATS_MODE:-standalone}"
echo "EthicsEngine: http://${EEE_HOST:-0.0.0.0}:${EEE_PORT:-8080}"
echo "=============================================="

# Create data directories if they don't exist
mkdir -p /data/benchmarks /data/traces /data/results
mkdir -p /config

# Generate AgentBeats manifest if not exists
if [ ! -f /config/agentbeats.json ]; then
    cp /app/agentbeats.json /config/agentbeats.json
    echo "Generated AgentBeats manifest at /config/agentbeats.json"
fi

# Wait for dependent services if URLs provided
if [ -n "$DATABASE_URL" ]; then
    echo "Waiting for database..."
    timeout 30 bash -c 'until curl -sf "${DATABASE_URL%/*}" 2>/dev/null; do sleep 1; done' || true
fi

if [ -n "$REDIS_URL" ]; then
    echo "Waiting for Redis..."
    timeout 30 bash -c 'until curl -sf "${REDIS_URL}" 2>/dev/null; do sleep 1; done' || true
fi

if [ "$LLM_PROVIDER" = "openrouter" ]; then
    echo "Using OpenRouter API"
elif [ -n "$OLLAMA_BASE_URL" ]; then
    echo "Waiting for Ollama..."
    timeout 60 bash -c "until curl -sf ${OLLAMA_BASE_URL}/api/tags 2>/dev/null; do sleep 2; done" || echo "Ollama not available, continuing..."
fi

# ---------------------------------------------------------------------------
# Standalone mode: wire AGENTBEATS_API_KEY -> ENGINE_API_KEYS so the Engine
# validates incoming API keys from the AgentBeats platform.
# ---------------------------------------------------------------------------
if [ -n "$AGENTBEATS_API_KEY" ] && [ -z "$ENGINE_API_KEYS" ]; then
    export ENGINE_API_KEYS="$AGENTBEATS_API_KEY"
    echo "Engine API key set from AGENTBEATS_API_KEY"
fi

# Run database migrations
echo "Running database migrations..."
# Engine alembic migrations (if alembic is available and DB is reachable)
if [ -f /app/engine/alembic.ini ] || [ -f /app/alembic.ini ]; then
    cd /app && python -m alembic upgrade head 2>/dev/null || echo "Engine alembic migrations: skipped or failed (non-fatal)"
fi

# Execute command
exec "$@"
