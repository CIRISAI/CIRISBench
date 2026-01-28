#!/bin/bash
# =============================================================================
# CIRISBench AgentBeats Entrypoint
# =============================================================================

set -e

echo "=============================================="
echo "CIRISBench for AgentBeats"
echo "=============================================="
echo "Mode: ${AGENTBEATS_MODE:-standalone}"
echo "CIRISNode: http://${CIRISNODE_HOST:-0.0.0.0}:${CIRISNODE_PORT:-8000}"
echo "EthicsEngine: http://${EEE_HOST:-0.0.0.0}:${EEE_PORT:-8080}"
echo "MCP Enabled: ${MCP_ENABLED:-true}"
echo "A2A Enabled: ${A2A_ENABLED:-true}"
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

if [ -n "$OLLAMA_BASE_URL" ]; then
    echo "Waiting for Ollama..."
    timeout 60 bash -c "until curl -sf ${OLLAMA_BASE_URL}/api/tags 2>/dev/null; do sleep 2; done" || echo "Ollama not available, continuing..."
fi

# Run database migrations if needed
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    cd /app && python -m cirisnode.db.migrate || true
fi

# Execute command
exec "$@"
