#!/bin/bash
# =============================================================================
# CIRISBench Demo Startup Script
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🟢 CIRISBench Demo Setup"
echo "========================"
echo ""

# Check for OpenRouter key
if [ ! -f ~/.openrouter_key ]; then
    echo "❌ Error: ~/.openrouter_key not found"
    echo "   Create it with: echo 'sk-or-...' > ~/.openrouter_key"
    exit 1
fi

echo "✓ OpenRouter API key found"
echo ""

# Start Green Agent (CIRISBench)
echo "Starting Green Agent (CIRISBench)..."
cd "$PROJECT_DIR/docker/agentbeats"
docker compose up -d cirisbench
echo "✓ Green Agent starting on ports 8000 (Node) and 8080 (EEE)"
echo ""

# Wait for services
echo "Waiting for services to be ready..."
sleep 10

# Health checks
echo "Checking services..."
curl -sf http://localhost:8000/health > /dev/null && echo "✓ CIRISNode ready (http://localhost:8000)" || echo "⏳ CIRISNode starting..."
curl -sf http://localhost:8080/health > /dev/null && echo "✓ EthicsEngine ready (http://localhost:8080)" || echo "⏳ EthicsEngine starting..."
echo ""

# Instructions for purple agent
echo "=========================================="
echo "🟣 To start the Purple Agent, run:"
echo ""
echo "   cd $PROJECT_DIR"
echo "   source demo/.env.purple"
echo "   python tests/llm_purple_agent.py --port 9000"
echo ""
echo "=========================================="
echo "📊 To run a demo benchmark:"
echo ""
echo '   curl -X POST http://localhost:8080/he300/agentbeats/run \'
echo '     -H "Content-Type: application/json" \'
echo '     -d '"'"'{"agent_url": "http://host.docker.internal:9000/a2a", "agent_name": "GPT-4o-mini", "model": "gpt-4o-mini", "sample_size": 10, "concurrency": 5}'"'"
echo ""
echo "=========================================="
