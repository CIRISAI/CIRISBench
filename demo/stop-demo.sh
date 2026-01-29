#!/bin/bash
# =============================================================================
# CIRISBench Demo Stop Script
# =============================================================================
# Stops all demo services cleanly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "Stopping CIRISBench Demo..."
echo ""

# Stop Purple Agent
if [ -f /tmp/cirisbench-purple-agent.pid ]; then
    PID=$(cat /tmp/cirisbench-purple-agent.pid)
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null || true
        echo -e "${GREEN}✓${NC} Purple Agent stopped (PID: $PID)"
    fi
    rm -f /tmp/cirisbench-purple-agent.pid
else
    # Try to find and kill by process name
    pkill -f "llm_purple_agent.py" 2>/dev/null && echo -e "${GREEN}✓${NC} Purple Agent stopped" || true
fi

# Stop Docker containers
cd "$PROJECT_DIR/docker/agentbeats"
docker compose down 2>/dev/null
echo -e "${GREEN}✓${NC} Docker containers stopped"

echo ""
echo "Demo environment stopped."
