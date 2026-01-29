#!/bin/bash
# Stop CIRISBench Demo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Stopping CIRISBench..."
cd "$PROJECT_DIR/docker/agentbeats"
docker compose down
echo "✓ Stopped"
