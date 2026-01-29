#!/bin/bash
# =============================================================================
# CIRISBench Demo Startup Script
# =============================================================================
# Starts the full demo environment for recording/testing.
# Handles cross-platform networking (Linux vs Mac/Windows).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}CIRISBench Demo Setup${NC}"
echo "========================"
echo ""

# =============================================================================
# Helper Functions
# =============================================================================

get_host_ip() {
    # Detect the host IP for container-to-host communication
    case "$(uname -s)" in
        Darwin)
            # macOS - use host.docker.internal
            echo "host.docker.internal"
            ;;
        Linux)
            # Linux - get the primary IP address
            hostname -I 2>/dev/null | awk '{print $1}' || ip route get 1 | awk '{print $7;exit}'
            ;;
        *)
            # Windows/WSL - try host.docker.internal
            echo "host.docker.internal"
            ;;
    esac
}

wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-30}
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} $name ready ($url)"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    echo -e "${YELLOW}⏳${NC} $name not ready after ${max_attempts}s (may still be starting)"
    return 1
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo "Checking prerequisites..."

# Check for OpenRouter key
if [ ! -f ~/.openrouter_key ]; then
    echo -e "${RED}✗${NC} ~/.openrouter_key not found"
    echo ""
    echo "  Get an API key from https://openrouter.ai"
    echo "  Then create the file:"
    echo ""
    echo "    echo 'sk-or-v1-your-key-here' > ~/.openrouter_key"
    echo ""
    exit 1
fi
echo -e "${GREEN}✓${NC} OpenRouter API key found"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗${NC} Docker not found. Please install Docker."
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker installed"

# Check for docker compose
if ! docker compose version &> /dev/null; then
    echo -e "${RED}✗${NC} Docker Compose not found. Please install Docker Compose V2."
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker Compose installed"

# Detect host IP for container networking
HOST_IP=$(get_host_ip)
echo -e "${GREEN}✓${NC} Host IP for containers: ${BLUE}${HOST_IP}${NC}"
echo ""

# =============================================================================
# Start Services
# =============================================================================

echo "Starting CIRISBench services..."
cd "$PROJECT_DIR/docker/agentbeats"

# Pull latest image if needed
echo "Pulling latest CIRISBench image..."
docker compose pull cirisbench 2>/dev/null || true

# Start the green agent (EthicsEngine)
docker compose --env-file .env up -d cirisbench
echo -e "${GREEN}✓${NC} CIRISBench container started"
echo ""

# Wait for services to be ready
echo "Waiting for services..."
wait_for_service "http://localhost:8080/health" "EthicsEngine" 30

# CIRISNode might not start due to packaging issue in current image - that's OK
wait_for_service "http://localhost:8000/health" "CIRISNode" 5 || true
echo ""

# =============================================================================
# Start Purple Agent
# =============================================================================

echo -e "${BLUE}Starting Purple Agent...${NC}"
cd "$PROJECT_DIR"

# Check for required Python packages
if ! python3 -c "import fastapi, uvicorn, httpx" 2>/dev/null; then
    echo "Installing required Python packages..."
    pip3 install fastapi uvicorn httpx pydantic --quiet
fi

# Source purple agent config and start in background
source demo/.env.purple
python3 tests/llm_purple_agent.py --port 9000 &
PURPLE_PID=$!
echo $PURPLE_PID > /tmp/cirisbench-purple-agent.pid

sleep 3
if wait_for_service "http://localhost:9000/health" "Purple Agent (GPT-4o-mini)" 10; then
    echo ""
else
    echo -e "${RED}✗${NC} Purple agent failed to start"
    echo "  Check if port 9000 is already in use: lsof -i :9000"
    exit 1
fi

# =============================================================================
# Ready!
# =============================================================================

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}  Demo Environment Ready!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Services:"
echo -e "  ${BLUE}EthicsEngine${NC}  http://localhost:8080"
echo -e "  ${BLUE}Purple Agent${NC}  http://localhost:9000"
echo ""
echo "Quick Commands:"
echo ""
echo -e "${YELLOW}# Run small demo (10 scenarios):${NC}"
echo "curl -X POST http://localhost:8080/he300/agentbeats/run \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"agent_url\": \"http://${HOST_IP}:9000/a2a\", \"agent_name\": \"GPT-4o-mini\", \"model\": \"gpt-4o-mini\", \"sample_size\": 10, \"concurrency\": 5}'"
echo ""
echo -e "${YELLOW}# Run full benchmark (300 scenarios):${NC}"
echo "curl -X POST http://localhost:8080/he300/agentbeats/run \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"agent_url\": \"http://${HOST_IP}:9000/a2a\", \"agent_name\": \"GPT-4o-mini\", \"model\": \"gpt-4o-mini\", \"sample_size\": 300, \"concurrency\": 10}'"
echo ""
echo -e "${YELLOW}# View results:${NC}"
echo "curl http://localhost:8080/he300/traces | python3 -m json.tool"
echo ""
echo -e "${YELLOW}# Stop everything:${NC}"
echo "./demo/stop-demo.sh"
echo ""
