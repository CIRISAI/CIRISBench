#!/bin/bash
# =============================================================================
# CIRISBench Benchmark Runner
# =============================================================================
# Convenience script to run HE-300 benchmarks with proper host detection.

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
SAMPLE_SIZE=${1:-10}
CONCURRENCY=${2:-5}
AGENT_NAME=${3:-"GPT-4o-mini"}
MODEL=${4:-"gpt-4o-mini"}

# Detect host IP
get_host_ip() {
    case "$(uname -s)" in
        Darwin) echo "host.docker.internal" ;;
        Linux) hostname -I 2>/dev/null | awk '{print $1}' || ip route get 1 | awk '{print $7;exit}' ;;
        *) echo "host.docker.internal" ;;
    esac
}

HOST_IP=$(get_host_ip)

echo -e "${BLUE}CIRISBench HE-300 Benchmark${NC}"
echo "=============================="
echo ""
echo "Configuration:"
echo "  Sample Size:  $SAMPLE_SIZE scenarios"
echo "  Concurrency:  $CONCURRENCY parallel requests"
echo "  Agent:        $AGENT_NAME ($MODEL)"
echo "  Agent URL:    http://${HOST_IP}:9000/a2a"
echo ""

# Check services
echo "Checking services..."
if ! curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    echo -e "${RED}✗${NC} EthicsEngine not running. Start with: ./demo/start-demo.sh"
    exit 1
fi
echo -e "${GREEN}✓${NC} EthicsEngine running"

if ! curl -sf http://localhost:9000/health > /dev/null 2>&1; then
    echo -e "${RED}✗${NC} Purple Agent not running. Start with: ./demo/start-demo.sh"
    exit 1
fi
echo -e "${GREEN}✓${NC} Purple Agent running"
echo ""

# Run benchmark
echo -e "${YELLOW}Running benchmark...${NC}"
echo ""

RESULT=$(curl -sX POST http://localhost:8080/he300/agentbeats/run \
  -H "Content-Type: application/json" \
  -d "{
    \"agent_url\": \"http://${HOST_IP}:9000/a2a\",
    \"agent_name\": \"${AGENT_NAME}\",
    \"model\": \"${MODEL}\",
    \"sample_size\": ${SAMPLE_SIZE},
    \"concurrency\": ${CONCURRENCY}
  }")

# Parse and display results
BATCH_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('batch_id','unknown'))" 2>/dev/null)
ACCURACY=$(echo "$RESULT" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin).get('accuracy',0)*100:.1f}%\")" 2>/dev/null)
TOTAL=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_scenarios',0))" 2>/dev/null)
CORRECT=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('correct',0))" 2>/dev/null)
ERRORS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('errors',0))" 2>/dev/null)

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}  Benchmark Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Results:"
echo "  Batch ID:     $BATCH_ID"
echo "  Accuracy:     $ACCURACY ($CORRECT/$TOTAL correct)"
echo "  Errors:       $ERRORS"
echo ""

# Show category breakdown
echo "Category Breakdown:"
echo "$RESULT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
cats = d.get('categories', {})
for name, data in cats.items():
    acc = data.get('accuracy', 0) * 100
    correct = data.get('correct', 0)
    total = data.get('total', 0)
    print(f'  {name:20} {acc:5.1f}% ({correct}/{total})')
" 2>/dev/null || echo "  (unable to parse)"

echo ""
echo "View full trace:"
echo "  curl http://localhost:8080/he300/trace/$BATCH_ID | python3 -m json.tool"
echo ""
