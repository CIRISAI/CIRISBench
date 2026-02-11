#!/usr/bin/env bash
# =============================================================================
# HE-300 Benchmark Runner
# =============================================================================
#
# End-to-end script to run the full HE-300 benchmark using CIRISNode and
# EthicsEngine Enterprise.
#
# Usage:
#   ./run_he300_benchmark.sh [options]
#
# Options:
#   --full              Run full 300 scenarios (default)
#   --quick             Run quick 30 scenarios (for CI)
#   --sample-size N     Number of scenarios to run
#   --mock              Use mock LLM (for testing)
#   --model MODEL       LLM model to use (e.g., ollama/llama3.2)
#   --seed N            Random seed (default: 42)
#   --no-cleanup        Don't stop services after benchmark
#   --skip-build        Skip rebuilding containers
#   --output FILE       Output file for results
#   --help              Show this help
#
# Prerequisites:
#   - Docker and Docker Compose
#   - Ollama running with desired model (or API keys for OpenAI/Anthropic)
#
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
SAMPLE_SIZE=300
SEED=42
USE_MOCK=false
MODEL=""
CLEANUP=true
SKIP_BUILD=false
OUTPUT_FILE=""
CIRISNODE_URL="http://localhost:8000"
EEE_URL="http://localhost:8080"
TIMEOUT=3600
POLL_INTERVAL=5

# Compose file (now relative to staging/)
COMPOSE_FILE="${COMPOSE_FILE:-$PROJECT_ROOT/docker/docker-compose.he300.yml}"

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

log_header() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $*${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
    exit 1
}

usage() {
    head -30 "$0" | grep -E "^#" | sed 's/^# *//'
    exit 0
}

cleanup() {
    if [ "$CLEANUP" = true ]; then
        log_info "Cleaning up..."
        docker compose -f "$COMPOSE_FILE" down -v 2>/dev/null || true
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            SAMPLE_SIZE=300
            shift
            ;;
        --quick)
            SAMPLE_SIZE=30
            shift
            ;;
        --sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --mock)
            USE_MOCK=true
            MODEL="mock/deterministic"
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --no-cleanup)
            CLEANUP=false
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            ;;
    esac
done

# Set up trap for cleanup
trap cleanup EXIT

# Set default output file
if [ -z "$OUTPUT_FILE" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    mkdir -p "$PROJECT_ROOT/results"
    OUTPUT_FILE="$PROJECT_ROOT/results/he300-results-$TIMESTAMP.json"
fi

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

log_header "HE-300 Benchmark Runner"

echo ""
log_info "Configuration:"
echo "  Sample Size:    $SAMPLE_SIZE"
echo "  Seed:           $SEED"
echo "  Model:          ${MODEL:-<default>}"
echo "  Mock Mode:      $USE_MOCK"
echo "  Output File:    $OUTPUT_FILE"
echo "  Cleanup:        $CLEANUP"
echo ""

# Check compose file
if [ ! -f "$COMPOSE_FILE" ]; then
    log_warn "Compose file not found: $COMPOSE_FILE"
    log_info "Looking for alternative compose files..."

    if [ -f "$PROJECT_ROOT/docker-compose.yml" ]; then
        COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
    elif [ -f "$PROJECT_ROOT/CIRISNode/docker-compose.yml" ]; then
        COMPOSE_FILE="$PROJECT_ROOT/CIRISNode/docker-compose.yml"
    else
        log_error "No Docker Compose file found"
    fi
fi

log_info "Using compose file: $COMPOSE_FILE"

# -----------------------------------------------------------------------------
# Step 1: Build and Start Services
# -----------------------------------------------------------------------------

log_header "Step 1: Starting Services"

if [ "$SKIP_BUILD" = false ]; then
    log_info "Building containers..."
    docker compose -f "$COMPOSE_FILE" build --quiet 2>/dev/null || true
fi

log_info "Starting services..."
docker compose -f "$COMPOSE_FILE" up -d

# -----------------------------------------------------------------------------
# Step 2: Wait for Services
# -----------------------------------------------------------------------------

log_header "Step 2: Waiting for Services"

log_info "Waiting for CIRISNode..."
for i in {1..60}; do
    if curl -sf "$CIRISNODE_URL/health" > /dev/null 2>&1; then
        log_success "CIRISNode is healthy"
        break
    fi
    echo -ne "\r  Waiting... ($i/60)"
    sleep 2
done
echo ""

if ! curl -sf "$CIRISNODE_URL/health" > /dev/null 2>&1; then
    log_error "CIRISNode failed to start"
fi

log_info "Waiting for EthicsEngine Enterprise..."
for i in {1..60}; do
    if curl -sf "$EEE_URL/health" > /dev/null 2>&1; then
        log_success "EthicsEngine Enterprise is healthy"
        break
    fi
    echo -ne "\r  Waiting... ($i/60)"
    sleep 2
done
echo ""

if ! curl -sf "$EEE_URL/health" > /dev/null 2>&1; then
    log_error "EthicsEngine Enterprise failed to start"
fi

# -----------------------------------------------------------------------------
# Step 3: Authenticate
# -----------------------------------------------------------------------------

log_header "Step 3: Authenticating"

# Try to get a real token
TOKEN=$(curl -sf -X POST "$CIRISNODE_URL/api/v1/auth/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=benchmark&password=benchmark" 2>/dev/null | jq -r '.access_token' || echo "")

if [ -z "$TOKEN" ] || [ "$TOKEN" = "null" ]; then
    log_warn "Could not get real token, using development token"
    # Generate a test JWT (for development - should be replaced with real auth)
    TOKEN="${JWT_TEST_TOKEN:-test-development-token}"
fi

log_success "Got authentication token"

# -----------------------------------------------------------------------------
# Step 4: Start Benchmark
# -----------------------------------------------------------------------------

log_header "Step 4: Starting Benchmark"

# Build request payload
PAYLOAD="{
    \"benchmark_type\": \"he300\",
    \"n_scenarios\": $SAMPLE_SIZE,
    \"seed\": $SEED
}"

if [ -n "$MODEL" ]; then
    PAYLOAD=$(echo "$PAYLOAD" | jq --arg model "$MODEL" '. + {model: $model}')
fi

if [ "$USE_MOCK" = true ]; then
    PAYLOAD=$(echo "$PAYLOAD" | jq '. + {use_mock: true}')
fi

log_info "Starting benchmark with $SAMPLE_SIZE scenarios..."
log_info "Payload: $PAYLOAD"

RESPONSE=$(curl -sf -X POST "$CIRISNODE_URL/api/v1/benchmarks/run" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "$PAYLOAD" 2>&1) || {
    log_error "Failed to start benchmark. Response: $RESPONSE"
}

# Check for job_id
JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id // empty')

if [ -z "$JOB_ID" ]; then
    # Might be synchronous response
    if echo "$RESPONSE" | jq -e '.results' > /dev/null 2>&1; then
        log_success "Benchmark completed synchronously"
        echo "$RESPONSE" > "$OUTPUT_FILE"
    else
        log_error "Unexpected response: $RESPONSE"
    fi
else
    log_success "Benchmark job started: $JOB_ID"

    # -----------------------------------------------------------------------------
    # Step 5: Poll for Completion
    # -----------------------------------------------------------------------------

    log_header "Step 5: Waiting for Completion"

    START_TIME=$(date +%s)

    while true; do
        ELAPSED=$(($(date +%s) - START_TIME))

        if [ $ELAPSED -gt $TIMEOUT ]; then
            log_error "Timeout waiting for benchmark (${TIMEOUT}s)"
        fi

        STATUS_RESPONSE=$(curl -sf "$CIRISNODE_URL/api/v1/benchmarks/status/$JOB_ID" \
            -H "Authorization: Bearer $TOKEN" 2>/dev/null || echo '{"status":"unknown"}')

        STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status // "unknown"')
        PROGRESS=$(echo "$STATUS_RESPONSE" | jq -r '.progress // ""')

        case $STATUS in
            completed)
                echo ""
                log_success "Benchmark completed!"
                break
                ;;
            failed|error)
                echo ""
                log_error "Benchmark failed: $STATUS_RESPONSE"
                ;;
            pending|running|queued)
                # Show progress
                if [ -n "$PROGRESS" ]; then
                    printf "\r  Status: %-10s | Progress: %-10s | Elapsed: %ds     " "$STATUS" "$PROGRESS" "$ELAPSED"
                else
                    printf "\r  Status: %-10s | Elapsed: %ds                        " "$STATUS" "$ELAPSED"
                fi
                sleep $POLL_INTERVAL
                ;;
            *)
                printf "\r  Status: %-10s | Elapsed: %ds                        " "$STATUS" "$ELAPSED"
                sleep $POLL_INTERVAL
                ;;
        esac
    done

    # -----------------------------------------------------------------------------
    # Step 6: Fetch Results
    # -----------------------------------------------------------------------------

    log_header "Step 6: Fetching Results"

    RESULTS=$(curl -sf "$CIRISNODE_URL/api/v1/benchmarks/results/$JOB_ID" \
        -H "Authorization: Bearer $TOKEN" 2>/dev/null)

    if [ -z "$RESULTS" ]; then
        log_error "Failed to fetch results"
    fi

    # Save results
    echo "$RESULTS" | jq '.' > "$OUTPUT_FILE"
    log_success "Results saved to: $OUTPUT_FILE"
fi

# -----------------------------------------------------------------------------
# Step 7: Display Summary
# -----------------------------------------------------------------------------

log_header "Benchmark Results"

# Extract and display summary
if [ -f "$OUTPUT_FILE" ]; then
    echo ""

    # Try different JSON paths for summary
    SUMMARY=$(jq -r '.result.summary // .summary // {}' "$OUTPUT_FILE")

    TOTAL=$(echo "$SUMMARY" | jq -r '.total // 0')
    CORRECT=$(echo "$SUMMARY" | jq -r '.correct // 0')
    ACCURACY=$(echo "$SUMMARY" | jq -r '.accuracy // 0')

    # Format accuracy as percentage
    ACCURACY_PCT=$(awk "BEGIN {printf \"%.2f\", $ACCURACY * 100}")

    echo "  ┌─────────────────────────────────────────────┐"
    echo "  │           HE-300 BENCHMARK RESULTS          │"
    echo "  ├─────────────────────────────────────────────┤"
    printf "  │  Total Scenarios:     %20s │\n" "$TOTAL"
    printf "  │  Correct:             %20s │\n" "$CORRECT"
    printf "  │  Accuracy:            %19s%% │\n" "$ACCURACY_PCT"
    echo "  └─────────────────────────────────────────────┘"
    echo ""

    # Category breakdown if available
    CATEGORIES=$(echo "$SUMMARY" | jq -r '.by_category // .categories // null')
    if [ "$CATEGORIES" != "null" ] && [ -n "$CATEGORIES" ]; then
        echo "  Category Breakdown:"
        echo "$CATEGORIES" | jq -r 'to_entries[] | "    \(.key): \(.value.correct // .value)/\(.value.total // 1) (\(if .value.accuracy then (.value.accuracy * 100 | floor) else "N/A" end)%)"'
    fi

    echo ""
    log_success "Benchmark complete!"
else
    log_warn "No results file found"
fi

# -----------------------------------------------------------------------------
# Signature Verification (if available)
# -----------------------------------------------------------------------------

SIGNATURE=$(jq -r '.signature // empty' "$OUTPUT_FILE" 2>/dev/null)
if [ -n "$SIGNATURE" ]; then
    log_info "Results are cryptographically signed"
    echo "  Signature: ${SIGNATURE:0:32}..."
fi

echo ""
log_info "Results saved to: $OUTPUT_FILE"
echo ""
