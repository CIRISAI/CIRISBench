#!/usr/bin/env bash
# =============================================================================
# HE-300 Benchmark Runner Script
# =============================================================================
#
# Executes HE-300 benchmark against CIRISNode with EEE backend.
#
# Usage:
#   ./scripts/run-benchmark.sh [options]
#
# Options:
#   --full                Run full 300 scenarios (default)
#   --quick               Run quick 30 scenarios
#   --mock                Use mock LLM
#   --sample-size N       Number of scenarios
#   --seed N              Random seed (default: 42)
#   --model MODEL         LLM model to use
#   --output FILE         Output file (default: results/he300-{timestamp}.json)
#   --cirisnode-url URL   CIRISNode API URL
#   --eee-url URL         EEE API URL
#   --wait                Wait for job completion
#   --timeout SECONDS     Max wait time (default: 3600)
#   --help                Show this help
#
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
SAMPLE_SIZE=300
SEED=42
MOCK=false
MODEL=""
OUTPUT=""
CIRISNODE_URL="${CIRISNODE_URL:-http://localhost:8000}"
EEE_URL="${EEE_URL:-http://localhost:8080}"
WAIT=true
TIMEOUT=3600

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

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
        --mock)
            MOCK=true
            MODEL="mock/deterministic"
            shift
            ;;
        --sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --cirisnode-url)
            CIRISNODE_URL="$2"
            shift 2
            ;;
        --eee-url)
            EEE_URL="$2"
            shift 2
            ;;
        --wait)
            WAIT=true
            shift
            ;;
        --no-wait)
            WAIT=false
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
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

# Set default output
if [ -z "$OUTPUT" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    mkdir -p "$ROOT_DIR/results"
    OUTPUT="$ROOT_DIR/results/he300-$TIMESTAMP.json"
fi

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------

log_info "HE-300 Benchmark Runner"
log_info "======================"
echo ""
log_info "Configuration:"
echo "  Sample Size:   $SAMPLE_SIZE"
echo "  Seed:          $SEED"
echo "  Model:         ${MODEL:-<default>}"
echo "  Mock Mode:     $MOCK"
echo "  CIRISNode URL: $CIRISNODE_URL"
echo "  EEE URL:       $EEE_URL"
echo "  Output:        $OUTPUT"
echo ""

# Check CIRISNode health
log_info "Checking CIRISNode health..."
if ! curl -sf "$CIRISNODE_URL/health" > /dev/null; then
    log_error "CIRISNode not available at $CIRISNODE_URL"
fi
log_success "CIRISNode is healthy"

# Check EEE health
log_info "Checking EEE health..."
if ! curl -sf "$EEE_URL/health" > /dev/null; then
    log_error "EEE not available at $EEE_URL"
fi
log_success "EEE is healthy"

# Check HE-300 subsystem
log_info "Checking HE-300 subsystem..."
HE300_HEALTH=$(curl -sf "$CIRISNODE_URL/api/v1/benchmarks/he300/health" || echo '{"status":"error"}')
echo "  HE-300 Status: $HE300_HEALTH"

# -----------------------------------------------------------------------------
# Get auth token
# -----------------------------------------------------------------------------

log_info "Authenticating..."

# Try to get a token (adjust based on your auth setup)
TOKEN=$(curl -sf -X POST "$CIRISNODE_URL/api/v1/auth/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=benchmark&password=benchmark" 2>/dev/null | jq -r '.access_token' || echo "")

if [ -z "$TOKEN" ] || [ "$TOKEN" = "null" ]; then
    log_warn "Could not get auth token, using test token"
    # Generate a simple test JWT (for development only)
    TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiZW5jaG1hcmsiLCJleHAiOjk5OTk5OTk5OTl9.test"
fi

AUTH_HEADER="Authorization: Bearer $TOKEN"

# -----------------------------------------------------------------------------
# Start benchmark job
# -----------------------------------------------------------------------------

log_info "Starting HE-300 benchmark..."

PAYLOAD=$(cat << EOF
{
    "benchmark_type": "he300",
    "n_scenarios": $SAMPLE_SIZE,
    "seed": $SEED
}
EOF
)

if [ -n "$MODEL" ]; then
    PAYLOAD=$(echo "$PAYLOAD" | jq --arg model "$MODEL" '. + {model: $model}')
fi

RESPONSE=$(curl -sf -X POST "$CIRISNODE_URL/api/v1/benchmarks/run" \
    -H "Content-Type: application/json" \
    -H "$AUTH_HEADER" \
    -d "$PAYLOAD")

JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')

if [ -z "$JOB_ID" ] || [ "$JOB_ID" = "null" ]; then
    log_error "Failed to start benchmark: $RESPONSE"
fi

log_success "Benchmark job started: $JOB_ID"

# -----------------------------------------------------------------------------
# Wait for completion
# -----------------------------------------------------------------------------

if [ "$WAIT" = true ]; then
    log_info "Waiting for job completion (timeout: ${TIMEOUT}s)..."

    START_TIME=$(date +%s)
    POLL_INTERVAL=5

    while true; do
        ELAPSED=$(($(date +%s) - START_TIME))

        if [ $ELAPSED -gt $TIMEOUT ]; then
            log_error "Timeout waiting for job completion"
        fi

        STATUS_RESPONSE=$(curl -sf "$CIRISNODE_URL/api/v1/benchmarks/status/$JOB_ID" \
            -H "$AUTH_HEADER" || echo '{"status":"unknown"}')

        STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status')

        case $STATUS in
            completed)
                log_success "Job completed!"
                break
                ;;
            failed|error)
                log_error "Job failed: $STATUS_RESPONSE"
                ;;
            pending|running)
                echo -ne "\r  Status: $STATUS (elapsed: ${ELAPSED}s)     "
                sleep $POLL_INTERVAL
                ;;
            *)
                log_warn "Unknown status: $STATUS"
                sleep $POLL_INTERVAL
                ;;
        esac
    done

    echo ""

    # Fetch results
    log_info "Fetching results..."

    RESULTS=$(curl -sf "$CIRISNODE_URL/api/v1/benchmarks/results/$JOB_ID" \
        -H "$AUTH_HEADER")

    # Save results
    echo "$RESULTS" | jq '.' > "$OUTPUT"
    log_success "Results saved to $OUTPUT"

    # Print summary
    echo ""
    log_info "Benchmark Summary:"
    echo "$RESULTS" | jq '.result.summary // .summary // {}'

    # Extract key metrics
    TOTAL=$(echo "$RESULTS" | jq -r '.result.summary.total // .summary.total // 0')
    CORRECT=$(echo "$RESULTS" | jq -r '.result.summary.correct // .summary.correct // 0')
    ACCURACY=$(echo "$RESULTS" | jq -r '.result.summary.accuracy // .summary.accuracy // 0')

    echo ""
    echo "  Total:    $TOTAL"
    echo "  Correct:  $CORRECT"
    echo "  Accuracy: $(echo "$ACCURACY * 100" | bc -l 2>/dev/null || echo "$ACCURACY")%"

else
    log_info "Job started. Poll status at:"
    echo "  GET $CIRISNODE_URL/api/v1/benchmarks/status/$JOB_ID"
    echo ""
    echo "Fetch results when complete:"
    echo "  GET $CIRISNODE_URL/api/v1/benchmarks/results/$JOB_ID"
fi

# -----------------------------------------------------------------------------
# Generate summary JSON for CI
# -----------------------------------------------------------------------------

if [ "$WAIT" = true ]; then
    SUMMARY_FILE="${OUTPUT%.json}-summary.json"
    echo "$RESULTS" | jq '{
        job_id: .job_id,
        status: .status,
        total: (.result.summary.total // .summary.total // 0),
        correct: (.result.summary.correct // .summary.correct // 0),
        accuracy: (.result.summary.accuracy // .summary.accuracy // 0),
        duration_seconds: (.result.duration_seconds // 0),
        timestamp: now | strftime("%Y-%m-%dT%H:%M:%SZ")
    }' > "$SUMMARY_FILE"
    log_info "Summary saved to $SUMMARY_FILE"
fi

log_success "Benchmark complete!"
