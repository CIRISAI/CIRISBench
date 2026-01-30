#!/bin/bash
# =============================================================================
# HE-300 Leaderboard Benchmark Runner
# =============================================================================
# Runs full HE-300 benchmarks across multiple frontier models for AgentBeats
# leaderboard submission.
#
# Usage:
#   ./scripts/run_leaderboard_benchmarks.sh [runs_per_model]
#
# Default: 5 runs per model
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/benchmark_results"

# Configuration
RUNS_PER_MODEL=${1:-5}
SAMPLE_SIZE=300
CONCURRENCY=10
AGENT_PORT=9000

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get host IP for container networking
get_host_ip() {
    case "$(uname -s)" in
        Darwin) echo "host.docker.internal" ;;
        Linux) hostname -I 2>/dev/null | awk '{print $1}' ;;
        *) echo "host.docker.internal" ;;
    esac
}

HOST_IP=$(get_host_ip)

# Models to benchmark (provider:model_id:display_name)
MODELS=(
    # Frontier models via OpenRouter
    "openrouter:meta-llama/llama-4-maverick:Llama-4-Maverick"
    "openrouter:anthropic/claude-sonnet-4:Claude-Sonnet-4"
    "openrouter:openai/gpt-4o:GPT-4o"
    "openrouter:google/gemini-2.5-pro-preview:Gemini-2.5-Pro"
    # Open source via Together.ai
    "together:meta-llama/Llama-3.3-70B-Instruct-Turbo:Llama-3.3-70B"
    "together:Qwen/Qwen2.5-72B-Instruct-Turbo:Qwen-2.5-72B"
)

mkdir -p "$RESULTS_DIR"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  HE-300 Leaderboard Benchmark Suite${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo "Configuration:"
echo "  Runs per model:  $RUNS_PER_MODEL"
echo "  Sample size:     $SAMPLE_SIZE scenarios"
echo "  Concurrency:     $CONCURRENCY"
echo "  Host IP:         $HOST_IP"
echo "  Results dir:     $RESULTS_DIR"
echo ""
echo "Models to benchmark:"
for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r provider model name <<< "$model_spec"
    echo "  - $name ($provider)"
done
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check EthicsEngine
if ! curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    echo -e "${RED}✗${NC} EthicsEngine not running"
    echo "  Starting CIRISBench..."
    cd "$PROJECT_DIR/docker/agentbeats"
    docker compose --env-file .env up -d cirisbench
    sleep 15
fi
echo -e "${GREEN}✓${NC} EthicsEngine ready"

# Check API keys
check_key() {
    local key_name=$1
    local key_file=$2
    if [ -n "${!key_name}" ] || [ -f "$key_file" ]; then
        echo -e "${GREEN}✓${NC} $key_name available"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $key_name not found (will skip related models)"
        return 1
    fi
}

HAS_OPENROUTER=false
HAS_TOGETHER=false

check_key "OPENROUTER_API_KEY" ~/.openrouter_key && HAS_OPENROUTER=true
check_key "TOGETHER_API_KEY" ~/.together_key && HAS_TOGETHER=true

echo ""

# Function to run a single benchmark
run_benchmark() {
    local provider=$1
    local model=$2
    local name=$3
    local run_num=$4

    echo -e "${BLUE}[$name] Run $run_num/$RUNS_PER_MODEL${NC}"

    # Start the agent
    cd "$PROJECT_DIR"

    # Source the appropriate key
    if [ "$provider" = "openrouter" ]; then
        export OPENROUTER_API_KEY=$(cat ~/.openrouter_key 2>/dev/null || echo "$OPENROUTER_API_KEY")
    elif [ "$provider" = "together" ]; then
        export TOGETHER_API_KEY=$(cat ~/.together_key 2>/dev/null || echo "$TOGETHER_API_KEY")
    fi

    # Kill any existing agent
    pkill -f "multi_provider_agent.py" 2>/dev/null || true
    sleep 1

    # Start agent in background
    python3 tests/multi_provider_agent.py \
        --provider "$provider" \
        --model "$model" \
        --port $AGENT_PORT > /tmp/agent_$$.log 2>&1 &
    AGENT_PID=$!

    # Wait for agent to be ready
    for i in {1..30}; do
        if curl -sf "http://localhost:$AGENT_PORT/health" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    if ! curl -sf "http://localhost:$AGENT_PORT/health" > /dev/null 2>&1; then
        echo -e "${RED}  ✗ Agent failed to start${NC}"
        cat /tmp/agent_$$.log
        kill $AGENT_PID 2>/dev/null || true
        return 1
    fi

    # Run benchmark
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local result_file="$RESULTS_DIR/${name// /_}_run${run_num}_${timestamp}.json"

    RESULT=$(curl -sX POST http://localhost:8080/he300/agentbeats/run \
        -H "Content-Type: application/json" \
        -d "{
            \"agent_url\": \"http://${HOST_IP}:${AGENT_PORT}/a2a\",
            \"agent_name\": \"$name\",
            \"model\": \"$model\",
            \"sample_size\": $SAMPLE_SIZE,
            \"concurrency\": $CONCURRENCY
        }")

    # Save result
    echo "$RESULT" > "$result_file"

    # Parse and display
    local accuracy=$(echo "$RESULT" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin).get('accuracy',0)*100:.1f}%\")" 2>/dev/null || echo "N/A")
    local errors=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('errors',0))" 2>/dev/null || echo "N/A")
    local batch_id=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('batch_id','unknown'))" 2>/dev/null || echo "unknown")

    echo -e "  ${GREEN}✓${NC} Accuracy: $accuracy | Errors: $errors | Batch: $batch_id"

    # Cleanup
    kill $AGENT_PID 2>/dev/null || true
    rm -f /tmp/agent_$$.log

    return 0
}

# Main benchmark loop
echo -e "${CYAN}Starting benchmark runs...${NC}"
echo ""

TOTAL_RUNS=0
SUCCESSFUL_RUNS=0

for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r provider model name <<< "$model_spec"

    # Skip if provider not available
    if [ "$provider" = "openrouter" ] && [ "$HAS_OPENROUTER" = "false" ]; then
        echo -e "${YELLOW}Skipping $name (no OpenRouter key)${NC}"
        continue
    fi
    if [ "$provider" = "together" ] && [ "$HAS_TOGETHER" = "false" ]; then
        echo -e "${YELLOW}Skipping $name (no Together.ai key)${NC}"
        continue
    fi

    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $name${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    for run in $(seq 1 $RUNS_PER_MODEL); do
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
        if run_benchmark "$provider" "$model" "$name" "$run"; then
            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        fi
        # Small delay between runs
        sleep 2
    done
    echo ""
done

# Summary
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Benchmark Complete${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo "Results: $SUCCESSFUL_RUNS/$TOTAL_RUNS runs successful"
echo "Results saved to: $RESULTS_DIR"
echo ""

# List results
echo "Result files:"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null | tail -20
echo ""

# Generate summary
echo -e "${YELLOW}Generating summary...${NC}"
python3 << 'PYEOF'
import json
import os
from collections import defaultdict

results_dir = os.environ.get("RESULTS_DIR", "benchmark_results")
if not os.path.exists(results_dir):
    print("No results directory found")
    exit()

# Aggregate results by model
model_results = defaultdict(list)

for fname in os.listdir(results_dir):
    if fname.endswith(".json"):
        try:
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)
                model = data.get("agent_name", "unknown")
                accuracy = data.get("accuracy", 0)
                model_results[model].append(accuracy)
        except:
            pass

if not model_results:
    print("No results to summarize")
    exit()

print("\n" + "="*60)
print("  LEADERBOARD SUMMARY")
print("="*60)
print(f"{'Model':<25} {'Runs':>5} {'Mean':>8} {'Min':>8} {'Max':>8}")
print("-"*60)

sorted_models = sorted(model_results.items(), key=lambda x: -sum(x[1])/len(x[1]))
for model, accs in sorted_models:
    mean = sum(accs) / len(accs) * 100
    min_acc = min(accs) * 100
    max_acc = max(accs) * 100
    print(f"{model:<25} {len(accs):>5} {mean:>7.1f}% {min_acc:>7.1f}% {max_acc:>7.1f}%")

print("="*60)
PYEOF

echo ""
echo -e "${GREEN}Done!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review results in $RESULTS_DIR"
echo "  2. Submit to AgentBeats leaderboard"
echo ""
