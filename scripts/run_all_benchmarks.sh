#!/bin/bash
# Run HE-300 benchmarks for all LLMs (5 runs each)
# Usage: ./scripts/run_all_benchmarks.sh

set -e

OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$(cat ~/.openrouter_key 2>/dev/null)}"
BENCH_URL="http://localhost:8180"
PURPLE_PORT=9000

# Models to test (OpenRouter format)
MODELS=(
    "meta-llama/llama-4-maverick:Llama-4-Maverick"
    "anthropic/claude-sonnet-4:Claude-Sonnet-4"
    "openai/gpt-4o:GPT-4o"
    "google/gemini-2.5-pro-preview-03-25:Gemini-2.5-Pro"
    "meta-llama/llama-3.3-70b-instruct:Llama-3.3-70B"
    "qwen/qwen-2.5-72b-instruct:Qwen-2.5-72B"
)

RUNS_PER_MODEL=5
SAMPLE_SIZE=60  # 60 per category = 300 total

echo "=== HE-300 Benchmark Suite ==="
echo "Models: ${#MODELS[@]}"
echo "Runs per model: $RUNS_PER_MODEL"
echo "Sample size: $SAMPLE_SIZE per category"
echo ""

for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r model_id display_name <<< "$model_spec"

    echo ""
    echo "================================================================"
    echo "=== Testing $display_name ($model_id) ==="
    echo "================================================================"

    # Kill existing purple agent
    pkill -f "multi_provider_agent.py" 2>/dev/null || true
    sleep 2

    # Start purple agent for this model
    OPENROUTER_API_KEY="$OPENROUTER_API_KEY" python3 tests/multi_provider_agent.py \
        --provider openrouter \
        --model "$model_id" \
        --port $PURPLE_PORT > /tmp/purple_agent.log 2>&1 &

    sleep 3

    # Verify agent is running
    if ! curl -sf "http://localhost:$PURPLE_PORT/health" > /dev/null; then
        echo "ERROR: Purple agent failed to start for $display_name"
        continue
    fi

    echo "Purple agent running for $display_name"

    for run in $(seq 1 $RUNS_PER_MODEL); do
        echo ""
        echo "--- Run $run/$RUNS_PER_MODEL for $display_name ---"

        # Call benchmark API (synchronous - returns when complete)
        result=$(curl -s -X POST "$BENCH_URL/he300/agentbeats/run" \
            -H "Content-Type: application/json" \
            -H "X-API-Key: bench-key-2026" \
            --max-time 600 \
            -d "{
                \"agent_url\": \"http://host.docker.internal:$PURPLE_PORT/a2a\",
                \"agent_name\": \"$display_name\",
                \"model\": \"$model_id\",
                \"sample_size\": $SAMPLE_SIZE,
                \"concurrency\": 50,
                \"categories\": [\"commonsense\", \"commonsense_hard\", \"deontology\", \"justice\", \"virtue\"]
            }" 2>/dev/null)

        if [ $? -eq 0 ] && [ -n "$result" ]; then
            accuracy=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d.get(\"accuracy\",0)*100:.1f}%')" 2>/dev/null)
            batch_id=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('batch_id','unknown'))" 2>/dev/null)
            echo "    Completed: $batch_id - Accuracy: $accuracy"
        else
            echo "    ERROR: Benchmark failed"
        fi
    done
done

# Cleanup
pkill -f "multi_provider_agent.py" 2>/dev/null || true

echo ""
echo "================================================================"
echo "=== All Benchmarks Complete ==="
echo "================================================================"
echo "Results saved in container at /app/engine/data/benchmark_results/"
echo ""
echo "To submit to leaderboard:"
echo "  docker cp cirisbench-runner:/app/engine/data/benchmark_results/*.json benchmark_results/"
echo "  python scripts/submit_to_agentbeats.py --all-results"
