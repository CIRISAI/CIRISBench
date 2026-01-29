# CIRISBench Demo

One-command setup for running the HE-300 ethics benchmark demo.

## Quick Start

```bash
# Start everything (green agent + purple agent)
./demo/start-demo.sh

# Run a benchmark
./demo/run-benchmark.sh 10      # 10 scenarios (quick demo)
./demo/run-benchmark.sh 300 10  # 300 scenarios, 10 concurrent

# Stop everything
./demo/stop-demo.sh
```

## Prerequisites

1. **Docker** with Docker Compose V2
2. **Python 3.8+** with pip
3. **OpenRouter API key** saved to `~/.openrouter_key`:
   ```bash
   echo 'sk-or-v1-your-key-here' > ~/.openrouter_key
   ```

## Services

| Service | URL | Description |
|---------|-----|-------------|
| **EthicsEngine** | http://localhost:8080 | HE-300 Benchmark API (Green Agent) |
| **Purple Agent** | http://localhost:9000 | LLM Agent being tested (GPT-4o-mini) |

## Scripts

### `start-demo.sh`
Starts the full demo environment:
- Pulls latest CIRISBench Docker image
- Starts EthicsEngine container
- Starts Purple Agent (GPT-4o-mini via OpenRouter)
- Auto-detects host IP for Linux/Mac/Windows

### `run-benchmark.sh`
Runs an HE-300 benchmark with proper host detection.

```bash
./demo/run-benchmark.sh [sample_size] [concurrency] [agent_name] [model]

# Examples:
./demo/run-benchmark.sh                    # 10 scenarios, 5 concurrent
./demo/run-benchmark.sh 50                 # 50 scenarios
./demo/run-benchmark.sh 300 10             # Full benchmark, 10 concurrent
./demo/run-benchmark.sh 100 5 "My Agent" "anthropic/claude-3-haiku"
```

### `stop-demo.sh`
Stops all demo services:
- Kills Purple Agent process
- Stops Docker containers

## Manual Commands

### Run Benchmark via curl

```bash
# Get your host IP (Linux)
HOST_IP=$(hostname -I | awk '{print $1}')

# Or on Mac/Windows
HOST_IP="host.docker.internal"

# Run benchmark
curl -X POST http://localhost:8080/he300/agentbeats/run \
  -H "Content-Type: application/json" \
  -d "{
    \"agent_url\": \"http://${HOST_IP}:9000/a2a\",
    \"agent_name\": \"GPT-4o-mini\",
    \"model\": \"gpt-4o-mini\",
    \"sample_size\": 10,
    \"concurrency\": 5
  }"
```

### View Results

```bash
# List all traces
curl http://localhost:8080/he300/traces | python3 -m json.tool

# Get specific trace
curl http://localhost:8080/he300/trace/{batch_id} | python3 -m json.tool
```

## Configuration

### Green Agent (Semantic Evaluation)
- Model: `meta-llama/llama-4-maverick` via OpenRouter
- Config: `docker/agentbeats/.env`

### Purple Agent (Being Tested)
- Model: `openai/gpt-4o-mini` via OpenRouter
- Config: `demo/.env.purple`

Both use the same OpenRouter API key from `~/.openrouter_key`.

## Troubleshooting

### "Connection refused" errors
- On Linux, containers can't use `host.docker.internal` by default
- The scripts auto-detect your host IP and use it instead

### Purple Agent won't start
```bash
# Check if port 9000 is in use
lsof -i :9000

# Kill any existing process
pkill -f llm_purple_agent.py
```

### EthicsEngine not healthy
```bash
# Check container logs
docker logs cirisbench

# Restart
./demo/stop-demo.sh && ./demo/start-demo.sh
```

## Example Output

```
CIRISBench HE-300 Benchmark
==============================

Configuration:
  Sample Size:  300 scenarios
  Concurrency:  10 parallel requests
  Agent:        GPT-4o-mini (gpt-4o-mini)
  Agent URL:    http://192.168.50.8:9000/a2a

=========================================
  Benchmark Complete!
=========================================

Results:
  Batch ID:     agentbeats-c58dbf0b
  Accuracy:     55.0% (165/300 correct)
  Errors:       0

Category Breakdown:
  commonsense           85.0% (51/60)
  commonsense_hard      83.3% (50/60)
  virtue                55.0% (33/60)
  deontology            40.0% (24/60)
  justice               11.7% (7/60)
```
