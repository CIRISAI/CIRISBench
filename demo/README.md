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

## Web UI

Access the HE-300 Web Interface at **http://localhost:3000/he300** for:

### Three Agent Demo Tabs

| Tab | Agent Type | Protocol | Endpoint | Description |
|-----|------------|----------|----------|-------------|
| **Base LLM** | Direct API | HTTP REST | `POST /he300/batch` | Raw LLM evaluation without reasoning pipeline |
| **EEE Purple** | EEE Pipeline | A2A | `POST http://localhost:9000/a2a` | Full reasoning pipeline with dual evaluation |
| **CIRIS Agent** | H3ERE Pipeline | A2A | `POST http://localhost:9001/a2a` | CIRIS agent with ethical reasoning |

### Benchmark Tab

Run full HE-300 benchmarks against any A2A or MCP compatible agent:
- **Supported Protocols**: A2A (Agent-to-Agent), MCP (Model Context Protocol)
- **Auto-distribution**: Scenarios automatically divided across 5 categories per spec
- **Reproducibility**: Optional random seed for deterministic scenario selection

### Report Features

- **Dual Evaluation**: Each scenario shows both heuristic and semantic classification
- **Agent Card Badge**: Purple agents display their A2A identity from `.well-known/agent.json`
- **Export Formats**: HTML, Markdown, JSON, CSV, XML, PDF

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
| **CIRISNode UI** | http://localhost:3000 | Web Interface |
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

Agent Card (A2A Identity):
  Name:         EEE Purple Agent
  Version:      1.0.0
  Provider:     CIRIS AI
  DID:          did:web:ciris.ai:purple-agent

=========================================
  Benchmark Complete!
=========================================

Results:
  Batch ID:     agentbeats-c58dbf0b
  Accuracy:     55.0% (165/300 correct)
  Errors:       0

Dual Evaluation:
  Heuristic:    Pattern-based classification
  Semantic:     LLM-based classification
  Agreement:    92.3% (277/300 scenarios)

Category Breakdown:
  commonsense           85.0% (51/60)
  commonsense_hard      83.3% (50/60)
  virtue                55.0% (33/60)
  deontology            40.0% (24/60)
  justice               11.7% (7/60)
```
