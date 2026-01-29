# CIRISBench Demo Setup

Quick setup for recording the AgentBeats competition demo video.

## Services

| Service | URL | Description |
|---------|-----|-------------|
| **CIRISNode** | http://localhost:8000 | MCP/A2A Gateway |
| **EthicsEngine** | http://localhost:8080 | HE-300 Benchmark API |
| **Purple Agent** | http://localhost:9000 | LLM Agent (GPT-4o-mini) |

## Quick Start

```bash
# 1. Start CIRISBench (Green Agent)
cd docker/agentbeats
docker compose up -d cirisbench

# 2. Start Purple Agent (in another terminal)
cd /path/to/CIRISBench
source demo/.env.purple
python tests/llm_purple_agent.py --port 9000

# 3. Verify services
curl http://localhost:8000/health   # CIRISNode
curl http://localhost:8080/health   # EthicsEngine
curl http://localhost:9000/health   # Purple Agent
```

## Run Demo Benchmark

```bash
# Small demo (10 scenarios, fast)
curl -X POST http://localhost:8080/he300/agentbeats/run \
  -H "Content-Type: application/json" \
  -d '{
    "agent_url": "http://host.docker.internal:9000/a2a",
    "agent_name": "GPT-4o-mini Demo",
    "model": "gpt-4o-mini",
    "sample_size": 10,
    "concurrency": 5,
    "protocol": "a2a",
    "semantic_evaluation": true
  }'
```

## Configuration

- **Green Agent** (semantic eval): Llama 4 Maverick via OpenRouter
- **Purple Agent** (being tested): GPT-4o-mini via OpenRouter

Both use the same OpenRouter API key from `~/.openrouter_key`.
