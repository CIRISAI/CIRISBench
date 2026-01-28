# CIRISBench for AgentBeats

Unified Docker deployment for the HE-300 Ethics Benchmark with parallel execution, MCP/A2A protocol support, and semantic LLM evaluation.

## Quick Start

```bash
# Pull pre-built image
docker pull ghcr.io/cirisai/cirisbench:agentbeats

# Run with default settings
docker run -p 8000:8000 -p 8080:8080 \
  -e LLM_PROVIDER=openrouter \
  -e OPENROUTER_API_KEY=sk-or-... \
  ghcr.io/cirisai/cirisbench:agentbeats

# Or use docker-compose
docker compose up -d
```

## Running Benchmarks

### AgentBeats Parallel Endpoint

The primary endpoint for AgentBeats integration:

```bash
curl -X POST http://localhost:8080/he300/agentbeats/run \
  -H "Content-Type: application/json" \
  -d '{
    "agent_url": "https://your-agent.example.com/a2a",
    "agent_name": "My Agent",
    "model": "gpt-4o",
    "sample_size": 300,
    "concurrency": 50,
    "protocol": "a2a",
    "semantic_evaluation": true
  }'
```

### Parallelization Levels

| Level | Concurrency | Use Case | Throughput |
|-------|-------------|----------|------------|
| Conservative | 10 | Rate-limited APIs | ~10/sec |
| **Default** | **50** | **Balanced** | **~30/sec** |
| Aggressive | 100 | High-capacity | ~50+/sec |

### Response Format

```json
{
  "batch_id": "agentbeats-abc12345",
  "accuracy": 0.847,
  "total_scenarios": 300,
  "correct": 254,
  "errors": 0,
  "categories": {
    "commonsense": {"accuracy": 0.88, "correct": 66, "total": 75},
    "deontology": {"accuracy": 0.82, "correct": 62, "total": 75},
    "justice": {"accuracy": 0.85, "correct": 64, "total": 75},
    "virtue": {"accuracy": 0.83, "correct": 62, "total": 75}
  },
  "processing_time_ms": 5600.0,
  "concurrency_used": 50
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CIRISBench Container                              │
│                                                                          │
│  ┌────────────────────┐    ┌────────────────────┐    ┌───────────────┐ │
│  │    CIRISNode       │    │   EthicsEngine     │    │  HE-300       │ │
│  │    :8000           │───►│   :8080            │───►│  Runner       │ │
│  │                    │    │                    │    │               │ │
│  │  • MCP Gateway     │    │  • /he300/...      │    │  • Parallel   │ │
│  │  • A2A Gateway     │    │  • Trace Gen       │    │  • Semantic   │ │
│  │  • Auth            │    │  • Validation      │    │  • SSL/TLS    │ │
│  └────────────────────┘    └────────────────────┘    └───────────────┘ │
│                                       │                                  │
│                           ┌───────────┴───────────┐                     │
│                           │   Purple Agent        │                     │
│                           │   (Being Tested)      │                     │
│                           │   via A2A or MCP      │                     │
│                           └───────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
```

## Semantic Evaluation

CIRISBench uses LLM-based semantic evaluation to accurately classify agent responses, even when they're verbose or mis-formatted:

```
Agent Response: "After weighing the ethical considerations, I believe
                 this action is morally acceptable given the circumstances."

Semantic Classification: ETHICAL (confidence: 0.9)
```

### Supported LLM Providers

| Provider | Variable | Model Examples |
|----------|----------|----------------|
| **OpenRouter** | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini`, `anthropic/claude-3-haiku` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini`, `gpt-4o` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-haiku-20240307` |
| Ollama | `OLLAMA_BASE_URL` | `llama3.2`, `mistral` |

```bash
# Recommended: OpenRouter (access to multiple models)
docker run -p 8080:8080 \
  -e LLM_PROVIDER=openrouter \
  -e OPENROUTER_API_KEY=sk-or-... \
  -e LLM_MODEL=openai/gpt-4o-mini \
  ghcr.io/cirisai/cirisbench:agentbeats
```

## API Endpoints

### EthicsEngine (:8080)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/he300/agentbeats/run` | POST | **Run parallel benchmark** |
| `/he300/agentbeats/status` | GET | Runner capabilities |
| `/he300/catalog` | GET | List scenarios |
| `/he300/batch` | POST | Sequential batch |
| `/he300/run` | POST | Full 300 compliant run |
| `/he300/ciris/validate/{id}` | POST | CIRIS trace validation |
| `/he300/ciris/sign/{id}` | POST | Ed25519 signing |

### CIRISNode (:8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/mcp` | POST | MCP protocol |
| `/a2a` | POST | A2A protocol |
| `/api/benchmarks` | GET/POST | Benchmark management |

## Agent Protocol Requirements

### A2A Protocol

Your agent must handle:

```json
POST /a2a
{
  "jsonrpc": "2.0",
  "method": "benchmark.evaluate",
  "params": {
    "scenario_id": "HE-CO-0001",
    "scenario": "You find a wallet with $500 and the owner's ID...",
    "question": "Is this action ethical or unethical?"
  },
  "id": "HE-CO-0001"
}
```

Expected response:

```json
{
  "jsonrpc": "2.0",
  "result": {
    "response": "ETHICAL\n\nReturning the wallet is the right thing to do because..."
  },
  "id": "HE-CO-0001"
}
```

### MCP Protocol

```json
POST /mcp
{
  "method": "tools/call",
  "params": {
    "name": "evaluate_scenario",
    "arguments": {
      "scenario": "You find a wallet with $500...",
      "scenario_id": "HE-CO-0001"
    }
  }
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | Provider: `ollama`, `openai`, `anthropic`, `openrouter` |
| `LLM_MODEL` | `llama3.2` | Model name |
| `OPENROUTER_API_KEY` | - | OpenRouter API key |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama server URL |
| `HE300_SAMPLE_SIZE` | `300` | Default scenarios |
| `MCP_ENABLED` | `true` | Enable MCP |
| `A2A_ENABLED` | `true` | Enable A2A |

### AgentBeats Platform Variables

When AgentBeats runs the container, it provides:

| Variable | Description |
|----------|-------------|
| `AGENTBEATS_API_KEY` | API key for incoming requests |
| `AGENTBEATS_WEBHOOK_SECRET` | Secret for result callbacks |
| `AGENTBEATS_CALLBACK_URL` | URL to send results |
| `AGENTBEATS_RUN_ID` | Unique run identifier |

### SSL/TLS for Purple Agents

For agents requiring custom certificates:

```json
{
  "verify_ssl": true,
  "ca_cert_path": "/config/ca.crt",
  "client_cert_path": "/config/client.crt",
  "client_key_path": "/config/client.key"
}
```

## Docker Compose Profiles

```bash
# Default: CIRISBench only
docker compose up -d

# With GPU inference (Ollama)
docker compose --profile gpu up -d

# Full stack (Postgres + Redis + Ollama)
docker compose --profile full up -d
```

## AgentBeats Manifest

The container includes `agentbeats.json` manifest:

```json
{
  "name": "cirisbench",
  "capabilities": {
    "mcp": {"enabled": true, "endpoint": "/mcp"},
    "a2a": {"enabled": true, "endpoint": "/a2a"},
    "benchmark": {
      "enabled": true,
      "endpoint": "/he300/agentbeats/run",
      "parallelization": {
        "levels": {"conservative": 10, "default": 50, "aggressive": 100}
      },
      "semantic_evaluation": {"enabled": true}
    }
  }
}
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `run_benchmark` | Run HE-300 benchmark with parallelization |
| `validate_trace` | Validate CIRIS Ed25519 signed trace |
| `get_scenario` | Get specific ethical scenario |

### A2A Capabilities

- `benchmark-request` - Request benchmark evaluation
- `trace-validation` - Validate agent traces
- `result-streaming` - Stream benchmark results

## Authentication Flow

### Incoming Requests (AgentBeats → CIRISBench)

```http
POST /he300/agentbeats/run
Authorization: Bearer ${AGENTBEATS_API_KEY}
X-AgentBeats-Run-ID: ${AGENTBEATS_RUN_ID}
```

### Outgoing Callbacks (CIRISBench → AgentBeats)

```http
POST ${AGENTBEATS_CALLBACK_URL}
X-AgentBeats-Signature: sha256=${HMAC_SIGNATURE}
X-AgentBeats-Run-ID: ${AGENTBEATS_RUN_ID}
```

See [auth.md](auth.md) for detailed authentication documentation.

## Volumes

| Volume | Container Path | Description |
|--------|----------------|-------------|
| `cirisbench_data` | `/data` | Benchmark results, traces |
| `cirisbench_config` | `/config` | Configuration, certificates |
| `ollama_models` | `/root/.ollama` | Downloaded LLM models |

## Testing

### Quick Verification

```bash
# Health checks
curl http://localhost:8000/health  # CIRISNode
curl http://localhost:8080/health  # EthicsEngine

# Check AgentBeats runner status
curl http://localhost:8080/he300/agentbeats/status

# List available scenarios
curl http://localhost:8080/he300/catalog?limit=5
```

### Run Test Benchmark

```bash
# Start mock agent (for testing)
docker run -d --name mock-agent -p 9000:9000 \
  ghcr.io/cirisai/cirisbench:mock-agent

# Run benchmark against mock agent
curl -X POST http://localhost:8080/he300/agentbeats/run \
  -H "Content-Type: application/json" \
  -d '{
    "agent_url": "http://host.docker.internal:9000/a2a",
    "sample_size": 50,
    "concurrency": 10
  }'
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "LLM call failed" | Check `LLM_PROVIDER` and API key |
| "Connection refused" to agent | Verify agent URL is accessible from container |
| Slow benchmark | Increase `concurrency` (max 100) |
| SSL errors | Add CA cert to `/config/ca.crt` |

### Logs

```bash
docker compose logs -f cirisbench
docker compose logs -f eee  # EthicsEngine only
```

## Performance Benchmarks

Tested with mock purple agent:

| Scenarios | Concurrency | Time | Throughput |
|-----------|-------------|------|------------|
| 50 | 10 | 3.1s | 16/sec |
| 100 | 50 | 3.0s | 33/sec |
| 300 | 100 | 5.6s | 54/sec |

*Actual performance depends on purple agent response time and LLM provider latency.*
