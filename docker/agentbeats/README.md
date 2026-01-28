# CIRISBench for AgentBeats

Unified Docker deployment of CIRISBench with MCP and A2A protocol support for AgentBeats integration.

## Quick Start

```bash
# Minimal deployment (uses external LLM)
docker compose up -d

# With local GPU inference (Ollama)
docker compose --profile gpu up -d

# Full stack (Postgres + Redis + Ollama)
docker compose --profile full up -d
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CIRISBench Container                        │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │      CIRISNode          │  │     EthicsEngine            │  │
│  │      :8000              │  │        :8080                │  │
│  │                         │  │                             │  │
│  │  • MCP Protocol         │  │  • HE-300 Benchmarks        │  │
│  │  • A2A Protocol         │  │  • Trace Validation         │  │
│  │  • Agent Gateway        │  │  • Ethics Evaluation        │  │
│  └────────────┬────────────┘  └──────────────┬──────────────┘  │
│               │                              │                  │
│               └──────────────┬───────────────┘                  │
│                              │                                  │
└──────────────────────────────┼──────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
   ┌─────────┐          ┌───────────┐          ┌──────────┐
   │ Ollama  │          │ Postgres  │          │  Redis   │
   │ (GPU)   │          │           │          │          │
   └─────────┘          └───────────┘          └──────────┘
   [gpu profile]        [full profile]         [full profile]
```

## Endpoints

| Service | Port | Description |
|---------|------|-------------|
| CIRISNode | 8000 | Agent gateway, MCP/A2A protocols |
| EthicsEngine | 8080 | Benchmark evaluation API |

### CIRISNode Endpoints

```
GET  /health              Health check
GET  /info                Service info
POST /mcp                 MCP protocol endpoint
POST /a2a                 A2A protocol endpoint
GET  /api/benchmarks      List benchmarks
POST /api/benchmarks      Create benchmark run
```

### EthicsEngine Endpoints

```
GET  /health              Health check
GET  /he300/catalog       List available scenarios
POST /he300/batch         Run batch evaluation
GET  /he300/results/{id}  Get benchmark results
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_ENABLED` | `true` | Enable MCP protocol |
| `A2A_ENABLED` | `true` | Enable A2A protocol |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama server |
| `LLM_MODEL` | `llama3.2` | Default LLM model |
| `HE300_SAMPLE_SIZE` | `50` | Benchmark sample size |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |

### Using External LLM

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini
docker compose up -d

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_MODEL=claude-3-haiku
docker compose up -d
```

## AgentBeats Integration

The container exposes an AgentBeats manifest at `/config/agentbeats.json`:

```json
{
  "name": "cirisbench",
  "capabilities": {
    "mcp": { "enabled": true, "endpoint": "/mcp" },
    "a2a": { "enabled": true, "endpoint": "/a2a" }
  }
}
```

### MCP Tools Available

| Tool | Description |
|------|-------------|
| `run_benchmark` | Run HE-300 ethics benchmark |
| `validate_trace` | Validate CIRIS Ed25519 signed trace |
| `get_scenario` | Get specific ethical scenario |

### A2A Capabilities

- `benchmark-request` - Request benchmark evaluation
- `trace-validation` - Validate agent traces
- `result-streaming` - Stream benchmark results

## Development

```bash
# Build locally
docker compose build

# View logs
docker compose logs -f cirisbench

# Shell access
docker compose exec cirisbench bash

# Run tests
docker compose exec cirisbench pytest
```

## Volumes

| Volume | Path | Description |
|--------|------|-------------|
| `cirisbench_data` | `/data` | Benchmark results, traces |
| `cirisbench_config` | `/config` | Configuration overrides |
| `ollama_models` | `/root/.ollama` | Downloaded LLM models |
