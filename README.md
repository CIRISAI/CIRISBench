# CIRISBench

Benchmarking and evaluation infrastructure for CIRIS AI agents with MCP and A2A protocol support.

## AgentBeats Quick Start

```bash
# Single command deployment
docker compose -f docker/agentbeats/docker-compose.yml up -d

# With GPU inference (Ollama)
docker compose -f docker/agentbeats/docker-compose.yml --profile gpu up -d

# Verify
curl http://localhost:8000/health  # CIRISNode (MCP/A2A)
curl http://localhost:8080/health  # EthicsEngine (Benchmarks)
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              AgentBeats                                  │
│                                  │                                       │
│                          MCP / A2A Protocols                             │
│                                  │                                       │
│  ┌───────────────────────────────┴───────────────────────────────────┐  │
│  │                         CIRISBench                                 │  │
│  │  ┌─────────────────────┐       ┌─────────────────────────────┐   │  │
│  │  │     CIRISNode       │       │     EthicsEngine            │   │  │
│  │  │     :8000           │◄─────►│        :8080                │   │  │
│  │  │                     │       │                             │   │  │
│  │  │  • MCP Endpoint     │       │  • HE-300 Benchmark         │   │  │
│  │  │  • A2A Endpoint     │       │  • Trace Validation         │   │  │
│  │  │  • Agent Gateway    │       │  • Ethics Evaluation        │   │  │
│  │  └─────────────────────┘       └─────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                  │                                       │
│                          ┌───────┴───────┐                              │
│                          │    Ollama     │                              │
│                          │   (LLM/GPU)   │                              │
│                          └───────────────┘                              │
└──────────────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
CIRISBench/
├── cirisbench/           # Python SDK
│   ├── cli.py           # CLI tools
│   └── config.py        # Configuration
├── docker/
│   └── agentbeats/      # Unified Docker deployment
│       ├── Dockerfile
│       ├── docker-compose.yml
│       └── agentbeats.json  # AgentBeats manifest
├── engine/              # EthicsEngine Enterprise
│   ├── api/            # FastAPI endpoints
│   ├── core/           # Evaluation engine
│   └── data/           # HE-300 scenarios
├── infra/              # CI/CD infrastructure
│   ├── docker/         # Docker Compose stacks
│   ├── terraform/      # IaC deployments
│   └── .github/        # Actions workflows
└── tests/              # Test suite
```

## Components

| Component | Port | Description |
|-----------|------|-------------|
| **CIRISNode** | 8000 | Agent gateway - MCP tools and A2A protocols |
| **EthicsEngine** | 8080 | HE-300 benchmark evaluation engine |

### CIRISNode (Agent Gateway)

Handles agent connections via:
- **MCP** (Model Context Protocol) - Tool access for AI agents
- **A2A** (Agent-to-Agent) - Inter-agent communication

### EthicsEngine (Benchmark Engine)

Evaluates AI agents using:
- **HE-300 Benchmark** - 300 ethical reasoning scenarios
- **CIRIS Trace Validation** - Ed25519 signed audit trails

## Development Setup

```bash
# Clone and setup
git clone https://github.com/CIRISAI/CIRISBench.git
cd CIRISBench

# Install dependencies
make install-dev

# Run tests
make test

# Start services locally
make engine-up
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_ENABLED` | `true` | Enable MCP protocol |
| `A2A_ENABLED` | `true` | Enable A2A protocol |
| `LLM_MODEL` | `llama3.2` | Default LLM for benchmarks |
| `HE300_SAMPLE_SIZE` | `50` | Scenarios per benchmark |

## API Endpoints

### CIRISNode (:8000)

```
POST /mcp              MCP protocol
POST /a2a              A2A protocol
GET  /api/benchmarks   List benchmarks
POST /api/benchmarks   Run benchmark
```

### EthicsEngine (:8080)

```
GET  /he300/catalog    Available scenarios
POST /he300/batch      Batch evaluation
GET  /he300/results    Benchmark results
```

## Ecosystem

```
CIRISNode (agent framework)
    ↓ connects via MCP/A2A
CIRISBench (this repo)
    ↓ deploys on
CIRISBridge (infrastructure)
```

## License

See individual component licenses:
- `engine/` - EthicsEngine EULA
- `infra/` - Apache 2.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `make check`
4. Submit a pull request

See [docker/agentbeats/README.md](docker/agentbeats/README.md) for AgentBeats integration details.
