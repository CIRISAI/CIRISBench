# HE-300 Benchmark Integration - Staging & CI/CD Infrastructure

This directory contains the CI/CD infrastructure for integrating CIRISNode with EthicsEngine Enterprise for HE-300 benchmark execution.

## Component Repositories & Branches

| Component | Repository | Branch | Description |
|-----------|------------|--------|-------------|
| **CIRISNode** | `rng-ops/CIRISNode` | `feature/eee-integration` | EthicsEngine Enterprise integration |
| **EthicsEngine** | `rng-ops/ethicsengine_enterprise` | `feature/he300-api` | HE-300 batch API endpoints |
| **Integration** | `rng-ops/he300-integration` | `main` | This repository - CI/CD infrastructure |

## Directory Structure

```
staging/
├── README.md                    # This file
├── Makefile                     # Unified build/test/deploy commands
├── VERSION                      # Current version tag
├── .env.example                 # Template environment file
│
├── config/                      # Configuration files
│   ├── feature-flags.yaml       # Feature flag definitions
│   ├── models.yaml              # Supported LLM model configurations
│   ├── forks.yaml               # Fork/branch targeting configuration
│   └── environments/            # Environment-specific configs
│       ├── development.yaml
│       ├── staging.yaml
│       └── production.yaml
│
├── docker/                      # Docker configurations
│   ├── docker-compose.he300.yml # HE-300 benchmark stack
│   ├── docker-compose.dev.yml   # Development stack
│   ├── docker-compose.test.yml  # Test stack
│   └── docker-compose.prod.yml  # Production stack
│
├── .github/                     # GitHub Actions workflows
│   └── workflows/
│       ├── ci.yml               # Main CI pipeline
│       ├── he300-benchmark.yml  # HE-300 benchmark runner
│       ├── regression.yml       # Regression test suite
│       ├── release.yml          # Release automation
│       ├── benchmark.yml        # Generic benchmark runner
│       └── submodule-sync.yml   # Submodule synchronization
│
├── scripts/                     # CI/CD utility scripts
│   ├── setup-submodules.sh      # Initialize git submodules
│   ├── run-tests.sh             # Unified test runner
│   ├── run-benchmark.sh         # HE-300 benchmark runner
│   ├── run_he300_benchmark.sh   # Full HE-300 benchmark script
│   ├── version-bump.sh          # Version management
│   ├── build-images.sh          # Docker image builder
│   ├── deploy.sh                # Deployment script
│   └── sync-forks.sh            # Fork synchronization
│
├── tests/                       # Integration & E2E tests
│   ├── conftest.py              # Pytest configuration
│   ├── test_e2e_he300.py        # End-to-end HE-300 tests
│   ├── test_integration.py      # Cross-service integration
│   └── fixtures/                # Test fixtures
│
├── docs/                        # Documentation
│   └── HE300_BENCHMARK.md       # HE-300 benchmark guide
│
├── QUICKSTART.md                # Architecture and usage overview
├── RUN.md                       # Step-by-step benchmark guide
│
├── tasks/                       # Task specifications
│   └── task.md                  # Integration task document
│
├── jenkins/                     # Jenkins pipeline support
│   └── Jenkinsfile              # Jenkins pipeline definition
│
├── submodules/                  # Git submodules (managed)
│   ├── .gitmodules              # Submodule configuration
│   ├── cirisnode/               # CIRISNode submodule
│   └── ethicsengine/            # EthicsEngine Enterprise submodule
│
└── releases/                    # Release artifacts
    └── .gitkeep
```

## Documentation

| Document | Description |
|----------|-------------|
| **[RUN.md](RUN.md)** | Step-by-step guide to run the HE-300 benchmark |
| **[QUICKSTART.md](QUICKSTART.md)** | Architecture overview and component guide |
| **[docs/HE300_BENCHMARK.md](docs/HE300_BENCHMARK.md)** | Detailed benchmark documentation |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Git

### Run the Benchmark

```bash
# Clone and setup
git clone https://github.com/rng-ops/he300-integration.git
cd he300-integration
./scripts/setup-submodules.sh

# Start services
docker compose -f docker/docker-compose.he300.yml up -d

# Run benchmark (mock mode for testing)
./scripts/run_he300_benchmark.sh --mock --sample-size 50

# Run benchmark (with Ollama)
ollama pull llama3.2
./scripts/run_he300_benchmark.sh --model ollama/llama3.2 --sample-size 300
```

See **[RUN.md](RUN.md)** for complete instructions.

### Common Commands

```bash
# Development
make dev-up          # Start dev environment
make dev-down        # Stop dev environment
make dev-logs        # View logs

# Testing
make test            # Run all tests
make test-unit       # Unit tests only
make test-e2e        # E2E tests only

# Benchmarks
make benchmark       # Run HE-300 benchmark
make benchmark-quick # Quick 30-scenario test

# Docker
make build           # Build all images
make push            # Push to registry

# Releases
make version-patch   # Bump patch version
make version-minor   # Bump minor version
make release         # Create release
```

## Configuration

### Feature Flags

Feature flags are defined in `config/feature-flags.yaml`:

```yaml
flags:
  EEE_ENABLED: true      # Enable EthicsEngine Enterprise integration
  HE300_ENABLED: true    # Enable HE-300 benchmark endpoints
  MOCK_LLM: false        # Use mock LLM for testing
  DEBUG_MODE: false      # Enable debug logging
```

### Fork/Branch Targeting

Configure which forks and branches to use in `config/forks.yaml`:

```yaml
repositories:
  cirisnode:
    upstream: CIRISAI/CIRISNode
    fork: rng-ops/CIRISNode
    branch: feature/eee-integration

  ethicsengine:
    upstream: emooreatx/ethicsengine_enterprise
    fork: rng-ops/ethicsengine_enterprise
    branch: feature/he300-api
```

### Model Configuration

Configure LLM models in `config/models.yaml`:

```yaml
models:
  default: ollama/llama3.2
  available:
    - ollama/llama3.2
    - ollama/mistral
    - openai/gpt-4o-mini
```

## CI/CD Pipelines

### GitHub Actions

The primary CI/CD is implemented via GitHub Actions:

- **ci.yml**: Runs on every push/PR - linting, unit tests, build
- **he300-tests.yml**: **Standalone HE-300 test suite** - visible separately in Actions UI
- **regression.yml**: Scheduled nightly regression tests
- **release.yml**: Automated releases on version tags
- **benchmark.yml**: Manual/scheduled HE-300 benchmarks
- **submodule-sync.yml**: Sync submodules with upstream

### HE-300 Test Suite

The HE-300 test suite runs as a **separate, standalone workflow** in GitHub Actions:

```yaml
# Tested Components:
- HE-300 API Tests (EthicsEngine /he300/* endpoints)
- EEE Client Tests (CIRISNode eee_client.py)
- Integration Tests (Full stack CIRISNode + EthicsEngine)
- E2E Tests (Complete benchmark workflow)

# Default Branch Configuration:
- CIRISNode: rng-ops/CIRISNode @ feature/eee-integration
- EthicsEngine: rng-ops/ethicsengine_enterprise @ feature/he300-api
```

### Jenkins

For Jenkins environments, use the provided `jenkins/Jenkinsfile`.

### GitLab CI

For GitLab, convert workflows using `scripts/convert-to-gitlab.sh`.

## Versioning

Versions follow SemVer: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes

Current version is stored in `VERSION` file.

## Submodule Management

Both CIRISNode and EthicsEngine Enterprise are managed as git submodules:

```bash
# Update to latest from configured branches
make submodule-update

# Switch to different branch
./scripts/setup-submodules.sh --cirisnode-branch main --eee-branch main

# Sync with upstream (no local changes)
make submodule-sync
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EEE_ENABLED` | Enable EEE integration | `false` |
| `EEE_BASE_URL` | EEE API URL | `http://eee:8080` |
| `EEE_TIMEOUT_SECONDS` | API timeout | `300` |
| `LLM_MODEL` | LLM model to use | `ollama/llama3.2` |
| `OLLAMA_BASE_URL` | Ollama API URL | `http://ollama:11434` |
| `HE300_BATCH_SIZE` | Batch size for HE-300 | `50` |
| `HE300_SAMPLE_SIZE` | Total scenarios | `300` |
| `HE300_SEED` | Random seed | `42` |

## Contributing

1. Create feature branch
2. Make changes
3. Run `make test`
4. Submit PR

## License

See individual repository licenses.
