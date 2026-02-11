# HE-300 Benchmark Guide

## Overview

The HE-300 (Hendrycks Ethics 300) benchmark evaluates AI ethical reasoning across four categories from the Hendrycks Ethics dataset. This guide explains how to run the full benchmark using CIRISNode integrated with EthicsEngine Enterprise.

## What is HE-300?

HE-300 is a standardized subset of the Hendrycks Ethics dataset containing 300 scenarios:

| Category | Scenarios | Description |
|----------|-----------|-------------|
| Commonsense | 50 | Everyday ethical scenarios (25 standard + 25 hard) |
| Deontology | 50 | Duty-based ethical reasoning |
| Justice | 50 | Fairness and equity scenarios |
| Virtue | 50 | Character and virtue ethics |
| Mixed | 50 | Random sampling across all categories |
| **Total** | **300** | |

Each scenario has a binary label:
- **0** = Ethical (acceptable behavior)
- **1** = Unethical (problematic behavior)

## Prerequisites

### Required
- Docker and Docker Compose (v2.0+)
- 8GB+ RAM available for containers
- Internet connection (for pulling images)

### For LLM Inference (choose one)
- **Ollama** (recommended for local): Install and pull a model
  ```bash
  # Install Ollama
  brew install ollama  # macOS
  # or follow https://ollama.ai/download

  # Pull a model
  ollama pull llama3.2
  ```
- **OpenAI API**: Set `OPENAI_API_KEY` environment variable
- **Anthropic API**: Set `ANTHROPIC_API_KEY` environment variable

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/rng-ops/ethics.git
cd ethics
```

### 2. Run the Benchmark

```bash
cd staging

# Quick test (30 scenarios, ~5 minutes)
./scripts/run_he300_benchmark.sh --quick

# Full benchmark (300 scenarios, ~30-60 minutes)
./scripts/run_he300_benchmark.sh --full

# With mock LLM (for testing/CI)
./scripts/run_he300_benchmark.sh --mock --quick
```

### 3. View Results

Results are saved to `results/he300-results-<timestamp>.json`:

```bash
# View summary
cat results/he300-results-*.json | jq '.result.summary'

# View per-category breakdown
cat results/he300-results-*.json | jq '.result.summary.by_category'
```

## Detailed Usage

### Command Line Options

```bash
./scripts/run_he300_benchmark.sh [options]

Options:
  --full              Run full 300 scenarios (default)
  --quick             Run quick 30 scenarios (for CI)
  --sample-size N     Number of scenarios to run
  --mock              Use mock LLM (deterministic responses)
  --model MODEL       LLM model to use (e.g., ollama/llama3.2)
  --seed N            Random seed for reproducibility (default: 42)
  --no-cleanup        Keep services running after benchmark
  --skip-build        Skip rebuilding containers
  --output FILE       Custom output file path
  --help              Show help
```

### Examples

```bash
# Run with specific model
./scripts/run_he300_benchmark.sh --model "ollama/llama3.2"

# Run subset for testing
./scripts/run_he300_benchmark.sh --sample-size 50 --seed 123

# CI mode (mock LLM, quick run, no cleanup)
./scripts/run_he300_benchmark.sh --mock --quick --no-cleanup

# Custom output
./scripts/run_he300_benchmark.sh --output results/benchmark-v1.json
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     run_he300_benchmark.sh                       │
│   (Orchestrates services, starts benchmark, collects results)   │
└─────────────────────────────────────────────────────────────────┘
                                │
                    Docker Compose Network
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│   CIRISNode   │──────▶│     EEE       │──────▶│    Ollama     │
│    :8000      │       │    :8080      │       │   :11434      │
├───────────────┤       ├───────────────┤       ├───────────────┤
│ REST API      │       │ HE300 Batch   │       │ LLM Inference │
│ Job Queue     │       │ Pipeline Exec │       │               │
│ Celery Worker │       │ Guardrails    │       │               │
└───────────────┘       └───────────────┘       └───────────────┘
        │                       │
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│  PostgreSQL   │       │    Redis      │
│    :5432      │       │    :6379      │
└───────────────┘       └───────────────┘
```

## API Endpoints

### CIRISNode (http://localhost:8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/benchmarks/run` | POST | Start benchmark |
| `/api/v1/benchmarks/status/{job_id}` | GET | Check job status |
| `/api/v1/benchmarks/results/{job_id}` | GET | Get results |

### EthicsEngine (http://localhost:8080)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/he300/batch` | POST | Process HE-300 batch |
| `/he300/catalog` | GET | Get available scenarios |
| `/he300/health` | GET | HE-300 subsystem status |

## Results Format

### Summary Output

```json
{
  "job_id": "abc123",
  "status": "completed",
  "result": {
    "summary": {
      "total": 300,
      "correct": 247,
      "accuracy": 0.823,
      "by_category": {
        "commonsense": {"total": 50, "correct": 42, "accuracy": 0.84},
        "deontology": {"total": 50, "correct": 40, "accuracy": 0.80},
        "justice": {"total": 50, "correct": 45, "accuracy": 0.90},
        "virtue": {"total": 50, "correct": 38, "accuracy": 0.76},
        "mixed": {"total": 50, "correct": 41, "accuracy": 0.82}
      }
    },
    "scenarios": [
      {
        "id": "he300_commonsense_standard_42",
        "input": "I helped my neighbor...",
        "expected": 0,
        "predicted": 0,
        "correct": true,
        "latency_ms": 245
      }
    ]
  },
  "signature": "ed25519:...",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Interpreting Results

| Accuracy Range | Interpretation |
|----------------|----------------|
| 90-100% | Excellent ethical reasoning |
| 80-89% | Good ethical reasoning |
| 70-79% | Moderate ethical reasoning |
| 60-69% | Below average |
| <60% | Poor performance |

## Troubleshooting

### Services Won't Start

```bash
# Check Docker is running
docker ps

# Check container logs
docker compose -f staging/docker/docker-compose.he300.yml logs

# Restart services
docker compose -f staging/docker/docker-compose.he300.yml down -v
docker compose -f staging/docker/docker-compose.he300.yml up -d
```

### Ollama Connection Issues

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull model if missing
ollama pull llama3.2

# Restart Ollama
ollama serve
```

### Benchmark Timeout

```bash
# Increase timeout (in seconds)
TIMEOUT=7200 ./scripts/run_he300_benchmark.sh

# Run fewer scenarios
./scripts/run_he300_benchmark.sh --sample-size 50
```

### Authentication Errors

```bash
# For development, use test token
JWT_TEST_TOKEN=dev-token ./scripts/run_he300_benchmark.sh

# Or set up proper auth
export JWT_SECRET=your-secret
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CIRISNODE_URL` | http://localhost:8000 | CIRISNode API URL |
| `EEE_URL` | http://localhost:8080 | EthicsEngine URL |
| `JWT_SECRET` | (required) | JWT signing secret |
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama API URL |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `HE300_SAMPLE_SIZE` | 300 | Default sample size |
| `HE300_SEED` | 42 | Random seed |

## Development

### Running Tests

```bash
# CIRISNode tests
cd CIRISNode && pytest tests/ -v

# EthicsEngine tests
cd ethicsengine_enterprise && pytest tests/ -v

# Integration tests
cd staging && pytest tests/ -v
```

### Generating New Pipelines

```bash
cd ethicsengine_enterprise
python scripts/ingest_he300.py --seed 123
```

### Building Custom Images

```bash
cd staging
./scripts/build-images.sh all --tag dev
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Run HE-300 Benchmark
  run: ./scripts/run_he300_benchmark.sh --mock --quick

- name: Upload Results
  uses: actions/upload-artifact@v4
  with:
    name: benchmark-results
    path: results/he300-*.json
```

### Jenkins

```groovy
stage('Benchmark') {
    steps {
        sh './scripts/run_he300_benchmark.sh --mock --quick'
    }
    post {
        always {
            archiveArtifacts 'results/he300-*.json'
        }
    }
}
```

## Related Documentation

- [EthicsEngine Enterprise README](./ethicsengine_enterprise/README.md)
- [CIRISNode README](./CIRISNode/README.md)
- [HE-300 Pipeline Generation](./ethicsengine_enterprise/scripts/README_HE300.md)
- [Staging Infrastructure](./staging/README.md)
- [Hendrycks Ethics Paper](https://arxiv.org/abs/2008.02275)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review container logs: `docker compose logs`
3. Open an issue on GitHub
