# HE-300 Ethics Benchmark - Quick Start Guide

## Overview

This system benchmarks LLMs on ethical reasoning using the **Hendrycks Ethics** dataset. It measures how well models handle moral scenarios across categories like commonsense ethics, deontology, justice, and virtue ethics.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GitHub Actions CI/CD                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │ ubuntu-gpu  │  │ macos-m2    │  │ self-hosted │  │ cloud-gpu  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                         CIRISNode                                    │
│  • Orchestrates benchmark runs                                       │
│  • Manages jobs via Celery + Redis                                   │
│  • Stores results in PostgreSQL                                      │
│  • Exposes REST API for triggering benchmarks                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                   EthicsEngine Enterprise                            │
│  • Contains HE-300 scenarios (260 ethical dilemmas)                  │
│  • Evaluates model responses against ground truth                    │
│  • Supports batch processing via /he300/batch API                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      LLM Inference Layer                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐ │
│  │ Ollama  │  │  vLLM   │  │llama.cpp│  │ OpenAI  │  │ Anthropic │ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### CIRISNode (`rng-ops/CIRISNode` @ `feature/eee-integration`)

**Role**: Job orchestrator and API gateway

| File | Purpose |
|------|---------|
| `cirisnode/utils/eee_client.py` | HTTP client for EthicsEngine |
| `cirisnode/celery_tasks.py` | Async benchmark job execution |
| `cirisnode/api/benchmarks/routes.py` | REST API endpoints |

**Key Config** (environment variables):
```bash
EEE_ENABLED=true              # Enable EthicsEngine integration
EEE_BASE_URL=http://eee:8080  # EthicsEngine endpoint
EEE_BATCH_SIZE=50             # Scenarios per batch
```

### EthicsEngine Enterprise (`rng-ops/ethicsengine_enterprise` @ `feature/he300-api`)

**Role**: Ethics evaluation engine

| File | Purpose |
|------|---------|
| `api/routers/he300.py` | HE-300 batch evaluation API |
| `schemas/he300.py` | Request/response models |
| `data/pipelines/he300/` | 260 scenario definitions |
| `core/engine.py` | Pipeline execution engine |

**Key Endpoints**:
```
GET  /he300/health   → Check service status
GET  /he300/catalog  → List available scenarios  
POST /he300/batch    → Evaluate scenarios against LLM
```

---

## LLM Integration Points

### Where Models Connect

```
EthicsEngine
    │
    ├─► open_llm/config_llm.py     # LLM configuration
    │       │
    │       ├─► Ollama (local)     # OLLAMA_BASE_URL
    │       ├─► OpenAI API         # OPENAI_API_KEY
    │       ├─► Anthropic API      # ANTHROPIC_API_KEY
    │       └─► Custom endpoint    # LLM_BASE_URL
    │
    └─► core/engine.py             # Prompt construction & response parsing
```

### Inference Flow

1. **Scenario Loading**: EthicsEngine loads scenario from `data/pipelines/he300/`
2. **Prompt Construction**: Scenario text formatted into model prompt
3. **LLM Call**: Request sent to configured inference backend
4. **Response Parsing**: Model output parsed for ethical judgment
5. **Evaluation**: Response compared to ground truth label
6. **Aggregation**: Results collected across all scenarios

---

## Weights, Agents & Quantization

### Model Configuration

Models are specified as `provider/model-name`:

```yaml
# config/models.yaml
providers:
  ollama:
    models:
      - llama3.2
      - mistral
      - mixtral:8x7b
  openai:
    models:
      - gpt-4o-mini
      - gpt-4o
  anthropic:
    models:
      - claude-3-haiku
```

### Quantization Levels

For local inference (Ollama, llama.cpp), specify quantization:

| Level | Description | Memory | Speed |
|-------|-------------|--------|-------|
| `fp16` | Full precision | High | Baseline |
| `int8` | 8-bit integer | ~50% | Faster |
| `q4_k_m` | 4-bit (recommended) | ~25% | Fast |
| `q4_0` | 4-bit basic | ~25% | Fastest |

Example: `ollama/llama3.2:q4_k_m`

### Agent Pipelines

EthicsEngine uses configurable pipelines for different evaluation strategies:

```
data/pipelines/he300/
├── manifest.json           # Index of all scenarios
├── he300_commonsense_*.json  # 50 commonsense scenarios
├── he300_deontology_*.json   # 50 deontology scenarios
├── he300_justice_*.json      # 50 justice scenarios
├── he300_virtue_*.json       # 50 virtue scenarios
└── he300_mixed_*.json        # 60 mixed scenarios
```

---

## Running Benchmarks in CI/CD

### GitHub Actions Workflow

Navigate to **Actions** → **HE-300 Benchmark** → **Run workflow**

![Workflow dispatch options](https://img.shields.io/badge/GitHub-Actions-blue)

| Parameter | Options | Description |
|-----------|---------|-------------|
| **model** | `ollama/llama3.2`, `openai/gpt-4o-mini` | Model to evaluate |
| **quantization** | `default`, `fp16`, `int4`, `q4_k_m` | Precision level |
| **sample_size** | `10`, `30`, `50`, `100`, `300` | Number of scenarios |
| **inference_engine** | `ollama`, `vllm`, `openai`, `mock` | Backend engine |
| **runner** | `ubuntu-latest`, `self-hosted-gpu` | Hardware target |

### Self-Hosted Runners

To benchmark on your own hardware:

1. **Set up GitHub Runner**:
   ```bash
   # On your GPU machine
   ./config.sh --url https://github.com/rng-ops/he300-integration \
               --token YOUR_TOKEN \
               --labels self-hosted-gpu
   ```

2. **Run benchmark targeting your hardware**:
   - Select `self-hosted-gpu` as runner
   - Choose your model and quantization
   - Compare results across different setups

### Comparing Configurations

Run the same benchmark on different setups:

| Run | Hardware | Model | Quant | Accuracy | Throughput |
|-----|----------|-------|-------|----------|------------|
| #1 | A100 40GB | llama3.2 | fp16 | 87.3% | 12.4/sec |
| #2 | A100 40GB | llama3.2 | int4 | 85.1% | 28.7/sec |
| #3 | M2 Max | llama3.2 | q4_k_m | 84.8% | 8.2/sec |
| #4 | CPU only | llama3.2 | q4_0 | 84.5% | 0.8/sec |

---

## Quick Commands

### Local Development

```bash
# Start all services
cd staging
make dev-up

# Run quick test (mock LLM)
make benchmark-quick

# Run full benchmark with Ollama
./scripts/run_he300_benchmark.sh --model ollama/llama3.2 --sample-size 50
```

### Docker Compose

```bash
# Start HE-300 stack
docker compose -f docker/docker-compose.he300.yml up -d

# Check health
curl http://localhost:8000/health   # CIRISNode
curl http://localhost:8080/health   # EthicsEngine

# Trigger benchmark
curl -X POST http://localhost:8000/api/benchmarks/he300 \
  -H "Content-Type: application/json" \
  -d '{"model": "ollama/llama3.2", "sample_size": 50}'
```

---

## Results & Artifacts

Benchmark results are saved as JSON artifacts:

```json
{
  "timestamp": "2026-01-03T12:00:00Z",
  "model": "ollama/llama3.2",
  "quantization": "q4_k_m",
  "runner": "self-hosted-gpu",
  "summary": {
    "total": 260,
    "correct": 221,
    "accuracy": 0.85,
    "elapsed_seconds": 45.2,
    "scenarios_per_second": 5.75
  },
  "categories": {
    "commonsense": {"accuracy": 0.88},
    "deontology": {"accuracy": 0.82},
    "justice": {"accuracy": 0.86},
    "virtue": {"accuracy": 0.84}
  }
}
```

Download from **Actions** → **Run** → **Artifacts** → `he300-results-{run}`

---

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `ollama/llama3.2` | Model identifier |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API |
| `OPENAI_API_KEY` | - | OpenAI credentials |
| `ANTHROPIC_API_KEY` | - | Anthropic credentials |
| `HE300_SAMPLE_SIZE` | `300` | Scenarios to evaluate |
| `HE300_SEED` | `42` | Random seed for reproducibility |
| `EEE_ENABLED` | `true` | Enable EthicsEngine |
| `FF_MOCK_LLM` | `false` | Use mock (for testing) |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Connection refused" to EEE | Check `EEE_BASE_URL`, ensure EthicsEngine is running |
| "Model not found" in Ollama | Run `ollama pull <model>` first |
| Slow inference | Try smaller quantization (`q4_0`) or reduce batch size |
| OOM errors | Reduce `HE300_BATCH_SIZE` or use quantized model |
| Tests skipped in CI | Ensure feature branches are pushed to remote repos |

---

## Next Steps

1. **Run your first benchmark**: Use GitHub Actions with `mock` engine
2. **Set up local Ollama**: `ollama pull llama3.2` and run with real model
3. **Add self-hosted runner**: Benchmark on your GPU hardware
4. **Compare configurations**: Track accuracy vs speed trade-offs
