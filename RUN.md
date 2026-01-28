# Running the HE-300 Benchmark

This guide explains how to run the full HE-300 ethics benchmark using CIRISNode and EthicsEngine Enterprise.

## Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Git
- (Optional) Ollama for local LLM inference
- (Optional) OpenAI/Anthropic API keys for cloud inference

## Quick Start (5 minutes)

### 1. Clone the Integration Repo

```bash
git clone https://github.com/rng-ops/he300-integration.git
cd he300-integration
```

### 2. Set Up Component Repos

```bash
# Clone CIRISNode with EEE integration
git clone --branch feature/eee-integration \
  https://github.com/rng-ops/CIRISNode.git submodules/cirisnode

# Clone EthicsEngine Enterprise with HE-300 API
git clone --branch feature/he300-api \
  https://github.com/rng-ops/ethicsengine_enterprise.git submodules/ethicsengine
```

### 3. Start Services

```bash
# Copy environment template
cp .env.example .env

# Start the full stack
docker compose -f docker/docker-compose.he300.yml up -d
```

### 4. Run the Benchmark

```bash
# Quick test with mock LLM (no GPU required)
./scripts/run_he300_benchmark.sh --mock --sample-size 30

# Full benchmark with Ollama (requires ollama running locally)
./scripts/run_he300_benchmark.sh --model ollama/llama3.2 --sample-size 300
```

---

## Component Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     he300-integration                           │
│                     (this repo - CI/CD)                         │
└───────────────────────────┬────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   CIRISNode   │   │  EthicsEngine │   │  LLM Backend  │
│   :8000       │──▶│  Enterprise   │──▶│  (Ollama,     │
│               │   │  :8080        │   │   OpenAI,etc) │
└───────────────┘   └───────────────┘   └───────────────┘
     │                    │
     ▼                    ▼
┌─────────┐         ┌─────────────────┐
│ Postgres│         │ HE-300 Scenarios│
│ + Redis │         │ (260 pipelines) │
└─────────┘         └─────────────────┘
```

### Repositories

| Component | Repository | Branch |
|-----------|------------|--------|
| **CIRISNode** | `rng-ops/CIRISNode` | `feature/eee-integration` |
| **EthicsEngine** | `rng-ops/ethicsengine_enterprise` | `feature/he300-api` |
| **Integration** | `rng-ops/he300-integration` | `main` |

### Upstream Repositories

| Component | Upstream |
|-----------|----------|
| CIRISNode | `CIRISAI/CIRISNode` |
| EthicsEngine | `emooreatx/ethicsengine_enterprise` |

---

## Running Tests

### Unit Tests (Component Level)

These test the individual components in isolation.

#### EthicsEngine HE-300 API Tests

```bash
cd submodules/ethicsengine
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx

# Run HE-300 API tests
pytest tests/test_he300_api.py -v

# What it tests:
# - /he300/health endpoint
# - /he300/catalog endpoint (lists 260 scenarios)
# - /he300/batch endpoint (batch evaluation)
# - Request/response schema validation
```

#### CIRISNode EEE Integration Tests

```bash
cd submodules/cirisnode
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx respx

# Start Redis (required)
docker run -d -p 6379:6379 redis:7-alpine

# Run integration tests
export REDIS_URL=redis://localhost:6379/0
export JWT_SECRET=test-secret
pytest tests/test_he300_integration.py tests/test_celery_he300.py -v

# What it tests:
# - EEEClient HTTP communication
# - Retry logic and error handling
# - Celery task for async benchmark execution
# - Result aggregation
```

### Integration Tests (Full Stack)

Tests CIRISNode and EthicsEngine working together.

```bash
cd he300-integration

# Start test stack
docker compose -f docker/docker-compose.test.yml up -d

# Wait for services
sleep 30

# Run integration tests
pip install pytest pytest-asyncio httpx
pytest tests/test_integration.py -v

# Cleanup
docker compose -f docker/docker-compose.test.yml down -v
```

### End-to-End Benchmark Test

Full benchmark run with result validation.

```bash
# Start full stack
docker compose -f docker/docker-compose.he300.yml up -d

# Run E2E test
pytest tests/test_e2e_he300.py -v

# This validates:
# - Service health checks
# - Scenario catalog retrieval
# - Batch evaluation execution
# - Result format and accuracy calculation
```

---

## CI/CD Pipelines

### GitHub Actions (Recommended)

The `he300-benchmark.yml` workflow provides the full CI/CD pipeline.

#### Automatic Runs (on push to main)

- **EthicsEngine HE-300 API Tests** - Validates API endpoints
- **CIRISNode EEE Client Tests** - Validates integration client
- **Quick Smoke Test** - Verifies components are in place

#### Manual Benchmark Run

1. Go to **Actions** → **HE-300 Benchmark**
2. Click **Run workflow**
3. Configure:
   - **model**: `ollama/llama3.2`, `openai/gpt-4o-mini`, etc.
   - **sample_size**: `30`, `50`, `100`, `300`
   - **inference_engine**: `ollama`, `openai`, `anthropic`, `mock`
   - **runner**: `ubuntu-latest`, `self-hosted-gpu`
4. View results in job summary and download artifacts

### Jenkins

```groovy
// Jenkinsfile example
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh './scripts/setup-submodules.sh'
            }
        }
        
        stage('Test EthicsEngine') {
            steps {
                dir('submodules/ethicsengine') {
                    sh 'pip install -r requirements.txt'
                    sh 'pytest tests/test_he300_api.py -v'
                }
            }
        }
        
        stage('Test CIRISNode') {
            steps {
                dir('submodules/cirisnode') {
                    sh 'pip install -r requirements.txt'
                    sh 'pytest tests/test_he300_integration.py -v'
                }
            }
        }
        
        stage('Benchmark') {
            steps {
                sh './scripts/run_he300_benchmark.sh --mock --sample-size 50'
            }
        }
    }
    
    post {
        always {
            archiveArtifacts 'results/*.json'
        }
    }
}
```

### GitLab CI

```yaml
# .gitlab-ci.yml example
stages:
  - test
  - benchmark

test-ethicsengine:
  stage: test
  script:
    - cd submodules/ethicsengine
    - pip install -r requirements.txt
    - pytest tests/test_he300_api.py -v

test-cirisnode:
  stage: test
  services:
    - redis:7-alpine
  script:
    - cd submodules/cirisnode
    - pip install -r requirements.txt
    - pytest tests/test_he300_integration.py -v

benchmark:
  stage: benchmark
  needs: [test-ethicsengine, test-cirisnode]
  script:
    - ./scripts/run_he300_benchmark.sh --mock --sample-size 50
  artifacts:
    paths:
      - results/
```

### Local CI with Make

```bash
# Run all tests
make test

# Run specific test suites
make test-unit          # Unit tests only
make test-integration   # Integration tests
make test-e2e          # End-to-end tests

# Run benchmark
make benchmark          # Full 300 scenarios
make benchmark-quick    # Quick 30 scenarios
```

---

## Benchmark Configurations

### Mock Mode (Testing/CI)

No LLM required - uses deterministic mock responses.

```bash
./scripts/run_he300_benchmark.sh --mock --sample-size 50
```

### Ollama (Local GPU/CPU)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull llama3.2

# Run benchmark
./scripts/run_he300_benchmark.sh --model ollama/llama3.2 --sample-size 300
```

### OpenAI API

```bash
export OPENAI_API_KEY=sk-...
./scripts/run_he300_benchmark.sh --model openai/gpt-4o-mini --sample-size 300
```

### Anthropic API

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./scripts/run_he300_benchmark.sh --model anthropic/claude-3-haiku --sample-size 300
```

---

## Test Suites Summary

| Test File | Location | What It Tests |
|-----------|----------|---------------|
| `test_he300_api.py` | `ethicsengine/tests/` | HE-300 API endpoints |
| `test_he300_integration.py` | `cirisnode/tests/` | EEE client communication |
| `test_celery_he300.py` | `cirisnode/tests/` | Async benchmark tasks |
| `test_integration.py` | `staging/tests/` | Full stack integration |
| `test_e2e_he300.py` | `staging/tests/` | End-to-end benchmark |
| `test_infrastructure.py` | `staging/tests/` | CI/CD config validation |

### Running All Tests

```bash
# From staging directory
cd he300-integration

# 1. Infrastructure tests (this repo)
pytest tests/test_infrastructure.py -v

# 2. EthicsEngine tests
cd submodules/ethicsengine && pytest tests/test_he300_api.py -v && cd ../..

# 3. CIRISNode tests (requires Redis)
docker run -d -p 6379:6379 redis:7-alpine
cd submodules/cirisnode && \
  REDIS_URL=redis://localhost:6379/0 JWT_SECRET=test \
  pytest tests/test_he300_integration.py tests/test_celery_he300.py -v && \
  cd ../..

# 4. Integration tests (requires Docker)
docker compose -f docker/docker-compose.test.yml up -d
sleep 30
pytest tests/test_integration.py -v
docker compose -f docker/docker-compose.test.yml down -v
```

---

## Expected Results

### Successful Benchmark Output

```
=== HE-300 Benchmark Results ===
Model: ollama/llama3.2
Scenarios: 260
Correct: 221
Accuracy: 85.00%
Time: 142.5s
Throughput: 1.82 scenarios/sec

Categories:
  - Commonsense: 88.0% (44/50)
  - Deontology: 82.0% (41/50)
  - Justice: 86.0% (43/50)
  - Virtue: 84.0% (42/50)
  - Mixed: 85.0% (51/60)

Results saved to: results/he300-20260103-120000.json
```

### Result JSON Format

```json
{
  "timestamp": "2026-01-03T12:00:00Z",
  "model": "ollama/llama3.2",
  "sample_size": 260,
  "summary": {
    "total": 260,
    "correct": 221,
    "accuracy": 0.85,
    "elapsed_seconds": 142.5
  },
  "categories": {
    "commonsense": {"total": 50, "correct": 44, "accuracy": 0.88},
    "deontology": {"total": 50, "correct": 41, "accuracy": 0.82},
    "justice": {"total": 50, "correct": 43, "accuracy": 0.86},
    "virtue": {"total": 50, "correct": 42, "accuracy": 0.84},
    "mixed": {"total": 60, "correct": 51, "accuracy": 0.85}
  }
}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Connection refused :8080` | EthicsEngine not running. Check `docker compose logs eee` |
| `Connection refused :8000` | CIRISNode not running. Check `docker compose logs cirisnode` |
| `No scenarios found` | EthicsEngine missing HE-300 pipelines. Verify `feature/he300-api` branch |
| `Test skipped in CI` | Push feature branches to remote repos |
| `Redis connection error` | Start Redis: `docker run -d -p 6379:6379 redis:7-alpine` |
| `Model not found` | Pull model first: `ollama pull llama3.2` |

---

## Original Request Fulfilled

> "$200 USD for you to stand up https://github.com/CIRISAI/CIRISNode with https://github.com/emooreatx/ethicsengine_enterprise to the point I can run a full HE-300 benchmark"

✅ **Completed:**

1. **CIRISNode Integration** (`feature/eee-integration` branch)
   - `cirisnode/utils/eee_client.py` - HTTP client for EthicsEngine
   - `cirisnode/celery_tasks.py` - Async benchmark execution
   - `tests/test_he300_integration.py` - Integration tests
   - `tests/test_celery_he300.py` - Celery task tests

2. **EthicsEngine HE-300 API** (`feature/he300-api` branch)
   - `api/routers/he300.py` - Batch evaluation endpoints
   - `schemas/he300.py` - Request/response models
   - `data/pipelines/he300/` - 260 benchmark scenarios
   - `tests/test_he300_api.py` - API tests

3. **Integration Repository** (`he300-integration`)
   - Docker Compose configurations
   - GitHub Actions CI/CD workflows
   - Benchmark runner scripts
   - Comprehensive documentation

4. **To run the full HE-300 benchmark:**
   ```bash
   git clone https://github.com/rng-ops/he300-integration.git
   cd he300-integration
   ./scripts/setup-submodules.sh
   docker compose -f docker/docker-compose.he300.yml up -d
   ./scripts/run_he300_benchmark.sh --model ollama/llama3.2 --sample-size 300
   ```
