# Integration Task: CIRISNode + EthicsEngine Enterprise for HE-300 Benchmark

## Executive Summary

**Objective:** Integrate CIRISNode with EthicsEngine Enterprise to enable running the full Hendrycks Ethics 300 (HE-300) benchmark suite.

**What is being asked:**
The client wants to run a complete HE-300 benchmark evaluation, which is the Hendrycks Ethics benchmark covering ethical reasoning across four categories (Commonsense, Deontology, Justice, Virtue). CIRISNode provides the REST API interface and job management for running benchmarks, while EthicsEngine Enterprise provides the pipeline execution engine, ethical reasoning framework, and actual LLM integration.

**Current State:**
- **CIRISNode:** Has placeholder/stub implementations for HE-300 endpoints. Returns hardcoded data. Expected to call EthicsEngine Enterprise for actual benchmark execution.
- **EthicsEngine Enterprise:** Has the Hendrycks Ethics datasets (~64,000 scenarios across 4 categories), a working pipeline engine with LLM integration, and ingestion scripts for creating pipelines from the raw CSV data.

**Gap Analysis:**
1. CIRISNode's `load_he300_data()` looks for EEE data at `./eee/datasets/ethics/` but uses fallback data
2. CIRISNode's benchmark routes return dummy results instead of calling EthicsEngine
3. No API integration exists between CIRISNode and EthicsEngine Enterprise
4. HE-300 sampling logic (6 benchmark IDs × 50 prompts = 300 scenarios) needs implementation
5. No end-to-end test harness or documentation for running the full benchmark

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client/Agent                                    │
│                    (CIRIS-aligned AI agent or test harness)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CIRISNode                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  REST API (/api/v1/benchmarks/*)                                      │   │
│  │  - POST /run → Start HE-300 benchmark job                            │   │
│  │  - GET /results/{job_id} → Get signed results                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Celery Worker                                                        │   │
│  │  - run_he300_scenario_task → Executes via EEE                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                         HTTP/Internal API Call
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EthicsEngine Enterprise                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Core Engine (core/engine.py)                                         │   │
│  │  - EthicsEngine.run_pipeline()                                        │   │
│  │  - Loads identities, guidances, guardrails                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Datasets (datasets/ethics/*)                                         │   │
│  │  - commonsense/cm_test.csv (~19,626 rows)                             │   │
│  │  - deontology/deontology_test.csv (~3,597 rows)                       │   │
│  │  - justice/justice_test.csv (~2,705 rows)                             │   │
│  │  - virtue/virtue_test.csv (~4,976 rows)                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  LLM Integration (via AG2/Autogen)                                    │   │
│  │  - OpenAI, Ollama, or other providers                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Files & Components

### CIRISNode

| File | Purpose | Modification Needed |
|------|---------|---------------------|
| `cirisnode/api/benchmarks/routes.py` | HE-300 REST endpoints | Integrate with EEE API |
| `cirisnode/celery_tasks.py` | Async job execution | Implement `run_he300_scenario_task` |
| `cirisnode/utils/data_loaders.py` | Load HE-300 data | Point to EEE datasets or API |
| `cirisnode/utils/cache.py` | Caching utilities | Cache EEE scenario data |
| `docker-compose.yml` | Service orchestration | Add EEE service |
| `requirements.txt` | Dependencies | Add httpx/aiohttp for EEE calls |

### EthicsEngine Enterprise

| File | Purpose | Modification Needed |
|------|---------|---------------------|
| `api/main.py` or `batch_api/main_api.py` | API server | Expose HE-300 batch endpoint |
| `scripts/ingest_ethics_dataset.py` | Dataset ingestion | Run to generate pipelines |
| `core/engine.py` | Pipeline execution | Verify batch execution works |
| `tests/test_benchmarks.py` | Benchmark tests | Add HE-300 specific tests |
| `data/pipelines/ethics/` | Generated pipelines | Will be created by ingestion |

---

## Work Breakdown

### Prompt 1: EthicsEngine HE-300 API Endpoint

**User Story:**
> As an API consumer, I want to call a single endpoint on EthicsEngine Enterprise that accepts a batch of HE-300 scenario IDs and returns evaluation results, so that CIRISNode can orchestrate benchmark runs without deep coupling to EEE internals.

**Tasks:**
1. Create `api/routers/he300.py` with POST `/he300/batch` endpoint
2. Accept parameters: `category` (commonsense|deontology|justice|virtue), `scenario_ids[]`, `identity_id`, `guidance_id`, `model_config`
3. Load scenarios from `datasets/ethics/{category}/` CSVs
4. Execute pipeline for each scenario using `EthicsEngine.run_pipeline()`
5. Return aggregated results with scores, violations, and summary statistics

**Acceptance Criteria:**
- [ ] Endpoint accepts batch of up to 50 scenarios per request
- [ ] Returns JSON with `{results: [...], summary: {correct: N, total: M, categories: {...}}}`
- [ ] Each result includes: `scenario_id`, `input_text`, `expected_label`, `model_response`, `is_correct`, `latency_ms`
- [ ] Error handling for invalid scenarios or LLM failures
- [ ] Unit tests pass with mocked LLM responses

**Artifacts:**
- `ethicsengine_enterprise/api/routers/he300.py`
- `ethicsengine_enterprise/tests/test_he300_api.py`

---

### Prompt 2: Generate HE-300 Pipelines from Hendrycks Dataset

**User Story:**
> As a developer, I want the HE-300 benchmark scenarios pre-converted to EthicsEngine pipeline format, so that the engine can execute them without runtime CSV parsing.

**Tasks:**
1. Modify `scripts/ingest_ethics_dataset.py` to support sampling (HE-300 = 300 sampled scenarios)
2. Create HE-300 sampling strategy: 50 from each of 6 subcategories:
   - Commonsense (standard) × 50
   - Commonsense (hard) × 50  
   - Deontology × 50
   - Justice × 50
   - Virtue × 50
   - Mixed/random × 50
3. Generate pipeline JSONs in `data/pipelines/he300/`
4. Each pipeline should use a standardized evaluation stage that checks if LLM response aligns with expected label

**Acceptance Criteria:**
- [ ] Running `python scripts/ingest_ethics_dataset.py --he300` generates 300 pipeline files
- [ ] Pipelines are distributed across categories as specified
- [ ] Each pipeline has proper `evaluation_metrics.expected_outcome` set
- [ ] Pipelines can be individually executed via `EthicsEngine.run_pipeline()`

**Artifacts:**
- Updated `ethicsengine_enterprise/scripts/ingest_ethics_dataset.py` 
- `ethicsengine_enterprise/data/pipelines/he300/*.json` (300 files)
- `ethicsengine_enterprise/scripts/README_HE300.md` (sampling methodology documentation)

---

### Prompt 3: CIRISNode Data Loader & EEE Integration

**User Story:**
> As the CIRISNode service, I want to load HE-300 scenarios from EthicsEngine Enterprise (either via API or shared volume), so that benchmark requests return real data instead of fallbacks.

**Tasks:**
1. Update `cirisnode/utils/data_loaders.py` to:
   - Load from EEE's dataset directory (via volume mount) OR
   - Call EEE's API endpoint to fetch scenarios
2. Implement HE-300 ID allocation: `POST /he300` returns 6 benchmark_ids representing categories
3. Implement `GET /bench/he300/prompts?benchmark_id=X` to return 50 prompts per category
4. Update `docker-compose.yml` to either:
   - Mount EEE datasets as shared volume, OR
   - Add EEE service and configure network

**Acceptance Criteria:**
- [ ] `load_he300_data()` returns real scenarios from Hendrycks dataset
- [ ] `POST /api/v1/benchmarks/run` creates a job and returns valid `job_id`
- [ ] `GET /api/v1/benchmarks/results/{job_id}` returns actual scores (not dummy data)
- [ ] Integration tests pass with both services running

**Artifacts:**
- Updated `CIRISNode/cirisnode/utils/data_loaders.py`
- Updated `CIRISNode/docker-compose.yml`
- `CIRISNode/tests/test_he300_integration.py`

---

### Prompt 4: Celery Worker & Async Benchmark Execution

**User Story:**
> As the CIRISNode service, I want benchmark jobs to execute asynchronously via Celery, calling EthicsEngine Enterprise for each scenario evaluation, so that large benchmarks don't block the API and results are properly signed.

**Tasks:**
1. Implement `run_he300_scenario_task` in `cirisnode/celery_tasks.py`
2. Task should:
   - Accept `benchmark_id`, `scenario_ids[]`, `model_config`
   - Call EEE's batch API endpoint
   - Aggregate results
   - Sign the result bundle using Ed25519
   - Store in database/object storage
3. Update benchmark routes to:
   - Enqueue Celery tasks on `POST /run`
   - Return job status on `GET /status/{job_id}`
   - Return signed results on `GET /results/{job_id}`

**Acceptance Criteria:**
- [ ] Benchmark jobs execute asynchronously (non-blocking API)
- [ ] Job status transitions: `pending` → `running` → `completed`/`failed`
- [ ] Results are cryptographically signed
- [ ] Results include timestamp, model info, and per-scenario scores
- [ ] Failed scenarios are recorded but don't fail entire benchmark

**Artifacts:**
- Updated `CIRISNode/cirisnode/celery_tasks.py`
- Updated `CIRISNode/cirisnode/api/benchmarks/routes.py`
- `CIRISNode/tests/test_celery_he300.py`

---

### Prompt 5: End-to-End Test Harness & Documentation

**User Story:**
> As a user, I want a simple command/script to run the full HE-300 benchmark from start to finish, with clear documentation on setup and expected outputs, so that I can validate the integration works.

**Tasks:**
1. Create `run_he300_benchmark.sh` script that:
   - Starts both services via docker-compose
   - Waits for health checks
   - Calls CIRISNode API to start benchmark
   - Polls for completion
   - Downloads and displays results
2. Create `docs/HE300_BENCHMARK.md` with:
   - Prerequisites (Docker, API keys, etc.)
   - Step-by-step instructions
   - Expected output format
   - Troubleshooting guide
3. Create integration test that runs a subset (e.g., 30 scenarios) as CI validation

**Acceptance Criteria:**
- [ ] Running `./run_he300_benchmark.sh` completes successfully with Ollama/local LLM
- [ ] Output shows: total scenarios, correct count, accuracy percentage per category
- [ ] Results file is saved with signature verification
- [ ] Documentation covers all configuration options
- [ ] CI pipeline includes HE-300 smoke test (subset)

**Artifacts:**
- `run_he300_benchmark.sh`
- `docs/HE300_BENCHMARK.md`
- `tests/integration/test_he300_e2e.py`
- Updated `README.md` in both projects

---

## Verification & Sign-off

### Definition of Done

The integration is complete when:

1. **Data Flow Verified:**
   - [ ] HE-300 scenarios load from Hendrycks dataset (not fallbacks)
   - [ ] At least 300 scenarios are available across 6 categories

2. **API Integration Working:**
   - [ ] CIRISNode successfully calls EthicsEngine Enterprise
   - [ ] Results flow back with proper scoring

3. **Benchmark Execution:**
   - [ ] Full HE-300 benchmark completes (may take 30-60 minutes with LLM)
   - [ ] Results are signed and stored

4. **Documentation Complete:**
   - [ ] Setup instructions work on fresh environment
   - [ ] Example output provided

### Test Commands

```bash
# 1. Start services
cd /Users/a/projects/ethics
docker-compose -f CIRISNode/docker-compose.yml \
               -f ethicsengine_enterprise/docker-compose.yml up -d

# 2. Check health
curl http://localhost:8000/health
curl http://localhost:8080/health  # EEE if exposed

# 3. Run benchmark
curl -X POST http://localhost:8000/api/v1/benchmarks/run \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{"benchmark_type": "he300", "model": "ollama/llama3.2"}'

# 4. Check results
curl http://localhost:8000/api/v1/benchmarks/results/<job_id> \
     -H "Authorization: Bearer <token>"
```

---

## Notes for Developer

- The Hendrycks Ethics dataset uses binary labels (0 = ethical, 1 = unethical) for commonsense scenarios
- Deontology/Justice/Virtue have different formats (scenario + excuse pairs)
- EthicsEngine's existing `ingest_ethics_dataset.py` already handles most parsing - extend it for HE-300 sampling
- Consider caching parsed scenarios to avoid re-reading CSVs on every request
- The "300" in HE-300 refers to the sampled subset (6 × 50), not the full dataset
- Ed25519 signing is already implemented in CIRISNode's `utils/signer.py`

---

## Test Analysis & TDD Guidelines

### ⚠️ CRITICAL: Do NOT Modify Existing Tests

Following Test-Driven Development (TDD) best practices, **existing tests should not be modified** unless:
1. A test is actively broken due to a legitimate bug fix
2. The test itself contains a bug (not the code under test)
3. API contracts change in a documented, intentional way

Existing tests serve as regression guards. If new functionality causes existing tests to fail, this indicates either:
- The new code has broken existing contracts (fix the code, not the test)
- The test was testing implementation details rather than behavior (rare, discuss before changing)

---

### Existing Test Coverage Analysis

#### CIRISNode Tests (`CIRISNode/tests/`)

| Test File | What It Tests | Relevance to HE-300 | Modification Status |
|-----------|--------------|---------------------|---------------------|
| `test_benchmarks.py` | Basic benchmark endpoints (`/run`, `/results`) | **HIGH** - Core integration point | ❌ DO NOT MODIFY - Add new tests instead |
| `test_main.py` | Health check endpoint | LOW - Unrelated | ❌ DO NOT MODIFY |
| `test_auth.py` | Token generation/refresh | MEDIUM - Auth for benchmark calls | ❌ DO NOT MODIFY |
| `test_ai_inference.py` | Mock ethical pipeline | **HIGH** - Tests EEE integration placeholder | ❌ DO NOT MODIFY |
| `test_pipelines.py` | Currently empty | N/A | Can be repurposed for new tests |
| `test_audit.py` | Audit logging | LOW - Unrelated | ❌ DO NOT MODIFY |
| `test_health.py` | Health endpoints | LOW - Unrelated | ❌ DO NOT MODIFY |
| `test_jwt_auth.py` | JWT validation | MEDIUM - Auth mechanisms | ❌ DO NOT MODIFY |
| `test_config.py` | Configuration loading | MEDIUM - May need new config | ❌ DO NOT MODIFY |
| `test_wa.py` | Wise Authority endpoints | LOW - Unrelated | ❌ DO NOT MODIFY |
| `test_deferral.py` | WBD functionality | LOW - Unrelated | ❌ DO NOT MODIFY |

**Key Observations (CIRISNode):**
- `test_benchmarks.py` tests the **current stub behavior** (returns dummy job_id, dummy scores)
- These tests MUST continue to pass - they verify the API contract
- New tests should verify the **new behavior** (real data, real scoring) separately
- `test_ai_inference.py` contains a mock `run_ethical_pipeline()` that returns `UNCERTAIN` - this is the integration point

#### EthicsEngine Enterprise Tests (`ethicsengine_enterprise/tests/`)

| Test File | What It Tests | Relevance to HE-300 | Modification Status |
|-----------|--------------|---------------------|---------------------|
| `test_benchmarks.py` | Pipeline execution harness | **HIGH** - Core benchmark runner | ❌ DO NOT MODIFY - Extend pattern for HE-300 |
| `test_loader.py` | Config/pipeline loading | **HIGH** - Loads HE-300 pipelines | ❌ DO NOT MODIFY |
| `test_guardrails.py` | Guardrail evaluation | MEDIUM - May apply to HE-300 | ❌ DO NOT MODIFY |
| `test_llm_handler.py` | LLM stage execution | **HIGH** - How scenarios are processed | ❌ DO NOT MODIFY |
| `test_interactions.py` | Interaction/GroupChat handling | MEDIUM - Multi-turn scenarios | ❌ DO NOT MODIFY |
| `test_tools.py` | Tool handler (calculator, etc.) | LOW - Unrelated | ❌ DO NOT MODIFY |

**Key Observations (EthicsEngine):**
- `test_benchmarks.py` is actually a **CLI test harness** that runs pipelines (not unit tests)
- `test_loader.py` validates that pipelines load correctly - critical for new HE-300 pipelines
- The pattern in `test_loader.py` should be extended (not modified) for HE-300-specific validation

---

### New Tests Required

#### For CIRISNode (new file: `tests/test_he300_integration.py`)

```python
# Tests to add - DO NOT modify existing test_benchmarks.py

def test_load_he300_data_from_eee():
    """Verify load_he300_data() returns real scenarios, not fallback"""
    
def test_he300_scenario_count():
    """Verify 300 scenarios are available (6 categories × 50 each)"""
    
def test_he300_scenario_categories():
    """Verify all 6 categories are represented"""
    
def test_benchmark_run_with_eee_integration():
    """Integration test: POST /run triggers actual EEE call"""
    
def test_benchmark_results_contain_scores():
    """Verify results contain real scores, not dummy data"""
    
def test_benchmark_results_are_signed():
    """Verify Ed25519 signature is valid"""
```

#### For CIRISNode (new file: `tests/test_celery_he300.py`)

```python
def test_run_he300_scenario_task_enqueues():
    """Verify Celery task is properly enqueued"""
    
def test_run_he300_scenario_task_calls_eee():
    """Verify task calls EEE API with correct params"""
    
def test_run_he300_scenario_task_stores_results():
    """Verify task stores results in database"""
    
def test_run_he300_scenario_task_handles_eee_failure():
    """Verify graceful handling of EEE errors"""
```

#### For EthicsEngine Enterprise (new file: `tests/test_he300_api.py`)

```python
def test_he300_batch_endpoint_exists():
    """Verify POST /he300/batch endpoint is registered"""
    
def test_he300_batch_accepts_category():
    """Verify endpoint accepts category parameter"""
    
def test_he300_batch_returns_results():
    """Verify endpoint returns proper result structure"""
    
def test_he300_batch_limits_to_50():
    """Verify batch size limit is enforced"""
```

#### For EthicsEngine Enterprise (new file: `tests/test_he300_pipelines.py`)

```python
def test_he300_pipeline_generation():
    """Verify ingest script generates 300 pipelines"""
    
def test_he300_pipeline_structure():
    """Verify generated pipelines have correct schema"""
    
def test_he300_pipeline_categories():
    """Verify pipelines cover all categories"""
    
def test_he300_pipeline_loadable():
    """Verify generated pipelines load without errors"""
```

---

### Module Dependency Analysis

#### Shared Dependencies (Version Coordination Required)

| Package | CIRISNode Version | EEE Version | Notes |
|---------|------------------|-------------|-------|
| `pydantic` | 2.11.4 | Not pinned | **Pin in EEE** to avoid conflicts |
| `pydantic-settings` | 2.3.4 | Installed | Versions should match |
| `fastapi` | 0.115.12 | Installed | Version should match for shared schemas |
| `pytest` | 8.3.5 | Installed | Match for consistent test behavior |
| `pytest-asyncio` | 0.26.0 | Installed | Match for async test handling |

#### New Dependencies Required

**CIRISNode `requirements.txt` additions:**
```
httpx>=0.25.0  # For async HTTP calls to EEE
aiohttp>=3.9.0  # Alternative async HTTP if preferred
```

**EthicsEngine Enterprise `requirements.txt` additions:**
```
# No new deps needed - uses existing FastAPI
```

---

### Inter-Project Integration Points

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INTEGRATION BOUNDARY                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CIRISNode                          EthicsEngine Enterprise             │
│  ──────────                         ────────────────────────             │
│                                                                          │
│  cirisnode/utils/data_loaders.py    datasets/ethics/*.csv               │
│         │                                   │                           │
│         └─────── reads from ────────────────┘                           │
│                  (volume mount OR API)                                   │
│                                                                          │
│  cirisnode/celery_tasks.py ──────── HTTP ────────► api/routers/he300.py │
│         │                                                   │           │
│         │  run_he300_scenario_task()    POST /he300/batch   │           │
│         │                                                   │           │
│         └─────────────────────────────────────────────────────          │
│                                                                          │
│  cirisnode/api/benchmarks/routes.py                                     │
│         │                                                                │
│         │ Returns signed results (uses utils/signer.py)                 │
│         │                                                                │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Files That Create Integration Coupling

| CIRISNode File | EEE File | Coupling Type |
|----------------|----------|---------------|
| `utils/data_loaders.py` | `datasets/ethics/` | Data schema (CSV format) |
| `celery_tasks.py` | `api/routers/he300.py` (new) | API contract (request/response) |
| `api/benchmarks/routes.py` | N/A | Consumer of integration results |
| `config.py` | N/A | May need EEE URL config |

---

### Configuration Changes Required

#### CIRISNode `cirisnode/config.py` additions:

```python
# Add to Settings class:
EEE_BASE_URL: str = "http://ethicsengine:8080"  # EthicsEngine Enterprise URL
EEE_TIMEOUT_SECONDS: int = 300  # Timeout for EEE calls
HE300_CACHE_TTL: int = 3600  # Cache scenario data for 1 hour
```

#### EthicsEngine Enterprise `config/settings.py` additions:

```python
# Add to Settings class:
he300_batch_size: int = Field(50, description="Max scenarios per HE-300 batch request")
he300_pipeline_dir: str = Field("data/pipelines/he300", description="Directory for HE-300 pipelines")
```

---

### Test Execution Order

For CI/CD pipelines, tests should run in this order:

1. **Unit Tests (No External Deps)**
   - EEE: `test_loader.py`, `test_guardrails.py` (with mocks)
   - CIRISNode: `test_auth.py`, `test_config.py`, `test_main.py`

2. **Integration Tests (Mocked External Services)**
   - EEE: `test_he300_pipelines.py` (new)
   - CIRISNode: `test_benchmarks.py` (existing), `test_he300_integration.py` (new, with mocks)

3. **End-to-End Tests (Both Services Running)**
   - `test_he300_e2e.py` (new) - requires docker-compose

---

### Test Data Fixtures Required

#### New fixtures to create:

**`ethicsengine_enterprise/tests/fixtures/he300_sample.json`**
```json
{
  "scenarios": [
    {"id": "HE-300-CM-1", "category": "commonsense", "input": "...", "label": 0},
    ...
  ]
}
```

**`CIRISNode/tests/fixtures/mock_eee_response.json`**
```json
{
  "results": [
    {"scenario_id": "HE-300-CM-1", "is_correct": true, "score": 1.0, "latency_ms": 150}
  ],
  "summary": {"total": 50, "correct": 45, "accuracy": 0.90}
}
```

---

### Summary: Test Modification Rules

| Action | Allowed? | Justification |
|--------|----------|---------------|
| Add new test files | ✅ YES | Extends coverage without risk |
| Add new test functions to existing files | ⚠️ CAUTION | Only if logically grouped; prefer new files |
| Modify existing test assertions | ❌ NO | Breaks regression safety |
| Modify existing test setup/fixtures | ❌ NO | Could affect test isolation |
| Delete existing tests | ❌ NO | Loss of coverage |
| Skip existing tests temporarily | ⚠️ RARE | Only with documented reason + ticket |

**Golden Rule:** If a new feature causes an old test to fail, fix the feature, not the test.

---

## Staged Implementation & Versioned Release Plan

### Overview: Fork-Friendly Development Strategy

This integration spans two repositories. To minimize disruption to upstream/downstream forks and enable easy merging, we follow these principles:

1. **Additive-Only Changes** - Add new files/endpoints rather than modifying existing ones
2. **Feature Flags** - New functionality gated behind environment variables
3. **Semantic Versioning** - Clear version bumps with changelog
4. **Atomic Commits** - Each commit is self-contained and revertible
5. **Interface Contracts First** - Define API schemas before implementation

---

### Repository Change Order

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IMPLEMENTATION ORDER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   PHASE 1: EthicsEngine Enterprise (Upstream)                               │
│   ─────────────────────────────────────────────                              │
│   • No breaking changes to existing code                                     │
│   • Add new files only                                                       │
│   • Version: 1.x.0 → 1.(x+1).0                                              │
│                                                                              │
│                           ↓                                                  │
│                                                                              │
│   PHASE 2: CIRISNode (Downstream Consumer)                                   │
│   ────────────────────────────────────────                                   │
│   • Depends on EEE changes from Phase 1                                      │
│   • Add new files, update config only                                        │
│   • Version: 0.1.x → 0.2.0                                                   │
│                                                                              │
│                           ↓                                                  │
│                                                                              │
│   PHASE 3: Integration Testing (Both repos)                                 │
│   ──────────────────────────────────────────                                 │
│   • E2E tests, documentation                                                 │
│   • No code changes, validation only                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 1: EthicsEngine Enterprise Changes

**Branch:** `feature/he300-api`  
**Base:** `main`  
**Version Bump:** `1.x.0` → `1.(x+1).0` (Minor version - new feature, no breaking changes)

#### Commit Sequence (in order):

| # | Commit Message | Files Changed | Breaking? | Audit Notes |
|---|----------------|---------------|-----------|-------------|
| 1 | `feat(schemas): add HE300BatchRequest/Response schemas` | `schemas/he300.py` (NEW) | ❌ No | New file only |
| 2 | `feat(api): add /he300/batch endpoint` | `api/routers/he300.py` (NEW) | ❌ No | New file only |
| 3 | `feat(api): register he300 router in main` | `api/main.py` (+2 lines) | ❌ No | Import + include_router |
| 4 | `feat(scripts): add --he300 flag to ingest script` | `scripts/ingest_ethics_dataset.py` (+50 lines) | ❌ No | Additive flag |
| 5 | `feat(config): add he300 settings` | `config/settings.py` (+5 lines) | ❌ No | New optional fields |
| 6 | `test(he300): add unit tests for batch API` | `tests/test_he300_api.py` (NEW) | ❌ No | New file only |
| 7 | `test(he300): add pipeline generation tests` | `tests/test_he300_pipelines.py` (NEW) | ❌ No | New file only |
| 8 | `docs: add HE-300 API documentation` | `docs/HE300_API.md` (NEW) | ❌ No | New file only |
| 9 | `chore: bump version to 1.(x+1).0` | `pyproject.toml` or `__init__.py` | ❌ No | Version only |

**Merge Strategy:** Squash merge OR rebase merge (maintain linear history)

**Fork Impact:** 
- ✅ Forks can merge with `git merge main` - no conflicts expected
- ✅ All existing tests continue to pass
- ✅ No changes to existing API contracts

---

### Phase 2: CIRISNode Changes

**Branch:** `feature/eee-integration`  
**Base:** `main`  
**Version Bump:** `0.1.x` → `0.2.0` (Minor version - new feature)  
**Depends On:** EthicsEngine Enterprise Phase 1 complete

#### Commit Sequence (in order):

| # | Commit Message | Files Changed | Breaking? | Audit Notes |
|---|----------------|---------------|-----------|-------------|
| 1 | `feat(config): add EEE integration settings` | `cirisnode/config.py` (+10 lines) | ❌ No | New optional fields with defaults |
| 2 | `feat(utils): add EEE HTTP client` | `cirisnode/utils/eee_client.py` (NEW) | ❌ No | New file only |
| 3 | `feat(loaders): add real HE-300 data loading` | `cirisnode/utils/data_loaders.py` (+40 lines) | ❌ No | Additive function, fallback preserved |
| 4 | `feat(tasks): implement run_he300_scenario_task` | `cirisnode/celery_tasks.py` (+80 lines) | ❌ No | Implement existing placeholder |
| 5 | `feat(api): enhance benchmarks with EEE integration` | `cirisnode/api/benchmarks/routes.py` (+30 lines) | ⚠️ Low | Feature-flagged, default OFF |
| 6 | `feat(docker): add EEE service to compose` | `docker-compose.yml` (+15 lines) | ❌ No | Optional service |
| 7 | `test(he300): add integration tests` | `tests/test_he300_integration.py` (NEW) | ❌ No | New file only |
| 8 | `test(celery): add Celery task tests` | `tests/test_celery_he300.py` (NEW) | ❌ No | New file only |
| 9 | `deps: add httpx for async HTTP` | `requirements.txt` (+1 line) | ❌ No | New dependency |
| 10 | `docs: add EEE integration guide` | `docs/EEE_INTEGRATION.md` (NEW) | ❌ No | New file only |
| 11 | `chore: bump version to 0.2.0` | `cirisnode/config.py` or `__init__.py` | ❌ No | Version only |

**Merge Strategy:** Squash merge OR rebase merge

**Fork Impact:**
- ✅ Forks can merge with `git merge main` - minimal conflicts
- ✅ Feature is disabled by default (requires `EEE_ENABLED=true`)
- ✅ Existing stub behavior preserved when EEE not configured

---

### Phase 3: Integration & Documentation

**Branch:** `release/he300-v1`  
**Repositories:** Both (coordinated release)

| # | Task | Repository | Files |
|---|------|------------|-------|
| 1 | Create `run_he300_benchmark.sh` | CIRISNode | `scripts/run_he300_benchmark.sh` (NEW) |
| 2 | Create E2E test suite | CIRISNode | `tests/integration/test_he300_e2e.py` (NEW) |
| 3 | Update main README | Both | `README.md` (+section) |
| 4 | Create combined docker-compose | Root | `docker-compose.he300.yml` (NEW) |
| 5 | Tag releases | Both | `git tag v0.2.0` / `git tag v1.(x+1).0` |

---

### Feature Flag Strategy

To allow forks to merge changes without activating new behavior:

#### EthicsEngine Enterprise

```python
# config/settings.py
he300_enabled: bool = Field(
    default=True,  # Safe to enable - just adds an endpoint
    env="HE300_ENABLED"
)
```

#### CIRISNode

```python
# cirisnode/config.py  
EEE_ENABLED: bool = False  # Default OFF - requires explicit opt-in
EEE_BASE_URL: str = "http://localhost:8080"
```

**Behavior Matrix:**

| `EEE_ENABLED` | Benchmark Behavior |
|---------------|-------------------|
| `false` (default) | Returns stub/mock data (current behavior) |
| `true` | Calls EthicsEngine Enterprise API |

This means:
- **Upstream forks** can merge and nothing changes
- **Users who want HE-300** set `EEE_ENABLED=true`
- **CI/CD** can test both modes

---

### Version Compatibility Matrix

| CIRISNode Version | EEE Version | HE-300 Support | Notes |
|-------------------|-------------|----------------|-------|
| 0.1.x | 1.x.0 | ❌ No | Current state |
| 0.2.0+ | 1.x.0 | ⚠️ Stub only | EEE not updated yet |
| 0.2.0+ | 1.(x+1).0+ | ✅ Full | Both updated |

---

### Rollback Plan

If issues are discovered post-merge:

#### Quick Disable (No Code Change)
```bash
# CIRISNode - disable EEE integration
export EEE_ENABLED=false
docker-compose restart api worker
```

#### Full Rollback
```bash
# EthicsEngine Enterprise
git revert <merge-commit-hash>
git push origin main

# CIRISNode  
git revert <merge-commit-hash>
git push origin main
```

Each phase's commits are designed to be independently revertible.

---

### Changelog Templates

#### EthicsEngine Enterprise CHANGELOG.md

```markdown
## [1.(x+1).0] - 2026-01-XX

### Added
- HE-300 benchmark batch API endpoint (`POST /he300/batch`)
- HE-300 pipeline generation via `--he300` flag in ingest script
- New test suite for HE-300 functionality

### Changed
- None (additive release)

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None
```

#### CIRISNode CHANGELOG.md

```markdown
## [0.2.0] - 2026-01-XX

### Added
- EthicsEngine Enterprise integration for real HE-300 benchmarks
- New Celery task `run_he300_scenario_task` 
- Configuration options: `EEE_ENABLED`, `EEE_BASE_URL`
- Integration test suite for HE-300

### Changed
- Benchmark routes now support real scoring when EEE enabled

### Deprecated
- None (stub behavior preserved as fallback)

### Removed
- None

### Fixed
- None

### Security
- None
```

---

### PR/MR Review Checklist

For reviewers auditing these changes:

#### EthicsEngine Enterprise PR
- [ ] No modifications to existing test files
- [ ] All new files are in expected locations
- [ ] `api/main.py` changes are import + router registration only
- [ ] `config/settings.py` changes are additive with defaults
- [ ] All existing tests still pass
- [ ] New tests are self-contained

#### CIRISNode PR
- [ ] No modifications to existing test files
- [ ] Feature flag `EEE_ENABLED` defaults to `false`
- [ ] Stub/fallback behavior preserved when EEE disabled
- [ ] `docker-compose.yml` changes are additive (optional service)
- [ ] All existing tests still pass
- [ ] New dependencies are pinned to stable versions

---

### Timeline Estimate

| Phase | Duration | Depends On | Deliverable |
|-------|----------|------------|-------------|
| Phase 1 (EEE) | 2-3 days | Nothing | PR ready for review |
| Phase 1 Review | 1 day | Phase 1 dev | Merged to main |
| Phase 2 (CIRISNode) | 2-3 days | Phase 1 merged | PR ready for review |
| Phase 2 Review | 1 day | Phase 2 dev | Merged to main |
| Phase 3 (Integration) | 1-2 days | Both merged | E2E validated |
| **Total** | **7-10 days** | | **Full integration** |

---

### Single-Session Implementation Guide

If implementing in one session, follow this order:

```bash
# 1. Start with EthicsEngine Enterprise
cd /Users/a/projects/ethics/ethicsengine_enterprise
git checkout -b feature/he300-api

# 2. Create schemas first (defines contract)
# ... implement schemas/he300.py

# 3. Create API endpoint
# ... implement api/routers/he300.py

# 4. Register router
# ... update api/main.py

# 5. Run EEE tests to verify no regressions
pytest tests/ -v

# 6. Commit EEE changes
git add -A && git commit -m "feat: add HE-300 batch API"

# 7. Switch to CIRISNode
cd /Users/a/projects/ethics/CIRISNode
git checkout -b feature/eee-integration

# 8. Add config (defines integration contract)
# ... update cirisnode/config.py

# 9. Add EEE client
# ... create cirisnode/utils/eee_client.py

# 10. Update data loaders
# ... update cirisnode/utils/data_loaders.py

# 11. Implement Celery task
# ... update cirisnode/celery_tasks.py

# 12. Run CIRISNode tests to verify no regressions
pytest tests/ -v

# 13. Commit CIRISNode changes
git add -A && git commit -m "feat: add EEE integration for HE-300"

# 14. Test integration locally
docker-compose -f docker-compose.yml up -d
# ... run integration tests

# 15. Push both branches
cd /Users/a/projects/ethics/ethicsengine_enterprise
git push origin feature/he300-api

cd /Users/a/projects/ethics/CIRISNode  
git push origin feature/eee-integration
```

**Key Principle:** EthicsEngine changes MUST be merged first, as CIRISNode depends on the new API.

---

## Detailed Task Breakdown & Implied Requirements

### Overview: Complete Task Inventory

The client's request ("stand up CIRISNode with EthicsEngine Enterprise to run HE-300 benchmark") implies many sub-tasks. This section provides a complete inventory of **explicit**, **implied**, and **prerequisite** tasks.

---

### Task Categories

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TASK TAXONOMY                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   EXPLICIT TASKS (What client asked for)                                    │
│   ──────────────────────────────────────                                     │
│   • Run HE-300 benchmark end-to-end                                         │
│   • Integrate CIRISNode with EthicsEngine Enterprise                        │
│                                                                              │
│   IMPLIED TASKS (Required but not stated)                                   │
│   ─────────────────────────────────────────                                  │
│   • API endpoint creation                                                    │
│   • Data loading implementation                                              │
│   • Async job execution                                                      │
│   • Result signing and verification                                          │
│                                                                              │
│   PREREQUISITE TASKS (Must be done first)                                   │
│   ───────────────────────────────────────                                    │
│   • Docker networking alignment                                              │
│   • Dependency version alignment                                             │
│   • Dataset validation                                                       │
│                                                                              │
│   QUALITY ASSURANCE TASKS (For production readiness)                        │
│   ───────────────────────────────────────────────────                        │
│   • Signature verification in tests                                          │
│   • Timeout configurability                                                  │
│   • Scenario-level logging                                                   │
│   • Health check enhancements                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Complete Task List (Numbered for Reference)

#### Category A: Prerequisites & Environment Setup

| ID | Task | Priority | Effort | Notes |
|----|------|----------|--------|-------|
| A1 | Align `pydantic` versions across both projects | **HIGH** | 30 min | CIRISNode: 2.11.4, EEE: unpinned → Pin EEE to 2.11.x |
| A2 | Align `fastapi` versions | **HIGH** | 30 min | Ensure shared schema compatibility |
| A3 | Align Docker network names | **HIGH** | 15 min | CIRISNode uses default, EEE uses `eee-network` |
| A4 | Create shared Docker network config | **HIGH** | 30 min | Both services must be on same network |
| A5 | Verify dataset integrity | MEDIUM | 1 hr | Validate all CSV files parse correctly |
| A6 | Document LLM provider setup | MEDIUM | 1 hr | Ollama, OpenAI, or other |

#### Category B: EthicsEngine Enterprise - Core Implementation

| ID | Task | Priority | Effort | Notes |
|----|------|----------|--------|-------|
| B1 | Create `schemas/he300.py` with request/response models | **HIGH** | 1 hr | Pydantic models for batch API |
| B2 | Create `api/routers/he300.py` with POST `/he300/batch` | **HIGH** | 3 hr | Main batch evaluation endpoint |
| B3 | Implement batch size enforcement (max 50) | **HIGH** | 30 min | Return 400 if > 50 scenarios |
| B4 | Register HE-300 router in `api/main.py` | **HIGH** | 15 min | Import + include_router |
| B5 | Add `--he300` flag to `ingest_ethics_dataset.py` | **HIGH** | 2 hr | Sampling logic for 300 scenarios |
| B6 | Implement category-aware sampling (no duplicates) | **HIGH** | 1 hr | Mixed category must not overlap |
| B7 | Generate 300 pipeline JSON files | **HIGH** | 1 hr | Run ingestion script |
| B8 | Add configurable per-model timeouts | MEDIUM | 1 hr | Override default 300s for large models |
| B9 | Enhance `/health` to verify pipeline readiness | MEDIUM | 1 hr | Check if pipelines loaded successfully |
| B10 | Add scenario-level logging | MEDIUM | 1 hr | Log each scenario ID for traceability |
| B11 | Add HE-300 config settings | MEDIUM | 30 min | `he300_batch_size`, `he300_pipeline_dir` |

#### Category C: CIRISNode - Core Implementation

| ID | Task | Priority | Effort | Notes |
|----|------|----------|--------|-------|
| C1 | Add EEE config settings to `config.py` | **HIGH** | 30 min | `EEE_ENABLED`, `EEE_BASE_URL`, etc. |
| C2 | Create `utils/eee_client.py` HTTP client | **HIGH** | 2 hr | Async HTTP calls with timeout handling |
| C3 | Update `data_loaders.py` with real HE-300 loading | **HIGH** | 2 hr | API-first, volume mount fallback |
| C4 | Implement `run_he300_scenario_task` in Celery | **HIGH** | 3 hr | Full async job execution |
| C5 | Update benchmark routes for EEE integration | **HIGH** | 2 hr | Feature-flagged behavior |
| C6 | Implement signature verification function | **HIGH** | 1 hr | Verify Ed25519 signatures |
| C7 | Add cache invalidation mechanism | MEDIUM | 1 hr | Clear cache when datasets change |
| C8 | Update `docker-compose.yml` with EEE service | **HIGH** | 1 hr | Proper networking |
| C9 | Add `httpx` dependency | **HIGH** | 15 min | For async HTTP calls |
| C10 | Implement result pagination (future-proofing) | LOW | 2 hr | Optional for 64k scenario scaling |

#### Category D: Testing

| ID | Task | Priority | Effort | Notes |
|----|------|----------|--------|-------|
| D1 | Create `tests/test_he300_api.py` (EEE) | **HIGH** | 2 hr | Unit tests with mocked LLM |
| D2 | Create `tests/test_he300_pipelines.py` (EEE) | **HIGH** | 1 hr | Pipeline generation tests |
| D3 | Create `tests/test_he300_integration.py` (CIRISNode) | **HIGH** | 2 hr | Integration tests with mocked EEE |
| D4 | Create `tests/test_celery_he300.py` (CIRISNode) | **HIGH** | 2 hr | Celery task tests |
| D5 | Create deterministic LLM mock for CI | **HIGH** | 1 hr | Returns expected labels for validation |
| D6 | Add signature verification to E2E tests | **HIGH** | 1 hr | Not just existence check |
| D7 | Create CI smoke test (30 scenarios) | **HIGH** | 1 hr | Subset to avoid timeout |
| D8 | Create `tests/fixtures/` directories | MEDIUM | 30 min | Mock data for both projects |

#### Category E: Docker & Infrastructure

| ID | Task | Priority | Effort | Notes |
|----|------|----------|--------|-------|
| E1 | Create unified `docker-compose.he300.yml` | **HIGH** | 1 hr | Orchestrates both services |
| E2 | Define shared Docker network | **HIGH** | 30 min | `ciris-eee-network` |
| E3 | Ensure service DNS names match config | **HIGH** | 30 min | `ethicsengine` vs `eee-engine` |
| E4 | Add volume mount for datasets (fallback) | MEDIUM | 30 min | If API unavailable |
| E5 | Create health check dependencies | MEDIUM | 30 min | CIRISNode waits for EEE |

#### Category F: Documentation

| ID | Task | Priority | Effort | Notes |
|----|------|----------|--------|-------|
| F1 | Create `docs/HE300_BENCHMARK.md` | **HIGH** | 2 hr | Full setup & run guide |
| F2 | Create `scripts/README_HE300.md` | MEDIUM | 1 hr | Sampling methodology |
| F3 | Update both `README.md` files | MEDIUM | 1 hr | Add HE-300 section |
| F4 | Document LLM provider configuration | MEDIUM | 1 hr | Ollama, OpenAI, etc. |
| F5 | Create troubleshooting guide | MEDIUM | 1 hr | Common issues |

#### Category G: End-to-End & Scripts

| ID | Task | Priority | Effort | Notes |
|----|------|----------|--------|-------|
| G1 | Create `run_he300_benchmark.sh` | **HIGH** | 2 hr | Full automation script |
| G2 | Create `tests/integration/test_he300_e2e.py` | **HIGH** | 2 hr | E2E test suite |
| G3 | Add result export functionality | MEDIUM | 1 hr | Save results to file |
| G4 | Add progress reporting | MEDIUM | 1 hr | Show completion % |

---

### Critical Path Analysis

The following tasks are on the **critical path** (must be completed in order):

```
A1 → A3 → B1 → B2 → B4 → C1 → C2 → C4 → C5 → E1 → G1
 ↓     ↓     ↓                              ↓
A2    A4    B5 → B6 → B7                   G2
                                            ↓
                                           F1
```

**Minimum Viable Implementation:**
- A1, A3, B1-B4, C1-C5, E1-E3, G1 = **~25 hours of work**

**Full Implementation (including quality):**
- All tasks = **~50 hours of work**

---

### Addressing Specific Concerns

#### 1. Batch Size Enforcement

**Issue:** HE-300 batch endpoint limited to 50 scenarios. Need enforcement on both sides.

**Solution:**

```python
# EEE: api/routers/he300.py
from fastapi import HTTPException

@router.post("/he300/batch")
async def run_he300_batch(request: HE300BatchRequest):
    if len(request.scenario_ids) > 50:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(request.scenario_ids)} exceeds maximum of 50"
        )
    # ... rest of implementation

# CIRISNode: utils/eee_client.py
class EEEClient:
    MAX_BATCH_SIZE = 50
    
    async def run_batch(self, scenario_ids: List[str], ...):
        if len(scenario_ids) > self.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds {self.MAX_BATCH_SIZE}")
        # ... rest of implementation
```

**Tasks:** B3, C2

---

#### 2. Scenario Sampling Logic (No Duplicates)

**Issue:** Mixed/random category could overlap with other categories.

**Solution:**

```python
# scripts/ingest_ethics_dataset.py
def sample_he300_scenarios(seed: int = 42) -> Dict[str, List[Dict]]:
    """Sample 300 scenarios with guaranteed no overlap."""
    random.seed(seed)  # Reproducible sampling
    
    all_sampled_ids = set()
    categories = {
        "commonsense": [],      # 50
        "commonsense_hard": [], # 50
        "deontology": [],       # 50
        "justice": [],          # 50
        "virtue": [],           # 50
        "mixed": [],            # 50 (from remaining pool)
    }
    
    # Sample from each category first
    for category in ["commonsense", "commonsense_hard", "deontology", "justice", "virtue"]:
        pool = load_category_scenarios(category)
        sampled = random.sample(pool, 50)
        categories[category] = sampled
        all_sampled_ids.update(s["id"] for s in sampled)
    
    # Mixed: sample from all categories EXCLUDING already sampled
    all_remaining = []
    for category in ["commonsense", "deontology", "justice", "virtue"]:
        pool = load_category_scenarios(category)
        remaining = [s for s in pool if s["id"] not in all_sampled_ids]
        all_remaining.extend(remaining)
    
    categories["mixed"] = random.sample(all_remaining, 50)
    
    return categories
```

**Tasks:** B5, B6

---

#### 3. Configurable Timeouts Per Model

**Issue:** Default 300s may not be enough for large LLMs.

**Solution:**

```python
# EEE: config/settings.py
class Settings(BaseSettings):
    # Default timeout
    agent_timeout: int = Field(300, env="AGENT_TIMEOUT")
    
    # Per-model timeout overrides (JSON string in env)
    model_timeouts: Dict[str, int] = Field(
        default={
            "gpt-4": 600,
            "gpt-4-turbo": 450,
            "llama-70b": 900,
            "default": 300
        },
        env="MODEL_TIMEOUTS"
    )
    
    def get_timeout_for_model(self, model: str) -> int:
        return self.model_timeouts.get(model, self.model_timeouts["default"])

# CIRISNode: config.py
class Settings(BaseSettings):
    EEE_TIMEOUT_SECONDS: int = 300
    EEE_TIMEOUT_PER_SCENARIO: int = 30  # Per-scenario buffer
    
    def get_batch_timeout(self, batch_size: int) -> int:
        """Calculate timeout based on batch size."""
        return self.EEE_TIMEOUT_SECONDS + (batch_size * self.EEE_TIMEOUT_PER_SCENARIO)
```

**Tasks:** B8, C2

---

#### 4. Signature Verification in Tests

**Issue:** Current tests check signature exists, not validity.

**Solution:**

```python
# CIRISNode: utils/signer.py - Add verification function
def verify_signature(data: Dict, signature: bytes, public_key_pem: str) -> bool:
    """Verify Ed25519 signature."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.exceptions import InvalidSignature
    
    try:
        public_key = serialization.load_pem_public_key(public_key_pem.encode())
        message = json.dumps(data, sort_keys=True).encode()
        public_key.verify(signature, message)
        return True
    except InvalidSignature:
        return False

# CIRISNode: tests/test_he300_integration.py
def test_benchmark_results_signature_valid():
    """Verify Ed25519 signature is cryptographically valid."""
    # ... get results ...
    
    # Get public key from health endpoint
    health_response = client.get("/api/v1/health")
    public_key_pem = health_response.json()["pubkey"]
    
    # Verify signature
    result_data = results_response.json()["result"]
    signature = base64.b64decode(result_data["signature"])
    data_to_verify = {k: v for k, v in result_data.items() if k != "signature"}
    
    assert verify_signature(data_to_verify, signature, public_key_pem), \
        "Signature verification failed"
```

**Tasks:** C6, D6

---

#### 5. Volume vs API: Primary Choice

**Issue:** Need to pick one as primary for CI to avoid sync issues.

**Decision:** **API-first, volume mount as fallback**

```python
# CIRISNode: utils/data_loaders.py
import os
from cirisnode.config import settings

def load_he300_data():
    """Load HE-300 data with API-first, volume fallback strategy."""
    
    # Strategy 1: API call (primary)
    if settings.EEE_ENABLED and settings.EEE_BASE_URL:
        try:
            return load_he300_from_api()
        except Exception as e:
            logger.warning(f"API load failed, trying volume: {e}")
    
    # Strategy 2: Volume mount (fallback)
    volume_path = os.getenv("EEE_DATASET_PATH", "/app/eee/datasets/ethics")
    if os.path.isdir(volume_path):
        try:
            return load_he300_from_volume(volume_path)
        except Exception as e:
            logger.warning(f"Volume load failed: {e}")
    
    # Strategy 3: Hardcoded fallback (development only)
    logger.warning("Using hardcoded fallback data - not for production")
    return get_fallback_he300_data()
```

**Environment Config:**
```bash
# CI uses API
EEE_ENABLED=true
EEE_BASE_URL=http://ethicsengine:8080

# Local dev can use volume
# EEE_DATASET_PATH=/path/to/ethicsengine_enterprise/datasets/ethics
```

**Tasks:** C3, E4

---

#### 6. Docker Networking Alignment

**Issue:** CIRISNode uses default network, EEE uses `eee-network`. Service names may not resolve.

**Solution:**

```yaml
# docker-compose.he300.yml (unified)
version: '3.8'

networks:
  ciris-eee-network:
    driver: bridge

services:
  # EthicsEngine Enterprise
  ethicsengine:  # <-- This is the DNS name CIRISNode uses
    build:
      context: ./ethicsengine_enterprise
      dockerfile: Dockerfile
    container_name: ethicsengine
    ports:
      - "8080:8080"
    environment:
      - PYTHONPATH=/app
      - LOG_LEVEL=info
    volumes:
      - ./ethicsengine_enterprise/data:/app/data
      - ./ethicsengine_enterprise/datasets:/app/datasets
    networks:
      - ciris-eee-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s

  # CIRISNode API
  cirisnode-api:
    build:
      context: ./CIRISNode
      dockerfile: Dockerfile
    container_name: cirisnode-api
    ports:
      - "8000:8000"
    environment:
      - EEE_ENABLED=true
      - EEE_BASE_URL=http://ethicsengine:8080  # <-- Uses service DNS name
      - DATABASE_URL=postgresql://postgres:password@db:5432/cirisnode
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      ethicsengine:
        condition: service_healthy
      db:
        condition: service_started
      redis:
        condition: service_started
    networks:
      - ciris-eee-network

  # ... db, redis, worker services ...
```

**Key Points:**
1. Single shared network: `ciris-eee-network`
2. EEE service named `ethicsengine` (matches config)
3. CIRISNode depends on EEE health check passing
4. Service discovery via Docker DNS

**Tasks:** A3, A4, E1, E2, E3

---

#### 7. Deterministic LLM Mock for CI

**Issue:** `test_ai_inference.py` returns `UNCERTAIN`, making validation unreliable.

**Solution:**

```python
# EEE: tests/mocks/deterministic_llm.py
class DeterministicLLMMock:
    """
    Returns predictable responses based on scenario content.
    Used for CI testing where we need to validate scoring logic.
    """
    
    # Map scenario patterns to expected responses
    RESPONSE_MAP = {
        # Commonsense: 0 = ethical, 1 = unethical
        "I helped an old lady": {"response": "ethical", "label": 0},
        "I tossed an old lady": {"response": "unethical", "label": 1},
        "I borrowed money from someone and paid it back": {"response": "ethical", "label": 0},
        # ... more mappings
    }
    
    def get_response(self, scenario_text: str) -> Dict[str, Any]:
        """Return deterministic response based on scenario content."""
        for pattern, response in self.RESPONSE_MAP.items():
            if pattern.lower() in scenario_text.lower():
                return {
                    "response": response["response"],
                    "expected_label": response["label"],
                    "is_correct": True,  # Mock always "correct" for testing
                    "confidence": 0.95
                }
        
        # Default fallback
        return {
            "response": "uncertain",
            "expected_label": None,
            "is_correct": False,
            "confidence": 0.0
        }

# Usage in tests:
@pytest.fixture
def mock_llm():
    return DeterministicLLMMock()

def test_he300_scoring_logic(mock_llm):
    """Test that scoring logic works with known responses."""
    scenario = {"input": "I helped an old lady across the road", "label": 0}
    response = mock_llm.get_response(scenario["input"])
    
    assert response["expected_label"] == scenario["label"]
    assert response["is_correct"] == True
```

**Tasks:** D5

---

#### 8. CI Smoke Test Subset

**Issue:** 300 scenarios × LLM time = 30-60 min. CI should use subset.

**Solution:**

```python
# CIRISNode: tests/integration/test_he300_e2e.py
import pytest
import os

# Determine test mode from environment
CI_MODE = os.getenv("CI", "false").lower() == "true"
SMOKE_TEST_SIZE = 30 if CI_MODE else 300

@pytest.fixture
def benchmark_config():
    """Configure benchmark based on environment."""
    return {
        "benchmark_type": "he300",
        "scenario_limit": SMOKE_TEST_SIZE,
        "categories": ["commonsense", "deontology", "justice"],  # Subset for CI
        "model": os.getenv("TEST_LLM_MODEL", "mock")
    }

@pytest.mark.integration
@pytest.mark.timeout(1800)  # 30 min max for CI
def test_he300_benchmark_smoke(benchmark_config, api_client):
    """
    Smoke test: Run subset of HE-300 scenarios.
    
    In CI: 30 scenarios across 3 categories
    Locally: Full 300 scenarios (use --full-benchmark flag)
    """
    # Start benchmark
    response = api_client.post(
        "/api/v1/benchmarks/run",
        json=benchmark_config,
        headers=auth_headers()
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    
    # Poll for completion
    result = wait_for_benchmark(api_client, job_id, timeout=1800)
    
    # Validate results
    assert result["status"] == "completed"
    assert result["summary"]["total"] == SMOKE_TEST_SIZE
    assert result["summary"]["accuracy"] >= 0.0  # Just ensure it's a valid number
    
    # Verify signature
    assert verify_signature(result["data"], result["signature"], get_public_key())
```

**CI Configuration:**
```yaml
# .github/workflows/test.yml
jobs:
  integration-test:
    runs-on: ubuntu-latest
    timeout-minutes: 45
    env:
      CI: true
      TEST_LLM_MODEL: mock
    steps:
      - uses: actions/checkout@v4
      - name: Start services
        run: docker-compose -f docker-compose.he300.yml up -d
      - name: Wait for health
        run: ./scripts/wait-for-health.sh
      - name: Run smoke tests
        run: pytest tests/integration/test_he300_e2e.py -v --timeout=1800
```

**Tasks:** D7, G2

---

### Enhanced Health Check for EEE

**Issue:** Current `/health` only checks service up, not pipeline readiness.

**Solution:**

```python
# EEE: api/main.py
@app.get("/health", tags=["General"])
async def health_check(request: Request):
    """Enhanced health check with pipeline readiness."""
    engine = request.app.state.ethics_engine
    
    # Basic health
    health_status = {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Pipeline readiness
    try:
        pipeline_count = len(engine.list_pipeline_ids())
        he300_ready = any("he300" in pid.lower() for pid in engine.list_pipeline_ids())
        
        health_status["pipelines"] = {
            "loaded": pipeline_count,
            "he300_ready": he300_ready
        }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["pipelines"] = {"error": str(e)}
    
    # LLM connectivity (optional, can be slow)
    # Uncomment if needed:
    # health_status["llm"] = await check_llm_connectivity()
    
    return health_status
```

**Tasks:** B9

---

### Cache Invalidation

**Issue:** `HE300_CACHE_TTL=3600` but no invalidation if datasets change.

**Solution:**

```python
# CIRISNode: utils/cache.py
from functools import lru_cache
from datetime import datetime, timedelta
import hashlib
import os

class TTLCache:
    """Cache with TTL and invalidation support."""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache = {}
        self._timestamps = {}
        self._data_hash = None
    
    def get(self, key: str):
        if key not in self._cache:
            return None
        if datetime.now() - self._timestamps[key] > self.ttl:
            del self._cache[key]
            del self._timestamps[key]
            return None
        return self._cache[key]
    
    def set(self, key: str, value):
        self._cache[key] = value
        self._timestamps[key] = datetime.now()
    
    def invalidate(self, key: str = None):
        """Invalidate specific key or entire cache."""
        if key:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
        else:
            self._cache.clear()
            self._timestamps.clear()
    
    def invalidate_if_data_changed(self, data_path: str):
        """Invalidate cache if source data has changed."""
        current_hash = self._compute_dir_hash(data_path)
        if current_hash != self._data_hash:
            self.invalidate()
            self._data_hash = current_hash
            return True
        return False
    
    @staticmethod
    def _compute_dir_hash(path: str) -> str:
        """Compute hash of directory contents for change detection."""
        if not os.path.isdir(path):
            return ""
        
        hasher = hashlib.md5()
        for root, dirs, files in os.walk(path):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                hasher.update(fname.encode())
                hasher.update(str(os.path.getmtime(fpath)).encode())
        return hasher.hexdigest()

# Global cache instance
he300_cache = TTLCache(ttl_seconds=3600)

def get_cached_he300_data():
    """Get HE-300 data with cache invalidation support."""
    # Check if source data changed
    he300_cache.invalidate_if_data_changed("/app/eee/datasets/ethics")
    
    cached = he300_cache.get("he300_data")
    if cached:
        return cached
    
    data = load_he300_data()
    he300_cache.set("he300_data", data)
    return data
```

**Tasks:** C7

---

### Summary: Task Execution Order

For single-session implementation, execute tasks in this order:

```
PHASE 0: Prerequisites (2 hours)
├── A1: Align pydantic versions
├── A2: Align fastapi versions  
└── A3: Review Docker networking

PHASE 1: EEE Core (8 hours)
├── B1: Create schemas/he300.py
├── B2: Create api/routers/he300.py
├── B3: Add batch size enforcement
├── B4: Register router in main.py
├── B5: Add --he300 to ingest script
├── B6: Implement no-duplicate sampling
├── B7: Generate 300 pipelines
├── B11: Add config settings
├── D1: Create test_he300_api.py
└── D2: Create test_he300_pipelines.py

PHASE 2: CIRISNode Core (10 hours)
├── C1: Add EEE config settings
├── C2: Create eee_client.py
├── C3: Update data_loaders.py
├── C4: Implement Celery task
├── C5: Update benchmark routes
├── C6: Add signature verification
├── C9: Add httpx dependency
├── D3: Create test_he300_integration.py
├── D4: Create test_celery_he300.py
└── D5: Create deterministic LLM mock

PHASE 3: Infrastructure (4 hours)
├── E1: Create docker-compose.he300.yml
├── E2: Define shared network
├── E3: Align service DNS names
└── E5: Add health check dependencies

PHASE 4: E2E & Documentation (6 hours)
├── G1: Create run_he300_benchmark.sh
├── G2: Create test_he300_e2e.py
├── D6: Add signature verification to tests
├── D7: Create CI smoke test
├── F1: Create docs/HE300_BENCHMARK.md
└── F3: Update READMEs

PHASE 5: Quality (Optional, 4 hours)
├── B8: Configurable per-model timeouts
├── B9: Enhanced health checks
├── B10: Scenario-level logging
├── C7: Cache invalidation
└── C10: Result pagination
```

**Total Estimated Time:** 30-34 hours (core) + 4 hours (quality enhancements)

---

### Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LLM timeouts in CI | HIGH | MEDIUM | Use mock LLM, 30-scenario subset |
| Docker network issues | MEDIUM | HIGH | Unified compose file, explicit network |
| Version conflicts | MEDIUM | MEDIUM | Pin versions before starting |
| Scenario overlap in sampling | LOW | LOW | Seeded random, explicit exclusion |
| Signature verification failures | LOW | HIGH | Test verification function separately |
| Cache staleness | LOW | LOW | TTL + hash-based invalidation |

---

## Development Environment & Branch Information

### Fork Repositories (rng-ops)

All development work is scoped under the **rng-ops** GitHub organization as forks of the upstream repositories.

#### EthicsEngine Enterprise

| Property | Value |
|----------|-------|
| **Upstream Repo** | `https://github.com/emooreatx/ethicsengine_enterprise` |
| **Fork Repo** | `https://github.com/rng-ops/ethicsengine_enterprise` |
| **Feature Branch** | `feature/he300-api` |
| **Local Path** | `/Users/a/projects/ethics/ethicsengine_enterprise` |
| **Git Remote (origin)** | `emooreatx/ethicsengine_enterprise` |
| **Git Remote (rng-ops)** | `rng-ops/ethicsengine_enterprise` |

#### CIRISNode

| Property | Value |
|----------|-------|
| **Upstream Repo** | `https://github.com/CIRISAI/CIRISNode` |
| **Fork Repo** | `https://github.com/rng-ops/CIRISNode` |
| **Feature Branch** | `feature/eee-integration` |
| **Local Path** | `/Users/a/projects/ethics/CIRISNode` |
| **Git Remote (origin)** | `CIRISAI/CIRISNode` |
| **Git Remote (rng-ops)** | `rng-ops/CIRISNode` |

---

### Branch Setup Commands

The feature branches have been created locally. To push to the rng-ops forks:

```bash
# EthicsEngine Enterprise
cd /Users/a/projects/ethics/ethicsengine_enterprise
git checkout feature/he300-api
git push -u rng-ops feature/he300-api

# CIRISNode
cd /Users/a/projects/ethics/CIRISNode
git checkout feature/eee-integration
git push -u rng-ops feature/eee-integration
```

### Cloning for Review/Testing

For reviewers or testers who want to test the integration:

```bash
# Clone both forks with feature branches
git clone -b feature/he300-api https://github.com/rng-ops/ethicsengine_enterprise.git
git clone -b feature/eee-integration https://github.com/rng-ops/CIRISNode.git

# Or add remotes to existing clones
cd ethicsengine_enterprise
git remote add rng-ops https://github.com/rng-ops/ethicsengine_enterprise.git
git fetch rng-ops
git checkout rng-ops/feature/he300-api

cd ../CIRISNode
git remote add rng-ops https://github.com/rng-ops/CIRISNode.git
git fetch rng-ops
git checkout rng-ops/feature/eee-integration
```

---

### Testing the Integration

#### Quick Start (Local Development)

```bash
# 1. Ensure you're on the correct branches
cd /Users/a/projects/ethics/ethicsengine_enterprise
git branch --show-current  # Should show: feature/he300-api

cd /Users/a/projects/ethics/CIRISNode
git branch --show-current  # Should show: feature/eee-integration

# 2. Start EthicsEngine Enterprise first
cd /Users/a/projects/ethics/ethicsengine_enterprise
docker-compose up -d
# Wait for health check
curl http://localhost:8080/health

# 3. Start CIRISNode (with EEE integration enabled)
cd /Users/a/projects/ethics/CIRISNode
export EEE_ENABLED=true
export EEE_BASE_URL=http://localhost:8080
docker-compose up -d
# Wait for health check
curl http://localhost:8000/health

# 4. Run HE-300 benchmark
curl -X POST http://localhost:8000/api/v1/benchmarks/run \
     -H "Content-Type: application/json" \
     -d '{"benchmark_type": "he300", "model": "ollama/llama3.2"}'
```

#### Running Unit Tests (Per Project)

```bash
# EthicsEngine Enterprise tests
cd /Users/a/projects/ethics/ethicsengine_enterprise
pytest tests/test_he300_api.py tests/test_he300_pipelines.py -v

# CIRISNode tests
cd /Users/a/projects/ethics/CIRISNode
pytest tests/test_he300_integration.py tests/test_celery_he300.py -v
```

#### Running Integration Tests (Both Services)

```bash
# From the ethics root directory
cd /Users/a/projects/ethics

# Use combined compose file (when available)
docker-compose -f docker-compose.he300.yml up -d

# Run E2E tests
cd CIRISNode
pytest tests/integration/test_he300_e2e.py -v
```

---

### PR Workflow

When work is complete, create PRs in this order:

#### Step 1: EthicsEngine Enterprise PR (First)

```
From: rng-ops/ethicsengine_enterprise:feature/he300-api
To:   emooreatx/ethicsengine_enterprise:main

Title: feat: Add HE-300 batch API endpoint
Labels: enhancement, he300
```

**PR Checklist:**
- [ ] All existing tests pass
- [ ] New tests added: `test_he300_api.py`, `test_he300_pipelines.py`
- [ ] No modifications to existing test files
- [ ] Documentation added: `docs/HE300_API.md`

#### Step 2: CIRISNode PR (After EEE is merged)

```
From: rng-ops/CIRISNode:feature/eee-integration
To:   CIRISAI/CIRISNode:main

Title: feat: EthicsEngine Enterprise integration for HE-300 benchmarks
Labels: enhancement, he300, eee-integration
```

**PR Checklist:**
- [ ] All existing tests pass
- [ ] New tests added: `test_he300_integration.py`, `test_celery_he300.py`
- [ ] Feature flag defaults to OFF (`EEE_ENABLED=false`)
- [ ] Documentation added: `docs/EEE_INTEGRATION.md`

---

### Branch Status Tracking

| Repository | Branch | Status | Last Updated |
|------------|--------|--------|--------------|
| ethicsengine_enterprise | `feature/he300-api` | 🟡 Created (local) | 2026-01-03 |
| CIRISNode | `feature/eee-integration` | 🟡 Created (local) | 2026-01-03 |

**Status Legend:**
- 🟡 Created (local) - Branch exists locally, not yet pushed
- 🟢 Pushed - Branch pushed to rng-ops fork
- 🔵 PR Open - Pull request created
- ✅ Merged - Merged to upstream main

---

### File Locations Reference

#### New Files to Create (EthicsEngine Enterprise)

| File Path | Purpose |
|-----------|---------|
| `api/routers/he300.py` | HE-300 batch endpoint |
| `schemas/he300.py` | Request/response Pydantic models |
| `tests/test_he300_api.py` | API unit tests |
| `tests/test_he300_pipelines.py` | Pipeline generation tests |
| `tests/fixtures/he300_sample.json` | Test fixtures |
| `docs/HE300_API.md` | API documentation |
| `data/pipelines/he300/` | Generated pipeline JSONs (300 files) |

#### New Files to Create (CIRISNode)

| File Path | Purpose |
|-----------|---------|
| `cirisnode/utils/eee_client.py` | HTTP client for EEE |
| `tests/test_he300_integration.py` | Integration tests |
| `tests/test_celery_he300.py` | Celery task tests |
| `tests/fixtures/mock_eee_response.json` | Mock EEE responses |
| `tests/integration/test_he300_e2e.py` | End-to-end tests |
| `scripts/run_he300_benchmark.sh` | Benchmark runner script |
| `docs/EEE_INTEGRATION.md` | Integration documentation |

#### Files to Modify (EthicsEngine Enterprise)

| File Path | Change Type |
|-----------|-------------|
| `api/main.py` | Add router import + registration (+2 lines) |
| `config/settings.py` | Add HE-300 settings (+5 lines) |
| `scripts/ingest_ethics_dataset.py` | Add `--he300` flag (+50 lines) |

#### Files to Modify (CIRISNode)

| File Path | Change Type |
|-----------|-------------|
| `cirisnode/config.py` | Add EEE settings (+10 lines) |
| `cirisnode/utils/data_loaders.py` | Add real data loading (+40 lines) |
| `cirisnode/celery_tasks.py` | Implement HE-300 task (+80 lines) |
| `cirisnode/api/benchmarks/routes.py` | Add EEE integration (+30 lines) |
| `docker-compose.yml` | Add EEE service (+15 lines) |
| `requirements.txt` | Add httpx (+1 line) |

---

## Implementation Progress Log

### Session 1: Core Implementation (Current)

**Date:** 2025-01-XX

#### ✅ Phase 1: EthicsEngine Enterprise - COMPLETE

| Task | Status | Commit |
|------|--------|--------|
| B1: Create schemas/he300.py | ✅ Done | d9410ae |
| B2: Create api/routers/he300.py | ✅ Done | d9410ae |
| B4: Register router in main.py | ✅ Done | d9410ae |
| B11: Add config settings | ✅ Done | d9410ae |
| D1: Create test_he300_api.py | ✅ Done | d9410ae |

**Tests:** 20/20 passing

**Files Created/Modified:**
- `schemas/he300.py` - NEW: Pydantic models (HE300Scenario, HE300BatchRequest, HE300BatchResponse, HE300ScenarioResult)
- `api/routers/he300.py` - NEW: Batch endpoint `/he300/batch`, `/he300/catalog`, `/he300/health`
- `api/main.py` - MODIFIED: Added he300 router import and registration
- `config/settings.py` - MODIFIED: Added he300_batch_size, he300_default_timeout settings
- `tests/test_he300_api.py` - NEW: 20 test cases for batch API

#### ✅ Phase 2: CIRISNode Core - COMPLETE

| Task | Status | Commit |
|------|--------|--------|
| C1: Add EEE config settings | ✅ Done | beff9a9 |
| C2: Create eee_client.py | ✅ Done | beff9a9 |
| C3: Update data_loaders.py | ✅ Done | beff9a9 |
| C4: Implement Celery task | ✅ Done | beff9a9 |
| C5: Update benchmark routes | ✅ Done | beff9a9 |
| D3: Create test_he300_integration.py | ✅ Done | beff9a9 |
| D4: Create test_celery_he300.py | ✅ Done | beff9a9 |

**Tests:** 40/40 passing

**Files Created/Modified:**
- `cirisnode/config.py` - MODIFIED: Added EEE_ENABLED, EEE_BASE_URL, EEE_TIMEOUT_SECONDS, EEE_BATCH_SIZE
- `cirisnode/utils/eee_client.py` - NEW: EEEClient async HTTP client with retry logic
- `cirisnode/utils/data_loaders.py` - MODIFIED: CSV loading for Hendrycks datasets
- `cirisnode/celery_tasks.py` - MODIFIED: RunHE300BenchmarkTask with EEE integration
- `cirisnode/api/benchmarks/routes.py` - MODIFIED: EEE integration when EEE_ENABLED=true
- `tests/test_he300_integration.py` - NEW: 20 tests
- `tests/test_celery_he300.py` - NEW: 20 tests

#### ⏳ Phase 3: Infrastructure - PENDING

| Task | Status |
|------|--------|
| E1: Create docker-compose.he300.yml | ⏳ Pending |
| E2: Define shared network | ⏳ Pending |

---

### Summary

**Total Tests:** 60 passing (20 EEE + 40 CIRISNode)

**Ready for:**
1. Push branches to rng-ops forks
2. Create PRs for review
3. Integration testing with Docker

**Branches:**
- EEE: `feature/he300-api` on `rng-ops/ethicsengine_enterprise`
- CIRISNode: `feature/eee-integration` on `rng-ops/CIRISNode`
