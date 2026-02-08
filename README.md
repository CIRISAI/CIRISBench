# CIRISBench

AI Agent Ethics Benchmarking Platform implementing the **HE-300 (Hendrycks Ethics)** benchmark with a unified evaluation pipeline, frontier model scoring, and managed benchmarking services via [ethicsengine.org](https://ethicsengine.org).

## Overview

CIRISBench is the **write path** of the ethical benchmarking platform. It evaluates AI models against 300 ethical scenarios, persists results to PostgreSQL, and publishes scores to the public leaderboard via [CIRISNode](https://github.com/CIRISAI/CIRISNode) (the read path).

```
ethicsengine.org (frontend)  -->  CIRISNode (read path)  -->  PostgreSQL
                                                                  ^
                                                          CIRISBench (write path)
                                                          Celery Beat frontier sweep
```

## HE-300 Benchmark

300 ethical scenarios evaluated across five categories:

| Category | Scenarios | Description |
|----------|-----------|-------------|
| **Justice** | 50 | Fairness, desert, and equitable treatment |
| **Deontology** | 50 | Duty-based moral reasoning |
| **Virtue Ethics** | 50 | Character-based moral reasoning |
| **Commonsense** | 75 | Everyday moral intuitions |
| **Commonsense (Hard)** | 75 | Challenging everyday moral intuitions |

### Evaluation Pipeline

- **Parallel execution** with configurable concurrency (default: 15, up to 100)
- **Incremental checkpointing** — results persisted every 25 scenarios for crash recovery
- **Dual-method scoring** — heuristic classification + semantic analysis
- **Cryptographic trace binding** — every evaluation produces a unique auditable trace ID
- **Badge computation at write time** — excellence (>=90%), balanced (all categories >=80%), category mastery (>=95%)

## Quick Start

```bash
git clone https://github.com/CIRISAI/CIRISBench.git
cd CIRISBench

# Start infrastructure
docker compose -f infra/docker/docker-compose.he300.yml up -d db redis

# Run the engine
cd engine
pip install -r requirements.txt
uvicorn api.main:app --port 8080

# Run a benchmark
curl -X POST http://localhost:8080/he300/run \
  -H "Content-Type: application/json" \
  -d '{
    "batch_id": "my-test",
    "model_name": "gpt-4o-mini",
    "random_seed": 42,
    "concurrency": 15
  }'
```

## Unified Evaluation Pipeline

All evaluations (frontier sweeps, client benchmarks, promotional runs) flow through the same pipeline and are stored in a single `evaluations` table:

| Eval Type | Trigger | Visibility | Purpose |
|-----------|---------|------------|---------|
| `frontier` | Celery Beat (weekly) | Always public | Frontier model leaderboard |
| `client` | API request | Private (toggle) | Paid/free customer evaluations |

### Evaluation Lifecycle

```
queued --> running (checkpoints every 25 scenarios) --> completed | failed
```

- **Create** — eval row created before run starts
- **Checkpoint** — atomic JSONB append of scenario results
- **Complete** — final accuracy, badges, cache invalidation
- **Crash recovery** — stale `running` evals marked `failed` on startup

## Frontier Model Sweep

CIRISBench evaluates 15+ frontier models weekly via Celery Beat:

```
GPT-4o, GPT-4o-mini, GPT-5, Claude Opus 4, Claude Sonnet 4,
Gemini 2.5 Pro, Gemini 2.5 Flash, Llama 4 Maverick, Llama 4 Scout,
DeepSeek-R1, DeepSeek-V3, Mistral Large, Command R+, Grok-3, Grok-3 Mini
```

Results are published to the public leaderboard at [ethicsengine.org/scores](https://ethicsengine.org/scores).

## Architecture

### CIRISBench (this repo) — Write Path

| Component | Location | Purpose |
|-----------|----------|---------|
| Evaluation service | `engine/db/eval_service.py` | Create/start/checkpoint/complete/fail lifecycle |
| HE-300 runner | `engine/api/routers/he300.py` | Parallel `/run` endpoint with checkpointing |
| Celery tasks | `engine/celery_tasks.py` | Frontier sweep fan-out |
| Badge engine | `engine/core/badges.py` | Compute badges at write time |
| Models | `engine/db/models.py` | Evaluation + FrontierModel tables |
| Migrations | `engine/db/alembic/versions/` | Schema evolution |

### CIRISNode (submodule) — Read Path

| Component | Location | Purpose |
|-----------|----------|---------|
| Scores API | `cirisnode/api/scores/routes.py` | `/scores`, `/scores/{model}`, `/embed/scores` |
| Evaluations API | `cirisnode/api/evaluations/routes.py` | Auth-filtered eval listing + visibility toggle |
| Redis cache | `cirisnode/utils/redis_cache.py` | 1-hour cache for scores, 5-min for leaderboard |

### Shared Infrastructure

| Service | Purpose |
|---------|---------|
| PostgreSQL | `evaluations` + `frontier_models` + `tenant_tiers` tables |
| Redis | Cache + Celery broker |
| Celery Worker | Processes evaluation tasks |
| Celery Beat | Weekly frontier sweep schedule |

### Billing (Stripe)

| Component | Location | Purpose |
|-----------|----------|---------|
| Stripe router | `engine/api/routers/stripe_billing.py` | Checkout session, customer portal, webhook handler |
| TenantTier model | `engine/db/models.py` | Subscription tier per tenant |
| Billing proxy | `CIRISNode/cirisnode/api/billing/routes.py` | Proxies checkout/portal/webhook from frontend to Engine |
| Migration | `engine/db/alembic/versions/004_add_tenant_tiers.py` | Schema for `tenant_tiers` table |

## API Reference

### Benchmark Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/he300/run` | POST | Run full 300-scenario HE-300 evaluation |
| `/he300/catalog` | GET | List available scenarios |
| `/he300/validate` | POST | Validate a previous batch run |
| `/he300/agentbeats/run` | POST | AgentBeats-compatible parallel benchmark |
| `/health` | GET | Service health check |

### Billing Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/billing/checkout` | POST | Create Stripe Checkout session (auth required) |
| `/billing/portal` | GET | Create Stripe Customer Portal session (auth required) |
| `/billing/webhook` | POST | Stripe webhook handler (signature verified, no auth) |

### Run Request

```json
{
  "batch_id": "my-evaluation",
  "model_name": "gpt-4o-mini",
  "random_seed": 42,
  "concurrency": 15,
  "validate_after_run": true
}
```

### Response

```json
{
  "batch_response": {
    "status": "completed",
    "results": [...],
    "summary": {
      "total": 300,
      "correct": 248,
      "accuracy": 0.827,
      "by_category": {
        "virtue": {"total": 150, "correct": 128, "accuracy": 0.853},
        "commonsense_hard": {"total": 150, "correct": 120, "accuracy": 0.800}
      }
    }
  },
  "trace_id": "he300-...",
  "is_he300_compliant": true
}
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL_ASYNC` | - | PostgreSQL connection (asyncpg) |
| `REDIS_URL` | `redis://localhost:6379` | Redis for cache + Celery |
| `LLM_PROVIDER` | `openai` | LLM provider for evaluation |
| `LLM_MODEL` | `gpt-4o-mini` | Model for evaluation |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `HE300_CONCURRENCY` | `15` | Default parallel evaluation limit |
| `FRONTIER_SWEEP_ENABLED` | `false` | Enable weekly frontier sweep |
| `STRIPE_SECRET_KEY` | - | Stripe secret API key (`sk_live_...` or `sk_test_...`) |
| `STRIPE_WEBHOOK_SECRET` | - | Stripe webhook signing secret (`whsec_...`) |
| `STRIPE_PRO_PRICE_ID` | - | Stripe Price ID for Pro monthly subscription |

## Docker Deployment

```bash
# Full stack: CIRISNode + EthicsEngine + Worker + Beat + DB + Redis
docker compose -f infra/docker/docker-compose.he300.yml up -d
```

## Badges

| Badge | Requirement |
|-------|-------------|
| `excellence` | >= 90% overall accuracy |
| `balanced` | >= 80% in all categories |
| `{category}-mastery` | >= 95% in a specific category |

## Links

- [EthicsEngine.org](https://ethicsengine.org) — Managed benchmarking platform
- [CIRIS Framework](https://ciris.ai) — Ethical scoring methodology
- [CIRISNode](https://github.com/CIRISAI/CIRISNode) — Read path / API gateway
- [Hendrycks Ethics Paper](https://arxiv.org/abs/2008.02275) — Original dataset

## License

AGPL-3.0 — CIRIS L3C

## Citation

```bibtex
@article{hendrycks2021ethics,
  title={Aligning AI With Shared Human Values},
  author={Hendrycks, Dan and others},
  journal={arXiv preprint arXiv:2008.02275},
  year={2021}
}
```
