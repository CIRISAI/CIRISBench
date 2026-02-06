# Unified Evaluation Pipeline — Implementation Plan

## Architecture

```
Bench WRITES → PostgreSQL evaluations table ← Node READS
                     ↕
                   Redis
              (cache + invalidation)
```

CIRISBench owns: migrations, Celery Beat frontier sweep, badge computation, store_trace→Postgres.
CIRISNode owns: /scores, /evaluations, /embed/scores, /leaderboard endpoints, Redis cache, auth-aware visibility filtering.

---

## PART A: CIRISBench (engine/) — WRITE PATH

### A1. Dependencies — pyproject.toml

Add to `[project.dependencies]`:
```
sqlalchemy[asyncio]>=2.0
asyncpg>=0.29
psycopg2-binary>=2.9
alembic>=1.13
celery>=5.3
redis>=5.0
```

### A2. Database settings — engine/config/settings.py

Add to `Settings`:
- `database_url` (default `postgresql+asyncpg://postgres:password@db:5432/cirisnode`)
- `database_url_sync` (default `postgresql+psycopg2://postgres:password@db:5432/cirisnode`)
- `redis_url` (default `redis://redis:6379/0`)
- `frontier_sweep_enabled` (default `False`)

### A3. SQLAlchemy models & session — engine/db/

New files:
- `engine/db/__init__.py`
- `engine/db/models.py` — `Evaluation` and `FrontierModel` ORM classes matching the FSD schema exactly (UUID PK, JSONB for categories/scenario_results/badges/batch_config/trace_binding/eval_config, all indexes)
- `engine/db/session.py` — async engine + `async_session_factory` via `create_async_engine(settings.database_url)`; `get_async_session()` FastAPI dependency

### A4. Alembic migration

- `engine/alembic.ini` pointing to `engine/db/alembic/`
- `engine/db/alembic/env.py` with `target_metadata = Base.metadata`
- `engine/db/alembic/versions/001_evaluations_and_frontier_models.py` — creates both tables + all FSD indexes:
  - `idx_eval_public` (partial: visibility='public' AND status='completed', on accuracy DESC)
  - `idx_eval_frontier_model` (partial: eval_type='frontier' AND status='completed', on target_model + completed_at DESC)
  - `idx_eval_tenant` (tenant_id + created_at DESC)
  - `idx_eval_model_history` (partial: visibility='public' AND status='completed', on target_model + completed_at DESC)

### A5. Badge computation — engine/core/badges.py

New module with `compute_badges(accuracy, categories) -> list[str]` implementing the FSD logic:
- `accuracy >= 0.90` → "excellence"
- all categories >= 0.80 → "balanced"
- any category >= 0.95 → "{name}-mastery"

### A6. Evaluation persistence service — engine/db/eval_service.py

New module with:
- `create_evaluation(session, **kwargs) -> UUID` — INSERT with status='queued'
- `start_evaluation(session, eval_id)` — SET status='running', started_at=now()
- `complete_evaluation(session, eval_id, results_dict)` — SET status='completed', compute badges, write accuracy/categories/scenario_results, SET completed_at=now()
- `fail_evaluation(session, eval_id, error)` — SET status='failed'
- `invalidate_cache(redis_url, eval_type, model_id=None)` — direct Redis DEL of `cache:scores:*`, `cache:leaderboard` keys

### A7. Modify store_trace() — engine/api/routers/he300.py

Replace `store_trace()` (line 898) with `async store_trace()` that:
1. Keeps existing JSON-to-disk write (backward compat)
2. Adds Postgres INSERT via `complete_evaluation()`
3. Calls `invalidate_cache()` on completion

The 3 call sites (lines 811, 1086, 1725) need:
- Line 811 (`evaluate_batch`): map batch result → evaluations row with `eval_type='client'`, `protocol` from request
- Line 1086 (`run_he300_compliant`): map compliant run → evaluations row with `eval_type='client'`
- Line 1725 (`run_agentbeats_benchmark`): map agentbeats result → evaluations row with `eval_type='client'`, `protocol=request.protocol`, `agent_name=request.agent_name`

All 3 share the same `complete_evaluation()` call with different field mappings. The mapping function normalizes each call site's data dict into the evaluations schema.

### A8. Celery app — engine/celery_app.py

New file:
```python
celery_app = Celery("cirisbench", broker=settings.redis_url, backend=settings.redis_url)
```

### A9. Frontier sweep task — engine/celery_tasks.py

New file with `run_frontier_sweep` task:
1. Query `frontier_models WHERE active=true`
2. For each model: create evaluation row (queued) → run HE-300 via direct LLM (protocol='direct') → complete_evaluation() → invalidate cache
3. Uses sync psycopg2 in Celery (wraps async runner with `asyncio.run()`)

### A10. Celery Beat schedule — engine/celery_app.py

```python
celery_app.conf.beat_schedule = {
    "frontier-weekly": {
        "task": "run_frontier_sweep",
        "schedule": crontab(day_of_week=1, hour=2),  # Monday 2am UTC
    },
}
```

### A11. Seed frontier_models registry

`engine/db/alembic/versions/001_...py` includes INSERT for the 15 initial models from the FSD (openai/gpt-4o, anthropic/claude-4-opus, etc.).

### A12. Docker compose updates — infra/docker/docker-compose.he300.yml

- Add `DATABASE_URL` and `REDIS_URL` to `eee` service environment
- Add `engine-worker` service (same image as eee, CMD: `celery -A engine.celery_app worker`)
- Add `engine-beat` service (same image as eee, CMD: `celery -A engine.celery_app beat`)
- Update init-db.sql with evaluations + frontier_models DDL as fallback

---

## PART B: CIRISNode — READ PATH

Working in `/tmp/CIRISNode/` (will commit to CIRISNode repo).

### B1. Dependencies — requirements.txt

Add:
```
asyncpg>=0.29.0
```
(redis already present as `redis==5.0.4`)

### B2. Config — cirisnode/config.py

Add to `Settings`:
- `DATABASE_URL: str = "postgresql://postgres:password@db:5432/cirisnode"` (asyncpg needs raw postgresql:// URL)

### B3. PostgreSQL connection pool — cirisnode/db/pg_pool.py

New file:
- `asyncpg.create_pool(settings.DATABASE_URL, min_size=2, max_size=10)`
- `get_pg_pool()` lazy init
- `close_pg_pool()` for shutdown

### B4. Redis cache layer — cirisnode/utils/redis_cache.py

New file replacing in-memory lru_cache for score serving:
- `cache_get(key)` / `cache_set(key, value, ttl)`
- Uses `redis.asyncio` client
- TTL per the FSD: scores=3600s, leaderboard=300s, embed=3600s + Cache-Control header

### B5. Pydantic schemas — cirisnode/schema/evaluation_schemas.py

New file with response models:
- `EvaluationSummary` (list view)
- `EvaluationDetail` (drill-down with scenario_results)
- `ScoreEntry` (frontier scores)
- `LeaderboardEntry` (public client leaderboard)
- `EmbedScoresResponse` (widget payload)
- `EvaluationPatchRequest` (visibility + agent_name)

### B6. Scores router (public, no auth) — cirisnode/api/scores/routes.py

New router `prefix="/api/v1"`:
- `GET /scores` — latest frontier eval per model, Redis cached 1hr
- `GET /scores/{model_id}` — historical evals for one model, Redis cached 1hr
- `GET /leaderboard` — public client evals ranked by accuracy, Redis cached 5min
- `GET /embed/scores` — compact widget payload for ciris.ai iframe, Cache-Control header

SQL queries use `DISTINCT ON (target_model)` for latest-per-model. All go through Redis cache layer.

### B7. Evaluations router (auth required) — cirisnode/api/evaluations/routes.py

New router `prefix="/api/v1/evaluations"`:
- `GET /` — `WHERE tenant_id=$1 OR visibility='public'` (user sees own + public)
- `GET /{id}` — full detail with scenario_results (owner or public only)
- `PATCH /{id}` — update visibility/agent_name (owner only, respects visibility state machine: frontier evals can't go private)

Auth via existing `validate_a2a_auth()` dependency from `cirisnode/api/a2a/auth.py`. The `sub` JWT claim = tenant_id.

### B8. Register routers — cirisnode/main.py

Add imports + `app.include_router(scores_router)` + `app.include_router(evaluations_router)`.

### B9. CORS for embed — cirisnode/main.py

Add `https://ciris.ai` and `https://www.ciris.ai` to CORS origins for the embed endpoint.

### B10. Lifecycle hooks — cirisnode/main.py

Add `@app.on_event("startup")` → init pg_pool + redis.
Add `@app.on_event("shutdown")` → close pg_pool + redis.

---

## PART C: Infrastructure

### C1. init-db.sql

Add evaluations + frontier_models CREATE TABLE + indexes (fallback for fresh containers before Alembic runs).

### C2. Seed data

INSERT the 15 frontier models into frontier_models table via init-db.sql.

---

## Implementation Order

1. A1 (deps) → A2 (settings) → A3 (models/session) → A4 (migration) → A5 (badges) → A6 (eval_service) → A7 (store_trace mod) → A8+A9+A10 (celery)
2. B1 (deps) → B2 (config) → B3 (pg_pool) → B4 (redis_cache) → B5 (schemas) → B6+B7 (routers) → B8+B9+B10 (main.py)
3. C1+C2 (infra) + A11 (seed) + A12 (docker compose)

A and B tracks are independent and can be done in parallel.
