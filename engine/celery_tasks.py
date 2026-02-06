"""Celery tasks for CIRISBench — frontier model sweep.

The sweep iterates the frontier_models registry and enqueues one
HE-300 evaluation per active model via protocol='direct' (CIRISProxy).
"""

import asyncio
import logging
import uuid

from celery import Task

from engine.celery_app import celery_app

logger = logging.getLogger(__name__)


class RunFrontierSweepTask(Task):
    """Fan-out task: one eval per active frontier model."""

    name = "engine.celery_tasks.run_frontier_sweep"
    max_retries = 1
    default_retry_delay = 300

    def run(self):
        logger.info("Starting frontier model sweep")
        return asyncio.run(self._sweep_async())

    async def _sweep_async(self):
        from engine.db.session import async_session_factory
        from engine.db import eval_service

        async with async_session_factory() as session:
            models = await eval_service.get_frontier_models(session)

        if not models:
            logger.warning("No active frontier models found, skipping sweep")
            return {"status": "skipped", "reason": "no_active_models"}

        logger.info("Frontier sweep: %d models to evaluate", len(models))
        results = []

        for model in models:
            try:
                eval_id = await self._evaluate_model(model)
                results.append({"model_id": model.model_id, "eval_id": str(eval_id), "status": "completed"})
            except Exception as e:
                logger.error("Frontier eval failed for %s: %s", model.model_id, e)
                results.append({"model_id": model.model_id, "status": "failed", "error": str(e)})

        logger.info("Frontier sweep complete: %d/%d succeeded",
                     sum(1 for r in results if r["status"] == "completed"), len(results))
        return {"status": "completed", "results": results}

    async def _evaluate_model(self, model) -> uuid.UUID:
        """Run a single frontier model evaluation."""
        from engine.db.session import async_session_factory
        from engine.db import eval_service

        seed = 42  # deterministic for reproducibility
        sample_size = 300

        async with async_session_factory() as session:
            eval_id = await eval_service.create_evaluation(
                session,
                tenant_id="ciris-frontier",
                eval_type="frontier",
                protocol="direct",
                seed=seed,
                target_model=model.model_id,
                target_provider=model.provider,
                agent_name=model.display_name,
                sample_size=sample_size,
                visibility="public",  # frontier evals are always public
            )
            await session.commit()
            await eval_service.start_evaluation(session, eval_id)
            await session.commit()

        # Run HE-300 via the engine's batch runner
        try:
            result = await self._run_he300_direct(model, seed, sample_size)

            async with async_session_factory() as session:
                await eval_service.complete_evaluation(
                    session,
                    eval_id,
                    accuracy=result["accuracy"],
                    total_scenarios=result["total"],
                    correct=result["correct"],
                    errors=result["errors"],
                    categories=result["categories"],
                    scenario_results=result.get("results"),
                    avg_latency_ms=result.get("avg_latency_ms"),
                    processing_ms=result.get("processing_ms"),
                    trace_id=result.get("trace_id"),
                )
                await session.commit()
        except Exception as e:
            async with async_session_factory() as session:
                await eval_service.fail_evaluation(session, eval_id, str(e))
                await session.commit()
            raise

        return eval_id

    async def _run_he300_direct(self, model, seed: int, sample_size: int) -> dict:
        """Run HE-300 against a frontier model directly via LLM API.

        This calls the engine's HE-300 runner with protocol='direct',
        bypassing the A2A/MCP handshake. The model is reached via
        CIRISProxy using the model's proxy_route.
        """
        # Import here to avoid circular imports at module level
        from engine.core.he300_runner import run_batch, BatchConfig, ScenarioInput
        from engine.core.he300_data import sample_he300_scenarios

        scenarios = sample_he300_scenarios(seed=seed, n=sample_size)
        scenario_inputs = [
            ScenarioInput(
                scenario_id=s["scenario_id"],
                category=s["category"],
                input_text=s["input_text"],
                expected_label=s["expected_label"],
            )
            for s in scenarios
        ]

        batch_config = BatchConfig(
            batch_id=str(uuid.uuid4()),
            concurrency=50,
            agent_config={
                "model": model.proxy_route or model.model_id,
                "protocol": "direct",
            },
            timeout_per_scenario=120,
            semantic_evaluation=True,
        )

        result = await run_batch(scenario_inputs, batch_config)
        return {
            "accuracy": result.accuracy,
            "total": result.total,
            "correct": result.correct,
            "errors": result.errors,
            "categories": result.categories,
            "results": result.results,
            "avg_latency_ms": result.avg_latency_ms,
            "processing_ms": result.processing_time_ms,
            "trace_id": result.trace_id if hasattr(result, "trace_id") else None,
        }


run_frontier_sweep = celery_app.register_task(RunFrontierSweepTask())
