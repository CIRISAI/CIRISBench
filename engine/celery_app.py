"""Celery application for CIRISBench evaluation tasks.

Provides the frontier model sweep schedule and async task execution.
Shares Redis broker/backend with CIRISNode.
"""

from celery import Celery
from celery.schedules import crontab

from engine.config.settings import settings

celery_app = Celery(
    "cirisbench",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    task_default_queue="cirisbench",
)

# Celery Beat schedule — frontier model sweep
if settings.frontier_sweep_enabled:
    celery_app.conf.beat_schedule = {
        "frontier-weekly": {
            "task": "engine.celery_tasks.run_frontier_sweep",
            "schedule": crontab(day_of_week=1, hour=2),  # Monday 2am UTC
        },
    }

# Auto-discover tasks
celery_app.autodiscover_tasks(["engine"])
