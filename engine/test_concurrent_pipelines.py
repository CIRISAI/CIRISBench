import asyncio
import json
import logging
import os
import time
from typing import Dict, Any

# Import and configure logging using the central setup
from utils.logging_config import setup_logging
setup_logging() # Call the setup function

# Get logger for this module (after setup)
logger = logging.getLogger(__name__)

# Import necessary components
from core.engine import EthicsEngine
from schemas.pipeline import Pipeline
from config.settings import settings
from config import concurrency_monitor # Import the monitor

async def run_single_pipeline(engine: EthicsEngine, pipeline: Pipeline, run_index: int):
    """Runs a single instance of the pipeline."""
    logger.info(f"Starting pipeline run #{run_index} (Pipeline ID: {pipeline.id})")
    start_time = time.monotonic()
    try:
        results = await engine.run_pipeline(pipeline)
        end_time = time.monotonic()
        if results:
            logger.info(f"Finished pipeline run #{run_index} (Pipeline ID: {pipeline.id}). Outcome: {results.outcome}. Duration: {end_time - start_time:.2f}s")
        else:
            logger.error(f"Pipeline run #{run_index} (Pipeline ID: {pipeline.id}) did not return results.")
    except Exception as e:
        end_time = time.monotonic()
        logger.error(f"Error during pipeline run #{run_index} (Pipeline ID: {pipeline.id}): {e}. Duration: {end_time - start_time:.2f}s", exc_info=True)

async def main():
    """
    Loads the pipeline and runs multiple instances concurrently.
    """
    pipeline_id = "concurrency_test_pipeline" # Use the new simple sleep pipeline
    num_concurrent_runs = 20 # Keep the number of runs

    # Construct the pipeline file path
    pipeline_filename = f"{pipeline_id}.json"
    pipeline_file_path = os.path.join(settings.data_dir, "pipelines", pipeline_filename)

    logger.info(f"Loading pipeline definition: {pipeline_file_path}")

    # 1. Load the pipeline configuration
    try:
        with open(pipeline_file_path, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)
        pipeline = Pipeline(**pipeline_data)
        logger.info(f"Successfully loaded pipeline: {pipeline.id}")
    except Exception as e:
        logger.error(f"Failed to load or parse pipeline {pipeline_file_path}: {e}", exc_info=True)
        return

    # 2. Instantiate the EthicsEngine
    try:
        # Create a single engine instance to be shared by all runs
        # This ensures they all use the same semaphore and registries
        engine = EthicsEngine(load_pipelines_on_init=False)
        # Explicitly load identities, guidances, guardrails (but not pipelines)
        engine.load_configurations(load_pipelines=False)
    except Exception as e:
        logger.error(f"Failed to initialize EthicsEngine: {e}", exc_info=True)
        return

    # 3. Start the concurrency monitor
    concurrency_monitor.start() # Use default interval (now 2 seconds)
    # Removed diagnostic sleep

    # 4. Create and run concurrent pipeline tasks
    logger.info(f"--- Starting {num_concurrent_runs} concurrent runs of pipeline: {pipeline.id} ---")
    tasks = []
    for i in range(num_concurrent_runs):
        # Pass the same engine and pipeline object to each task
        task = asyncio.create_task(run_single_pipeline(engine, pipeline.model_copy(deep=True), i + 1))
        tasks.append(task)

    # 5. Wait for all tasks to complete
    await asyncio.gather(*tasks, return_exceptions=True) # Capture exceptions from tasks

    logger.info(f"--- All {num_concurrent_runs} pipeline runs attempted. ---")

    # 6. Stop the monitor
    # Monitor stop is now synchronous as it runs in a separate thread
    concurrency_monitor.stop()

if __name__ == "__main__":
    asyncio.run(main())
