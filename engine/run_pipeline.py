import asyncio # Import asyncio
import ujson
import logging
import os
import argparse # Import argparse
from typing import Dict, Any

# Import and configure logging using the central setup
from utils.logging_config import setup_logging
setup_logging() # Call the setup function

# Get logger for this module (after setup)
logger = logging.getLogger(__name__)

# Import necessary components
from core.engine import EthicsEngine
from schemas.pipeline import Pipeline
from config.settings import settings # To get default paths
from config import concurrency_monitor # Import the monitor

async def main(pipeline_id: str): # Accept pipeline_id as argument
    """
    Loads and runs the specified pipeline using the EthicsEngine.
    Starts and stops the concurrency monitor.
    """
    # Start the concurrency monitor
    concurrency_monitor.start() # Use default interval (now 2 seconds)

    # Construct the pipeline file path dynamically, handling potential full paths
    pipeline_filename_with_ext = f"{pipeline_id}.json"
    if '/' in pipeline_id or '\\' in pipeline_id:
        # Assume pipeline_id is a path relative to CWD or absolute
        # Ensure it ends with .json
        if not pipeline_id.endswith('.json'):
             pipeline_file_path = pipeline_filename_with_ext
        else:
             pipeline_file_path = pipeline_id # Already has .json if user provided it
        # Normalize the path
        pipeline_file_path = os.path.abspath(pipeline_file_path)
    else:
        # Assume pipeline_id is just the ID, construct path relative to default dir
        pipeline_file_path = os.path.join(settings.data_dir, "pipelines", pipeline_filename_with_ext)

    logger.info(f"Attempting to run pipeline: {pipeline_file_path}")

    # 1. Load the pipeline configuration from JSON
    try:
        with open(pipeline_file_path, 'r', encoding='utf-8') as f:
            pipeline_data = ujson.load(f)
        # Validate and parse using Pydantic model
        pipeline = Pipeline(**pipeline_data)
        logger.info(f"Successfully loaded and parsed pipeline: {pipeline.id}")
    except FileNotFoundError:
        logger.error(f"Pipeline file not found: {pipeline_file_path}")
        return
    except ujson.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {pipeline_file_path}: {e}")
        return
    except Exception as e: # Catch Pydantic validation errors or others
        logger.error(f"Error loading or parsing pipeline {pipeline_file_path}: {e}", exc_info=True)
        return

    # 2. Instantiate the EthicsEngine
    # It will load identities, guidances, guardrails automatically by default.
    # Pass load_pipelines_on_init=False because we load the specific pipeline manually here.
    try:
        engine = EthicsEngine(load_pipelines_on_init=False)
    except Exception as e:
        logger.error(f"Failed to initialize EthicsEngine: {e}", exc_info=True)
        return

    # 3. Run the pipeline
    logger.info(f"--- Running Pipeline: {pipeline.id} ---")
    try:
        results = await engine.run_pipeline(pipeline) # Await the async call

        if results: # Check if results object was returned
            logger.info(f"--- Pipeline Run Complete: {pipeline.id} ---")
            # 4. Print Results Summary (optional, results are also saved to file)
            print("\n--- Pipeline Results Summary ---")
            print(f"Run ID: {results.run_id}")
            print(f"Pipeline ID: {results.pipeline_id}")
            print(f"Outcome: {results.outcome}") # Print the string outcome directly
            print(f"Details: {results.outcome_details}")
            print(f"Violations: {len(results.violations)}")
            for i, v in enumerate(results.violations):
                print(f"  Violation {i+1}: ID={v.id}, Severity={v.severity}, Details={v.details}")
            print(f"Metrics: {results.metrics.model_dump_json(indent=2)}")
            print("--- End Summary ---")
            print(f"\nFull results saved to: {settings.results_dir}/{results.run_id}.json")
        else:
            logger.error(f"Pipeline run for {pipeline.id} did not return results (likely due to pre-run configuration error).")

    except Exception as e:
        print(f"Outcome: {results.outcome}") # Print the string outcome directly
        print(f"Details: {results.outcome_details}")
        print(f"Violations: {len(results.violations)}")
        for i, v in enumerate(results.violations):
            print(f"  Violation {i+1}: ID={v.id}, Severity={v.severity}, Details={v.details}")
        print(f"Metrics: {results.metrics.model_dump_json(indent=2)}")
        print("--- End Summary ---")
        print(f"\nFull results saved to: {settings.results_dir}/{results.run_id}.json")

    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
    finally:
        # Ensure the monitor is stopped when the main function exits
        # Monitor stop is now synchronous as it runs in a separate thread
        concurrency_monitor.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an EthicsEngine pipeline.")
    parser.add_argument("pipeline_id", help="The ID of the pipeline to run (corresponds to filename without .json)")
    args = parser.parse_args()

    asyncio.run(main(args.pipeline_id)) # Pass the pipeline_id to main
