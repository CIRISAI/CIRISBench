import asyncio
import json
import logging
import os
import sys # Add sys import
import argparse
import glob
from typing import List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('core').setLevel(logging.WARNING) # Keep core logs quieter unless debugging
logging.getLogger('config').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Import necessary components
from core.engine import EthicsEngine
from schemas.pipeline import Pipeline
from schemas.results import Results
from config.settings import settings # To get default paths

async def run_single_pipeline(engine: EthicsEngine, pipeline: Pipeline) -> Optional[Results]:
    """Runs a single pipeline and returns the results."""
    logger.info(f"Starting pipeline: {pipeline.id}")
    try:
        results = await engine.run_pipeline(pipeline)
        logger.info(f"Finished pipeline: {pipeline.id} Outcome: {results.outcome if results else 'N/A'}")
        return results
    except Exception as e:
        logger.error(f"Error running pipeline {pipeline.id}: {e}", exc_info=True)
        return None

async def main(args):
    """Main function to load pipelines, run them concurrently, and report results."""
    logger.info(f"Starting benchmark test run with args: {args}")

    # 1. Instantiate the EthicsEngine
    try:
        # Pass load_configs=False if we intend to override configs via args later,
        # but for now, let it load defaults. We'll override pipeline params directly.
        engine = EthicsEngine(load_configs=True)
    except Exception as e:
        logger.error(f"Failed to initialize EthicsEngine: {e}", exc_info=True)
        return

    # 2. Find and load pipeline files
    pipeline_files = []
    if args.pipelines:
        # User specified specific pipeline IDs or patterns
        for pattern in args.pipelines:
            full_pattern = os.path.join(settings.data_dir, "pipelines", f"{pattern}.json")
            pipeline_files.extend(glob.glob(full_pattern))
    else:
        # Default to all bench_q*.json files
        full_pattern = os.path.join(settings.data_dir, "pipelines", "bench_q*.json")
        pipeline_files = glob.glob(full_pattern)

    if not pipeline_files:
        logger.error(f"No pipeline files found matching the specified criteria.")
        return

    logger.info(f"Found {len(pipeline_files)} pipeline files to run.")

    pipelines_to_run: List[Pipeline] = []
    for p_file in pipeline_files:
        try:
            with open(p_file, 'r', encoding='utf-8') as f:
                pipeline_data = json.load(f)
            # --- Override pipeline config with CLI args if provided ---
            # Modify the raw dictionary *before* Pydantic validation/parsing
            if args.identity:
                pipeline_data['identity_id'] = args.identity
            if args.guidance:
                pipeline_data['ethical_guidance_id'] = args.guidance
            if args.guardrails is not None: # Allow empty list to override
                pipeline_data['guardrail_ids'] = args.guardrails

            # Override reasoning depth for the LLM stage if requested
            if args.reasoning_depth is not None and args.reasoning_depth > 0:
                for stage_data in pipeline_data.get('stages', []):
                    if stage_data.get('id') == 'llm_answer' and stage_data.get('type') == 'LLM':
                        if 'ag2_config' not in stage_data or not isinstance(stage_data['ag2_config'], dict):
                            stage_data['ag2_config'] = {}
                        stage_data['ag2_config']['max_depth'] = args.reasoning_depth
                        # Optionally set other reasoning params if needed, e.g., method
                        # stage_data['ag2_config']['method'] = "tree_of_thought" # Example
                        logger.info(f"Overriding reasoning depth for stage 'llm_answer' in pipeline {pipeline_data.get('id')} to {args.reasoning_depth}")
                        break # Assume only one LLM stage to modify

            # Now parse the potentially modified data
            pipeline = Pipeline(**pipeline_data)

            # The overrides are now handled *before* parsing, so this block is removed.

            pipelines_to_run.append(pipeline)
            logger.debug(f"Loaded pipeline: {pipeline.id} from {p_file}")
        except Exception as e:
            logger.error(f"Failed to load or parse pipeline {p_file}: {e}", exc_info=True)

    if not pipelines_to_run:
        logger.error("No pipelines were successfully loaded. Exiting.")
        return

    # 3. Run pipelines concurrently
    tasks = [run_single_pipeline(engine, p) for p in pipelines_to_run]
    logger.info(f"Running {len(tasks)} pipelines concurrently...")
    # Note: Concurrency is limited by the semaphore in config.py
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("All pipeline runs finished.")

    # 4. Process results and calculate correctness
    successful_runs = 0
    correct_runs = 0
    total_runs = len(pipelines_to_run)
    detailed_results_data = [] # List to store detailed results for the report

    # --- Process results ---
    for i, result_or_exc in enumerate(results_list):
        pipeline_id = pipelines_to_run[i].id
        pipeline_obj = pipelines_to_run[i] # Get the corresponding pipeline object

        if isinstance(result_or_exc, Results):
            results = result_or_exc
            # --- Extract detailed info for the report ---
            llm_output = "N/A"
            evaluation_details = "N/A"
            eval_content = "N/A"
            eval_metadata = "N/A"

            if results.interactions:
                for interaction in results.interactions:
                    if interaction.stage_id == 'llm_answer' and interaction.role == 'assistant':
                        llm_output = interaction.content or "N/A"
                    if interaction.stage_id == 'evaluation' and interaction.role == 'evaluator':
                        eval_content = interaction.content or "N/A"
                        eval_metadata = str(interaction.metadata) if interaction.metadata else "N/A" # Convert metadata dict to string

            correctness_score_num = results.metrics.correctness if results.metrics and results.metrics.correctness is not None else None
            correctness_score_str = f"{correctness_score_num:.2f}" if correctness_score_num is not None else "N/A"

            evaluation_details = f"Score: {correctness_score_str}, Content: {eval_content}, Metadata: {eval_metadata}"

            expected_outcome = "N/A"
            if pipeline_obj.evaluation_metrics and hasattr(pipeline_obj.evaluation_metrics, 'expected_outcome'):
                expected_outcome = pipeline_obj.evaluation_metrics.expected_outcome or "N/A"

            # Store extracted data
            detailed_results_data.append({
                "id": pipeline_id,
                "llm_response": llm_output,
                "expected_response": expected_outcome,
                "evaluation": evaluation_details,
                "is_correct": correctness_score_num is not None and correctness_score_num > 0.5,
                "outcome": results.outcome,
                "outcome_details": results.outcome_details
            })

            # --- Original logic for counting successful/correct runs ---
            if results.outcome == "success":
                successful_runs += 1
                if correctness_score_num is not None and correctness_score_num > 0.5:
                    correct_runs += 1
            elif results.outcome in ["error", "failure"]:
                 logger.error(f"Pipeline {pipeline_id}: Failed with outcome '{results.outcome}'. Details: {results.outcome_details}")
            else:
                 logger.warning(f"Pipeline {pipeline_id}: Completed with unexpected outcome '{results.outcome}'. Details: {results.outcome_details}")

        elif isinstance(result_or_exc, Exception):
            logger.error(f"Pipeline {pipeline_id}: Failed with exception: {result_or_exc}", exc_info=result_or_exc)
            # Store error info for the report
            detailed_results_data.append({
                "id": pipeline_id,
                "llm_response": "ERROR",
                "expected_response": "ERROR",
                "evaluation": f"Exception: {result_or_exc}",
                "is_correct": False,
                "outcome": "exception",
                "outcome_details": str(result_or_exc)
            })
        else:
             logger.error(f"Pipeline {pipeline_id}: Unknown result type: {type(result_or_exc)}")
             # Store error info for the report
             detailed_results_data.append({
                 "id": pipeline_id,
                 "llm_response": "ERROR",
                 "expected_response": "ERROR",
                 "evaluation": f"Unknown result type: {type(result_or_exc)}",
                 "is_correct": False,
                 "outcome": "unknown",
                 "outcome_details": f"Unknown result type: {type(result_or_exc)}"
             })

    # 5. Report Summary
    logger.info("Preparing to write summary report to file...")
    report_file_path = "summary_report.txt"
    try:
        with open(report_file_path, "w", encoding="utf-8") as f:
            f.write("--- Benchmark Test Summary ---\n")
            f.write(f"Total Pipelines Run: {total_runs}\n")
            f.write(f"Successful Runs (no errors): {successful_runs}\n")
            f.write(f"Correct Runs (based on evaluation): {correct_runs}\n")
            if total_runs > 0:
                percent_correct = (correct_runs / total_runs) * 100
                f.write(f"Percentage Correct: {percent_correct:.2f}%\n")
            else:
                f.write("Percentage Correct: N/A (No pipelines run)\n")
            f.write("--- End Summary ---\n\n")

            # --- Write Detailed Results ---
            f.write("--- Detailed Results per Question ---\n")
            # Sort results by pipeline ID for consistent order (e.g., bench_q1, bench_q2, ...)
            detailed_results_data.sort(key=lambda x: x.get('id', ''))
            for item in detailed_results_data:
                f.write(f"\n--- Pipeline: {item.get('id', 'N/A')} ---\n")
                f.write(f"Outcome: {item.get('outcome', 'N/A')} ({item.get('outcome_details', 'N/A')})\n")
                f.write(f"Correct: {'Yes' if item.get('is_correct') else 'No'}\n")
                f.write(f"LLM Response:\n{item.get('llm_response', 'N/A')}\n\n")
                f.write(f"Expected Response:\n{item.get('expected_response', 'N/A')}\n\n")
                f.write(f"Evaluation Details:\n{item.get('evaluation', 'N/A')}\n")
                f.write(f"--- End Pipeline: {item.get('id', 'N/A')} ---\n")

        logger.info(f"Summary and detailed report written to {report_file_path}")
    except Exception as write_err:
        logger.error(f"Failed to write summary report to {report_file_path}: {write_err}", exc_info=True)
        # Fallback to printing if file write fails (keeping original fallback structure)
        print("\n\n--- Benchmark Test Summary (Fallback Print) ---")
        print(f"Total Pipelines Run: {total_runs}")
        print(f"Successful Runs (no errors): {successful_runs}")
        print(f"Correct Runs (based on evaluation): {correct_runs}")
        if total_runs > 0:
            percent_correct = (correct_runs / total_runs) * 100
            print(f"Percentage Correct: {percent_correct:.2f}%")
        else:
            print("Percentage Correct: N/A (No pipelines run)")
        print("--- End Summary ---")
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EthicsEngine benchmark pipelines concurrently.")
    parser.add_argument(
        "--pipelines",
        nargs='+',
        help="Specific pipeline IDs or glob patterns (e.g., bench_q1 bench_q*) relative to data/pipelines. Default runs all 'bench_q*.json'."
    )
    parser.add_argument(
        "--identity",
        type=str,
        default=None, # Default uses the one specified in the pipeline file
        help="Override the identity ID for all pipelines run."
    )
    parser.add_argument(
        "--guidance",
        type=str,
        default=None, # Default uses the one specified in the pipeline file
        help="Override the ethical guidance ID for all pipelines run."
    )
    parser.add_argument(
        "--guardrails",
        nargs='*', # Allows zero or more arguments
        default=None, # Default uses the ones specified in the pipeline file
        help="Override the list of active guardrail IDs for all pipelines run. Provide no arguments to disable all guardrails."
    )
    parser.add_argument(
        "--reasoning-depth",
        type=int,
        default=0, # Default to 0 (basic call)
        help="Override the reasoning max_depth for the LLM stage."
    )
    # Add other arguments as needed (e.g., --concurrency, --output-dir)

    args = parser.parse_args()
    asyncio.run(main(args))
