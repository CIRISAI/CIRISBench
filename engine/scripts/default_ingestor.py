import ujson
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths (relative to project root)
DATASET_DIR = "datasets"
BASE_PIPELINE_DIR = os.path.join("data", "pipelines")
BENCH_PIPELINE_DIR = os.path.join(BASE_PIPELINE_DIR, "simple_bench_public") # New subdirectory for benchmarks
SCENARIO_PIPELINE_DIR = os.path.join(BASE_PIPELINE_DIR, "ethical_scenarios") # New subdirectory for scenarios
BENCH_FILE = os.path.join(DATASET_DIR, "simple_bench_public.json")
SCENARIO_FILE = os.path.join(DATASET_DIR, "sample_scenarios.json")

# Default configurations for generated pipelines
DEFAULT_IDENTITY = "Neutral" # Use the Neutral identity as default
DEFAULT_GUIDANCE = "Neutral" # Use the Neutral guidance as default
DEFAULT_GUARDRAILS = ["no_self_harm", "no_hate_speech", "no_illegal_acts"] # Basic safety

def sanitize_filename(name):
    """Removes or replaces characters unsuitable for filenames."""
    name = name.replace(' ', '_') # Replace spaces with underscores first
    name = re.sub(r'[^\w\-_\.]', '', name) # Remove remaining non-alphanumeric/hyphen/underscore/dot
    return name[:100] # Limit length

def create_benchmark_pipeline(question_data):
    """Generates a pipeline dictionary for a benchmark question."""
    q_id = question_data.get("question_id")
    prompt = question_data.get("prompt", "")
    answer = question_data.get("answer", "")

    if not q_id or not prompt or not answer:
        logger.warning(f"Skipping benchmark question due to missing data: {question_data}")
        return None

    pipeline_id = f"bench_q{q_id}"
    pipeline_desc = f"Benchmark question {q_id} from simple_bench_public."

    # Ensure prompt asks for only the letter
    prompt_suffix = "\n\nBased on the scenario, choose the most likely answer and respond with only the corresponding letter (A, B, C, D, E, or F)."
    full_prompt = prompt.strip() + prompt_suffix

    pipeline = {
        "schema_version": "1.0.0",
        "id": pipeline_id,
        "description": pipeline_desc,
        "identity_id": DEFAULT_IDENTITY,
        "ethical_guidance_id": DEFAULT_GUIDANCE,
        "guardrail_ids": DEFAULT_GUARDRAILS,
        "stages": [
            {
                "schema_version": "1.0.0",
                "id": "llm_answer",
                "type": "LLM",
                "role": "assistant",
                "prompt": full_prompt,
                "outputs": {"spec": {"answer_letter": "text"}}
            },
            {
                "schema_version": "1.0.0",
                "id": "evaluate_answer",
                "type": "evaluation",
                "function_id": "evaluate_result_with_llm", # Use the new LLM evaluator
                "inputs": { # Corrected: Removed extra 'spec' layer
                    "actual": "{llm_answer.answer_letter}", # LLM's output from previous stage
                    "expected": "{{ pipeline_expected_outcome }}" # Use {{ key }} format for direct context lookup
                },
                "outputs": {"spec": {"metrics": "object"}} # Output structure remains the same
            }
        ],
        "evaluation_metrics": {
            "expected_outcome": answer # Store expected outcome for reference
        }
    }
    return pipeline

def create_scenario_pipeline(scenario_data):
    """Generates a pipeline dictionary for an ethical scenario."""
    s_id = scenario_data.get("id", "")
    prompt = scenario_data.get("prompt", "")
    tags = scenario_data.get("tags", [])
    criteria = scenario_data.get("evaluation_criteria", {})

    if not s_id or not prompt:
        logger.warning(f"Skipping scenario due to missing id or prompt: {scenario_data}")
        return None

    pipeline_id = f"scenario_{sanitize_filename(s_id)}"
    pipeline_desc = f"Analysis of scenario: {s_id}"
    tag_string = ", ".join(tags)

    # Construct prompt for analysis
    analysis_prompt = (
        f"Analyze the following scenario, considering the ethical implications and trade-offs involved "
        f"based on principles like {tag_string}:\n\n{prompt}"
    )

    pipeline = {
        "schema_version": "1.0.0",
        "id": pipeline_id,
        "description": pipeline_desc,
        "identity_id": DEFAULT_IDENTITY,
        "ethical_guidance_id": DEFAULT_GUIDANCE,
        "guardrail_ids": DEFAULT_GUARDRAILS,
        "stages": [
            {
                "schema_version": "1.0.0",
                "id": "analyze_scenario",
                "type": "LLM",
                "role": "assistant",
                "prompt": analysis_prompt,
                "outputs": {"spec": {"analysis": "text"}}
            }
            # No automated evaluation stage added by default for scenarios
        ],
        "evaluation_metrics": criteria # Store the provided criteria
    }
    return pipeline

def write_pipeline_file(pipeline_data, filename, target_dir): # Added target_dir argument
    """Writes a pipeline dictionary to a JSON file in the specified directory."""
    filepath = os.path.join(target_dir, filename) # Use target_dir
    try:
        os.makedirs(target_dir, exist_ok=True) # Ensure target directory exists
        with open(filepath, 'w', encoding='utf-8') as f:
            ujson.dump(pipeline_data, f, indent=2)
        logger.info(f"Successfully wrote pipeline to {filepath}")
    except IOError as e:
        logger.error(f"Failed to write pipeline file {filepath}: {e}")
    except TypeError as e:
         logger.error(f"Failed to serialize pipeline data to JSON for {filepath}: {e}")

def main():
    """Main function to ingest datasets and create pipeline files."""
    logger.info("Starting dataset ingestion...")
    # Ensure base and subdirectories exist
    os.makedirs(BENCH_PIPELINE_DIR, exist_ok=True)
    os.makedirs(SCENARIO_PIPELINE_DIR, exist_ok=True)

    # Ingest Benchmarks
    try:
        with open(BENCH_FILE, 'r', encoding='utf-8') as f:
            bench_data = ujson.load(f)
        if "eval_data" in bench_data and isinstance(bench_data["eval_data"], list):
            logger.info(f"Processing {len(bench_data['eval_data'])} benchmark questions from {BENCH_FILE}...")
            for question in bench_data["eval_data"]:
                pipeline = create_benchmark_pipeline(question)
                if pipeline:
                    filename = f"{pipeline['id']}.json"
                    write_pipeline_file(pipeline, filename, BENCH_PIPELINE_DIR) # Pass benchmark dir
        else:
            logger.error(f"Invalid format in {BENCH_FILE}: 'eval_data' key missing or not a list.")
    except FileNotFoundError:
        logger.error(f"Benchmark file not found: {BENCH_FILE}")
    except ujson.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {BENCH_FILE}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error processing {BENCH_FILE}: {e}", exc_info=True)

    # Ingest Scenarios
    try:
        with open(SCENARIO_FILE, 'r', encoding='utf-8') as f:
            scenario_data = ujson.load(f)
        if isinstance(scenario_data, list):
            logger.info(f"Processing {len(scenario_data)} scenarios from {SCENARIO_FILE}...")
            for scenario in scenario_data:
                pipeline = create_scenario_pipeline(scenario)
                if pipeline:
                    filename = f"{pipeline['id']}.json"
                    write_pipeline_file(pipeline, filename, SCENARIO_PIPELINE_DIR) # Pass scenario dir
        else:
             logger.error(f"Invalid format in {SCENARIO_FILE}: Expected a JSON list.")
    except FileNotFoundError:
        logger.error(f"Scenario file not found: {SCENARIO_FILE}")
    except ujson.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {SCENARIO_FILE}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error processing {SCENARIO_FILE}: {e}", exc_info=True)

    logger.info("Dataset ingestion finished.")

if __name__ == "__main__":
    main()
