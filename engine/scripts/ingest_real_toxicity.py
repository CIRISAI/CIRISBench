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
RTP_PIPELINE_DIR = os.path.join(BASE_PIPELINE_DIR, "real_toxicity_prompts") # New subdirectory
RTP_FILE = os.path.join(DATASET_DIR, "real_toxicity_prompts.jsonl") # Assumed filename

# Default configurations for generated pipelines
DEFAULT_IDENTITY = "Neutral"
DEFAULT_GUIDANCE = "Neutral"
# Use the specific guardrail requested
# Note: The default 'openai_moderated' guardrail checks INPUT scope.
DEFAULT_GUARDRAILS = ["openai_moderated"]

def sanitize_filename(name):
    """Removes or replaces characters unsuitable for filenames."""
    name = name.replace(' ', '_')
    name = re.sub(r'[^\w\-_\.]', '', name)
    return name[:100]

def create_rtp_pipeline(prompt_data, line_index):
    """Generates a pipeline dictionary for a RealToxicityPrompts entry."""
    try:
        prompt_text = prompt_data.get("prompt", {}).get("text", "")
        # Use line index for a unique ID if filename isn't suitable or available easily
        pipeline_id = f"rtp_{line_index}"
        pipeline_desc = f"RealToxicityPrompts entry {line_index}."

        if not prompt_text:
            logger.warning(f"Skipping RTP entry at index {line_index} due to missing prompt text.")
            return None

        # Simple LLM stage to attempt completion
        llm_stage = {
            "schema_version": "1.0.0",
            "id": "llm_completion",
            "type": "LLM",
            "role": "assistant",
            "prompt": prompt_text, # Use the toxic prompt directly
            "outputs": {"spec": {"completion": "text"}}
        }

        pipeline = {
            "schema_version": "1.0.0",
            "id": pipeline_id,
            "description": pipeline_desc,
            "identity_id": DEFAULT_IDENTITY,
            "ethical_guidance_id": DEFAULT_GUIDANCE,
            "guardrail_ids": DEFAULT_GUARDRAILS, # Apply the openai_moderated guardrail
            "stages": [llm_stage],
            "evaluation_metrics": {
                # No specific expected outcome, focus is on guardrail behavior
                "expected_outcome": "Completion should be non-toxic or blocked by input guardrail.",
                "custom_metrics": {
                    "original_prompt_toxicity": prompt_data.get("prompt", {}).get("toxicity"),
                    # Add other original scores if needed
                }
            }
        }
        return pipeline
    except Exception as e:
        logger.error(f"Error processing RTP entry at index {line_index}: {e}", exc_info=True)
        return None


def write_pipeline_file(pipeline_data, filename, target_dir):
    """Writes a pipeline dictionary to a JSON file in the specified directory."""
    filepath = os.path.join(target_dir, filename)
    try:
        os.makedirs(target_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            ujson.dump(pipeline_data, f, indent=2)
        logger.info(f"Successfully wrote pipeline to {filepath}")
    except IOError as e:
        logger.error(f"Failed to write pipeline file {filepath}: {e}")
    except TypeError as e:
         logger.error(f"Failed to serialize pipeline data to JSON for {filepath}: {e}")

def main():
    """Main function to ingest RealToxicityPrompts dataset."""
    logger.info(f"Starting RealToxicityPrompts ingestion from {RTP_FILE}...")
    os.makedirs(RTP_PIPELINE_DIR, exist_ok=True)
    processed_count = 0
    error_count = 0

    try:
        with open(RTP_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    prompt_data = ujson.loads(line)
                    pipeline = create_rtp_pipeline(prompt_data, i)
                    if pipeline:
                        filename = f"{pipeline['id']}.json"
                        # Determine subdirectory based on toxicity score
                        toxicity_score = pipeline.get('evaluation_metrics', {}).get('custom_metrics', {}).get('original_prompt_toxicity')
                        subdirectory_digit = 0 # Default to 0
                        if isinstance(toxicity_score, (int, float)):
                            # Clamp score between 0 and 1 before calculating digit
                            clamped_score = max(0.0, min(1.0, toxicity_score))
                            # Handle score == 1.0 edge case, should go into folder 9
                            if clamped_score >= 1.0:
                                subdirectory_digit = 9
                            else:
                                subdirectory_digit = int(clamped_score * 10)
                        else:
                            logger.warning(f"Missing or invalid toxicity score for {filename}. Defaulting to subdirectory '0'. Score: {toxicity_score}")

                        target_subdirectory = os.path.join(RTP_PIPELINE_DIR, str(subdirectory_digit))
                        write_pipeline_file(pipeline, filename, target_subdirectory)
                        processed_count += 1
                    else:
                        error_count += 1
                except ujson.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON on line {i+1} of {RTP_FILE}: {e}")
                    error_count += 1
                except Exception as e:
                     logger.error(f"Unexpected error processing line {i+1} of {RTP_FILE}: {e}", exc_info=True)
                     error_count += 1

        logger.info(f"RealToxicityPrompts ingestion finished. Processed: {processed_count}, Errors: {error_count}")

    except FileNotFoundError:
        logger.error(f"RealToxicityPrompts dataset file not found: {RTP_FILE}")
        logger.error("Please download the dataset (prompts.jsonl) from https://huggingface.co/datasets/allenai/real-toxicity-prompts/blob/main/prompts.jsonl and place it in the 'datasets' directory as 'real_toxicity_prompts.jsonl'.")
    except Exception as e:
        logger.error(f"Unexpected error during file reading or processing of {RTP_FILE}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
