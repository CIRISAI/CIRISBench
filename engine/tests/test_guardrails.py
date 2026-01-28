import asyncio
import json
import logging
import argparse
from typing import List, Dict, Any

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Project Imports ---
# Assuming schemas and engine are accessible from the tests directory
# Adjust path if necessary based on execution context
try:
    from schemas.guardrail import Guardrail, GuardrailScope
    from schemas.results import ResultViolation
    from core.guardrails.engine import check_guardrails
    # Import the correct helper functions from config.loader
    from config.loader import _load_config_file, _validate_and_parse
except ImportError as e:
    logger.error(f"Failed to import necessary modules. Ensure PYTHONPATH is set correctly or run from project root. Error: {e}")
    exit(1)

# --- Test Content ---
# Add more diverse examples to test different guardrails
TEST_CONTENT = [
    ("input", "This is a perfectly normal sentence."),
    ("input", "I want to express my extreme dislike for this situation."), # Test 'hate' potentially
    ("output", "The system generated a response containing a forbidden keyword: pineapple."), # Test keyword filter
    ("output", "Here's some potentially harmful content involving self-harm instructions."), # Test classifier
    ("input", "Another benign input string."),
    ("output", "This output might trigger a violence classifier."), # Test classifier
]

async def run_guardrail_tests(guardrail_file: str):
    """Loads guardrails and runs checks against predefined test content."""
    logger.info(f"Loading guardrails from: {guardrail_file}")
    active_guardrails: List[Guardrail] = []
    try:
        # Step 1: Load raw data from the specified file
        raw_data = _load_config_file(guardrail_file)
        if raw_data is None:
            logger.error(f"Could not load data from {guardrail_file}.")
            return # Exit if file loading failed

        # Step 2: Validate the raw data against the Guardrail model
        # _validate_and_parse expects a list or dict, and returns a list of validated models
        validated_guardrails = _validate_and_parse(raw_data, Guardrail, guardrail_file)

        if not validated_guardrails:
             logger.error(f"No valid Guardrail objects found or parsed in {guardrail_file}.")
             # Check if raw_data was loaded but validation failed
             if raw_data:
                 logger.error("Raw data was loaded but failed validation. Check file content and schema.")
             return

        active_guardrails = validated_guardrails
        logger.info(f"Successfully loaded and validated {len(active_guardrails)} guardrails.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during guardrail loading/parsing from {guardrail_file}: {e}", exc_info=True)
        return

    if not active_guardrails:
        logger.warning("No active guardrails loaded. Exiting test.")
        return

    logger.info("\n--- Running Guardrail Checks ---")

    all_violations: Dict[int, List[ResultViolation]] = {}

    for i, (scope_str, content) in enumerate(TEST_CONTENT):
        logger.info(f"\nTest Case {i+1}: Scope='{scope_str}', Content='{content[:100]}...'") # Log truncated content
        # Remove the incorrect instantiation: scope = GuardrailScope(scope_str)

        try:
            # Call the async check_guardrails function, passing the string scope directly
            violations: List[ResultViolation] = await check_guardrails(
                content=content,
                active_guardrails=active_guardrails,
                scope=scope_str, # Pass the string directly
                engine_instance=None # Pass None as engine_instance if policy checks don't need it here
            )

            if violations:
                logger.warning(f"Found {len(violations)} violation(s) for Test Case {i+1}:")
                all_violations[i+1] = violations
                for v_idx, violation in enumerate(violations):
                    logger.warning(f"  Violation {v_idx+1}: ID='{violation.id}', Type='{violation.type}', Severity='{violation.severity}'")
                    # Attempt to parse and pretty-print details if it's JSON
                    try:
                        details_json = json.loads(violation.details)
                        logger.warning(f"    Details (JSON):\n{json.dumps(details_json, indent=4)}")
                    except (json.JSONDecodeError, TypeError):
                         logger.warning(f"    Details (Raw): {violation.details}") # Print raw details if not valid JSON
            else:
                logger.info(f"No violations found for Test Case {i+1}.")

        except Exception as e:
            logger.error(f"Error during guardrail check for Test Case {i+1}: {e}", exc_info=True)

    logger.info("\n--- Guardrail Test Summary ---")
    if all_violations:
        logger.warning(f"Violations detected in {len(all_violations)} out of {len(TEST_CONTENT)} test cases.")
        for case_num, violations in all_violations.items():
             logger.warning(f"  Case {case_num}: {len(violations)} violation(s)")
    else:
        logger.info("No violations detected in any test case.")
    logger.info("--- Test Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test guardrail definitions against sample content.")
    parser.add_argument(
        "-g", "--guardrails",
        default="data/guardrails/default_guardrails.json",
        help="Path to the guardrails JSON file to test."
    )
    args = parser.parse_args()

    # Run the async function using asyncio
    asyncio.run(run_guardrail_tests(args.guardrails))
