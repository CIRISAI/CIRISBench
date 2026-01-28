import asyncio
import json
from unittest.mock import MagicMock, AsyncMock # Removed patch import
from pathlib import Path
from typing import Dict, Any # Import Dict and Any for type hinting

# Assuming schemas are correctly defined and accessible
from schemas.stage import Stage, StageAG2Config, StageOutputSpec # Corrected imports
# PipelineContext is not a class, the handler expects a Dict[str, Any]
from schemas.identity import Identity
from schemas.ethical_guidance import EthicalGuidance
from schemas.interaction import Interaction # Import Interaction for return type hint checking if needed

# Import the handler function and necessary config/loaders
from core.stages.llm_handler import handle_llm_stage # Import the actual handler function
from config.loader import load_all_identities, load_all_guidances # Import correct loader functions
from config import settings # To get default paths if needed

# --- Test Setup ---

# 1. Define a simple LLM Stage Configuration
# Using valid IDs from the default JSON files
default_identity_id = "adult_general" # Changed to a valid ID
default_guidance_id = "VirtueEthics" # Changed to a valid ID

# Define the output label we expect
output_label = "llm_response"

test_stage_config = Stage( # Use Stage class
    id="test_llm_stage_basic",
    # name="Test Basic LLM Stage", # Name is not in Stage schema
    description="A stage to test the basic AssistantAgent functionality.",
    type="LLM", # Use 'type' field with correct literal
    # provider_config=LLMProviderConfig(provider="openai", model="gpt-3.5-turbo"), # Provider info likely comes from global config, not stage schema
    # identity_id and guidance_id are NOT part of the Stage schema. They are handled at the pipeline level.
    prompt="{{ initial_input }}", # Use 'prompt' field - Removed "User query: " prefix
    outputs=StageOutputSpec(spec={output_label: "text"}), # Use 'outputs' field
    ag2_config=StageAG2Config(max_depth=1), # Explicitly set max_depth=1 for reasoning test
    # Note: role is missing, but optional. function_id/inputs not needed for LLM type.
)

# 2. Create the pipeline context dictionary
# The handler expects a Dict[str, Any] containing outputs from previous stages.
# For the first stage, it might contain initial inputs.
initial_context_data = {"initial_input": "Tell me about ethical AI."}
# The context passed between stages is a dictionary.
# Keys might be stage IDs mapping to their output dictionaries, or initial input keys.
# For resolving "{{ initial_input }}", the key "initial_input" needs to be directly in the context dict.
pipeline_context: Dict[str, Any] = initial_context_data

# 3. Load necessary Identity and Guidance (using existing loaders)
# Assuming loader functions handle file paths based on settings
try:
    # Load all identities and guidances into dictionaries (registries)
    identity_registry = load_all_identities()
    guidance_registry = load_all_guidances()
    # Check if the specific IDs we need were loaded
    if default_identity_id not in identity_registry:
        raise ValueError(f"Default identity '{default_identity_id}' not found in loaded identities.")
    if default_guidance_id not in guidance_registry:
         raise ValueError(f"Default guidance '{default_guidance_id}' not found in loaded guidances.")
    print("Successfully loaded default identity and guidance registries.")
except Exception as e:
    print(f"Error loading identity/guidance registries: {e}")
    print("Proceeding with mock objects instead.")
    # Fallback to mock objects if loading fails
    identity_registry = {default_identity_id: MagicMock(spec=Identity, id=default_identity_id, name="Mock Identity", system_prompt="Mock System Prompt")}
    guidance_registry = {default_guidance_id: MagicMock(spec=EthicalGuidance, id=default_guidance_id, name="Mock Guidance", system_prompt="Mock Guidance Prompt")}


# 4. Create a Mock Engine Instance
# The handler expects engine_instance with registries and guardrail_engine
mock_engine = MagicMock()
mock_engine.identity_registry = identity_registry
mock_engine.guidance_registry = guidance_registry

# Mock the guardrail engine - needs a check_response method
mock_guardrail_engine = MagicMock()
# Make check_response an async function that accepts the correct arguments
# (response_content, active_ids) and returns the expected tuple (is_safe, modified_content, triggered_rules)
async def mock_check_response(response_content, active_ids):
    # Simulate a successful check with no modifications
    return (True, response_content, [])

mock_guardrail_engine.check_response = AsyncMock(side_effect=mock_check_response)
mock_engine.guardrail_engine = mock_guardrail_engine

# 5. No Handler Instantiation needed - we imported the function directly

# --- Test Execution ---

async def run_test(): # Removed mock argument and patch decorator
    # No mock configuration needed here anymore

    print(f"\n--- Running Test for Stage: {test_stage_config.id} ---")
    # Print the dictionary directly using json.dumps
    print(f"Initial Context: {json.dumps(pipeline_context, indent=2)}")

    try:
        # Execute the handler function directly
        # The handler expects stage, context, engine, identity, guidance
        identity_obj = identity_registry.get(default_identity_id)
        guidance_obj = guidance_registry.get(default_guidance_id)

        # The handler returns: updated_context, interactions, status
        updated_context, interactions_list, stage_status = await handle_llm_stage(
            stage=test_stage_config,
            pipeline_context=pipeline_context,
            engine_instance=mock_engine,
            identity=identity_obj,
            guidance=guidance_obj
        )

        print("\n--- Test Execution Complete ---")
        print(f"Stage Status: {stage_status}") # Print the returned status
        print("\nUpdated Context:")
        # Pretty print the updated context
        # Updated context is now the first element of the tuple returned by the handler
        print(json.dumps(updated_context, indent=2)) # Dump the dict directly

        # Basic Verification - Check using the defined output label
        output_labels = list(test_stage_config.outputs.spec.keys())
        # The handler stores output under context[stage_id][output_label]
        stage_output_data = updated_context.get(test_stage_config.id, {})

        if output_labels and output_labels[0] in stage_output_data:
            print(f"\nVerification PASSED: Output key '{output_labels[0]}' found in context data for stage '{test_stage_config.id}'.")
        elif output_labels:
            print(f"\nVerification FAILED: Output key '{output_labels[0]}' NOT found in context data for stage '{test_stage_config.id}'.")
        else:
              print(f"\nVerification SKIPPED: No output labels defined in stage config.")

        # Check the returned interactions list
        if interactions_list:
            print(f"\nVerification PASSED: Interaction list is not empty (length: {len(interactions_list)}).")
            # Further checks could inspect the content of the interaction record
            print("\n--- Interactions List ---")
            print(json.dumps([i.model_dump() for i in interactions_list], indent=2)) # Print interactions including metadata
            print("--- End Interactions List ---\n")
        else:
            print(f"\nVerification FAILED: Interaction list is empty.")


    except ImportError as e:
        print(f"\nImportError: {e}")
        print("Please ensure necessary libraries (like 'autogen') are installed.")
        print("You might need to run: pip install -r requirements.txt")
    except Exception as e:
        print(f"\nAn error occurred during handler execution: {e}")
        import traceback
        traceback.print_exc()

# Run the async test function
if __name__ == "__main__":
    # Note: This test will likely make a real LLM call unless autogen is mocked further.
    # Ensure OPENAI_API_KEY (or relevant key for the provider) is set in the environment.
    print("NOTE: This test may make a real LLM call.")
    print("Ensure necessary API keys (e.g., OPENAI_API_KEY) are set in your environment.")
    asyncio.run(run_test())
