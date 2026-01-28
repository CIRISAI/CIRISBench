import asyncio
import ast # Added for literal_eval
import io
import ujson
import logging
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# --- Autogen Imports ---
try:
    # Import AssistantAgent and UserProxyAgent
    from autogen import AssistantAgent, UserProxyAgent, ConversableAgent, GroupChat # GroupChat might not be needed but good to have
    # from autogen.agentchat.contrib.capabilities.tool_use import SimpleToolRegistry # If needed later - Removed as unused and causing import error
    AUTOGEN_AVAILABLE = True
except ImportError:
    # Define dummy classes if autogen is not available
    class AssistantAgent:
         def __init__(self, *args, **kwargs): self.name = kwargs.get("name"); self.chat_messages = {}
         async def a_generate_reply(self, *args, **kwargs): return "Dummy Reply - Autogen Import Failed"
    class UserProxyAgent:
         def __init__(self, *args, **kwargs): self.name = kwargs.get("name"); self.chat_messages = {}
         async def a_initiate_chat(self, *args, **kwargs): pass
         def register_function(self, *args, **kwargs): pass # Mock method
    class ConversableAgent: # Keep for type hints if needed elsewhere
         def __init__(self, *args, **kwargs): pass

    AUTOGEN_AVAILABLE = False
    # Keep the dummy class definitions above available
    # Do not set the actual class variables to None here
    # ConversableAgent = None # Keep this if ConversableAgent is only used for type hints

# --- Project Imports ---
from config import (AGENT_TIMEOUT, AG2_REASONING_SPECS, AUTOGEN_AVAILABLE,
                    llm_config, semaphore, settings) # Import shared configs
from schemas.ethical_guidance import EthicalGuidance
from schemas.identity import Identity
from schemas.interaction import Interaction, InteractionRole, Metadata
from schemas.stage import Stage
from schemas.guardrail import Guardrail # Import Guardrail schema
from utils.placeholder_resolver import resolve_placeholders
# Import the registry instead of a specific tool
from core.tools import TOOL_REGISTRY

from . import register_stage_handler # Import registry and decorator
from core.guardrails import check_guardrails # Import the guardrail checker function

logger = logging.getLogger(__name__)

@register_stage_handler("tool")
async def handle_tool_stage(
    stage: Stage,
    pipeline_context: Dict[str, Any],
    engine_instance: Any, # Reference to the EthicsEngine instance
    identity: Optional[Identity], # Passed directly from engine
    guidance: Optional[EthicalGuidance], # Passed directly from engine
    active_guardrails: List[Guardrail] # Passed directly from engine
) -> Tuple[Dict[str, Any], List[Interaction], Any]:
    """
    Handles the execution of a tool stage using Autogen's AssistantAgent and UserProxyAgent.

    Args:
        stage: The Stage object definition. Prompt should instruct tool use.
        pipeline_context: Dictionary containing outputs from previous stages.
        engine_instance: The EthicsEngine instance.
        identity: The active Identity profile.
        guidance: The active EthicalGuidance profile.
        active_guardrails: List of active guardrails.

    Returns:
        A tuple containing:
        - Updated pipeline_context with outputs from this stage.
        - List of interactions generated during this stage.
        - Stage status/outcome.
    """
    interactions: List[Interaction] = []
    stage_outputs: Dict[str, Any] = {}
    status: Optional[str] = None
    # Reasoning tree not applicable in this two-agent setup
    # reasoning_tree_dict: Optional[Dict] = None
    llm_metadata: Dict[str, Any] = {"model": None, "tokens_used": None, "guardrails_triggered": []}
    start_time = time.monotonic()
    stage_id = stage.id

    logger.info(f"Handling tool stage directly: {stage_id}") # Updated log message

    # --- Pre-checks ---
    # Removed Autogen/LLM config checks as we're calling directly
    # Removed identity/guidance checks as they aren't used in direct call

    # --- Look up Tool ---
    tool_id = stage.tool_id
    if not tool_id:
        logger.error(f"Tool stage {stage_id} is missing the 'tool_id' field.")
        status = "error: missing tool_id in stage definition"
        return pipeline_context, interactions, status

    tool_config = TOOL_REGISTRY.get(tool_id)
    if not tool_config:
        logger.error(f"Tool '{tool_id}' specified in stage {stage_id} not found in TOOL_REGISTRY.")
        status = f"error: specified tool '{tool_id}' not registered"
        return pipeline_context, interactions, status

    tool_schema = tool_config.get("schema")
    tool_function = tool_config.get("function")

    if not tool_schema or not tool_function:
        logger.error(f"Configuration for tool '{tool_id}' in TOOL_REGISTRY is incomplete (missing schema or function).")
        status = f"error: incomplete configuration for tool '{tool_id}'"
        return pipeline_context, interactions, status

    # --- Resolve Inputs ---
    resolved_inputs: Dict[str, Any] = {}
    if stage.inputs:
        try:
            resolved_inputs = resolve_placeholders(stage.inputs, pipeline_context)
            logger.debug(f"Tool Stage {stage_id} - Resolved Inputs: {resolved_inputs}")
        except Exception as e:
            logger.error(f"Error resolving inputs for tool stage {stage_id}: {e}", exc_info=True)
            status = f"error: input placeholder resolution failed: {e}"
            return pipeline_context, interactions, status
    else:
        logger.warning(f"Tool stage {stage_id} has no inputs defined.")
        # Continue, assuming the tool might not need inputs or gets them implicitly

    # --- Prepare Arguments and Call Tool ---
    tool_result: Dict[str, Any] = {"error": "Tool execution failed to produce a result."} # Default error
    final_response_content = tool_result["error"] # Default output content

    try:
        # --- Argument Formatting (Specific to simple_calculator) ---
        if tool_id == "simple_calculator":
            # Expecting 'grain_surplus' and 'percent_allocated' based on pipeline
            num1_str = resolved_inputs.get("grain_surplus")
            num2_str = resolved_inputs.get("percent_allocated")
            if num1_str is not None and num2_str is not None:
                expression = f"{num1_str} * {num2_str}"
                logger.info(f"Tool stage {stage_id}: Calling '{tool_id}' with expression: '{expression}'")
                tool_result = tool_function(expression=expression)
            else:
                missing_keys = [k for k in ["grain_surplus", "percent_allocated"] if resolved_inputs.get(k) is None]
                err_msg = f"Missing required inputs for simple_calculator: {', '.join(missing_keys)}"
                logger.error(f"Tool stage {stage_id}: {err_msg}")
                tool_result = {"error": err_msg}
        else:
            # --- Generic Tool Call (using resolved inputs as kwargs) ---
            # This assumes other tools accept keyword arguments matching the input keys
            logger.info(f"Tool stage {stage_id}: Calling generic tool '{tool_id}' with inputs: {resolved_inputs}")
            # Note: This might fail if the tool function doesn't accept these kwargs
            # Consider adding more robust argument mapping if needed
            tool_result = tool_function(**resolved_inputs)

        logger.debug(f"Tool stage {stage_id}: Raw tool result: {tool_result}")

        # --- Process Tool Result ---
        if isinstance(tool_result, dict):
            if "error" in tool_result:
                error_message = tool_result['error']
                final_response_content = f"Error from tool '{tool_id}': {error_message}"
                status = f"error: tool execution failed: {error_message}"
                logger.error(f"Tool stage {stage_id}: Tool execution returned an error: {error_message}")
            elif "result" in tool_result:
                # Use the successful result from the tool
                # Store the raw result for potential object output type
                raw_result = tool_result["result"]
                final_response_content = str(raw_result) # Ensure string conversion for interaction log
                status = None # Explicitly set status to None for success
                logger.info(f"Tool stage {stage_id}: Extracted successful tool result: {final_response_content}")

                # Store potentially non-string result in stage_outputs later
                # based on output spec type hint
                if stage.outputs and stage.outputs.spec:
                    for label, type_hint in stage.outputs.spec.items():
                        if type_hint == "object":
                            stage_outputs[label] = raw_result # Store raw result
                        else: # Default to string
                            stage_outputs[label] = final_response_content
                        logger.debug(f"Tool stage {stage_id}: Storing output '{label}' (type: {type_hint})")

            else:
                # If dict has no 'error' or 'result', treat as raw string representation
                final_response_content = str(tool_result)
                status = "error: unexpected tool result structure"
                logger.warning(f"Tool stage {stage_id}: Tool return dict has unexpected structure: {tool_result}. Using string representation.")
        else:
            # Handle cases where tool result is not a dict
            final_response_content = str(tool_result)
            status = "error: unexpected tool result type"
            logger.error(f"Tool stage {stage_id}: Tool return has unexpected type: {type(tool_result)}")

    except TypeError as te:
         logger.error(f"Tool stage {stage_id}: TypeError calling tool '{tool_id}'. Check if inputs match function signature. Inputs: {resolved_inputs}. Error: {te}", exc_info=True)
         status = f"error: tool call failed (TypeError): {te}"
         final_response_content = f"Error calling tool '{tool_id}': {te}"
    except Exception as e:
        logger.error(f"Unexpected error during direct tool call for stage {stage_id}: {e}", exc_info=True)
        status = f"error: unexpected failure during tool call: {e}"
        final_response_content = f"Error: Unexpected failure during tool call: {e}"


    # --- Apply Guardrails (on the final string representation) ---
    # Note: Guardrails are applied to the string version even if the output is stored as an object
    triggered_violation_ids: List[str] = []
    if status is None and isinstance(final_response_content, str): # Check status is success before guardrails
        logger.debug(f"Tool stage {stage_id}: Checking output against {len(active_guardrails)} active guardrails.")
        stage_violations = await check_guardrails( # Added await
            final_response_content,
            active_guardrails,
            "output",
            engine_instance=engine_instance # Pass engine_instance
        )
        if stage_violations:
            should_block = False
            block_message = "Content blocked by guardrail."
            violation_summary = []
            for violation in stage_violations:
                triggered_violation_ids.append(violation.id)
                violation_summary.append(f"{violation.id} ({violation.severity or 'N/A'})")
                triggered_guardrail = next((g for g in active_guardrails if g.id == violation.id), None)
                if triggered_guardrail and triggered_guardrail.action == 'block':
                    should_block = True
                    block_message = triggered_guardrail.message or f"Content blocked by guardrail '{violation.id}'."
            if should_block:
                logger.error(f"Blocking guardrail(s) triggered in tool stage '{stage_id}'. Violations: {', '.join(violation_summary)}. Stopping pipeline.")
                status = f"guardrail_violation: {', '.join(violation_summary)}"
                # Overwrite final content only if blocked
                final_response_content = block_message
                # Clear stage_outputs if blocked
                stage_outputs = {}
        else:
            logger.debug(f"Tool stage {stage_id}: No guardrail violations detected in output.")
    elif status is None:
         logger.warning(f"Cannot apply output guardrails for tool stage '{stage_id}': Final response content is not a string ({type(final_response_content)}).")


    # --- Record Interaction ---
    # Record the tool execution result (or error)
    interaction_metadata = Metadata(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=None, # No LLM involved in direct call
        tokens_used=None,
        guardrails_triggered=triggered_violation_ids,
    )
    interactions.append(Interaction(
        stage_id=stage_id,
        role="tool", # Use 'tool' role for direct execution result
        content=final_response_content, # Log the final string content
        metadata=interaction_metadata,
        schema_version=stage.schema_version
    ))

    # --- Store Outputs (if not already stored during result processing) ---
    if status is None or "modified" in (status or ""):
        if stage.outputs and stage.outputs.spec:
            # Ensure outputs are stored correctly based on type hint
            for label, type_hint in stage.outputs.spec.items():
                if label not in stage_outputs: # Only store if not already handled (e.g., object type)
                    stage_outputs[label] = final_response_content # Default to string
                    logger.debug(f"Tool stage {stage_id}: Storing output '{label}' (type: {type_hint}) with default string value.")
        elif stage.outputs:
             logger.warning(f"Tool stage {stage_id} has 'outputs' defined but missing 'spec'. Outputs may not be stored correctly.")

    # --- Finalize ---
    updated_context = pipeline_context.copy()
    if stage_outputs:
        updated_context[stage_id] = stage_outputs

    end_time = time.monotonic()
    logger.info(f"Finished tool stage: {stage_id} (Duration: {end_time - start_time:.2f}s) Status: {status or 'success'}")

    return updated_context, interactions, status
