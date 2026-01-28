import asyncio
import io
import ujson
import logging
import time
import re # Import re for guardrail modification
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# --- Autogen Imports ---
try:
    from autogen import AssistantAgent, ConversableAgent, UserProxyAgent
    from autogen.agents.experimental import ReasoningAgent, ThinkNode
    AUTOGEN_AVAILABLE = True
except ImportError:
    # Define dummy classes if autogen is not available
    class ThinkNode:
        def __init__(self, content, parent=None): self.content=content; self.depth=0; self.value=0; self.visits=0; self.children=[]
        def to_dict(self): return {"content": self.content, "children": []}
    class ReasoningAgent:
        def __init__(self, *args, **kwargs): self._root = None
        def generate_reply(self, *args, **kwargs): return "Dummy Reply - Autogen Import Failed"
    class AssistantAgent:
         def __init__(self, *args, **kwargs): pass
         def generate_reply(self, *args, **kwargs): return "Dummy Reply - Autogen Import Failed"
    class ConversableAgent:
         def __init__(self, *args, **kwargs): pass
    class UserProxyAgent:
         def __init__(self, *args, **kwargs): pass
         def initiate_chat(self, *args, **kwargs): pass
         def last_message(self, agent): return {"content": "Dummy Reply - Autogen Import Failed"}

    AUTOGEN_AVAILABLE = False
    ReasoningAgent = None # Ensure it's None if not imported
    AssistantAgent = None
    UserProxyAgent = None
    ConversableAgent = None

# --- Project Imports ---
from config import (AGENT_TIMEOUT, AG2_REASONING_SPECS, AUTOGEN_AVAILABLE,
                    llm_config, semaphore, settings, concurrency_monitor) # Import monitor
from schemas.ethical_guidance import EthicalGuidance
from schemas.identity import Identity
from schemas.interaction import Interaction, InteractionRole, Metadata
from schemas.stage import Stage
from schemas.guardrail import Guardrail # Import Guardrail schema
from utils.placeholder_resolver import resolve_placeholders

from . import register_stage_handler # Import registry and decorator
from core.guardrails import check_guardrails # Import the guardrail checker function

logger = logging.getLogger(__name__)

@register_stage_handler("LLM")
async def handle_llm_stage(
    stage: Stage,
    pipeline_context: Dict[str, Any],
    engine_instance: Any, # Reference to the EthicsEngine instance
    identity: Optional[Identity], # Passed directly from engine
    guidance: Optional[EthicalGuidance], # Passed directly from engine
    active_guardrails: List[Guardrail] # Passed directly from engine
) -> Tuple[Dict[str, Any], List[Interaction], Any]:
    """
    Handles the execution of an LLM stage using Autogen agents.

    Supports both basic LLM calls and advanced reasoning via ReasoningAgent,
    handles concurrency, timeouts, and guardrails.

    Args:
        stage: The Stage object definition.
        pipeline_context: Dictionary containing outputs from previous stages.
        engine_instance: The EthicsEngine instance (for accessing configs, guardrail engine, etc.).
        identity: The active Identity profile.
        guidance: The active EthicalGuidance profile.

    Returns:
        A tuple containing:
        - Updated pipeline_context with outputs from this stage.
        - List of interactions generated during this stage.
        - Stage status/outcome (e.g., None for success, or an error/violation object/string).
    """
    interactions: List[Interaction] = []
    stage_outputs: Dict[str, Any] = {}
    status: Optional[str] = None
    reasoning_tree_dict: Optional[Dict] = None # Initialize as None
    llm_metadata: Dict[str, Any] = {"model": None, "tokens_used": None, "guardrails_triggered": []}
    start_time = time.monotonic()
    stage_id = stage.id

    logger.info(f"Handling LLM stage: {stage_id}")

    # --- Pre-checks ---
    if not AUTOGEN_AVAILABLE or not llm_config:
        logger.error(f"Autogen/LLMConfig not available for stage {stage_id}. Skipping.")
        status = "error: autogen library or LLM config unavailable"
        return pipeline_context, interactions, status

    if not identity or not guidance:
        logger.error(f"Identity or Guidance not provided to LLM stage handler for stage {stage_id}")
        status = "error: missing identity/guidance context"
        return pipeline_context, interactions, status

    # --- Determine Reasoning Mode and Config ---
    reasoning_level = "basic" # Default to basic non-reasoning call
    stage_ag2_config = stage.ag2_config.model_dump() if stage.ag2_config else {}
    # Determine the effective reasoning spec based on stage config or defaults
    if stage_ag2_config.get("max_depth", 0) > 0:
        reasoning_level = "custom"
        reason_config_spec = stage_ag2_config
        logger.info(f"Stage {stage_id} using custom AG2 reasoning config: {reason_config_spec}")
    elif stage_ag2_config.get("reasoning_level") in AG2_REASONING_SPECS:
        reasoning_level = stage_ag2_config["reasoning_level"]
        reason_config_spec = AG2_REASONING_SPECS[reasoning_level]
        logger.info(f"Stage {stage_id} using predefined reasoning level: {reasoning_level}")
    else:
        # Fallback to basic spec if no specific reasoning is requested
        reason_config_spec = AG2_REASONING_SPECS.get("basic", {"max_depth": 0}) # Ensure basic exists
        logger.info(f"Stage {stage_id} using default 'basic' reasoning spec.")


    # --- Prepare Prompt and System Message ---
    prompt_text = stage.prompt or ""
    try:
        resolved_prompt = resolve_placeholders(prompt_text, pipeline_context)
    except Exception as e:
        logger.error(f"Error resolving placeholders for stage {stage_id}: {e}", exc_info=True)
        status = f"error: placeholder resolution failed: {e}"
        return pipeline_context, interactions, status

    # --- Prepare Prompt and System Message ---
    system_message = "You are a helpful AI assistant." # Base message
    if guidance.prompt_template:
        system_message += f" {guidance.prompt_template}"
    if identity.description:
        system_message += f" You are interacting with/considering the perspective of: {identity.description}."
        if identity.notes:
            system_message += f" Keep in mind: {identity.notes}"

    logger.debug(f"LLM Stage {stage_id} - System Message: {system_message}")
    logger.debug(f"LLM Stage {stage_id} - User Prompt: {resolved_prompt}")

    # --- Configure Agent ---
    agent_name = f"ethics_agent_{stage_id}_{time.time_ns()}" # Unique name
    agent_llm_config = llm_config.model_copy(deep=True) # Create a copy to modify

    # Override temperature if specified in stage or reasoning spec
    temp_override = stage_ag2_config.get("temperature", reason_config_spec.get("temperature"))
    if temp_override is not None:
        agent_llm_config.temperature = temp_override
        logger.info(f"Stage {stage_id} overriding LLM temperature to: {temp_override}")

    agent: Optional[ConversableAgent] = None
    try:
        # Always use ReasoningAgent, control behavior via reason_config
        reason_config = {
            "method": stage_ag2_config.get("method", "beam_search"),
            "max_depth": reason_config_spec.get("max_depth", 0), # Use depth from spec (default 0 for basic)
            "beam_size": stage_ag2_config.get("beam_size", 3),
            "answer_approach": stage_ag2_config.get("answer_approach", "pool")
            # Add other ReasoningAgent params if needed
        }
        logger.info(f"Stage {stage_id} using ReasoningAgent with effective config: {reason_config}")

        # Convert the LLMConfig object to a dictionary for the agent constructor
        llm_config_dict = agent_llm_config.model_dump() if agent_llm_config else None
        print("LLM Config Dict:", llm_config_dict)
        agent = ReasoningAgent(
                name=agent_name,
                system_message=system_message,
                llm_config=llm_config_dict, # Pass the dictionary
                reason_config=reason_config,
                silent=True # Suppress Autogen's internal logging unless debugging
        )
    except Exception as e:
        logger.error(f"Error instantiating Autogen agent for stage {stage_id}: {e}", exc_info=True)
        status = f"error: agent instantiation failed: {e}"
        return pipeline_context, interactions, status

    # --- Execute Agent Call Asynchronously ---
    raw_response_content = "Error: Agent execution failed."
    captured_output = ""
    dummy_io = io.StringIO()

    if agent:
        # Outer try/finally to ensure decrement happens
        try:
            sema_acquire_start = time.monotonic()
            concurrency_monitor.increment_active() # Increment before acquiring
            async with semaphore:
                sema_acquire_end = time.monotonic()
                logger.debug(f"Stage {stage_id}: Semaphore acquired (wait={sema_acquire_end - sema_acquire_start:.2f}s)")

                # Inner try/except block for the agent call itself remains inside 'async with'
                with redirect_stdout(dummy_io), redirect_stderr(dummy_io):
                    thread_call_start = time.monotonic()
                    # Inner try for agent call timeout/errors
                    try:
                        # Use generate_reply directly
                        reply = await asyncio.wait_for(
                            asyncio.to_thread(
                                agent.generate_reply,
                                messages=[{"role": "user", "content": resolved_prompt}],
                                sender=None # Pass None as sender
                            ),
                            timeout=AGENT_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"Stage {stage_id}: Agent call timed out after {AGENT_TIMEOUT} seconds.")
                        raise # Re-raise to be caught by outer exception handler
                    thread_call_end = time.monotonic()
                    logger.debug(f"Stage {stage_id}: generate_reply completed (duration={thread_call_end - thread_call_start:.2f}s)")

                    # Process reply (generate_reply usually returns message content string)
                    if isinstance(reply, str):
                        raw_response_content = reply.strip()
                        # TODO: Extract metadata (model, tokens) if possible.
                        llm_metadata["model"] = agent_llm_config.config_list[0].get("model", "unknown") # Assume first model
                    else:
                         logger.warning(f"Stage {stage_id}: Unexpected reply type from generate_reply: {type(reply)}. Content: {reply}")
                         raw_response_content = str(reply).strip() # Attempt conversion

                    # Always attempt to extract reasoning tree if using ReasoningAgent
                    logger.debug(f"Checking for reasoning tree. Agent type: {type(agent)}")
                    if isinstance(agent, ReasoningAgent):
                        logger.debug("Agent is ReasoningAgent. Trying getattr...")
                        reasoning_tree_root: Optional[ThinkNode] = getattr(agent, '_root', None)
                        logger.debug(f"getattr(agent, '_root', None) result: {type(reasoning_tree_root)}")
                        if reasoning_tree_root:
                            logger.debug("_root attribute found. Calling .to_dict()...")
                            try:
                                reasoning_tree_dict = reasoning_tree_root.to_dict() # Store the dict representation
                            except Exception as tree_err:
                                logger.error(f"Error converting reasoning tree to dict: {tree_err}", exc_info=True)
                                reasoning_tree_dict = {"error": f"Failed to convert tree: {tree_err}"}
                            logger.debug(f"Stage {stage_id}: Reasoning tree extracted (Depth might be 0).")
                        else:
                            logger.debug(f"Stage {stage_id}: No reasoning tree (_root attribute) found on ReasoningAgent instance.")
                    else:
                        logger.debug(f"Stage {stage_id}: Agent is not a ReasoningAgent, skipping tree extraction.")

            # Semaphore released implicitly by exiting 'async with' block
            sema_release_time = time.monotonic()
            logger.debug(f"Stage {stage_id}: Semaphore released (held for {sema_release_time - sema_acquire_end:.2f}s)")
            captured_output = dummy_io.getvalue()

        except asyncio.TimeoutError: # Catch timeout from wait_for
            status = f"error: agent execution timed out after {AGENT_TIMEOUT}s"
            raw_response_content = f"Error: Agent execution timed out." # Ensure error message is set
            captured_output = dummy_io.getvalue()
            if captured_output: logger.error(f"Stage {stage_id}: Captured stdio before timeout: {captured_output}")
        except Exception as e: # Catch other exceptions during semaphore acquisition or agent call
            logger.error(f"Error during agent execution for stage {stage_id}: {e}", exc_info=True)
            status = f"error: agent execution failed: {e}"
            raw_response_content = f"Error: Agent execution failed." # Ensure error message is set
            captured_output = dummy_io.getvalue()
            if captured_output: logger.error(f"Stage {stage_id}: Captured stdio before error: {captured_output}")
        finally: # This finally corresponds to the outer try block
             concurrency_monitor.decrement_active() # Decrement after releasing or if an error occurred

        if captured_output:
            # Log captured stdio as INFO instead of WARNING
            logger.info(f"Stage {stage_id}: Captured stdio during agent run: {captured_output}")

    logger.debug(f"LLM Stage {stage_id} - Raw Response: {raw_response_content}")

    # --- Log reasoning_tree_dict before metadata ---
    logger.debug(f"Value of reasoning_tree_dict before Metadata: {type(reasoning_tree_dict)}")
    if reasoning_tree_dict:
        # Log the structure as debug if it exists
        logger.debug(f"Reasoning tree structure:\n{ujson.dumps(reasoning_tree_dict, indent=2, default=str)}")
    # --- END Log ---

    # --- Apply Guardrails ---
    final_response_content = raw_response_content
    triggered_violation_ids: List[str] = [] # Store IDs of triggered guardrails
    if status is None and isinstance(raw_response_content, str): # Only apply if agent didn't fail and response is string
        logger.debug(f"Stage {stage_id}: Checking output against {len(active_guardrails)} active guardrails.")
        # TODO: Add logic to merge/override with stage.guardrails if defined in stage schema
        stage_violations = await check_guardrails( # Added await
            raw_response_content,
            active_guardrails,
            "output",
            engine_instance=engine_instance # Pass engine_instance
        )

        if stage_violations:
            should_block = False
            modification_applied = False # Flag to track if modification happened
            modified_content = final_response_content # Start with original content
            block_message = "Content blocked by guardrail." # Default block message
            violation_summary = []

            for violation in stage_violations:
                triggered_violation_ids.append(violation.id)
                violation_summary.append(f"{violation.id} ({violation.severity or 'N/A'})")
                logger.warning(f"Guardrail violation in stage '{stage_id}': ID={violation.id}, Details={violation.details}")

                # Find the corresponding guardrail definition to check action
                triggered_guardrail = next((g for g in active_guardrails if g.id == violation.id), None)
                if triggered_guardrail:
                    if triggered_guardrail.action == 'block':
                        should_block = True
                        block_message = triggered_guardrail.message or f"Content blocked by guardrail '{violation.id}'."
                        # Break if we find a blocking action? Or collect all violations first?
                        # Current logic collects all, then blocks if any required it.
                    elif triggered_guardrail.action == 'modify':
                        logger.info(f"Guardrail '{violation.id}' triggered with action 'modify'. Attempting modification...")
                        if triggered_guardrail.type == 'content_filter':
                            try:
                                # Basic modification: Replace triggered content with [MODIFIED]
                                modification_placeholder = "[MODIFIED]"
                                if triggered_guardrail.trigger.regex:
                                     # Use re.sub for regex replacement (case-insensitive)
                                     modified_content = re.sub(triggered_guardrail.trigger.regex, modification_placeholder, modified_content, flags=re.IGNORECASE)
                                     modification_applied = True
                                elif triggered_guardrail.trigger.keywords:
                                     # Simple keyword replacement (case-insensitive)
                                     # This is basic; might replace parts of words if not careful.
                                     # A more robust approach would use word boundaries.
                                     temp_content = modified_content
                                     for keyword in triggered_guardrail.trigger.keywords:
                                         # Use regex for case-insensitive replacement of whole words
                                         pattern = r'\b' + re.escape(keyword) + r'\b'
                                         temp_content = re.sub(pattern, modification_placeholder, temp_content, flags=re.IGNORECASE)
                                     if temp_content != modified_content:
                                         modified_content = temp_content
                                         modification_applied = True

                                if modification_applied:
                                     logger.info(f"Applied basic modification for content_filter guardrail '{violation.id}'.")
                                else:
                                     logger.warning(f"Content filter modification for '{violation.id}' did not change content (regex/keyword might not have matched after prior modifications).")

                            except re.error as e:
                                logger.error(f"Regex error during modification for guardrail '{violation.id}': {e}")
                            except Exception as mod_exc:
                                logger.error(f"Unexpected error during modification for guardrail '{violation.id}': {mod_exc}", exc_info=True)
                        else:
                            logger.warning(f"Modification action for guardrail type '{triggered_guardrail.type}' (ID: {violation.id}) is not implemented. Using original content for this violation.")
                        # Do NOT set should_block = True for modify actions
                    # 'flag' action requires no specific handling here, logging is sufficient.
                    # 'escalate' action would need specific implementation.

            if should_block:
                logger.error(f"Blocking guardrail(s) triggered in stage '{stage_id}'. Violations: {', '.join(violation_summary)}. Stopping pipeline.")
                status = f"guardrail_violation: blocked - {', '.join(violation_summary)}" # Set status to indicate block
                final_response_content = block_message # Overwrite response with block message
            elif modification_applied:
                 logger.info(f"Content modified by guardrail(s) in stage '{stage_id}'. Violations: {', '.join(violation_summary)}.")
                 final_response_content = modified_content # Use the modified content
                 # Optionally set a specific status for modification
                 status = f"guardrail_violation: modified - {', '.join(violation_summary)}"
            # If only 'flag' violations occurred, status remains None (success)
        else:
            logger.debug(f"Stage {stage_id}: No guardrail violations detected in output.")

    elif status is None and not isinstance(raw_response_content, str):
         logger.warning(f"Cannot apply output guardrails for stage '{stage_id}': Raw response is not a string ({type(raw_response_content)}).")


    # --- Record Interactions ---
    interaction_metadata = Metadata(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=llm_metadata.get("model"),
        tokens_used=llm_metadata.get("tokens_used"), # Still needs proper extraction
        guardrails_triggered=triggered_violation_ids, # Use the collected list of triggered IDs
        reasoning_tree=reasoning_tree_dict # Always include the tree dict (will be None if not found)
    )

    # Record the final assistant response (potentially modified or error message)
    # Use string literal for role as InteractionRole is a Literal type
    interactions.append(Interaction(
        stage_id=stage_id,
        role=stage.role or "assistant", # Use string literal
        content=final_response_content,
        metadata=interaction_metadata,
        schema_version=stage.schema_version
    ))

    # --- Store Outputs ---
    if status is None or "modified" in (status or ""): # Store output if successful or modified
        output_spec = stage.outputs.spec
        for label, type_hint in output_spec.items():
            # Simple assumption: the entire response is the output
            # TODO: Add more sophisticated type handling (JSON parsing for 'object', etc.)
            stage_outputs[label] = final_response_content
            logger.debug(f"Stage {stage_id}: Storing output '{label}' (type: {type_hint})")

    # --- Finalize ---
    updated_context = pipeline_context.copy()
    if stage_outputs:
        updated_context[stage_id] = stage_outputs

    end_time = time.monotonic()
    logger.info(f"Finished LLM stage: {stage_id} (Duration: {end_time - start_time:.2f}s) Status: {status or 'success'}")

    return updated_context, interactions, status
