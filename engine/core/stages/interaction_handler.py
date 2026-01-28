import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

import asyncio # Added for async operations
import time # Added for timing GroupChat
import re # Import re for guardrail modification
from datetime import timedelta # Added for timestamp incrementing

# --- Autogen Imports ---
try:
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, ConversableAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    # Define dummy classes if autogen is not available
    class AssistantAgent:
         def __init__(self, *args, **kwargs): self.name = kwargs.get("name")
    class UserProxyAgent:
         def __init__(self, *args, **kwargs): self.name = kwargs.get("name")
    class GroupChat:
         def __init__(self, agents, messages, **kwargs): self.messages = []
    class GroupChatManager:
         def __init__(self, groupchat, llm_config): self.groupchat = groupchat # Store groupchat
         async def a_initiate_chat(self, *args, **kwargs): # Mock async initiate_chat if needed
             self.groupchat.messages.append({"role": "assistant", "content": "Dummy GroupChat Reply - Autogen Import Failed"})
             return None # Simulate ChatResult or similar structure if needed by caller
    class ConversableAgent:
         def __init__(self, *args, **kwargs): pass
         async def a_initiate_chat(self, *args, **kwargs): # Mock async initiate_chat
             return None # Simulate ChatResult or similar structure if needed by caller


    AUTOGEN_AVAILABLE = False

# Project Imports
# Updated import to include concurrency_monitor
from config import llm_config, semaphore, AGENT_TIMEOUT, concurrency_monitor
from schemas.stage import Stage, StageRole
from schemas.interaction import Interaction, InteractionRole, Metadata
from schemas.identity import Identity
from schemas.ethical_guidance import EthicalGuidance
from schemas.guardrail import Guardrail
from utils.placeholder_resolver import resolve_placeholders
from . import register_stage_handler # Import registry and decorator
# Import guardrail checker if needed for group chat messages
# from core.guardrails import check_guardrails

logger = logging.getLogger(__name__)

@register_stage_handler("interaction")
async def handle_interaction_stage(
    stage: Stage,
    pipeline_context: Dict[str, Any],
    engine_instance: Any, # Reference to the EthicsEngine instance
    identity: Optional[Identity], # Passed directly from engine
    guidance: Optional[EthicalGuidance], # Passed directly from engine
    active_guardrails: List[Guardrail] # Passed directly from engine
) -> Tuple[Dict[str, Any], List[Interaction], Any]:
    """
    Handles the execution of an interaction stage.

    Supports two modes:
    1. Multi-Agent Group Chat: If stage.participants has > 1 entry and Autogen is available.
    2. Script Processing: If not using multi-agent chat, processes the prompt as a script.

    Args:
        stage: The Stage object definition.
        pipeline_context: Dictionary containing outputs from previous stages.
        engine_instance: The EthicsEngine instance.
        identity: The active Identity profile.
        guidance: The active EthicalGuidance profile.
        active_guardrails: List of active guardrails (may be used for input checks).

    Returns:
        A tuple containing:
        - Updated pipeline_context with outputs from this stage.
        - List of interactions generated during this stage.
        - Stage status/outcome (usually None for success).
    """
    interactions: List[Interaction] = []
    stage_outputs: Dict[str, Any] = {}
    status: Optional[str] = None
    stage_id = stage.id

    logger.info(f"Handling interaction stage: {stage_id}")

    # --- Check for Multi-Agent Group Chat ---
    if AUTOGEN_AVAILABLE and stage.participants and len(stage.participants) > 1:
        logger.info(f"Stage {stage_id}: Detected multi-participant interaction ({len(stage.participants)} participants). Using GroupChat.")
        start_time = time.monotonic() # Start timer for group chat part

        # --- Pre-checks for GroupChat ---
        if not llm_config:
            logger.error(f"LLMConfig not available for GroupChat stage {stage_id}. Skipping.")
            status = "error: LLM config unavailable for group chat"
            return pipeline_context, interactions, status

        # --- Resolve Initial Prompt/Context ---
        initial_prompt_template = stage.prompt or "Please discuss."
        try:
            # Resolve context placeholders first if context is provided
            resolved_context = {}
            if stage.context:
                 resolved_context = resolve_placeholders(stage.context, pipeline_context)
            # Combine pipeline context and resolved stage context for prompt resolution
            full_context_for_prompt = {**pipeline_context, **resolved_context}
            initial_message_content = resolve_placeholders(initial_prompt_template, full_context_for_prompt)
            logger.debug(f"Stage {stage_id}: Resolved initial message for GroupChat: {initial_message_content}")
        except Exception as e:
            logger.error(f"Error resolving placeholders for GroupChat stage {stage_id}: {e}", exc_info=True)
            status = f"error: placeholder resolution failed: {e}"
            return pipeline_context, interactions, status

        # --- Create Agents ---
        agents: List[ConversableAgent] = []
        try:
            # Check if engine_instance has the get_identity method
            get_identity_method = getattr(engine_instance, 'get_identity', None)
            if not callable(get_identity_method):
                 logger.warning("Engine instance does not have a callable 'get_identity' method. Cannot load specific identities for participants.")
                 get_identity_method = None # Ensure it's None if not callable

            for participant_name in stage.participants:
                active_identity_for_agent = identity # Default to pipeline identity
                participant_guidance = guidance # Default to pipeline guidance (can be customized later)

                # Check for participant-specific configurations
                if stage.participant_configs and isinstance(stage.participant_configs, dict) and participant_name in stage.participant_configs:
                    p_config = stage.participant_configs[participant_name]
                    if isinstance(p_config, dict):
                        # Load specific identity if configured and getter method available
                        specific_identity_id = p_config.get("identity_id")
                        if specific_identity_id and get_identity_method:
                            try:
                                # Use the engine's get_identity method
                                loaded_identity = get_identity_method(specific_identity_id)
                                if loaded_identity:
                                    active_identity_for_agent = loaded_identity
                                    logger.debug(f"Using specific identity '{specific_identity_id}' for participant '{participant_name}'.")
                                else:
                                    logger.warning(f"Specific identity '{specific_identity_id}' configured for '{participant_name}' but not found via engine.get_identity. Using pipeline default.")
                            except Exception as lookup_exc:
                                logger.error(f"Error loading specific identity '{specific_identity_id}' for '{participant_name}' via engine.get_identity: {lookup_exc}. Using pipeline default.", exc_info=True)
                        elif specific_identity_id:
                             logger.warning(f"Specific identity '{specific_identity_id}' configured for '{participant_name}' but engine's get_identity method is unavailable. Using pipeline default.")

                        # TODO: Add similar logic for loading specific ethical_guidance_id if needed using engine.get_guidance

                    else:
                         logger.warning(f"Configuration for participant '{participant_name}' in participant_configs is not a dictionary. Ignoring.")
                else:
                    logger.debug(f"No specific config found for participant '{participant_name}'. Using pipeline defaults.")


                # Construct system message using the determined identity and guidance
                system_message = f"You are {participant_name}."
                if participant_guidance and participant_guidance.prompt_template:
                    system_message += f" Follow this guidance: {participant_guidance.prompt_template}"

                # Get description, handling both object and dict types (for testing)
                identity_description = None
                if isinstance(active_identity_for_agent, dict):
                    identity_description = active_identity_for_agent.get("description")
                elif hasattr(active_identity_for_agent, 'description'):
                    identity_description = active_identity_for_agent.description

                if identity_description:
                    system_message += f" Consider this identity context: {identity_description}"
                # Add termination instruction
                system_message += " After the discussion concludes or reaches a natural stopping point, respond with TERMINATE."


                agent = AssistantAgent(
                    name=participant_name,
                    system_message=system_message,
                    llm_config=llm_config, # Pass llm_config object/dict directly
                    # Ensure agents can terminate
                    is_termination_msg=lambda x: isinstance(x.get("content"), str) and "TERMINATE" in x.get("content", ""),
                )
                agents.append(agent)
            logger.info(f"Stage {stage_id}: Created {len(agents)} agents for GroupChat: {[a.name for a in agents]}")
        except Exception as e:
            logger.error(f"Error creating agents for GroupChat stage {stage_id}: {e}", exc_info=True)
            status = f"error: agent creation failed: {e}"
            return pipeline_context, interactions, status

        if not agents:
             logger.error(f"No agents created for GroupChat stage {stage_id}.")
             status = "error: no agents created"
             return pipeline_context, interactions, status

        # --- Setup and Run GroupChat ---
        # Add speaker_selection_method for smaller groups as suggested by warning
        groupchat = GroupChat(
            agents=agents,
            messages=[],
            max_round=10, # Limit rounds
            speaker_selection_method='round_robin' # Use round robin for 2 agents
        )
        # Use llm_config directly for the manager as well
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=llm_config, # Pass llm_config object/dict directly
            # Ensure manager can also terminate or recognize termination
            is_termination_msg=lambda x: isinstance(x.get("content"), str) and "TERMINATE" in x.get("content", "")
        )

        # Use a UserProxyAgent to initiate the chat with the manager
        # This agent just sends the initial message and doesn't participate further
        initiator_proxy = UserProxyAgent(
            name="InitiatorProxy",
            human_input_mode="NEVER",
            llm_config=False, # No LLM needed for this proxy
            code_execution_config=False, # Explicitly disable code execution/docker
            default_auto_reply="", # Don't reply automatically
            is_termination_msg=lambda x: True, # Terminate immediately after sending
        )

        chat_history = []
        final_chat_outcome = "Error: GroupChat execution failed."

        # Outer try/finally for monitor decrement
        try:
            sema_acquire_start = time.monotonic()
            concurrency_monitor.increment_active() # Increment before acquiring
            async with semaphore:
                sema_acquire_end = time.monotonic()
                logger.debug(f"Stage {stage_id} GroupChat: Semaphore acquired (wait={sema_acquire_end - sema_acquire_start:.2f}s)")

                # Inner try/except for chat execution
                try:
                    # Initiate chat using the proxy sending to the manager
                    await initiator_proxy.a_initiate_chat(
                        manager,
                        message=initial_message_content,
                        max_turns=1 # Proxy only sends one message
                    )
                    # The actual conversation happens within the manager's processing triggered by the initial message
                    # We retrieve the history from the groupchat object after initiation
                    chat_history = groupchat.messages
                    logger.info(f"Stage {stage_id}: GroupChat finished. History length: {len(chat_history)}")

                except asyncio.TimeoutError:
                    status = f"error: GroupChat execution timed out after {AGENT_TIMEOUT}s"
                    final_chat_outcome = status
                    logger.error(status)
                    # Re-raise or handle as needed, maybe set status and continue to finally
                    raise # Re-raise to be caught by outer handler
                except Exception as e:
                    logger.error(f"Error during GroupChat execution for stage {stage_id}: {e}", exc_info=True)
                    status = f"error: GroupChat execution failed: {e}"
                    final_chat_outcome = status
                    # Re-raise or handle as needed
                    raise # Re-raise to be caught by outer handler

            sema_release_time = time.monotonic()
            logger.debug(f"Stage {stage_id} GroupChat: Semaphore released (held for {sema_release_time - sema_acquire_end:.2f}s)")

            # --- Process Chat History (Only if no exception occurred) ---
            if status is None: # Check if chat execution was successful before processing
                current_timestamp = datetime.now(timezone.utc)
                for i, msg in enumerate(chat_history):
                    # Convert Autogen message format to Interaction schema
                    role = msg.get("role", "system") # Default role
                    sender_name = msg.get("name")

                    # Refine role based on sender name and map to allowed InteractionRole values
                    if sender_name == initiator_proxy.name:
                        interaction_role = "system" # Treat initiator message as system setup
                    elif sender_name and sender_name in [a.name for a in agents]:
                        # Map any participating agent name to 'assistant' for the Interaction schema
                        interaction_role = "assistant"
                    elif role == "user": # If role is user but name isn't initiator, it might be manager speaking as user proxy
                         interaction_role = "system" # Or manager? Needs clarity based on Autogen behavior. Defaulting to system.
                    else:
                        interaction_role = "system" # Default fallback

                    interactions.append(Interaction(
                        stage_id=stage_id,
                        role=interaction_role, # Use the mapped role
                        content=str(msg.get("content", "")).replace("TERMINATE", "").strip(), # Ensure content is string and remove TERMINATE
                        metadata=Metadata(
                            timestamp=(current_timestamp + timedelta(microseconds=i)).isoformat(),
                            model=llm_config.config_list[0].get("model") if llm_config else None # Basic model info
                            # TODO: Add guardrail checks for each message if needed
                        ),
                        schema_version=stage.schema_version
                    ))

                # Determine final output (e.g., last message content, excluding TERMINATE)
                if chat_history:
                    # Find the last non-empty message from an agent participant
                    last_agent_message = None
                    for msg in reversed(chat_history):
                        sender_name = msg.get("name")
                        content = str(msg.get("content", "")).replace("TERMINATE", "").strip()
                        if sender_name in [a.name for a in agents] and content:
                            last_agent_message = content
                            break
                    if last_agent_message:
                        final_chat_outcome = last_agent_message
                        status = None # Success if chat completed and we found a message
                        logger.info(f"Stage {stage_id}: Using last agent message as final output: '{final_chat_outcome[:100]}...'")
                    else:
                        final_chat_outcome = "Error: No final message found from participating agents."
                        status = "error: no agent message found"
                        logger.warning(f"Stage {stage_id}: GroupChat completed but no final message from agents found.")

                else:
                    final_chat_outcome = "Error: GroupChat produced no history."
                    status = "error: empty chat history"

        except Exception as outer_e: # Catch exceptions from semaphore or inner block re-raises
             if not status: # If status wasn't already set by inner exception
                 logger.error(f"Outer exception during GroupChat handling for stage {stage_id}: {outer_e}", exc_info=True)
                 status = f"error: GroupChat handling failed: {outer_e}"
                 final_chat_outcome = status
        finally:
             concurrency_monitor.decrement_active() # Ensure decrement happens


        # --- Store Output ---
        if status is None: # Only store if successful
             if stage.outputs and stage.outputs.spec:
                 output_spec = stage.outputs.spec
                 for label, type_hint in output_spec.items():
                     if type_hint == "text":
                         stage_outputs[label] = final_chat_outcome
                         logger.debug(f"Interaction stage {stage_id} (GroupChat) produced text output '{label}'.")
                     else:
                         logger.warning(f"Interaction stage {stage_id} (GroupChat) defines output '{label}' of unhandled type '{type_hint}'.")
                         stage_outputs[label] = None
             elif stage.outputs:
                  logger.warning(f"Interaction stage {stage_id} (GroupChat) has 'outputs' defined but missing 'spec'.")

        # --- Finalize GroupChat Path ---
        end_time = time.monotonic()
        logger.info(f"Finished interaction stage (GroupChat): {stage_id}. Generated {len(interactions)} interactions. Duration: {end_time - start_time:.2f}s Status: {status or 'success'}")
        updated_context = pipeline_context.copy()
        if stage_outputs:
            updated_context[stage_id] = stage_outputs
        return updated_context, interactions, status

    # --- Fallback to Existing Logic (No Multi-Participant GroupChat) ---
    else:
        # This is the original logic for processing the prompt as a script
        logger.debug(f"Stage {stage_id}: Handling as single interaction or prompt script (participants not > 1 or Autogen unavailable).")
        full_transcript = "" # Reset transcript specific to this logic path
        start_time = time.monotonic() # Start timer for script part

        # 1. Get the raw prompt script template
        prompt_template = stage.prompt or ""

        # 2. Parse the script template into turns *before* resolving placeholders
        script_lines = prompt_template.strip().splitlines()
        logger.debug(f"Split prompt into {len(script_lines)} lines.")
        current_timestamp = datetime.now(timezone.utc)

        if len(script_lines) == 1 and ':' not in script_lines[0]:
            # Handle single-line prompt without Role: prefix (treat as system/instruction)
            logger.debug(f"Handling single-line prompt for stage {stage_id} as system role.")
            role_str = "system"
            content_template = script_lines[0].strip()
            try:
                resolved_content = resolve_placeholders(content_template, pipeline_context)
                logger.debug(f"  Resolved Content: '{resolved_content}'")
                interactions.append(Interaction(
                    stage_id=stage_id, role=role_str, content=resolved_content,
                    metadata=Metadata(timestamp=current_timestamp.isoformat()),
                    schema_version=stage.schema_version
                ))
                full_transcript += f"{role_str.capitalize()}: {resolved_content}\n"
            except Exception as e:
                logger.error(f"Error resolving placeholders for single-line prompt in stage {stage_id}: '{content_template}'. Error: {e}", exc_info=True)
                status = "error: placeholder resolution failed"
                interactions.append(Interaction(
                    stage_id=stage_id, role="system", content=f"Error resolving content for prompt: {e}",
                    metadata=Metadata(timestamp=datetime.now(timezone.utc).isoformat())
                ))
        else:
            # Process multi-line script with Role: prefixes
            for i, line in enumerate(script_lines):
                line = line.strip()
                logger.debug(f"Processing line {i+1}/{len(script_lines)}: '{line}'")
                role_str: Optional[str] = None
                content_template: str = ""
                if line.startswith("User:"):
                    role_str = "user"
                    content_template = line[len("User:"):].strip()
                elif line.startswith("Assistant:"):
                    role_str = "assistant"
                    content_template = line[len("Assistant:"):].strip()
                elif line.startswith("System:"):
                    role_str = "system"
                    content_template = line[len("System:"):].strip()
                elif ":" in line:
                     parts = line.split(":", 1)
                     role_str = parts[0].lower().strip()
                     content_template = parts[1].strip()
                else:
                    logger.warning(f"Interaction stage {stage_id}: Skipping line without recognized 'Role:' prefix: '{line}'")
                    continue

                if role_str is not None:
                    logger.debug(f"  Parsed Role: {role_str}, Template: '{content_template}'")
                    try:
                        resolved_content = resolve_placeholders(content_template, pipeline_context)
                        logger.debug(f"  Resolved Content: '{resolved_content}'")
                    except Exception as e:
                        logger.error(f"Error resolving placeholders for line in stage {stage_id}: '{line}'. Error: {e}", exc_info=True)
                        interactions.append(Interaction(
                            stage_id=stage_id, role="system", content=f"Error resolving content for line: {line}. Error: {e}",
                            metadata=Metadata(timestamp=datetime.now(timezone.utc).isoformat())
                        ))
                        status = "error: placeholder resolution failed for one or more lines"
                        continue

                    triggered_violation_ids: List[str] = []
                    # TODO: Add input guardrail check here if needed

                    turn_timestamp = current_timestamp + timedelta(microseconds=i)
                    interaction_metadata = Metadata(
                        timestamp=turn_timestamp.isoformat(),
                        guardrails_triggered=triggered_violation_ids
                    )
                    interactions.append(Interaction(
                        stage_id=stage_id,
                        role=role_str,
                        content=resolved_content,
                        metadata=interaction_metadata,
                        schema_version=stage.schema_version
                    ))
                    logger.debug(f"  Appended interaction. Current count: {len(interactions)}")
                    full_transcript += f"{role_str.capitalize()}: {resolved_content}\n"

        logger.debug(f"Finished processing lines for stage {stage_id}. Total interactions generated: {len(interactions)}")

        # 3. Store Defined Outputs (using full_transcript from script processing)
        if stage.outputs and stage.outputs.spec:
            output_spec = stage.outputs.spec
            for label, type_hint in output_spec.items():
                if type_hint == "text":
                    stage_outputs[label] = full_transcript.strip()
                    logger.debug(f"Interaction stage {stage_id} (Script Mode) produced text output '{label}'.")
                else:
                    logger.warning(f"Interaction stage {stage_id} (Script Mode) defines output '{label}' of unhandled type '{type_hint}'.")
                    stage_outputs[label] = None
        elif stage.outputs:
             logger.warning(f"Interaction stage {stage_id} (Script Mode) has 'outputs' defined but missing 'spec'.")

        # 4. Finalize Script Path
        end_time = time.monotonic()
        logger.info(f"Finished interaction stage (Script Mode): {stage_id}. Generated {len(interactions)} interactions. Duration: {end_time - start_time:.2f}s Status: {status or 'success'}")
        updated_context = pipeline_context.copy()
        if stage_outputs:
            updated_context[stage_id] = stage_outputs
        return updated_context, interactions, status
