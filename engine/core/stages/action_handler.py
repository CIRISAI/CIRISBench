import logging
import asyncio
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

# Project Imports
from schemas.stage import Stage
from schemas.interaction import Interaction, InteractionRole, Metadata
from schemas.identity import Identity
from schemas.ethical_guidance import EthicalGuidance
from schemas.guardrail import Guardrail
from utils.placeholder_resolver import resolve_placeholders
from . import register_stage_handler # Import registry and decorator
# Import shared semaphore and monitor
from config import semaphore, concurrency_monitor

logger = logging.getLogger(__name__)

# Default timeout for subprocess execution
ACTION_TIMEOUT = 120 # seconds

@register_stage_handler("action")
async def handle_action_stage(
    stage: Stage,
    pipeline_context: Dict[str, Any],
    engine_instance: Any, # Reference to the EthicsEngine instance
    identity: Optional[Identity], # Passed directly from engine
    guidance: Optional[EthicalGuidance], # Passed directly from engine
    active_guardrails: List[Guardrail] # Passed directly from engine
) -> Tuple[Dict[str, Any], List[Interaction], Any]:
    """
    Handles the execution of an action stage by running a shell command.

    Args:
        stage: The Stage object definition. Expects 'prompt' to contain the command.
        pipeline_context: Dictionary containing outputs from previous stages.
        engine_instance: The EthicsEngine instance.
        identity: The active Identity profile.
        guidance: The active EthicalGuidance profile.
        active_guardrails: List of active guardrails (potentially check command?).

    Returns:
        A tuple containing:
        - Updated pipeline_context with outputs (stdout, stderr, returncode).
        - List of interactions (recording command and result).
        - Stage status/outcome.
    """
    interactions: List[Interaction] = []
    stage_outputs: Dict[str, Any] = {}
    status: Optional[str] = None
    stage_id = stage.id

    logger.info(f"Handling action stage: {stage_id}")

    # 1. Get and Resolve Command from Prompt
    command_template = stage.prompt or ""
    if not command_template:
        status = "error: action stage requires a command in the 'prompt' field"
        logger.error(f"Action stage {stage_id} is missing command in 'prompt'.")
        interactions.append(Interaction(
            stage_id=stage_id, role="system", content=status,
            metadata=Metadata(timestamp=datetime.now(timezone.utc).isoformat())
        ))
        return pipeline_context, interactions, status

    try:
        command_to_run = resolve_placeholders(command_template, pipeline_context)
        logger.info(f"Action Stage {stage_id} - Resolved Command: {command_to_run}")
    except Exception as e:
        logger.error(f"Error resolving placeholders for action stage {stage_id}: {e}", exc_info=True)
        status = f"error: placeholder resolution failed: {e}"
        interactions.append(Interaction(
            stage_id=stage_id, role="system", content=status,
            metadata=Metadata(timestamp=datetime.now(timezone.utc).isoformat())
        ))
        return pipeline_context, interactions, status

    # 2. Optional: Guardrail Check on Command?
    # Consider if guardrails should apply to the command string itself.
    # For now, skipping guardrail check on the command.

    # 3. Execute Command Asynchronously
    stdout_str = ""
    stderr_str = ""
    return_code = -1
    start_time = datetime.now(timezone.utc)
    process = None # Initialize process

    # Wrap subprocess execution in semaphore and monitor tracking
    try:
        concurrency_monitor.increment_active() # Increment before acquiring
        async with semaphore:
            logger.debug(f"Action Stage {stage_id}: Semaphore acquired.")
            # Use asyncio.create_subprocess_shell for non-blocking execution
            process = await asyncio.create_subprocess_shell(
            command_to_run,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

            # Wait for the command to complete with a timeout *inside* the semaphore block
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=ACTION_TIMEOUT)
                return_code = process.returncode
                stdout_str = stdout_bytes.decode(errors='ignore') if stdout_bytes else ""
                stderr_str = stderr_bytes.decode(errors='ignore') if stderr_bytes else ""
                logger.info(f"Action stage {stage_id} command finished. Return Code: {return_code}")
                if stdout_str: logger.debug(f"Action Stage {stage_id} - STDOUT:\n{stdout_str[:500]}...") # Log truncated stdout
                if stderr_str: logger.warning(f"Action Stage {stage_id} - STDERR:\n{stderr_str[:500]}...") # Log truncated stderr

                if return_code != 0:
                    status = f"error: command exited with non-zero code {return_code}"
                    logger.error(f"Action stage {stage_id} command failed with code {return_code}.")

            except asyncio.TimeoutError:
                logger.error(f"Action stage {stage_id} command timed out after {ACTION_TIMEOUT} seconds.")
                status = f"error: command timed out"
                # Termination logic moved outside the semaphore block in finally
                # Still try to get any output captured before timeout - this needs care,
                # process might be None if create_subprocess_shell failed before timeout block
                if process:
                    try:
                        # This communicate might hang if the process is stuck after timeout
                        # Consider just using process.stdout.read() / process.stderr.read() if needed
                        stdout_bytes, stderr_bytes = await process.communicate()
                        stdout_str = stdout_bytes.decode(errors='ignore') if stdout_bytes else ""
                        stderr_str = stderr_bytes.decode(errors='ignore') if stderr_bytes else ""
                    except Exception as comm_err:
                         logger.error(f"Action stage {stage_id}: Error communicating with process after timeout: {comm_err}")


    except FileNotFoundError:
        logger.error(f"Action stage {stage_id}: Command not found: '{command_to_run.split()[0]}'")
        status = "error: command not found"
        stderr_str = "Command not found"
        return_code = 127 # Typical exit code for command not found
    except Exception as e:
        logger.error(f"Error executing action stage {stage_id} command: {e}", exc_info=True)
        status = f"error: command execution failed: {e}"
        stderr_str = str(e)
    finally:
        # Ensure monitor count is decremented even if errors occur
        concurrency_monitor.decrement_active()
        logger.debug(f"Action Stage {stage_id}: Semaphore released.")
        # Ensure process is handled if it exists and timed out or errored early
        if process and process.returncode is None:
             try:
                 process.terminate()
                 await process.wait()
             except Exception:
                 pass # Ignore errors during cleanup termination

    end_time = datetime.now(timezone.utc)

    # 4. Record Interaction
    result_summary = f"Command executed: '{command_to_run}'\nReturn Code: {return_code}\n"
    if stdout_str: result_summary += f"STDOUT (truncated):\n{stdout_str[:500]}...\n"
    if stderr_str: result_summary += f"STDERR (truncated):\n{stderr_str[:500]}...\n"

    interaction_metadata = Metadata(
        timestamp=start_time.isoformat(), # Use start time for the interaction record
        # Add duration?
    )
    interactions.append(Interaction(
        stage_id=stage_id,
        role="system", # Action is performed by the system
        content=result_summary.strip(),
        metadata=interaction_metadata,
        schema_version=stage.schema_version
    ))

    # 5. Store Outputs
    # Store stdout, stderr, and return code based on output spec
    output_spec = stage.outputs.spec
    for label, type_hint in output_spec.items():
        if label == "stdout":
            stage_outputs[label] = stdout_str
        elif label == "stderr":
            stage_outputs[label] = stderr_str
        elif label == "return_code":
            stage_outputs[label] = return_code
        else:
            logger.warning(f"Action stage {stage_id} encountered unknown output label '{label}' in spec.")
            stage_outputs[label] = None

    # 6. Finalize
    updated_context = pipeline_context.copy()
    if stage_outputs:
        updated_context[stage_id] = stage_outputs

    logger.info(f"Finished action stage: {stage_id}. Status: {status or 'success'}")

    return updated_context, interactions, status
