import asyncio # Import asyncio
import logging
import re
import os
import ujson # Added import
from typing import List, Dict, Optional, Tuple, Any
from open_llm.config_llm import LLMSetter
# Initialize logger first
logger = logging.getLogger(__name__)

settings = LLMSetter()

# --- OpenAI Integration ---
try:
    from openai import OpenAI, APIError
    # Initialize OpenAI client - assumes OPENAI_API_KEY is set in environment
    # Consider moving client initialization to a central config/engine if used frequently
    openai_client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI library not found. Classifier guardrails using OpenAI will not function.")
    openai_client = None
    OPENAI_AVAILABLE = False
except Exception as e:
     logger.error(f"Failed to initialize OpenAI client: {e}. Classifier guardrails using OpenAI may fail.")
     openai_client = None # Ensure client is None if init fails
     OPENAI_AVAILABLE = False # Treat as unavailable if init fails

# --- Project Imports ---
from schemas.guardrail import Guardrail, GuardrailScope, GuardrailAction, GuardrailTrigger
from schemas.results import ResultViolation # To create violation objects

# --- Known OpenAI Moderation Categories ---
# Used to validate classifier_id in guardrail triggers
OPENAI_MODERATION_CATEGORIES = {
    "hate", "hate/threatening", "harassment", "harassment/threatening",
    "self-harm", "self-harm/intent", "self-harm/instructions",
    "sexual", "sexual/minors", "violence", "violence/graphic"
}

def check_content_filter(trigger: GuardrailTrigger, content: str) -> bool:
    """Checks content against regex or keyword triggers."""
    if trigger.regex:
        try:
            if re.search(trigger.regex, content, re.IGNORECASE): # Default to case-insensitive
                return True
        except re.error as e:
            logger.error(f"Invalid regex in guardrail trigger: {trigger.regex}. Error: {e}")
            return False # Treat invalid regex as non-triggering? Or raise?
    if trigger.keywords:
        # Simple keyword check (case-insensitive)
        content_lower = content.lower()
        if any(keyword.lower() in content_lower for keyword in trigger.keywords):
            return True
    return False

async def check_classifier(trigger: GuardrailTrigger, content: str) -> bool: # Changed to async def
    """Checks content using the OpenAI Moderation API."""
    if not OPENAI_AVAILABLE or not openai_client:
        logger.error("OpenAI client not available or failed to initialize. Cannot check classifier guardrail.")
        return False # Cannot perform check

    classifier_id = trigger.classifier # This should match an OpenAI category key
    threshold = trigger.threshold

    if not classifier_id or threshold is None:
        logger.warning("Classifier guardrail trigger is missing classifier category ID or threshold.")
        return False
    if classifier_id not in OPENAI_MODERATION_CATEGORIES:
         logger.warning(f"Classifier ID '{classifier_id}' is not a recognized OpenAI moderation category.")
         return False, None # Invalid category specified, no response

    logger.debug(f"Running OpenAI Moderation check for category '{classifier_id}' on content...")
    full_response_dict: Optional[Dict[str, Any]] = None # Initialize response dict

    try:
        # Wrap the synchronous OpenAI call in asyncio.to_thread
        response = await asyncio.to_thread(
            openai_client.moderations.create,
            input=content
        )
        # Store the full response as a dictionary
        full_response_dict = response.model_dump() if response else None

        # Example response structure:
        # Moderation(id='modr-...', model='text-moderation-007', results=[ModerationResult(
        # categories=Categories(hate=False, hate_threatening=False, ...),
        # category_scores=CategoryScores(hate=4.7e-06, hate_threatening=1.2e-07, ...),
        # flagged=False)])

        if response and response.results:
            result = response.results[0]
            # Get the score for the specific category defined in the guardrail trigger
            # Replace '/' with '_' for attribute access (e.g., 'hate/threatening' -> 'hate_threatening')
            category_attr_name = classifier_id.replace('/', '_')
            category_score = getattr(result.category_scores, category_attr_name, None)

            if category_score is None:
                 logger.error(f"Could not find score for category '{classifier_id}' (attribute '{category_attr_name}') in OpenAI Moderation response.")
                 return False, full_response_dict # Return response even if category score missing

            logger.debug(f"OpenAI Moderation category '{classifier_id}' score: {category_score:.4f}. Threshold: {threshold}")
            # Trigger if the specific category score meets or exceeds the threshold
            triggered = category_score >= threshold
            return triggered, full_response_dict
        else:
            logger.error("OpenAI Moderation API returned no results or response was empty.")
            return False, full_response_dict # Return response dict even if no results

    except APIError as e:
        logger.error(f"OpenAI API error during moderation check: {e}")
        return False, None # Fail safe - assume not triggered, no response available
    except Exception as e:
         logger.error(f"Unexpected error during OpenAI moderation check: {e}", exc_info=True)
         return False, None # Fail safe, no response available

# --- Placeholder Check Functions ---

async def _evaluate_policy_with_llm(policy_description: str, content_to_check: str) -> bool:
    """Placeholder for a potentially async LLM call to evaluate policy compliance."""
    # TODO: Implement LLM call to check if content violates the policy_description.
    # This would likely involve constructing a prompt asking an LLM judge if the
    # content_to_check adheres to the policy_description.
    # Requires access to an LLM client (like openai_client or via engine_instance).
    logger.warning("LLM-based policy evaluation not implemented.")
    await asyncio.sleep(0) # Simulate async operation
    return False

async def check_policy(trigger: GuardrailTrigger, content: str, engine_instance: Optional[Any] = None) -> bool:
    """
    Checks content against a policy guardrail.

    Requires specific implementation based on the policy definition.
    Could involve rule engines, keyword checks, or LLM calls.
    """
    policy_description = trigger.policy_description # Assuming trigger schema has this field
    if not policy_description:
        logger.warning("Policy guardrail trigger is missing 'policy_description'. Cannot check.")
        return False

    # Example: Using an LLM for evaluation (requires engine_instance or LLM client)
    # if engine_instance and hasattr(engine_instance, 'generate_evaluation_response'):
    #     # This is just an example structure, the actual implementation might differ
    #     prompt = f"Does the following text violate this policy: '{policy_description}'?\n\nText: \"{content}\"\n\nAnswer strictly with YES or NO."
    #     # Need identity/guidance for the evaluation call - perhaps default ones?
    #     # response = await engine_instance.generate_evaluation_response(prompt, default_identity, default_guidance)
    #     # return "yes" in response.lower()
    #     logger.warning("LLM-based policy check structure present but not fully implemented.")
    #     # Fallback to placeholder LLM call
    #     return await _evaluate_policy_with_llm(policy_description, content)

    # --- Specific Policy Implementation Example ---
    # Check for the specific policy description we want to implement
    target_policy_desc = "Content must be respectful, safe, and appropriate for all audiences. Avoid controversial topics unless directly relevant and handled neutrally."
    if policy_description == target_policy_desc:
        logger.debug(f"Applying basic checks for policy: '{target_policy_desc}'")
        # Basic keyword checks for disrespectful/unsafe content (case-insensitive)
        # This is a very basic example and should be expanded significantly for real use.
        disrespectful_keywords = ["idiot", "stupid", "moron", "jerk"] # Add more as needed
        unsafe_keywords = ["bomb", "kill", "attack"] # Add more, be careful with context
        content_lower = content.lower()

        if any(keyword in content_lower for keyword in disrespectful_keywords):
            logger.warning(f"Policy '{target_policy_desc}' triggered by disrespectful keyword.")
            return True # Triggered

        if any(keyword in content_lower for keyword in unsafe_keywords):
             logger.warning(f"Policy '{target_policy_desc}' triggered by potentially unsafe keyword.")
             return True # Triggered

        # Controversial topic check is complex and context-dependent, skipping for this basic example.
        logger.debug(f"Basic checks passed for policy: '{target_policy_desc}'")
        return False # Not triggered by basic checks

    # --- Fallback for other/unimplemented policies ---
    else:
        # Log for any *other* policy descriptions that don't have specific logic yet.
        logger.info(f"Policy guardrail check logic for description '{policy_description}' not fully implemented.")
        return False # Default to not triggered for other policies

def check_rate_limit(trigger: GuardrailTrigger, state: Optional[Dict[str, Any]] = None) -> bool:
    """
    Checks if an action exceeds defined rate limits.

    Requires state management (e.g., tracking counts/timestamps per user/session/key).
    The 'state' dictionary must be provided by the calling context (e.g., engine or handler)
    and should contain the necessary tracking information.

    Args:
        trigger: The GuardrailTrigger containing rate limit parameters (e.g., max_count, time_window_seconds).
        state: A dictionary holding the current state for rate limiting (e.g., timestamps of previous events).

    Returns:
        True if the rate limit is exceeded, False otherwise.
    """
    # TODO: Implement actual rate limit logic using the 'state' dictionary.
    # Example logic:
    # 1. Get parameters from trigger (e.g., trigger.max_count, trigger.time_window_seconds).
    # 2. Get tracking data from state (e.g., state['rate_limit_timestamps']).
    # 3. Filter timestamps within the time window.
    # 4. Check if count exceeds max_count.
    # 5. IMPORTANT: The caller must also *update* the state after a successful action (e.g., add current timestamp).

    if state is None:
        logger.warning("Rate limit check skipped: 'state' dictionary not provided.")
        return False # Cannot check without state

    max_count = getattr(trigger, 'max_count', None)
    time_window = getattr(trigger, 'time_window_seconds', None)

    if max_count is None or time_window is None:
        logger.warning("Rate limit guardrail trigger is missing 'max_count' or 'time_window_seconds'.")
        return False

    # --- Example State Structure (caller needs to manage this) ---
    # state = {
    #     'rate_limit_history': {
    #         'guardrail_id_abc': [timestamp1, timestamp2, ...],
    #         'guardrail_id_xyz': [timestampA, timestampB, ...]
    #     }
    # }
    # guardrail_id = trigger.parent_guardrail_id # Need a way to link trigger back to guardrail ID
    # timestamps = state.get('rate_limit_history', {}).get(guardrail_id, [])
    # current_time = time.time()
    # recent_timestamps = [ts for ts in timestamps if current_time - ts <= time_window]
    # if len(recent_timestamps) >= max_count:
    #     logger.warning(f"Rate limit exceeded: {len(recent_timestamps)+1}/{max_count} in {time_window}s.")
    #     return True # Exceeded
    # else:
    #     # Caller should add current_time to state['rate_limit_history'][guardrail_id] if action proceeds
    #     return False # Not exceeded
    # --- End Example ---

    logger.warning(f"Rate limit guardrail check logic not fully implemented (requires state management by caller).")
    return False # Default to not triggered

# --- Main Guardrail Checking Function ---

async def check_guardrails( # Changed to async def
    content: str,
    active_guardrails: List[Guardrail],
    scope: GuardrailScope,
    engine_instance: Optional[Any] = None # Added engine_instance for policy checks
) -> List[ResultViolation]:
    """
    Checks content against a list of active guardrails for a given scope. (Async)

    Args:
        content: The text content to check (e.g., user input or LLM output).
        active_guardrails: A list of Guardrail objects to apply.
        scope: The scope to check against ('input', 'output', 'both').

    Returns:
        A list of ResultViolation objects for any triggered guardrails.
    """
    triggered_violations: List[ResultViolation] = []
    # Removed violation_details dict as we'll store full response directly

    for guardrail in active_guardrails:
        # Check if the guardrail applies to the current scope
        if guardrail.scope != scope and guardrail.scope != 'both':
            continue

        triggered = False
        check_type = guardrail.type

        try:
            if check_type == "content_filter":
                triggered = check_content_filter(guardrail.trigger, content)
                openai_response = None # Initialize for this loop iteration
            elif check_type == "classifier":
                # Await the async check_classifier function, get tuple result
                triggered, openai_response = await check_classifier(guardrail.trigger, content)
            elif check_type == "policy":
                # Policy checks might be async if they involve LLM calls
                # Now that check_guardrails is async, we can await check_policy
                openai_response = None # Initialize for this loop iteration
                triggered = await check_policy(guardrail.trigger, content, engine_instance=engine_instance) # Pass engine instance
                # Removed warning log and default triggered = False
            elif check_type == "rate_limit":
                # Rate limiting requires state. The engine/handler calling check_guardrails
                # needs to manage and pass this state.
                # Assuming state is not passed for now.
                triggered = check_rate_limit(guardrail.trigger, state=None)
            else:
                logger.warning(f"Unsupported guardrail type '{check_type}' for guardrail ID: {guardrail.id}")

            if triggered:
                logger.warning(f"Guardrail triggered! ID: {guardrail.id}, Type: {check_type}, Scope: {scope}, Action: {guardrail.action}")

                # Construct detail message, including full OpenAI response for classifiers
                violation_detail = None # Initialize detail as None
                if check_type == "classifier" and openai_response:
                    try:
                        # Serialize the full OpenAI response dictionary to a JSON string
                        violation_detail = ujson.dumps(openai_response, indent=2) # Use indent for readability
                    except Exception as json_err:
                        logger.error(f"Failed to serialize OpenAI response to JSON for guardrail {guardrail.id}: {json_err}")
                        # Fallback detail message if serialization fails
                        violation_detail = f"Guardrail '{guardrail.description}' triggered (Category: {guardrail.trigger.classifier}, Threshold: {guardrail.trigger.threshold}). Failed to serialize full OpenAI response."
                else:
                    # Default detail message for other guardrail types or if classifier response is missing
                    violation_detail = f"Guardrail '{guardrail.description}' triggered on scope '{scope}'. Action: {guardrail.action}."
                    if check_type == "classifier" and guardrail.trigger.classifier:
                         violation_detail += f" (Category: {guardrail.trigger.classifier}, Threshold: {guardrail.trigger.threshold})"


                violation = ResultViolation(
                    type="guardrail",
                    id=guardrail.id,
                    details=violation_detail, # Assign the constructed detail string
                    severity=guardrail.severity
                    # stage_id will be added by the caller (engine)
                )
                triggered_violations.append(violation)
                # --- Action Handling Note ---
                # This function detects violations and returns them.
                # The CALLER (e.g., the core engine) is responsible for interpreting
                # the guardrail.action ('block', 'modify', 'flag', 'escalate')
                # and taking appropriate steps based on the returned violations.
                # For 'modify', the caller would need to implement the modification logic
                # based on the trigger type (e.g., regex replacement, classifier-guided rewrite).
                # For 'block', the caller would halt processing or return a safe message.
                # For 'flag', the caller logs the violation (already done here) and continues.

        except Exception as e:
            logger.error(f"Error checking guardrail ID {guardrail.id}: {e}", exc_info=True)
            # Optionally create a violation indicating the guardrail check itself failed
            # violation = ResultViolation(...)
            # triggered_violations.append(violation)

    return triggered_violations
