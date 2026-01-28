# Stage handlers module initialization

# This module will contain implementations for different stage types (LLM, evaluation, etc.)
# We can use a registry pattern here to map stage types to handler functions/classes.

from typing import Dict, Callable, Any, List # Added List here
from schemas.stage import Stage
from schemas.interaction import Interaction

from schemas.identity import Identity
from schemas.ethical_guidance import EthicalGuidance

# Define a type hint for a stage handler function
# Takes the stage definition, current pipeline context, engine instance, identity, and guidance
# Returns updated context, list of interactions, and potentially status/violations
StageHandler = Callable[
    [Stage, Dict[str, Any], Any, Identity | None, EthicalGuidance | None],
    tuple[Dict[str, Any], List[Interaction], Any]
]

STAGE_HANDLER_REGISTRY: Dict[str, StageHandler] = {}

def register_stage_handler(stage_type: str):
    """Decorator to register a stage handler function."""
    def decorator(func: StageHandler):
        if stage_type in STAGE_HANDLER_REGISTRY:
            # Handle potential duplicate registration if needed
            pass
        STAGE_HANDLER_REGISTRY[stage_type] = func
        return func
    return decorator

# Import specific handlers to register them
from .llm_handler import handle_llm_stage
from .evaluation_handler import handle_evaluation_stage
from .interaction_handler import handle_interaction_stage # Add import for interaction handler
from .action_handler import handle_action_stage # Add import for action handler
from .tool_handler import handle_tool_stage # Add import for tool handler
