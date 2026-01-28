from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from .base import BaseSchema

# Define allowed stage types based on FSD
StageType = Literal["LLM", "interaction", "evaluation", "action", "tool"]
StageRole = Literal["assistant", "user", "system"] # Primarily for LLM stages

class StageOutputSpec(BaseModel):
    """Defines the expected output(s) of a stage and how to label them."""
    # Example: {"plan": "text"} or {"metrics": "object"}
    # Using Dict[str, str] where key is label, value is expected type hint (e.g., 'text', 'object', 'list')
    spec: Dict[str, str] = Field(..., description="Specification of stage outputs (label: type).")

    class Config:
        extra = 'forbid'

# Removed StageInputSpec as inputs will be a direct Dict in Stage model

class StageAG2Config(BaseModel):
    """Configuration specific to AG2 ReasoningAgent for a stage."""
    max_depth: Optional[int] = Field(None, description="Max depth for reasoning (e.g., tree-of-thought, reflection).")
    method: Optional[str] = Field(None, description="Reasoning method (e.g., 'tree_of_thought', 'reflection', 'beam_search').")
    # Other AG2 parameters like beam_size, temperature are allowed via extra='allow'

    class Config:
        extra = 'allow' # Allow other AG2 params not explicitly defined


class Stage(BaseSchema):
    """
    Represents a discrete step or phase within a pipeline (e.g., LLM call, evaluation).
    """
    id: str = Field(..., description="Unique identifier of the stage within the pipeline.")
    type: StageType = Field(..., description="The type of stage (LLM, evaluation, etc.).")
    description: Optional[str] = Field(None, description="Optional description of the stage's purpose.")

    # Fields primarily for LLM stages
    role: Optional[StageRole] = Field(None, description="Role of the agent at this stage (assistant, user, system).")
    prompt: Optional[str] = Field(None, description="Prompt template or instruction for the LLM stage. Can use placeholders like {prev_stage.output_label}.")

    # Fields for specific stage types
    function_id: Optional[str] = Field(None, description="Identifier for the function to execute (for 'evaluation' type).")
    tool_id: Optional[str] = Field(None, description="Identifier for the tool to execute (for 'tool' type).")
    participants: Optional[List[str]] = Field(None, description="List of participant names/roles (for 'interaction' type using GroupChat).")
    participant_configs: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Optional configurations per participant (e.g., identity_id, ethical_guidance_id). Keyed by participant name.")
    context: Optional[Dict[str, str]] = Field(None, description="Context data/references for the stage (e.g., for 'interaction' type).")

    # Input specification (simplified to direct dictionary)
    inputs: Optional[Dict[str, str]] = Field(None, description="References to data needed from prior stages (name: reference_string).")

    # Output specification
    outputs: StageOutputSpec = Field(..., description="Specification of what this stage produces and how to label it.")

    # Stage-specific overrides or additions
    guardrails: Optional[List[str]] = Field(None, description="List of Guardrail IDs specifically applied at this stage (overrides/adds to pipeline level).")
    ag2_config: Optional[StageAG2Config] = Field(None, description="Specific AG2 ReasoningAgent configuration for this stage.")
    timeout: Optional[int] = Field(None, description="Timeout in seconds for this stage.")
    retries: Optional[int] = Field(0, description="Number of retries if the stage fails.")

    class Config:
        extra = 'forbid'
