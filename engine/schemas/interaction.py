from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal
from .base import BaseSchema, Metadata

InteractionRole = Literal["user", "assistant", "system", "evaluator", "tool"]

class Interaction(BaseSchema):
    """
    Represents a single message exchange, action, or internal thought
    during pipeline execution. Used for logging and analysis.
    """
    stage_id: str = Field(..., description="The stage ID in which this interaction occurred.")
    role: InteractionRole = Field(..., description="Role of the entity in this interaction (user, assistant, system, etc.).")
    content: Optional[str] = Field(None, description="The actual message, output content, or internal thought.")
    metadata: Optional[Metadata] = Field(default_factory=Metadata, description="Additional metadata (model, tokens, timestamp, guardrail flags).")
    reasoning_trace: Optional[Dict[str, Any]] = Field(None, description="Internal reasoning trace (e.g., chain-of-thought, tree structure) if applicable.")
    guardrail_triggered: Optional[bool] = Field(False, description="Flag indicating if a guardrail was triggered by this interaction's content.")
    guardrail_details: Optional[Dict[str, Any]] = Field(None, description="Details about the triggered guardrail (ID, action taken).")


    class Config:
        extra = 'forbid'
