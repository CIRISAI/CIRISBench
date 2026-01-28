from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from .base import BaseSchema

GuardrailType = Literal["content_filter", "classifier", "policy", "rate_limit"]
GuardrailAction = Literal["block", "modify", "flag", "escalate"]
GuardrailScope = Literal["input", "output", "both"]
GuardrailSeverity = Literal["low", "medium", "high", "critical"]

class GuardrailTrigger(BaseModel):
    """Defines the condition that triggers a guardrail."""
    regex: Optional[str] = Field(None, description="Regex pattern to trigger the guardrail.")
    keywords: Optional[List[str]] = Field(None, description="List of keywords to trigger the guardrail.")
    classifier: Optional[str] = Field(None, description="Identifier for an ML classifier model.")
    threshold: Optional[float] = Field(None, description="Threshold for the classifier score (if applicable).")
    # --- Fields for specific guardrail types ---
    policy_description: Optional[str] = Field(None, description="Description of the policy to check against (for 'policy' type).")
    max_count: Optional[int] = Field(None, description="Maximum allowed count within the time window (for 'rate_limit' type).")
    time_window_seconds: Optional[int] = Field(None, description="Time window in seconds (for 'rate_limit' type).")

    # Note: Validation that the correct fields are present for a given Guardrail.type
    # would typically happen during Guardrail model validation or in the check functions.

    class Config:
        extra = 'forbid' # Keep forbidding unexpected fields


class Guardrail(BaseSchema):
    """
    Defines safety constraints and content rules that the LLM must obey.
    Acts as filters or triggers for disallowed or dangerous content.
    """
    id: str = Field(..., description="Unique name or identifier of the guardrail rule.")
    description: str = Field(..., description="Description of what the guardrail checks or enforces.")
    type: GuardrailType = Field(..., description="The mechanism or category of the guardrail.")
    trigger: GuardrailTrigger = Field(..., description="The condition(s) that trigger the guardrail.")
    action: GuardrailAction = Field(..., description="Action to take if the guardrail is triggered.")
    scope: Optional[GuardrailScope] = Field("output", description="Scope of enforcement (input, output, or both).")
    severity: Optional[GuardrailSeverity] = Field("high", description="Severity level of the violation.")
    message: Optional[str] = Field(None, description="Default refusal or correction message if triggered.")

    class Config:
        extra = 'forbid'
