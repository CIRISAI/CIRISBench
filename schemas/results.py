from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from .base import BaseSchema, Metadata
from .interaction import Interaction # To include the list of interactions

# Define possible outcome statuses
ResultOutcome = Literal["success", "failure", "guardrail_violation", "error", "timeout", "pending"]

class ResultViolation(BaseModel):
    """Details about a specific violation (guardrail or ethical principle)."""
    type: Literal["guardrail", "ethical_principle"] = Field(..., description="Type of violation.")
    id: str = Field(..., description="ID of the guardrail or principle violated.")
    stage_id: Optional[str] = Field(None, description="Stage where the violation occurred.")
    details: Optional[str] = Field(None, description="Additional details about the violation.")
    severity: Optional[str] = Field(None, description="Severity of the violation (e.g., from Guardrail).")

    class Config:
        extra = 'forbid'

class ResultMetrics(BaseModel):
    """Structure for quantitative evaluation metrics."""
    correctness: Optional[float] = Field(None, description="Score for correctness (e.g., 0.0 to 1.0).")
    principle_alignment: Optional[Dict[str, float]] = Field(default_factory=dict, description="Scores for alignment with specific ethical principles (e.g., {'fairness': 0.8}).")
    ethical_score: Optional[float] = Field(None, description="Overall composite ethical score, if calculated.")
    tokens_used_total: Optional[int] = Field(None, description="Total tokens used for the entire pipeline run.")
    latency_seconds: Optional[float] = Field(None, description="Total time taken for the pipeline run in seconds.")
    custom_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Any other custom metrics recorded.")

    class Config:
        extra = 'allow' # Allow flexibility for custom metrics


class Results(BaseSchema):
    """
    Records the outcomes of a pipeline execution, including configuration,
    interactions, violations, and evaluation metrics.
    """
    run_id: str = Field(..., description="Unique identifier for this specific pipeline run.")
    pipeline_id: str = Field(..., description="Identifier of the pipeline definition that was run.")
    timestamp_start: str = Field(..., description="Timestamp when the run started (ISO 8601).")
    timestamp_end: Optional[str] = Field(None, description="Timestamp when the run finished (ISO 8601).")

    # Configuration snapshot
    identity_id: str = Field(..., description="ID of the Identity profile used.")
    ethical_guidance_id: str = Field(..., description="ID of the Ethical Guidance framework used.")
    guardrail_ids_active: List[str] = Field(..., description="List of Guardrail IDs active during the run.")
    # Optionally include full config objects if needed for self-contained results
    # identity_config: Optional[Identity] = None
    # ethical_guidance_config: Optional[EthicalGuidance] = None
    # pipeline_config: Optional[Pipeline] = None # The full pipeline definition run

    # Execution details
    interactions: List[Interaction] = Field(..., description="Chronological list of interactions that occurred.")
    outcome: ResultOutcome = Field(..., description="Summary status of the final outcome.")
    outcome_details: Optional[str] = Field(None, description="Further details or explanation of the outcome.")

    # Violations and Metrics
    violations: List[ResultViolation] = Field(default_factory=list, description="List of any guardrail or ethical principle violations detected.")
    metrics: Optional[ResultMetrics] = Field(default_factory=ResultMetrics, description="Quantitative evaluation metrics.")

    # Optional fields
    comparison_baseline_run_id: Optional[str] = Field(None, description="ID of a baseline run for comparison purposes.")
    notes: Optional[str] = Field(None, description="Additional commentary or qualitative analysis from the run.")

    class Config:
        extra = 'forbid'
