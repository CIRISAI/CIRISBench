from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from .base import BaseSchema
from .stage import Stage # Assuming Stage schema is defined in stage.py

# Import other necessary schemas if needed later for validation/linking,
# but for now, we'll often reference them by ID strings in the pipeline definition.
# from .identity import Identity
# from .ethical_guidance import EthicalGuidance
# from .guardrail import Guardrail

class PipelineEvaluationMetrics(BaseModel):
    """Defines expected outcomes or criteria for evaluating a pipeline run."""
    expected_outcome: Optional[str] = Field(None, description="Description of the expected correct or ideal outcome.")
    principle_alignment: Optional[List[str]] = Field(default_factory=list, description="List of ethical principles expected to be upheld.")
    custom_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Any other custom evaluation criteria or metrics.")

    class Config:
        extra = 'allow' # Allow flexibility in defining custom metrics


class Pipeline(BaseSchema):
    """
    Defines a self-contained sequence of interactions and evaluations
    simulating an ethical scenario or benchmark task.
    """
    id: str = Field(..., description="Unique pipeline identifier or name.")
    description: str = Field(..., description="Brief summary of the pipeline's purpose or scenario.")

    # References to other configurations (using IDs for loose coupling)
    identity_id: str = Field(..., description="ID of the Identity profile to apply.")
    ethical_guidance_id: str = Field(..., description="ID of the Ethical Guidance framework to use.")
    guardrail_ids: Optional[List[str]] = Field(default_factory=list, description="List of Guardrail IDs to enforce/monitor during this pipeline.")

    # The sequence of stages
    stages: List[Stage] = Field(..., description="Ordered list of stage definitions comprising the pipeline.")

    # Evaluation criteria for the overall pipeline
    evaluation_metrics: Optional[PipelineEvaluationMetrics] = Field(None, description="Criteria or metrics to evaluate the pipeline's overall outcome.")

    class Config:
        extra = 'forbid'
