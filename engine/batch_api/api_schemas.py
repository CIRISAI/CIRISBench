# batch_api/api_schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class BatchRunRequest(BaseModel):
    pipeline_ids: List[str] = Field(
        ...,
        description="A list of pipeline IDs to include in the batch run.",
        examples=[["scenario_001", "scenario_007", "scenario_032"]],
    )


class IndividualRunSummary(BaseModel):
    pipeline_id: str = Field(..., description="ID of the pipeline that was run.")
    run_id: str = Field(..., description="Unique ID for this run.")
    status: str = Field(
        ..., description="Status of the run ('success', 'fail', 'error')."
    )
    guardrail_violation: bool = Field(
        ..., description="True if a hard guardrail was violated."
    )
    correctness: Optional[float] = Field(
        default=None, description="Correctness score (0-1)."
    )
    principle_alignment: Optional[Dict[str, float]] = Field(
        default=None, description="Alignment scores per principle."
    )
    latency_ms: Optional[float] = Field(
        default=None, description="Execution time in ms."
    )
    error_message: Optional[str] = Field(
        default=None, description="Error details if failed."
    )


class BatchRunResult(BaseModel):
    batch_run_id: str = Field(..., description="Unique ID for the entire batch.")
    overall_pass: bool = Field(
        ..., description="True if batch meets success thresholds."
    )
    total_scenarios_run: int = Field(..., description="Total number of pipelines run.")
    successful_scenarios: int = Field(
        ..., description="Number of successful pipelines."
    )
    failed_scenarios_execution: int = Field(
        ..., description="Number of execution failures."
    )
    guardrail_violations_count: int = Field(
        ..., description="Total guardrail violations."
    )
    guardrail_violation_rate: Optional[float] = Field(
        default=None, description="Guardrail violation rate."
    )
    mean_correctness: Optional[float] = Field(
        default=None, description="Average correctness across successful runs."
    )
    mean_principle_alignment: Optional[Dict[str, float]] = Field(
        default=None, description="Average principle alignment across successful runs."
    )
    latency_p90_ms: Optional[float] = Field(
        default=None, description="90th percentile latency."
    )
    run_summaries: List[IndividualRunSummary] = Field(
        ..., description="Summaries for each run."
    )
