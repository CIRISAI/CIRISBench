"""
HE-300 Validation Schemas

Pydantic models for HE-300 benchmark validation, trace management,
and compliance verification per the HE-300 FSD specification.
"""

from pydantic import BaseModel, Field, computed_field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from datetime import datetime, timezone
import hashlib
import uuid
import json


class ValidationStatus(str, Enum):
    """Overall validation status."""
    PASS = "PASS"
    FAIL = "FAIL"


class RuleStatus(str, Enum):
    """Individual rule validation status."""
    PASS = "PASS"
    FAIL = "FAIL"


class EthicalDimension(str, Enum):
    """Ethical dimensions from Hendrycks ETHICS dataset."""
    JUSTICE = "justice"
    DUTIES = "duties"  # deontology
    VIRTUES = "virtues"  # virtue ethics
    COMMONSENSE = "commonsense"  # commonsense morality


class HE300SpecRule(BaseModel):
    """
    A single validation rule from the HE-300 specification.
    
    Rules define what conditions must be met for a benchmark run
    to be considered valid and compliant.
    """
    rule_id: str = Field(..., description="Unique rule identifier (e.g., 'FR-4', 'NFR-1')")
    rule_name: str = Field(..., description="Human-readable rule name")
    rule_description: str = Field(..., description="Full rule description from spec")
    category: str = Field(..., description="Rule category (sampling, scoring, tracing, etc.)")
    severity: Literal["critical", "major", "minor"] = Field(
        default="major",
        description="Severity if rule is violated"
    )
    expected_value: Optional[Any] = Field(
        None,
        description="Expected value/output for this rule"
    )
    validation_logic: Optional[str] = Field(
        None,
        description="Pseudocode or description of validation logic"
    )


class RuleResult(BaseModel):
    """
    Result of validating a single rule against a benchmark run.
    
    Per FSD FR-9: For each failed rule, includes how/why it failed
    and the expected answer.
    """
    rule_id: str = Field(..., description="ID of the rule being validated")
    rule_name: str = Field(..., description="Name of the rule")
    status: RuleStatus = Field(..., description="PASS or FAIL")
    failure_how: Optional[str] = Field(
        None,
        description="Mechanism/condition that caused the failure"
    )
    failure_why: Optional[str] = Field(
        None,
        description="Rationale from the spec explaining why this matters"
    )
    expected_answer: Optional[Any] = Field(
        None,
        description="What the spec expected"
    )
    actual_value: Optional[Any] = Field(
        None,
        description="What was actually observed"
    )
    ethical_dimensions: Optional[List[EthicalDimension]] = Field(
        None,
        description="Ethical dimensions affected by this failure"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "rule_id": "FR-4",
                "rule_name": "Scenario Count",
                "status": "FAIL",
                "failure_how": "Evaluated 295 scenarios instead of required 300",
                "failure_why": "HE-300 requires exactly 300 scenarios per run for statistical validity",
                "expected_answer": 300,
                "actual_value": 295,
                "ethical_dimensions": None
            }
        }


class ScenarioTrace(BaseModel):
    """
    Execution trace for a single evaluated scenario.
    
    Provides verifiable audit trail for each scenario.
    """
    scenario_id: str
    category: str
    input_hash: str = Field(..., description="SHA-256 of input text")
    expected_label: Optional[int]
    predicted_label: Optional[int]
    is_correct: bool
    pipeline_version: str = Field(
        default="1.0.0",
        description="Version of the ethics pipeline used"
    )
    model_response_hash: str = Field(..., description="SHA-256 of model response")
    latency_ms: float
    timestamp: str = Field(..., description="ISO 8601 timestamp of evaluation")


class TraceBinding(BaseModel):
    """
    Cryptographic binding for the Trace ID per FSD FR-10.
    
    The Trace ID cryptographically binds:
    - Random seed used for scenario sampling
    - List of selected scenario identifiers
    - Ethics Engine pipeline versions
    - Resulting scores and judgments
    """
    random_seed: int = Field(..., description="Seed used for scenario sampling")
    scenario_ids: List[str] = Field(..., description="Ordered list of selected scenarios")
    pipeline_versions: Dict[str, str] = Field(
        ...,
        description="Pipeline versions by category"
    )
    scores_hash: str = Field(..., description="SHA-256 of aggregated scores JSON")
    judgments_hash: str = Field(..., description="SHA-256 of all judgments JSON")
    
    @computed_field
    @property
    def binding_hash(self) -> str:
        """Compute hash of all bound data."""
        data = {
            "seed": self.random_seed,
            "scenarios": self.scenario_ids,
            "pipelines": self.pipeline_versions,
            "scores": self.scores_hash,
            "judgments": self.judgments_hash,
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


class TraceID(BaseModel):
    """
    Globally unique Trace ID per FSD FR-9, FR-10, FR-11.
    
    Enables end-to-end auditability across:
    - Scenario selection
    - Model execution
    - Scoring
    - Signing
    """
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID or Ed25519-derived identifier"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When the trace was created"
    )
    binding: TraceBinding = Field(..., description="Cryptographic binding data")
    
    @classmethod
    def generate(
        cls,
        seed: int,
        scenario_ids: List[str],
        pipeline_versions: Dict[str, str],
        scores: Dict[str, Any],
        judgments: List[Dict[str, Any]],
    ) -> "TraceID":
        """
        Generate a new Trace ID with cryptographic binding.
        
        Args:
            seed: Random seed used for scenario sampling
            scenario_ids: List of selected scenario IDs
            pipeline_versions: Dictionary of pipeline versions by category
            scores: Aggregated scores dictionary
            judgments: List of individual scenario judgments
            
        Returns:
            TraceID with all bindings computed
        """
        scores_hash = hashlib.sha256(
            json.dumps(scores, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        judgments_hash = hashlib.sha256(
            json.dumps(judgments, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        binding = TraceBinding(
            random_seed=seed,
            scenario_ids=scenario_ids,
            pipeline_versions=pipeline_versions,
            scores_hash=scores_hash,
            judgments_hash=judgments_hash,
        )
        
        # Generate deterministic trace ID from binding
        trace_id = hashlib.sha256(
            binding.binding_hash.encode()
        ).hexdigest()[:32]
        
        return cls(
            trace_id=f"he300-{trace_id}",
            binding=binding,
        )


class HE300SpecMetadata(BaseModel):
    """
    Metadata about the HE-300 specification per FSD FR-2.
    
    Records version, hash, and retrieval timestamp for reproducibility.
    """
    spec_version: str = Field(..., description="Semantic version of the spec")
    spec_hash: str = Field(..., description="SHA-256 hash of spec content")
    retrieval_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When the spec was retrieved"
    )
    source_url: Optional[str] = Field(
        None,
        description="URL from which spec was retrieved"
    )
    total_rules: int = Field(..., description="Number of rules in the spec")
    curated_scenario_count: int = Field(
        default=15000,
        description="Approx. size of the EEE curated subset"
    )
    sample_size: int = Field(
        default=300,
        description="Required sample size per run"
    )


class HE300Spec(BaseModel):
    """
    The complete HE-300 specification.
    
    Contains all validation rules and metadata for compliance checking.
    """
    metadata: HE300SpecMetadata
    rules: List[HE300SpecRule]
    ethical_dimensions: List[EthicalDimension] = Field(
        default=[
            EthicalDimension.JUSTICE,
            EthicalDimension.DUTIES,
            EthicalDimension.VIRTUES,
            EthicalDimension.COMMONSENSE,
        ],
        description="Required ethical dimensions to cover"
    )
    sampling_requirements: Dict[str, Any] = Field(
        default_factory=lambda: {
            "total_scenarios": 300,
            "scenarios_per_category": 50,
            "categories": ["commonsense", "commonsense_hard", "deontology", "justice", "virtue", "mixed"],
            "must_be_seedable": True,
            "must_be_reproducible": True,
        }
    )


class ValidationResult(BaseModel):
    """
    Complete validation result for an HE-300 benchmark run.
    
    Per FSD section 7.1, includes:
    - trace_id (string, UUID/Ed25519-derived)
    - spec_version (string)
    - spec_hash (string)
    - overall_status (enum: PASS | FAIL)
    - rule_results (array)
    """
    trace_id: str = Field(..., description="Unique trace identifier")
    spec_version: str = Field(..., description="Version of spec used for validation")
    spec_hash: str = Field(..., description="Hash of spec content")
    overall_status: ValidationStatus = Field(..., description="PASS or FAIL")
    rule_results: List[RuleResult] = Field(
        ...,
        description="Results for each validated rule"
    )
    validated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When validation was performed"
    )
    batch_id: str = Field(..., description="ID of the benchmark batch validated")
    model_name: str = Field(default="", description="Model that was evaluated")
    total_rules: int = Field(..., description="Total rules checked")
    passed_rules: int = Field(..., description="Rules that passed")
    failed_rules: int = Field(..., description="Rules that failed")
    
    # Execution trace for auditability
    scenario_traces: Optional[List[ScenarioTrace]] = Field(
        None,
        description="Execution trace for each scenario (if requested)"
    )
    
    # Machine-readable output
    is_he300_compliant: bool = Field(
        ...,
        description="Whether the run is fully HE-300 compliant"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "trace_id": "he300-abc123def456",
                "spec_version": "1.0.0",
                "spec_hash": "sha256:abc...",
                "overall_status": "PASS",
                "rule_results": [],
                "batch_id": "batch-2026-01-03-001",
                "total_rules": 15,
                "passed_rules": 15,
                "failed_rules": 0,
                "is_he300_compliant": True
            }
        }


class ValidationRequest(BaseModel):
    """Request to validate a benchmark run against HE-300 spec."""
    batch_id: str = Field(..., description="Batch ID to validate")
    spec_version: Optional[str] = Field(
        None,
        description="Specific spec version to use (latest if not specified)"
    )
    include_scenario_traces: bool = Field(
        default=False,
        description="Include detailed execution trace for each scenario"
    )
    force_refresh_spec: bool = Field(
        default=False,
        description="Force re-fetch of spec from remote source"
    )


class ValidationError(BaseModel):
    """Error during validation (separate from rule failures)."""
    error_code: str
    error_message: str
    trace_id: Optional[str] = Field(
        None,
        description="Trace ID if available"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
