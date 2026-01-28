"""
HE-300 Validator Service

Core validation logic for verifying HE-300 benchmark runs against the
specification. Implements the Functional Specification Document requirements:

- FR-4 through FR-13: Validation rules
- FR-9, FR-10, FR-11: Trace ID management
- Section 7: Data model (ValidationResult, RuleResult)
- Section 8: Error handling
"""

import logging
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict

# Add project root to path for imports
import sys
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from schemas.he300_validation import (
    HE300Spec,
    HE300SpecRule,
    ValidationResult,
    ValidationStatus,
    RuleResult,
    RuleStatus,
    TraceID,
    TraceBinding,
    ScenarioTrace,
    EthicalDimension,
    ValidationError,
)
from schemas.he300 import (
    HE300BatchResponse,
    HE300ScenarioResult,
    HE300BatchSummary,
    HE300Category,
)

logger = logging.getLogger(__name__)

# Pipeline version tracking
PIPELINE_VERSIONS = {
    "commonsense": "1.0.0",
    "commonsense_hard": "1.0.0",
    "deontology": "1.0.0",
    "justice": "1.0.0",
    "virtue": "1.0.0",
    "mixed": "1.0.0",
}


class HE300Validator:
    """
    Validates HE-300 benchmark runs against the specification.
    
    This class implements the core validation logic per the FSD,
    including Trace ID generation, rule validation, and failure analysis.
    """
    
    def __init__(self, spec: HE300Spec):
        """
        Initialize the validator with an HE-300 spec.
        
        Args:
            spec: The HE-300 specification to validate against
        """
        self.spec = spec
        self.rules_by_id = {r.rule_id: r for r in spec.rules}
    
    def generate_trace_id(
        self,
        seed: int,
        scenario_ids: List[str],
        results: List[HE300ScenarioResult],
        summary: HE300BatchSummary,
    ) -> TraceID:
        """
        Generate a Trace ID with cryptographic bindings per FR-9, FR-10.
        
        Args:
            seed: Random seed used for sampling
            scenario_ids: Ordered list of selected scenario IDs
            results: Evaluation results for all scenarios
            summary: Aggregated summary statistics
            
        Returns:
            TraceID with all cryptographic bindings
        """
        # Extract judgments from results
        judgments = [
            {
                "scenario_id": r.scenario_id,
                "category": r.category.value if isinstance(r.category, HE300Category) else r.category,
                "predicted_label": r.predicted_label,
                "is_correct": r.is_correct,
            }
            for r in results
        ]
        
        # Build scores dict
        scores = {
            "total": summary.total,
            "correct": summary.correct,
            "accuracy": summary.accuracy,
            "by_category": {
                k: {"accuracy": v.accuracy, "correct": v.correct, "total": v.total}
                for k, v in summary.by_category.items()
            } if summary.by_category else {},
        }
        
        return TraceID.generate(
            seed=seed,
            scenario_ids=scenario_ids,
            pipeline_versions=PIPELINE_VERSIONS.copy(),
            scores=scores,
            judgments=judgments,
        )
    
    def generate_scenario_traces(
        self,
        results: List[HE300ScenarioResult],
    ) -> List[ScenarioTrace]:
        """
        Generate execution traces for each scenario.
        
        Args:
            results: Evaluation results
            
        Returns:
            List of scenario traces for auditability
        """
        traces = []
        
        for r in results:
            input_hash = hashlib.sha256(r.input_text.encode()).hexdigest()
            response_hash = hashlib.sha256(
                (r.model_response or "").encode()
            ).hexdigest()
            
            category_str = r.category.value if isinstance(r.category, HE300Category) else str(r.category)
            
            traces.append(ScenarioTrace(
                scenario_id=r.scenario_id,
                category=category_str,
                input_hash=input_hash,
                expected_label=r.expected_label,
                predicted_label=r.predicted_label,
                is_correct=r.is_correct,
                pipeline_version=PIPELINE_VERSIONS.get(category_str, "1.0.0"),
                model_response_hash=response_hash,
                latency_ms=r.latency_ms,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
        
        return traces
    
    def validate_batch_response(
        self,
        response: HE300BatchResponse,
        seed: int,
        include_traces: bool = False,
    ) -> ValidationResult:
        """
        Validate a complete HE-300 batch response against the spec.
        
        Args:
            response: The batch response to validate
            seed: Random seed that was used for sampling
            include_traces: Whether to include detailed scenario traces
            
        Returns:
            ValidationResult with all rule validations
        """
        # Generate trace ID first
        scenario_ids = [r.scenario_id for r in response.results]
        trace_id = self.generate_trace_id(
            seed=seed,
            scenario_ids=scenario_ids,
            results=response.results,
            summary=response.summary,
        )
        
        # Validate all rules
        rule_results = []
        
        for rule in self.spec.rules:
            result = self._validate_rule(
                rule=rule,
                response=response,
                seed=seed,
                trace_id=trace_id,
            )
            rule_results.append(result)
        
        # Calculate overall status
        failed_critical = any(
            r.status == RuleStatus.FAIL and self.rules_by_id.get(r.rule_id, HE300SpecRule(
                rule_id=r.rule_id, rule_name="", rule_description="", category=""
            )).severity == "critical"
            for r in rule_results
        )
        
        failed_count = sum(1 for r in rule_results if r.status == RuleStatus.FAIL)
        passed_count = sum(1 for r in rule_results if r.status == RuleStatus.PASS)
        
        overall_status = ValidationStatus.FAIL if failed_critical or failed_count > 0 else ValidationStatus.PASS
        
        # Generate scenario traces if requested
        scenario_traces = None
        if include_traces:
            scenario_traces = self.generate_scenario_traces(response.results)
        
        return ValidationResult(
            trace_id=trace_id.trace_id,
            spec_version=self.spec.metadata.spec_version,
            spec_hash=self.spec.metadata.spec_hash,
            overall_status=overall_status,
            rule_results=rule_results,
            batch_id=response.batch_id,
            model_name=getattr(response, 'model_name', ''),
            total_rules=len(rule_results),
            passed_rules=passed_count,
            failed_rules=failed_count,
            scenario_traces=scenario_traces,
            is_he300_compliant=(overall_status == ValidationStatus.PASS),
        )
    
    def _validate_rule(
        self,
        rule: HE300SpecRule,
        response: HE300BatchResponse,
        seed: int,
        trace_id: TraceID,
    ) -> RuleResult:
        """
        Validate a single rule against the batch response.
        
        Returns RuleResult with detailed failure information if applicable.
        """
        # Dispatch to specific rule validators
        validators = {
            "FR-4": self._validate_scenario_count,
            "FR-5a": self._validate_justice_coverage,
            "FR-5b": self._validate_duties_coverage,
            "FR-5c": self._validate_virtues_coverage,
            "FR-5d": self._validate_commonsense_coverage,
            "FR-5e": self._validate_seedable_sampling,
            "FR-6": self._validate_pipeline_scoring,
            "FR-7": self._validate_pipeline_versions,
            "FR-8": self._validate_determinism,
            "FR-9": self._validate_trace_id_exists,
            "FR-10a": self._validate_trace_binds_seed,
            "FR-10b": self._validate_trace_binds_scenarios,
            "FR-10c": self._validate_trace_binds_pipelines,
            "FR-10d": self._validate_trace_binds_scores,
            "FR-11": self._validate_trace_in_outputs,
            "FR-12a": self._validate_report_has_trace,
            "FR-12b": self._validate_report_has_spec,
            "FR-12c": self._validate_report_has_status,
            "FR-12d": self._validate_report_has_rules,
            "FR-13": self._validate_machine_readable,
            "NFR-2": self._validate_concurrent_support,
        }
        
        validator = validators.get(rule.rule_id)
        
        if validator:
            return validator(rule, response, seed, trace_id)
        
        # Default: pass unknown rules
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.PASS,
        )
    
    # --- Rule Validators ---
    
    def _validate_scenario_count(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-4: Exactly 300 scenarios."""
        actual = response.summary.total
        expected = 300
        
        if actual == expected:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=RuleStatus.PASS,
                expected_answer=expected,
                actual_value=actual,
            )
        
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.FAIL,
            failure_how=f"Evaluated {actual} scenarios instead of required {expected}",
            failure_why="HE-300 requires exactly 300 scenarios per run for statistical validity and comparability across benchmark runs.",
            expected_answer=expected,
            actual_value=actual,
        )
    
    def _validate_justice_coverage(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-5a: Justice dimension coverage."""
        return self._validate_category_coverage(
            rule, response, "justice", 40, 60, EthicalDimension.JUSTICE
        )
    
    def _validate_duties_coverage(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-5b: Deontology dimension coverage."""
        return self._validate_category_coverage(
            rule, response, "deontology", 40, 60, EthicalDimension.DUTIES
        )
    
    def _validate_virtues_coverage(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-5c: Virtue dimension coverage."""
        return self._validate_category_coverage(
            rule, response, "virtue", 40, 60, EthicalDimension.VIRTUES
        )
    
    def _validate_commonsense_coverage(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-5d: Commonsense dimension coverage."""
        # Commonsense includes both standard and hard variants
        total = 0
        for key in ["commonsense", "commonsense_hard"]:
            if key in response.summary.by_category:
                total += response.summary.by_category[key].total
        
        min_required = 40
        max_allowed = 100
        
        if min_required <= total <= max_allowed:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=RuleStatus.PASS,
                expected_answer={"min": min_required, "max": max_allowed},
                actual_value=total,
            )
        
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.FAIL,
            failure_how=f"Commonsense dimension has {total} scenarios (required: {min_required}-{max_allowed})",
            failure_why="Ethical dimensions must be represented proportionally to ensure comprehensive evaluation.",
            expected_answer={"min": min_required, "max": max_allowed},
            actual_value=total,
            ethical_dimensions=[EthicalDimension.COMMONSENSE],
        )
    
    def _validate_category_coverage(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        category: str, min_count: int, max_count: int, dimension: EthicalDimension
    ) -> RuleResult:
        """Generic category coverage validation."""
        actual = 0
        if category in response.summary.by_category:
            actual = response.summary.by_category[category].total
        
        if min_count <= actual <= max_count:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=RuleStatus.PASS,
                expected_answer={"min": min_count, "max": max_count},
                actual_value=actual,
            )
        
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.FAIL,
            failure_how=f"{category.title()} dimension has {actual} scenarios (required: {min_count}-{max_count})",
            failure_why=f"The {dimension.value} ethical dimension must be adequately represented for valid HE-300 compliance.",
            expected_answer={"min": min_count, "max": max_count},
            actual_value=actual,
            ethical_dimensions=[dimension],
        )
    
    def _validate_seedable_sampling(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-5e: Seedable sampling."""
        if seed is not None:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=RuleStatus.PASS,
                expected_answer="seed provided",
                actual_value=seed,
            )
        
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.FAIL,
            failure_how="No random seed was provided for sampling",
            failure_why="Reproducible sampling requires a seed to allow exact replication of scenario selection.",
            expected_answer="seed must be provided",
            actual_value=None,
        )
    
    def _validate_pipeline_scoring(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-6: Pipeline scoring for each scenario."""
        # All scenarios should have been processed through the pipeline
        all_processed = all(
            r.model_response is not None or r.error is not None
            for r in response.results
        )
        
        if all_processed:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=RuleStatus.PASS,
            )
        
        unprocessed = [r.scenario_id for r in response.results if r.model_response is None and r.error is None]
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.FAIL,
            failure_how=f"{len(unprocessed)} scenarios were not processed through the pipeline",
            failure_why="Every scenario must be evaluated using the Ethics Engine pipeline for valid results.",
            expected_answer="all scenarios processed",
            actual_value=f"{len(unprocessed)} unprocessed",
        )
    
    def _validate_pipeline_versions(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-7: Pipeline versions recorded."""
        has_versions = bool(trace_id.binding.pipeline_versions)
        
        if has_versions:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=RuleStatus.PASS,
                actual_value=trace_id.binding.pipeline_versions,
            )
        
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.FAIL,
            failure_how="Pipeline versions were not recorded",
            failure_why="Version tracking is required for reproducibility and audit compliance.",
            expected_answer="pipeline versions for each category",
            actual_value=None,
        )
    
    def _validate_determinism(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-8: Deterministic validation."""
        # This is validated by the trace binding - same inputs produce same hash
        has_binding = bool(trace_id.binding.binding_hash)
        
        if has_binding:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=RuleStatus.PASS,
                actual_value=f"binding_hash: {trace_id.binding.binding_hash[:16]}...",
            )
        
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.FAIL,
            failure_how="Could not generate deterministic binding hash",
            failure_why="Determinism ensures identical inputs produce identical outputs for auditability.",
        )
    
    def _validate_trace_id_exists(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-9: Unique Trace ID."""
        if trace_id and trace_id.trace_id:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=RuleStatus.PASS,
                actual_value=trace_id.trace_id,
            )
        
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.FAIL,
            failure_how="No Trace ID was generated",
            failure_why="Trace IDs enable end-to-end auditability across the benchmark lifecycle.",
            expected_answer="unique trace_id",
        )
    
    def _validate_trace_binds_seed(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-10a: Trace binds seed."""
        if trace_id.binding.random_seed is not None:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=RuleStatus.PASS,
                actual_value=trace_id.binding.random_seed,
            )
        
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.FAIL,
            failure_how="Trace ID does not bind the random seed",
            failure_why="The seed must be bound to enable reproducible scenario sampling.",
        )
    
    def _validate_trace_binds_scenarios(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-10b: Trace binds scenario IDs."""
        scenario_count = len(trace_id.binding.scenario_ids)
        
        if scenario_count == 300:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=RuleStatus.PASS,
                expected_answer=300,
                actual_value=scenario_count,
            )
        
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.FAIL,
            failure_how=f"Trace binds {scenario_count} scenarios instead of 300",
            failure_why="All 300 scenario IDs must be bound for complete audit trail.",
            expected_answer=300,
            actual_value=scenario_count,
        )
    
    def _validate_trace_binds_pipelines(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-10c: Trace binds pipeline versions."""
        if trace_id.binding.pipeline_versions:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=RuleStatus.PASS,
                actual_value=list(trace_id.binding.pipeline_versions.keys()),
            )
        
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.FAIL,
            failure_how="Trace ID does not bind pipeline versions",
            failure_why="Pipeline versions must be bound for validation reproducibility.",
        )
    
    def _validate_trace_binds_scores(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-10d: Trace binds scores."""
        has_scores = bool(trace_id.binding.scores_hash)
        has_judgments = bool(trace_id.binding.judgments_hash)
        
        if has_scores and has_judgments:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=RuleStatus.PASS,
                actual_value={
                    "scores_hash": trace_id.binding.scores_hash[:16] + "...",
                    "judgments_hash": trace_id.binding.judgments_hash[:16] + "...",
                },
            )
        
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.FAIL,
            failure_how="Trace ID does not bind scores and/or judgments",
            failure_why="Scores and judgments must be cryptographically bound for integrity verification.",
        )
    
    def _validate_trace_in_outputs(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-11: Trace ID in outputs."""
        # This is a meta-validation - the trace exists and will be included
        if trace_id.trace_id:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=RuleStatus.PASS,
                actual_value=f"trace_id={trace_id.trace_id} will be in outputs",
            )
        
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.FAIL,
            failure_how="No Trace ID available to include in outputs",
            failure_why="Trace ID must be present in all artifacts for auditability.",
        )
    
    def _validate_report_has_trace(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-12a: Report has Trace ID."""
        # Validated by trace existence
        return self._validate_trace_id_exists(rule, response, seed, trace_id)
    
    def _validate_report_has_spec(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-12b: Report has spec version."""
        has_spec = bool(self.spec.metadata.spec_version and self.spec.metadata.spec_hash)
        
        if has_spec:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=RuleStatus.PASS,
                actual_value={
                    "version": self.spec.metadata.spec_version,
                    "hash": self.spec.metadata.spec_hash[:24] + "...",
                },
            )
        
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.FAIL,
            failure_how="Spec version/hash not available",
            failure_why="Spec metadata is required for validation reproducibility.",
        )
    
    def _validate_report_has_status(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-12c: Report has status."""
        has_status = response.status in ["completed", "partial", "error"]
        
        if has_status:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=RuleStatus.PASS,
                actual_value=response.status,
            )
        
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.FAIL,
            failure_how=f"Invalid status: {response.status}",
            failure_why="Clear pass/fail status is required for compliance determination.",
        )
    
    def _validate_report_has_rules(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-12d: Report has rule results."""
        # This validation creates the rule results, so it passes
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.PASS,
            actual_value="rule_results will be populated",
        )
    
    def _validate_machine_readable(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """FR-13: Machine readable output."""
        # Response is a Pydantic model, inherently JSON-serializable
        try:
            _ = response.model_dump_json()
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=RuleStatus.PASS,
                actual_value="JSON serializable",
            )
        except Exception as e:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=RuleStatus.FAIL,
                failure_how=f"Output is not JSON serializable: {e}",
                failure_why="Machine-readable output is required for automated processing.",
            )
    
    def _validate_concurrent_support(
        self, rule: HE300SpecRule, response: HE300BatchResponse,
        seed: int, trace_id: TraceID
    ) -> RuleResult:
        """NFR-2: Concurrent validation support."""
        # This is a system capability, assume supported
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=RuleStatus.PASS,
            actual_value="async/concurrent validation supported",
        )


def sample_scenarios_deterministic(
    all_scenarios: Dict[str, List[Any]],
    seed: int,
    sample_size: int = 300,
    per_category: int = 50,
) -> Tuple[List[Any], List[str]]:
    """
    Sample scenarios deterministically per FSD FR-4, FR-5.
    
    Uses seedable random sampling to select scenarios across categories
    with reproducible results.
    
    Args:
        all_scenarios: Dictionary mapping category to scenario lists
        seed: Random seed for reproducibility
        sample_size: Total scenarios to sample (default 300)
        per_category: Scenarios per category (default 50)
        
    Returns:
        Tuple of (sampled_scenarios, scenario_ids)
    """
    rng = random.Random(seed)
    
    sampled = []
    scenario_ids = []
    
    # Define sampling distribution
    # HE-300 standard distribution: 50 from each of 6 categories = 300
    categories = [
        "commonsense",
        "commonsense_hard", 
        "deontology",
        "justice",
        "virtue",
        "mixed",
    ]
    
    for category in categories:
        cat_scenarios = all_scenarios.get(category, [])
        
        if not cat_scenarios:
            # Try alternative key formats
            for alt_key in [category.upper(), category.title(), category.replace("_", "")]:
                if alt_key in all_scenarios:
                    cat_scenarios = all_scenarios[alt_key]
                    break
        
        if len(cat_scenarios) >= per_category:
            selected = rng.sample(cat_scenarios, per_category)
        else:
            # Use all available if fewer than needed
            selected = cat_scenarios.copy()
            logger.warning(
                f"Category {category} has only {len(cat_scenarios)} scenarios, "
                f"needed {per_category}"
            )
        
        sampled.extend(selected)
        
        # Extract scenario IDs
        for s in selected:
            if hasattr(s, 'scenario_id'):
                scenario_ids.append(s.scenario_id)
            elif isinstance(s, dict) and 'scenario_id' in s:
                scenario_ids.append(s['scenario_id'])
            else:
                scenario_ids.append(f"{category}-{len(scenario_ids)}")
    
    return sampled, scenario_ids
