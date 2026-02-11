"""
HE-300 Specification API Router

Provides endpoints for retrieving and managing the HE-300 evaluation
specification per the FSD requirements:
- FR-1: Retrieve HE-300 spec from dedicated endpoint
- FR-2: Record spec metadata (version, hash, timestamp)
- FR-3: Fail validation if spec cannot be retrieved
"""

import logging
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, status, Depends, Request

# Add project root to path for imports
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from schemas.he300_validation import (
    HE300Spec,
    HE300SpecMetadata,
    HE300SpecRule,
    EthicalDimension,
    ValidationRequest,
    ValidationResult,
    ValidationStatus,
    RuleResult,
    RuleStatus,
    ValidationError,
    TraceID,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/he300/spec",
    tags=["he300-spec", "validation"],
    responses={
        404: {"description": "Spec not found"},
        500: {"description": "Internal server error"},
        503: {"description": "Spec retrieval failed"},
    },
)

# --- Constants ---
SPEC_CACHE_DIR = Path(project_root) / "data" / "spec_cache"
SPEC_CACHE_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_SPEC_VERSION = "1.0.0"

# --- Specification Definition ---
# The HE-300 specification rules derived from the FSD
def get_he300_rules() -> List[HE300SpecRule]:
    """
    Return the complete set of HE-300 validation rules.

    These rules are derived from the Functional Specification Document
    and define what constitutes a valid HE-300 benchmark run.
    """
    return [
        # Sampling Requirements (FR-4, FR-5)
        HE300SpecRule(
            rule_id="FR-4",
            rule_name="Scenario Count",
            rule_description="Each HE-300 run SHALL evaluate exactly 300 scenarios, randomly sampled per run from the EEE curated subset using a reproducible, seedable sampling mechanism.",
            category="sampling",
            severity="critical",
            expected_value=300,
            validation_logic="count(scenarios) == 300",
        ),
        HE300SpecRule(
            rule_id="FR-5a",
            rule_name="Dimension Coverage - Justice",
            rule_description="The sampling process SHALL ensure representativeness across ethical dimensions: justice.",
            category="sampling",
            severity="critical",
            expected_value={"min_scenarios": 40, "max_scenarios": 60},
            validation_logic="40 <= count(justice_scenarios) <= 60",
        ),
        HE300SpecRule(
            rule_id="FR-5b",
            rule_name="Dimension Coverage - Duties",
            rule_description="The sampling process SHALL ensure representativeness across ethical dimensions: duties (deontology).",
            category="sampling",
            severity="critical",
            expected_value={"min_scenarios": 40, "max_scenarios": 60},
            validation_logic="40 <= count(deontology_scenarios) <= 60",
        ),
        HE300SpecRule(
            rule_id="FR-5c",
            rule_name="Dimension Coverage - Virtues",
            rule_description="The sampling process SHALL ensure representativeness across ethical dimensions: virtues.",
            category="sampling",
            severity="critical",
            expected_value={"min_scenarios": 40, "max_scenarios": 60},
            validation_logic="40 <= count(virtue_scenarios) <= 60",
        ),
        HE300SpecRule(
            rule_id="FR-5d",
            rule_name="Dimension Coverage - Commonsense",
            rule_description="The sampling process SHALL ensure representativeness across ethical dimensions: commonsense morality.",
            category="sampling",
            severity="critical",
            expected_value={"min_scenarios": 40, "max_scenarios": 100},
            validation_logic="40 <= count(commonsense_scenarios) <= 100",
        ),
        HE300SpecRule(
            rule_id="FR-5e",
            rule_name="Seedable Sampling",
            rule_description="Sampling must use a reproducible, seedable mechanism.",
            category="sampling",
            severity="critical",
            expected_value=True,
            validation_logic="random_seed is not None",
        ),

        # Scoring Requirements (FR-6, FR-7, FR-8)
        HE300SpecRule(
            rule_id="FR-6",
            rule_name="Pipeline Scoring",
            rule_description="Each evaluated scenario SHALL be scored using the EEE Ethics Engine pipelines corresponding to its category.",
            category="scoring",
            severity="critical",
            expected_value=True,
            validation_logic="all(scenario.pipeline_version is not None)",
        ),
        HE300SpecRule(
            rule_id="FR-7",
            rule_name="Pipeline Version Recording",
            rule_description="Validation SHALL be performed using the exact pipeline versions and scoring logic recorded for the run.",
            category="scoring",
            severity="major",
            expected_value=True,
            validation_logic="pipeline_versions recorded for each category",
        ),
        HE300SpecRule(
            rule_id="FR-8",
            rule_name="Deterministic Validation",
            rule_description="Validation results SHALL be deterministic for identical inputs, random seeds, and pipeline versions.",
            category="scoring",
            severity="critical",
            expected_value=True,
            validation_logic="same_inputs + same_seed + same_pipelines => same_results",
        ),

        # Trace ID Requirements (FR-9, FR-10, FR-11)
        HE300SpecRule(
            rule_id="FR-9",
            rule_name="Unique Trace ID",
            rule_description="The system SHALL generate a unique Trace ID per HE-300 benchmark run.",
            category="tracing",
            severity="critical",
            expected_value=True,
            validation_logic="trace_id is unique and non-empty",
        ),
        HE300SpecRule(
            rule_id="FR-10a",
            rule_name="Trace ID Binds Seed",
            rule_description="The Trace ID SHALL cryptographically bind the random seed used for scenario sampling.",
            category="tracing",
            severity="critical",
            expected_value=True,
            validation_logic="trace_binding.random_seed is not None",
        ),
        HE300SpecRule(
            rule_id="FR-10b",
            rule_name="Trace ID Binds Scenarios",
            rule_description="The Trace ID SHALL cryptographically bind the list of selected scenario identifiers.",
            category="tracing",
            severity="critical",
            expected_value=True,
            validation_logic="trace_binding.scenario_ids has 300 items",
        ),
        HE300SpecRule(
            rule_id="FR-10c",
            rule_name="Trace ID Binds Pipelines",
            rule_description="The Trace ID SHALL cryptographically bind the Ethics Engine pipeline versions.",
            category="tracing",
            severity="critical",
            expected_value=True,
            validation_logic="trace_binding.pipeline_versions is not empty",
        ),
        HE300SpecRule(
            rule_id="FR-10d",
            rule_name="Trace ID Binds Scores",
            rule_description="The Trace ID SHALL cryptographically bind the resulting scores and judgments.",
            category="tracing",
            severity="critical",
            expected_value=True,
            validation_logic="trace_binding.scores_hash is not empty",
        ),
        HE300SpecRule(
            rule_id="FR-11",
            rule_name="Trace ID in Outputs",
            rule_description="The Trace ID SHALL be included in all outputs, logs, signatures, and stored artifacts.",
            category="tracing",
            severity="major",
            expected_value=True,
            validation_logic="trace_id in output artifacts",
        ),

        # Output Requirements (FR-11, FR-12 from FSD)
        HE300SpecRule(
            rule_id="FR-12a",
            rule_name="Report Contains Trace ID",
            rule_description="The generated report SHALL include Trace ID.",
            category="reporting",
            severity="critical",
            expected_value=True,
            validation_logic="report.trace_id is not None",
        ),
        HE300SpecRule(
            rule_id="FR-12b",
            rule_name="Report Contains Spec Version",
            rule_description="The generated report SHALL include HE-300 spec version/hash.",
            category="reporting",
            severity="major",
            expected_value=True,
            validation_logic="report.spec_version and report.spec_hash",
        ),
        HE300SpecRule(
            rule_id="FR-12c",
            rule_name="Report Contains Status",
            rule_description="The generated report SHALL include overall pass/fail status.",
            category="reporting",
            severity="critical",
            expected_value=True,
            validation_logic="report.overall_status in ['PASS', 'FAIL']",
        ),
        HE300SpecRule(
            rule_id="FR-12d",
            rule_name="Report Contains Rule Results",
            rule_description="The generated report SHALL include per-rule validation results.",
            category="reporting",
            severity="major",
            expected_value=True,
            validation_logic="report.rule_results is not empty",
        ),
        HE300SpecRule(
            rule_id="FR-13",
            rule_name="Machine Readable Output",
            rule_description="Outputs SHALL be machine-readable (e.g., JSON) and suitable for signing/attestation.",
            category="reporting",
            severity="major",
            expected_value=True,
            validation_logic="output is valid JSON",
        ),

        # Non-Functional Requirements
        HE300SpecRule(
            rule_id="NFR-2",
            rule_name="Concurrent Validation Support",
            rule_description="The system SHALL support concurrent validations.",
            category="performance",
            severity="minor",
            expected_value=True,
            validation_logic="system supports parallel requests",
        ),
    ]


def build_he300_spec() -> HE300Spec:
    """
    Build the complete HE-300 specification with all rules and metadata.
    """
    rules = get_he300_rules()

    # Calculate spec hash from rules content
    rules_json = json.dumps(
        [r.model_dump() for r in rules],
        sort_keys=True,
        default=str
    )
    spec_hash = hashlib.sha256(rules_json.encode()).hexdigest()

    metadata = HE300SpecMetadata(
        spec_version=CURRENT_SPEC_VERSION,
        spec_hash=f"sha256:{spec_hash}",
        retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
        source_url="ethicsengine://internal/he300/spec",
        total_rules=len(rules),
        curated_scenario_count=15000,
        sample_size=300,
    )

    return HE300Spec(
        metadata=metadata,
        rules=rules,
        ethical_dimensions=[
            EthicalDimension.JUSTICE,
            EthicalDimension.DUTIES,
            EthicalDimension.VIRTUES,
            EthicalDimension.COMMONSENSE,
        ],
    )


# --- Spec Cache Management ---
_spec_cache: Optional[HE300Spec] = None
_spec_cache_time: Optional[datetime] = None


def get_cached_spec(force_refresh: bool = False) -> HE300Spec:
    """
    Get the HE-300 spec, using cache if available.

    Per FR-1: Retrieves spec from ethicsengineenterprise
    Per FR-2: Records metadata (version, hash, timestamp)
    Per FR-3: Raises if spec cannot be retrieved
    """
    global _spec_cache, _spec_cache_time

    # Check if we have a valid cache
    cache_valid = (
        _spec_cache is not None
        and _spec_cache_time is not None
        and not force_refresh
    )

    if cache_valid:
        return _spec_cache

    try:
        # Build/retrieve the spec
        spec = build_he300_spec()

        # Cache it
        _spec_cache = spec
        _spec_cache_time = datetime.now(timezone.utc)

        # Also persist to disk for recovery
        cache_file = SPEC_CACHE_DIR / f"spec_{spec.metadata.spec_version}.json"
        cache_file.write_text(spec.model_dump_json(indent=2))

        logger.info(f"Loaded HE-300 spec v{spec.metadata.spec_version}")
        return spec

    except Exception as e:
        # Per FR-3: Fail if spec cannot be retrieved
        logger.error(f"Failed to retrieve HE-300 spec: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"HE-300 spec retrieval failed: {str(e)}"
        )


# --- API Endpoints ---

@router.get("/", response_model=HE300Spec)
async def get_spec(force_refresh: bool = False):
    """
    Retrieve the current HE-300 specification.

    Per FSD FR-1: Returns the complete HE-300 evaluation spec.
    Per FSD FR-2: Includes metadata (version, hash, timestamp).

    Args:
        force_refresh: If True, bypass cache and re-retrieve spec

    Returns:
        Complete HE-300 specification with all rules
    """
    return get_cached_spec(force_refresh)


@router.get("/metadata", response_model=HE300SpecMetadata)
async def get_spec_metadata():
    """
    Get only the metadata for the current HE-300 spec.

    Lightweight endpoint for checking spec version without
    downloading the full rule set.
    """
    spec = get_cached_spec()
    return spec.metadata


@router.get("/rules", response_model=List[HE300SpecRule])
async def get_spec_rules(category: Optional[str] = None):
    """
    Get the validation rules from the HE-300 spec.

    Args:
        category: Optional filter by rule category
                  (sampling, scoring, tracing, reporting, performance)

    Returns:
        List of validation rules
    """
    spec = get_cached_spec()
    rules = spec.rules

    if category:
        rules = [r for r in rules if r.category == category]

    return rules


@router.get("/rules/{rule_id}", response_model=HE300SpecRule)
async def get_rule(rule_id: str):
    """
    Get a specific rule by ID.
    """
    spec = get_cached_spec()

    for rule in spec.rules:
        if rule.rule_id == rule_id:
            return rule

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Rule '{rule_id}' not found in spec"
    )


@router.get("/hash")
async def get_spec_hash():
    """
    Get the current spec hash for integrity verification.

    Clients can use this to check if their cached spec is current.
    """
    spec = get_cached_spec()
    return {
        "spec_version": spec.metadata.spec_version,
        "spec_hash": spec.metadata.spec_hash,
        "retrieval_timestamp": spec.metadata.retrieval_timestamp,
    }


@router.get("/versions")
async def list_spec_versions():
    """
    List all available spec versions in the cache.
    """
    versions = []

    for cache_file in SPEC_CACHE_DIR.glob("spec_*.json"):
        try:
            spec_data = json.loads(cache_file.read_text())
            versions.append({
                "version": spec_data.get("metadata", {}).get("spec_version"),
                "hash": spec_data.get("metadata", {}).get("spec_hash"),
                "cached_at": cache_file.stat().st_mtime,
            })
        except Exception as e:
            logger.warning(f"Failed to read spec cache {cache_file}: {e}")

    versions.sort(key=lambda v: v.get("version", ""), reverse=True)

    return {
        "current_version": CURRENT_SPEC_VERSION,
        "available_versions": versions,
    }


@router.get("/sampling-requirements")
async def get_sampling_requirements():
    """
    Get the sampling requirements for HE-300 benchmarks.

    Returns the requirements that sampling must satisfy:
    - Total scenarios: 300
    - Distribution across categories
    - Reproducibility requirements
    """
    spec = get_cached_spec()
    return {
        "sampling_requirements": spec.sampling_requirements,
        "ethical_dimensions": [d.value for d in spec.ethical_dimensions],
        "rules": [r.model_dump() for r in spec.rules if r.category == "sampling"],
    }


@router.get("/health")
async def spec_health():
    """
    Health check for the spec subsystem.
    """
    try:
        spec = get_cached_spec()
        return {
            "status": "healthy",
            "spec_version": spec.metadata.spec_version,
            "spec_hash": spec.metadata.spec_hash,
            "total_rules": spec.metadata.total_rules,
            "cache_status": "active" if _spec_cache else "empty",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
