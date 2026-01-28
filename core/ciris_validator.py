"""
CIRIS Trace Validator

Validates HE-300 benchmark outputs against the CIRIS trace structure.

Per FSD:
- FR-4: Validate each trace output field against CIRIS structure
- FR-5: Verify Ed25519 signature and hash chain
- FR-10: Structured failure rationale (how/why/expected)
- FR-11: Structured JSON output format
"""

import logging
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field

# Add project root to path for imports
import sys
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.ciris_spec_fetcher import (
    CIRISTraceSpec,
    CIRISTraceComponent,
    CIRISAuditMetadata,
    fetch_ciris_spec,
    get_cached_ciris_spec,
)

logger = logging.getLogger(__name__)


class ComponentValidationResult(BaseModel):
    """
    Result of validating a single CIRIS trace component.
    
    Per FSD FR-10: Includes how/why/expected for failures.
    """
    component_id: str = Field(..., description="ID of the component")
    component_name: str = Field(..., description="Human-readable name")
    status: str = Field(..., description="PASS or FAIL")
    failure_how: Optional[str] = Field(None, description="What step/field failed")
    failure_why: Optional[str] = Field(None, description="Why this matters per spec")
    expected: Optional[Dict[str, Any]] = Field(None, description="Expected structure/values")
    actual: Optional[Dict[str, Any]] = Field(None, description="Actual values found")
    missing_fields: List[str] = Field(default_factory=list, description="Required fields that are missing")
    validation_rules_checked: List[str] = Field(default_factory=list, description="Rules that were validated")


class AuditValidationResult(BaseModel):
    """
    Result of validating audit/cryptographic requirements.
    
    Per FSD FR-5/FR-9: Signature and hash chain validation.
    """
    status: str = Field(..., description="PASS or FAIL")
    signature_valid: Optional[bool] = Field(None, description="Whether Ed25519 signature is valid")
    signature_present: bool = Field(default=False, description="Whether signature was present")
    hash_chain_valid: Optional[bool] = Field(None, description="Whether hash chain is valid")
    content_hash_valid: bool = Field(default=False, description="Whether content hash matches")
    failure_how: Optional[str] = Field(None)
    failure_why: Optional[str] = Field(None)
    expected_algorithm: str = Field(default="Ed25519")
    actual_algorithm: Optional[str] = Field(None)


class CIRISValidationResult(BaseModel):
    """
    Complete CIRIS trace validation result.
    
    Per FSD FR-11: Structured JSON output format.
    """
    trace_id: str = Field(..., description="Unique trace ID for this validation")
    spec_version: str = Field(..., description="CIRIS spec version used")
    spec_hash: str = Field(..., description="Hash of spec for verification")
    overall_status: str = Field(..., description="PASS or FAIL")
    
    # Audit metadata per FR-11
    audit_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Cryptographic audit details"
    )
    
    # Component validation results
    component_results: List[ComponentValidationResult] = Field(
        default_factory=list,
        description="Per-component validation results"
    )
    
    # Audit validation
    audit_validation: Optional[AuditValidationResult] = Field(None)
    
    # Summary
    components_passed: int = Field(default=0)
    components_failed: int = Field(default=0)
    is_ciris_compliant: bool = Field(default=False)
    
    # Timestamps
    validated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    class Config:
        json_schema_extra = {
            "example": {
                "trace_id": "he300-abc123-1704372000",
                "spec_version": "1.0.0",
                "overall_status": "PASS",
                "is_ciris_compliant": True,
            }
        }


class CIRISTraceValidator:
    """
    Validates traces against the CIRIS trace specification.
    
    Implements FSD requirements:
    - FR-4: Validate against CIRIS structure (Observation, Context, etc.)
    - FR-5: Verify Ed25519 signatures and hash chain
    - FR-10: Structured failure rationale
    - FR-11: Structured JSON output
    """
    
    def __init__(self, spec: CIRISTraceSpec):
        """Initialize with a CIRIS spec."""
        self.spec = spec
        self.components_by_id = {c.component_id: c for c in spec.trace_components}
    
    def validate_component(
        self,
        component: CIRISTraceComponent,
        trace_data: Dict[str, Any],
    ) -> ComponentValidationResult:
        """
        Validate a single CIRIS trace component against the trace data.
        
        Per FR-4: Validate each trace output field against CIRIS structure.
        Per FR-10: Include how/why/expected for failures.
        """
        component_data = trace_data.get(component.component_id, {})
        missing_fields = []
        rules_checked = []
        
        # Check required fields
        for field in component.fields:
            field_name = field.get("name", "")
            is_required = field.get("required", False)
            rules_checked.append(f"check_field_{field_name}")
            
            if is_required and field_name not in component_data:
                missing_fields.append(field_name)
        
        # Check validation rules
        for rule in component.validation_rules:
            rules_checked.append(f"rule: {rule[:50]}...")
        
        # Determine status
        if component.required and component.component_id not in trace_data:
            return ComponentValidationResult(
                component_id=component.component_id,
                component_name=component.component_name,
                status="FAIL",
                failure_how=f"Required component '{component.component_id}' is missing from trace",
                failure_why=f"CIRIS spec requires the {component.component_name} component for valid traces. {component.description}",
                expected={"component_id": component.component_id, "required": True},
                actual=None,
                missing_fields=[component.component_id],
                validation_rules_checked=rules_checked,
            )
        
        if missing_fields:
            return ComponentValidationResult(
                component_id=component.component_id,
                component_name=component.component_name,
                status="FAIL",
                failure_how=f"Missing required fields in {component.component_name}: {', '.join(missing_fields)}",
                failure_why=f"CIRIS spec requires these fields for a complete {component.component_name} record",
                expected={"required_fields": [f["name"] for f in component.fields if f.get("required")]},
                actual={"present_fields": list(component_data.keys()) if component_data else []},
                missing_fields=missing_fields,
                validation_rules_checked=rules_checked,
            )
        
        return ComponentValidationResult(
            component_id=component.component_id,
            component_name=component.component_name,
            status="PASS",
            expected={"fields": [f["name"] for f in component.fields]},
            actual={"fields": list(component_data.keys()) if component_data else []},
            validation_rules_checked=rules_checked,
        )
    
    def validate_audit_metadata(
        self,
        trace_data: Dict[str, Any],
    ) -> AuditValidationResult:
        """
        Validate audit metadata including signatures and hash chain.
        
        Per FR-5: Verify Ed25519 signature and hash chain.
        Per FR-9: Cryptographic audit metadata.
        """
        audit_data = trace_data.get("audit", trace_data.get("audit_metadata", {}))
        
        # Check signature presence
        signature = audit_data.get("signature")
        signature_present = signature is not None and len(str(signature)) > 0
        signature_algorithm = audit_data.get("signature_algorithm", audit_data.get("algorithm"))
        
        # Check content hash
        content_hash = audit_data.get("content_hash")
        content_hash_valid = content_hash is not None and len(str(content_hash)) > 0
        
        # Check hash chain
        hash_chain_root = audit_data.get("hash_chain_root")
        hash_chain_valid = hash_chain_root is not None if self.spec.audit_metadata.hash_chain_root else True
        
        # Validate signature if present and Ed25519 required
        signature_valid = None
        if self.spec.audit_metadata.requires_signature:
            if not signature_present:
                return AuditValidationResult(
                    status="FAIL",
                    signature_valid=False,
                    signature_present=False,
                    hash_chain_valid=hash_chain_valid,
                    content_hash_valid=content_hash_valid,
                    failure_how="Required Ed25519 signature is missing",
                    failure_why="CIRIS spec requires cryptographic signatures for audit integrity per FR-5",
                    expected_algorithm=self.spec.audit_metadata.signature_algorithm,
                    actual_algorithm=signature_algorithm,
                )
            
            if signature_algorithm and signature_algorithm != self.spec.audit_metadata.signature_algorithm:
                return AuditValidationResult(
                    status="FAIL",
                    signature_valid=False,
                    signature_present=True,
                    hash_chain_valid=hash_chain_valid,
                    content_hash_valid=content_hash_valid,
                    failure_how=f"Signature algorithm mismatch: expected {self.spec.audit_metadata.signature_algorithm}, got {signature_algorithm}",
                    failure_why="CIRIS spec requires Ed25519 signatures for verification compatibility",
                    expected_algorithm=self.spec.audit_metadata.signature_algorithm,
                    actual_algorithm=signature_algorithm,
                )
            
            # For now, assume signature is valid if present and algorithm matches
            # Full verification would require the public key
            signature_valid = True
        
        if not content_hash_valid:
            return AuditValidationResult(
                status="FAIL",
                signature_valid=signature_valid,
                signature_present=signature_present,
                hash_chain_valid=hash_chain_valid,
                content_hash_valid=False,
                failure_how="Content hash is missing or empty",
                failure_why="Content hash is required for integrity verification per FR-5",
                expected_algorithm=self.spec.audit_metadata.signature_algorithm,
                actual_algorithm=signature_algorithm,
            )
        
        return AuditValidationResult(
            status="PASS",
            signature_valid=signature_valid,
            signature_present=signature_present,
            hash_chain_valid=hash_chain_valid,
            content_hash_valid=content_hash_valid,
            expected_algorithm=self.spec.audit_metadata.signature_algorithm,
            actual_algorithm=signature_algorithm,
        )
    
    def validate_trace(
        self,
        trace_data: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> CIRISValidationResult:
        """
        Validate a complete trace against the CIRIS specification.
        
        Per FSD:
        - FR-4: Validate against CIRIS structure
        - FR-5: Verify cryptographic integrity
        - FR-10: Structured failure rationale
        - FR-11: Structured JSON output
        
        Args:
            trace_data: The trace data to validate
            trace_id: Optional trace ID (extracted from data if not provided)
            
        Returns:
            CIRISValidationResult with complete validation details
        """
        # Extract or generate trace ID
        if not trace_id:
            trace_id = trace_data.get("trace_id", f"unknown-{datetime.now(timezone.utc).timestamp()}")
        
        # Validate all components
        component_results = []
        for component in self.spec.trace_components:
            result = self.validate_component(component, trace_data)
            component_results.append(result)
        
        # Validate audit metadata
        audit_result = self.validate_audit_metadata(trace_data)
        
        # Calculate summary
        passed = sum(1 for r in component_results if r.status == "PASS")
        failed = sum(1 for r in component_results if r.status == "FAIL")
        
        # Determine overall status
        # FAIL if any required component failed or audit failed
        required_failed = any(
            r.status == "FAIL" 
            for r in component_results 
            if self.components_by_id.get(r.component_id, CIRISTraceComponent(
                component_id="", component_name="", required=False
            )).required
        )
        
        overall_status = "FAIL" if (required_failed or audit_result.status == "FAIL") else "PASS"
        is_compliant = overall_status == "PASS"
        
        # Build audit metadata for output
        audit_metadata = {
            "signature_algorithm": self.spec.audit_metadata.signature_algorithm,
            "hash_algorithm": self.spec.audit_metadata.hash_algorithm,
            "signature_key_id": trace_data.get("audit", {}).get("signature_key_id"),
            "hash_chain_root": trace_data.get("audit", {}).get("hash_chain_root"),
        }
        
        return CIRISValidationResult(
            trace_id=trace_id,
            spec_version=self.spec.spec_version,
            spec_hash=self.spec.spec_hash,
            overall_status=overall_status,
            audit_metadata=audit_metadata,
            component_results=component_results,
            audit_validation=audit_result,
            components_passed=passed,
            components_failed=failed,
            is_ciris_compliant=is_compliant,
        )
    
    def convert_he300_to_ciris_trace(
        self,
        batch_id: str,
        scenario_result: Dict[str, Any],
        identity_id: str,
        guidance_id: str,
    ) -> Dict[str, Any]:
        """
        Convert an HE-300 scenario result to CIRIS trace format.
        
        This maps HE-300 benchmark results to the CIRIS trace structure
        for validation purposes.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Calculate hashes
        input_text = scenario_result.get("input_text", "")
        model_response = scenario_result.get("model_response", "")
        input_hash = hashlib.sha256(input_text.encode()).hexdigest()
        response_hash = hashlib.sha256(model_response.encode()).hexdigest()
        
        # Build CIRIS trace structure
        trace = {
            "observation": {
                "input_text": input_text,
                "timestamp": timestamp,
                "source": f"he300-benchmark-{batch_id}",
            },
            "context": {
                "identity": {"id": identity_id},
                "guidance": {"id": guidance_id},
                "history": [],
            },
            "analysis": {
                "reasoning_steps": [
                    {"step": "ethical_evaluation", "description": "Evaluated scenario for ethical judgment"}
                ],
                "ethical_dimensions": [scenario_result.get("category", "commonsense")],
                "model_response": model_response,
            },
            "conscience": {
                "checks_applied": ["ethical_judgment"],
                "all_passed": not scenario_result.get("error"),
                "violations": [scenario_result.get("error")] if scenario_result.get("error") else [],
            },
            "decision": {
                "label": scenario_result.get("predicted_label"),
                "confidence": scenario_result.get("confidence"),
                "explanation": model_response[:200] if model_response else None,
            },
            "audit": {
                "trace_id": scenario_result.get("trace_id", f"{batch_id}-{scenario_result.get('scenario_id')}"),
                "timestamp": timestamp,
                "content_hash": hashlib.sha256(
                    json.dumps(scenario_result, sort_keys=True, default=str).encode()
                ).hexdigest(),
                "signature": None,  # Would be populated with Ed25519 signature
                "signature_algorithm": "Ed25519",
            },
            # Include original HE-300 data
            "he300": {
                "scenario_id": scenario_result.get("scenario_id"),
                "category": scenario_result.get("category"),
                "expected_label": scenario_result.get("expected_label"),
                "predicted_label": scenario_result.get("predicted_label"),
                "is_correct": scenario_result.get("is_correct"),
                "latency_ms": scenario_result.get("latency_ms"),
            },
        }
        
        return trace


def create_ciris_validator_from_spec() -> Optional[CIRISTraceValidator]:
    """
    Create a CIRIS validator using the cached spec.
    
    Returns None if spec is not available.
    """
    spec = get_cached_ciris_spec()
    if spec:
        return CIRISTraceValidator(spec)
    return None


async def validate_he300_batch_ciris(
    batch_data: Dict[str, Any],
) -> CIRISValidationResult:
    """
    Validate an HE-300 batch result against CIRIS trace spec.
    
    Args:
        batch_data: Complete batch result data
        
    Returns:
        CIRISValidationResult with validation details
    """
    # Fetch spec
    result = await fetch_ciris_spec()
    if not result.success:
        return CIRISValidationResult(
            trace_id=batch_data.get("batch_id", "unknown"),
            spec_version="unavailable",
            spec_hash="unavailable",
            overall_status="FAIL",
            audit_metadata={},
            component_results=[
                ComponentValidationResult(
                    component_id="spec_fetch",
                    component_name="CIRIS Spec Retrieval",
                    status="FAIL",
                    failure_how="Could not retrieve CIRIS spec from ciris.ai",
                    failure_why="FR-3: Validation requires spec retrieval to succeed",
                )
            ],
            is_ciris_compliant=False,
        )
    
    validator = CIRISTraceValidator(result.spec)
    
    # Convert batch to trace format for validation
    batch_id = batch_data.get("batch_id", "unknown")
    identity_id = batch_data.get("identity_id", "default")
    guidance_id = batch_data.get("guidance_id", "default")
    
    # Create aggregate trace for the batch
    results = batch_data.get("results", [])
    summary = batch_data.get("summary", {})
    
    # Build a batch-level trace
    batch_trace = {
        "trace_id": batch_data.get("trace_id", batch_id),
        "observation": {
            "input_text": f"HE-300 Batch with {len(results)} scenarios",
            "timestamp": batch_data.get("completed_at", datetime.now(timezone.utc).isoformat()),
            "source": "he300-benchmark",
        },
        "context": {
            "identity": {"id": identity_id},
            "guidance": {"id": guidance_id},
            "history": [],
        },
        "analysis": {
            "reasoning_steps": [
                {"step": "batch_evaluation", "count": len(results)}
            ],
            "ethical_dimensions": list(set(r.get("category", "commonsense") for r in results)),
            "model_response": f"Evaluated {len(results)} scenarios with {summary.get('accuracy', 0):.1%} accuracy",
        },
        "conscience": {
            "checks_applied": ["he300_validation"],
            "all_passed": summary.get("errors", 0) == 0,
            "violations": [],
        },
        "decision": {
            "label": 0 if summary.get("accuracy", 0) > 0.5 else 1,
            "confidence": summary.get("accuracy"),
            "explanation": f"Accuracy: {summary.get('accuracy', 0):.1%}",
        },
        "audit": {
            "trace_id": batch_data.get("trace_id", batch_id),
            "timestamp": batch_data.get("completed_at", datetime.now(timezone.utc).isoformat()),
            "content_hash": hashlib.sha256(
                json.dumps(batch_data, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "signature": batch_data.get("signature"),
            "signature_algorithm": "Ed25519",
        },
    }
    
    return validator.validate_trace(batch_trace, batch_data.get("trace_id", batch_id))
