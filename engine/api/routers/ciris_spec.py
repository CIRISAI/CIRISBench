"""
CIRIS Trace Spec API Router

Provides endpoints for retrieving and validating against the CIRIS trace spec.

Per FSD:
- FR-1: Fetch CIRIS trace schema from ciris.ai
- FR-2: Record spec metadata (schema version, retrieval timestamp)
- FR-3: Fail validation if spec cannot be retrieved
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, status, Query

# Add project root to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.ciris_spec_fetcher import (
    fetch_ciris_spec,
    get_cached_ciris_spec,
    clear_ciris_spec_cache,
    CIRISTraceSpec,
    CIRISSpecFetchResult,
    CIRISTraceComponent,
    CIRISAuditMetadata,
    CIRIS_TRACE_URL,
    CIRIS_API_URL,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ciris",
    tags=["ciris", "trace-spec"],
    responses={
        404: {"description": "Spec not found"},
        503: {"description": "CIRIS spec retrieval failed"},
    },
)


@router.get("/spec", response_model=CIRISTraceSpec)
async def get_ciris_spec(
    force_refresh: bool = Query(
        default=False,
        description="Force refresh from ciris.ai, bypassing cache"
    ),
):
    """
    Retrieve the CIRIS trace specification.

    Per FSD FR-1: Fetches the CIRIS trace schema from https://ciris.ai/explore-a-trace/
    Per FSD FR-2: Records metadata (schema version, retrieval timestamp)
    Per FSD FR-3: Fails with clear error if spec cannot be retrieved

    Args:
        force_refresh: If True, bypass cache and fetch fresh from ciris.ai

    Returns:
        Complete CIRIS trace specification
    """
    result = await fetch_ciris_spec(force_refresh=force_refresh)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to retrieve CIRIS spec from {CIRIS_TRACE_URL}: {result.error}"
        )

    return result.spec


@router.get("/spec/metadata")
async def get_ciris_spec_metadata():
    """
    Get CIRIS spec metadata without the full component definitions.

    Lightweight endpoint for checking spec version and cache status.
    """
    result = await fetch_ciris_spec()

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to retrieve CIRIS spec: {result.error}"
        )

    spec = result.spec
    return {
        "spec_version": spec.spec_version,
        "spec_hash": spec.spec_hash,
        "retrieval_timestamp": spec.retrieval_timestamp,
        "source_url": spec.source_url,
        "from_cache": result.from_cache,
        "fetch_duration_ms": result.fetch_duration_ms,
        "component_count": len(spec.trace_components),
        "required_fields": spec.required_fields,
    }


@router.get("/spec/components", response_model=List[CIRISTraceComponent])
async def get_ciris_components(
    component_id: Optional[str] = Query(
        default=None,
        description="Filter by specific component ID"
    ),
):
    """
    Get the CIRIS trace components that must be validated.

    Per FSD FR-4: These components define the trace structure
    (Observation, Context, Analysis, Conscience checks, etc.)
    """
    result = await fetch_ciris_spec()

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to retrieve CIRIS spec: {result.error}"
        )

    components = result.spec.trace_components

    if component_id:
        components = [c for c in components if c.component_id == component_id]
        if not components:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Component '{component_id}' not found in CIRIS spec"
            )

    return components


@router.get("/spec/audit", response_model=CIRISAuditMetadata)
async def get_ciris_audit_requirements():
    """
    Get the CIRIS audit metadata requirements.

    Per FSD FR-5: Includes Ed25519 signature requirements and hash chain specs.
    Per FSD FR-9: Cryptographic audit metadata requirements.
    """
    result = await fetch_ciris_spec()

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to retrieve CIRIS spec: {result.error}"
        )

    return result.spec.audit_metadata


@router.get("/spec/required-fields")
async def get_required_fields():
    """
    Get the list of required fields for a valid CIRIS trace.

    Per FSD FR-11: Outputs must include these structured fields.
    """
    result = await fetch_ciris_spec()

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to retrieve CIRIS spec: {result.error}"
        )

    return {
        "required_fields": result.spec.required_fields,
        "spec_version": result.spec.spec_version,
    }


@router.get("/spec/hash")
async def get_ciris_spec_hash():
    """
    Get the current CIRIS spec hash for integrity verification.

    Clients can use this to verify their cached spec matches.
    """
    result = await fetch_ciris_spec()

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to retrieve CIRIS spec: {result.error}"
        )

    return {
        "spec_version": result.spec.spec_version,
        "spec_hash": result.spec.spec_hash,
        "retrieval_timestamp": result.spec.retrieval_timestamp,
        "source_url": result.spec.source_url,
    }


@router.post("/spec/refresh")
async def refresh_ciris_spec():
    """
    Force refresh the CIRIS spec from ciris.ai.

    Clears all caches and fetches fresh from the source.
    """
    # Clear caches
    clear_ciris_spec_cache()

    # Fetch fresh
    result = await fetch_ciris_spec(force_refresh=True)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to refresh CIRIS spec: {result.error}"
        )

    return {
        "status": "refreshed",
        "spec_version": result.spec.spec_version,
        "spec_hash": result.spec.spec_hash,
        "retrieval_timestamp": result.spec.retrieval_timestamp,
        "fetch_duration_ms": result.fetch_duration_ms,
    }


@router.get("/cache/status")
async def get_cache_status():
    """
    Get the current CIRIS spec cache status.
    """
    cached_spec = get_cached_ciris_spec()

    if cached_spec:
        return {
            "cached": True,
            "spec_version": cached_spec.spec_version,
            "spec_hash": cached_spec.spec_hash,
            "retrieval_timestamp": cached_spec.retrieval_timestamp,
            "source_url": cached_spec.source_url,
        }
    else:
        return {
            "cached": False,
            "message": "No CIRIS spec currently cached",
        }


@router.delete("/cache")
async def clear_cache():
    """
    Clear the CIRIS spec cache.

    Next request will fetch fresh from ciris.ai.
    """
    clear_ciris_spec_cache()
    return {"status": "cache_cleared"}


@router.get("/health")
async def ciris_health():
    """
    Health check for the CIRIS spec subsystem.

    Verifies that the CIRIS spec can be retrieved.
    """
    try:
        result = await fetch_ciris_spec()

        return {
            "status": "healthy" if result.success else "degraded",
            "spec_available": result.success,
            "spec_version": result.spec.spec_version if result.spec else None,
            "from_cache": result.from_cache,
            "source_url": CIRIS_TRACE_URL,
            "api_url": CIRIS_API_URL,
            "error": result.error,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "spec_available": False,
            "error": str(e),
        }


@router.get("/validation-schema")
async def get_validation_schema():
    """
    Get the validation schema derived from the CIRIS spec.

    Returns a JSON Schema that can be used to validate traces.
    """
    result = await fetch_ciris_spec()

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to retrieve CIRIS spec: {result.error}"
        )

    spec = result.spec

    # Build JSON Schema from components
    properties = {}
    required = list(spec.required_fields)

    for component in spec.trace_components:
        component_properties = {}
        component_required = []

        for field in component.fields:
            field_type = field.get("type", "string")
            json_type = {
                "string": "string",
                "integer": "integer",
                "float": "number",
                "boolean": "boolean",
                "array": "array",
                "object": "object",
                "datetime": "string",
            }.get(field_type, "string")

            component_properties[field["name"]] = {
                "type": json_type,
                "description": field.get("description", ""),
            }

            if field.get("required", False):
                component_required.append(field["name"])

        properties[component.component_id] = {
            "type": "object",
            "description": component.description,
            "properties": component_properties,
            "required": component_required,
        }

        if component.required:
            required.append(component.component_id)

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "CIRIS Trace Validation Schema",
        "description": f"Generated from CIRIS spec v{spec.spec_version}",
        "type": "object",
        "properties": properties,
        "required": list(set(required)),
        "additionalProperties": True,
    }
