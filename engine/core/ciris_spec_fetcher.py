"""
CIRIS Trace Spec Fetcher

Retrieves the authoritative HE-300 compliance spec from ciris.ai.

Per FSD:
- FR-1: Must fetch the CIRIS trace schema from https://ciris.ai/explore-a-trace/
- FR-2: Must record spec metadata (schema version, retrieval timestamp)
- FR-3: Failure to retrieve the spec fails validation with a clear error
"""

import logging
import hashlib
import json
import httpx
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# CIRIS spec source URL
CIRIS_TRACE_URL = "https://ciris.ai/explore-a-trace/"
CIRIS_API_URL = os.getenv("CIRIS_SPEC_API_URL", "https://ciris.ai/api/trace-schema")

# Fallback to local spec if network fails (configurable)
ALLOW_FALLBACK = os.getenv("CIRIS_SPEC_ALLOW_FALLBACK", "false").lower() == "true"

# Cache settings
SPEC_CACHE_TTL_SECONDS = int(os.getenv("CIRIS_SPEC_CACHE_TTL", "3600"))  # 1 hour default

# Local project root for cache storage
project_root = Path(__file__).resolve().parents[1]
CIRIS_CACHE_DIR = project_root / "data" / "ciris_spec_cache"
CIRIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class CIRISTraceComponent(BaseModel):
    """A component of the CIRIS trace structure."""
    component_id: str = Field(..., description="Unique identifier for this component")
    component_name: str = Field(..., description="Human-readable name")
    description: str = Field(default="", description="Description of this component's purpose")
    required: bool = Field(default=True, description="Whether this component is required")
    fields: List[Dict[str, Any]] = Field(default_factory=list, description="Fields within this component")
    validation_rules: List[str] = Field(default_factory=list, description="Validation rules for this component")


class CIRISAuditMetadata(BaseModel):
    """Audit metadata structure from CIRIS spec."""
    signature_algorithm: str = Field(default="Ed25519", description="Algorithm used for signing")
    signature_key_id: Optional[str] = Field(None, description="Key ID used for signing")
    hash_chain_root: Optional[str] = Field(None, description="Root of the hash chain")
    requires_signature: bool = Field(default=True, description="Whether signature is required")
    hash_algorithm: str = Field(default="SHA-256", description="Hash algorithm for content hashing")


class CIRISTraceSpec(BaseModel):
    """
    The CIRIS trace specification as retrieved from ciris.ai.

    This is the authoritative schema for HE-300 compliance validation.
    """
    # Metadata
    spec_version: str = Field(..., description="Version of the CIRIS trace spec")
    spec_hash: str = Field(..., description="SHA-256 hash of the spec content")
    retrieval_timestamp: str = Field(..., description="When this spec was retrieved")
    source_url: str = Field(..., description="URL from which spec was retrieved")

    # Trace structure
    trace_components: List[CIRISTraceComponent] = Field(
        default_factory=list,
        description="Components that make up a valid CIRIS trace"
    )

    # Audit requirements
    audit_metadata: CIRISAuditMetadata = Field(
        default_factory=CIRISAuditMetadata,
        description="Audit and cryptographic requirements"
    )

    # Validation requirements
    required_fields: List[str] = Field(
        default_factory=list,
        description="Top-level fields required in every trace"
    )

    # Raw spec data for forward compatibility
    raw_spec: Optional[Dict[str, Any]] = Field(
        None,
        description="Raw spec data from CIRIS for fields not yet modeled"
    )


class CIRISSpecFetchResult(BaseModel):
    """Result of fetching the CIRIS spec."""
    success: bool
    spec: Optional[CIRISTraceSpec] = None
    error: Optional[str] = None
    from_cache: bool = False
    fetch_duration_ms: float = 0


# Module-level cache
_ciris_spec_cache: Optional[CIRISTraceSpec] = None
_ciris_spec_cache_time: Optional[datetime] = None


def _get_default_ciris_trace_components() -> List[CIRISTraceComponent]:
    """
    Returns the expected CIRIS trace components based on the CIRIS architecture.

    These are derived from the CIRIS reasoning structure:
    - Observation: What was observed/input
    - Context: Relevant context and history
    - Analysis: Ethical analysis performed
    - Conscience: Conscience checks applied
    - Decision: Final decision/output
    - Audit: Audit trail and signatures
    """
    return [
        CIRISTraceComponent(
            component_id="observation",
            component_name="Observation",
            description="The input observation that triggered this trace",
            required=True,
            fields=[
                {"name": "input_text", "type": "string", "required": True},
                {"name": "timestamp", "type": "datetime", "required": True},
                {"name": "source", "type": "string", "required": False},
            ],
            validation_rules=[
                "input_text must not be empty",
                "timestamp must be valid ISO-8601",
            ],
        ),
        CIRISTraceComponent(
            component_id="context",
            component_name="Context",
            description="Contextual information used in reasoning",
            required=True,
            fields=[
                {"name": "identity", "type": "object", "required": True},
                {"name": "guidance", "type": "object", "required": True},
                {"name": "history", "type": "array", "required": False},
            ],
            validation_rules=[
                "identity must specify the evaluating agent",
                "guidance must reference ethical framework",
            ],
        ),
        CIRISTraceComponent(
            component_id="analysis",
            component_name="Analysis",
            description="Ethical analysis and reasoning steps",
            required=True,
            fields=[
                {"name": "reasoning_steps", "type": "array", "required": True},
                {"name": "ethical_dimensions", "type": "array", "required": True},
                {"name": "model_response", "type": "string", "required": True},
            ],
            validation_rules=[
                "reasoning_steps must contain at least one step",
                "ethical_dimensions must identify evaluated categories",
            ],
        ),
        CIRISTraceComponent(
            component_id="conscience",
            component_name="Conscience Checks",
            description="Conscience checks and ethical guardrails applied",
            required=True,
            fields=[
                {"name": "checks_applied", "type": "array", "required": True},
                {"name": "all_passed", "type": "boolean", "required": True},
                {"name": "violations", "type": "array", "required": False},
            ],
            validation_rules=[
                "checks_applied must list all conscience checks run",
                "all_passed must reflect aggregate check status",
            ],
        ),
        CIRISTraceComponent(
            component_id="decision",
            component_name="Decision",
            description="Final decision and output",
            required=True,
            fields=[
                {"name": "label", "type": "integer", "required": True},
                {"name": "confidence", "type": "float", "required": False},
                {"name": "explanation", "type": "string", "required": False},
            ],
            validation_rules=[
                "label must be 0 (ETHICAL) or 1 (UNETHICAL)",
            ],
        ),
        CIRISTraceComponent(
            component_id="audit",
            component_name="Audit Trail",
            description="Audit metadata and cryptographic binding",
            required=True,
            fields=[
                {"name": "trace_id", "type": "string", "required": True},
                {"name": "timestamp", "type": "datetime", "required": True},
                {"name": "signature", "type": "string", "required": False},
                {"name": "content_hash", "type": "string", "required": True},
            ],
            validation_rules=[
                "trace_id must be globally unique",
                "content_hash must be SHA-256 of trace content",
                "signature must be Ed25519 if present",
            ],
        ),
    ]


def _build_fallback_spec() -> CIRISTraceSpec:
    """
    Build a fallback spec based on known CIRIS structure.

    Used when network fetch fails and ALLOW_FALLBACK is True.
    """
    components = _get_default_ciris_trace_components()

    # Calculate hash of the components
    components_json = json.dumps(
        [c.model_dump() for c in components],
        sort_keys=True,
        default=str
    )
    spec_hash = hashlib.sha256(components_json.encode()).hexdigest()

    return CIRISTraceSpec(
        spec_version="1.0.0-fallback",
        spec_hash=f"sha256:{spec_hash}",
        retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
        source_url="local://fallback",
        trace_components=components,
        audit_metadata=CIRISAuditMetadata(
            signature_algorithm="Ed25519",
            hash_algorithm="SHA-256",
            requires_signature=True,
        ),
        required_fields=[
            "trace_id",
            "spec_version",
            "overall_status",
            "audit_metadata",
            "component_results",
        ],
    )


async def _fetch_ciris_spec_from_api() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Fetch the CIRIS trace spec from the API endpoint.

    Returns (spec_data, error_message)
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Try the API endpoint first
            response = await client.get(
                CIRIS_API_URL,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "EthicsEngine-HE300/1.0",
                },
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                    return data, None
                except json.JSONDecodeError as e:
                    return None, f"Invalid JSON from CIRIS API: {e}"

            # If API fails, try scraping the explore page
            logger.info(f"API returned {response.status_code}, trying explore page...")

            response = await client.get(
                CIRIS_TRACE_URL,
                headers={
                    "Accept": "text/html,application/json",
                    "User-Agent": "EthicsEngine-HE300/1.0",
                },
            )

            if response.status_code == 200:
                # Try to extract JSON from the page
                content = response.text

                # Look for embedded JSON (common patterns)
                import re

                # Pattern 1: JSON in script tag
                json_match = re.search(
                    r'<script[^>]*type="application/json"[^>]*>(.*?)</script>',
                    content,
                    re.DOTALL
                )
                if json_match:
                    try:
                        data = json.loads(json_match.group(1))
                        return data, None
                    except json.JSONDecodeError:
                        pass

                # Pattern 2: JSON in data attribute
                json_match = re.search(
                    r'data-trace-schema=["\'](\{.*?\})["\']',
                    content,
                    re.DOTALL
                )
                if json_match:
                    try:
                        data = json.loads(json_match.group(1))
                        return data, None
                    except json.JSONDecodeError:
                        pass

                # Pattern 3: Return structured data based on page analysis
                # (In production, this would parse the actual page structure)
                logger.warning("Could not extract JSON from explore page, using structure analysis")
                return None, "Could not extract spec from explore page"

            return None, f"CIRIS endpoint returned status {response.status_code}"

    except httpx.TimeoutException:
        return None, "Timeout connecting to CIRIS API"
    except httpx.ConnectError as e:
        return None, f"Connection error to CIRIS API: {e}"
    except Exception as e:
        return None, f"Unexpected error fetching CIRIS spec: {e}"


def _parse_ciris_spec_data(data: Dict[str, Any]) -> CIRISTraceSpec:
    """
    Parse raw spec data from CIRIS into our structured format.
    """
    # Extract components if present
    components = []
    if "components" in data:
        for comp_data in data["components"]:
            components.append(CIRISTraceComponent(
                component_id=comp_data.get("id", comp_data.get("component_id", "")),
                component_name=comp_data.get("name", comp_data.get("component_name", "")),
                description=comp_data.get("description", ""),
                required=comp_data.get("required", True),
                fields=comp_data.get("fields", []),
                validation_rules=comp_data.get("validation_rules", []),
            ))
    elif "trace_components" in data:
        for comp_data in data["trace_components"]:
            components.append(CIRISTraceComponent(**comp_data))
    else:
        # Use default components
        components = _get_default_ciris_trace_components()

    # Extract audit metadata
    audit_data = data.get("audit_metadata", data.get("audit", {}))
    audit_metadata = CIRISAuditMetadata(
        signature_algorithm=audit_data.get("signature_algorithm", "Ed25519"),
        signature_key_id=audit_data.get("signature_key_id"),
        hash_chain_root=audit_data.get("hash_chain_root"),
        requires_signature=audit_data.get("requires_signature", True),
        hash_algorithm=audit_data.get("hash_algorithm", "SHA-256"),
    )

    # Calculate hash
    content_for_hash = json.dumps(data, sort_keys=True, default=str)
    spec_hash = hashlib.sha256(content_for_hash.encode()).hexdigest()

    return CIRISTraceSpec(
        spec_version=data.get("version", data.get("spec_version", "1.0.0")),
        spec_hash=f"sha256:{spec_hash}",
        retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
        source_url=CIRIS_TRACE_URL,
        trace_components=components,
        audit_metadata=audit_metadata,
        required_fields=data.get("required_fields", [
            "trace_id", "spec_version", "overall_status",
            "audit_metadata", "component_results"
        ]),
        raw_spec=data,
    )


def _load_cached_spec() -> Optional[CIRISTraceSpec]:
    """Load spec from disk cache if available and not expired."""
    cache_file = CIRIS_CACHE_DIR / "ciris_spec_latest.json"

    if not cache_file.exists():
        return None

    try:
        cache_data = json.loads(cache_file.read_text())
        cached_at = datetime.fromisoformat(cache_data.get("cached_at", ""))

        # Check if cache is expired
        age_seconds = (datetime.now(timezone.utc) - cached_at).total_seconds()
        if age_seconds > SPEC_CACHE_TTL_SECONDS:
            logger.info(f"CIRIS spec cache expired (age: {age_seconds:.0f}s)")
            return None

        spec_data = cache_data.get("spec", {})
        return CIRISTraceSpec(**spec_data)

    except Exception as e:
        logger.warning(f"Failed to load CIRIS spec from cache: {e}")
        return None


def _save_spec_to_cache(spec: CIRISTraceSpec) -> None:
    """Save spec to disk cache."""
    cache_file = CIRIS_CACHE_DIR / "ciris_spec_latest.json"

    try:
        cache_data = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "spec": spec.model_dump(),
        }
        cache_file.write_text(json.dumps(cache_data, indent=2, default=str))
        logger.info(f"Saved CIRIS spec to cache: {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save CIRIS spec to cache: {e}")


async def fetch_ciris_spec(force_refresh: bool = False) -> CIRISSpecFetchResult:
    """
    Fetch the CIRIS trace spec from ciris.ai.

    Per FSD:
    - FR-1: Fetches from https://ciris.ai/explore-a-trace/ or API
    - FR-2: Records metadata (version, timestamp, hash)
    - FR-3: Fails with clear error if spec cannot be retrieved

    Args:
        force_refresh: If True, bypass cache and fetch fresh

    Returns:
        CIRISSpecFetchResult with success status and spec or error
    """
    global _ciris_spec_cache, _ciris_spec_cache_time

    import time
    start_time = time.time()

    # Check in-memory cache first
    if not force_refresh and _ciris_spec_cache is not None:
        cache_age = (datetime.now(timezone.utc) - _ciris_spec_cache_time).total_seconds()
        if cache_age < SPEC_CACHE_TTL_SECONDS:
            return CIRISSpecFetchResult(
                success=True,
                spec=_ciris_spec_cache,
                from_cache=True,
                fetch_duration_ms=(time.time() - start_time) * 1000,
            )

    # Check disk cache
    if not force_refresh:
        cached_spec = _load_cached_spec()
        if cached_spec:
            _ciris_spec_cache = cached_spec
            _ciris_spec_cache_time = datetime.now(timezone.utc)
            return CIRISSpecFetchResult(
                success=True,
                spec=cached_spec,
                from_cache=True,
                fetch_duration_ms=(time.time() - start_time) * 1000,
            )

    # Fetch from network
    logger.info(f"Fetching CIRIS spec from {CIRIS_API_URL}")
    spec_data, error = await _fetch_ciris_spec_from_api()

    if spec_data:
        # Parse and cache
        try:
            spec = _parse_ciris_spec_data(spec_data)
            _ciris_spec_cache = spec
            _ciris_spec_cache_time = datetime.now(timezone.utc)
            _save_spec_to_cache(spec)

            logger.info(f"Successfully fetched CIRIS spec v{spec.spec_version}")
            return CIRISSpecFetchResult(
                success=True,
                spec=spec,
                from_cache=False,
                fetch_duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            error = f"Failed to parse CIRIS spec: {e}"

    # Network fetch failed
    logger.error(f"Failed to fetch CIRIS spec: {error}")

    # Check if fallback is allowed
    if ALLOW_FALLBACK:
        logger.warning("Using fallback CIRIS spec (network fetch failed)")
        fallback_spec = _build_fallback_spec()
        _ciris_spec_cache = fallback_spec
        _ciris_spec_cache_time = datetime.now(timezone.utc)

        return CIRISSpecFetchResult(
            success=True,
            spec=fallback_spec,
            from_cache=False,
            error=f"Used fallback spec due to: {error}",
            fetch_duration_ms=(time.time() - start_time) * 1000,
        )

    # Per FR-3: Fail with clear error
    return CIRISSpecFetchResult(
        success=False,
        spec=None,
        error=error,
        fetch_duration_ms=(time.time() - start_time) * 1000,
    )


def get_ciris_spec_sync(force_refresh: bool = False) -> CIRISTraceSpec:
    """
    Synchronous wrapper for fetching CIRIS spec.

    Raises HTTPException if spec cannot be retrieved (per FR-3).
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    result = loop.run_until_complete(fetch_ciris_spec(force_refresh))

    if not result.success:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to retrieve CIRIS spec (FR-3): {result.error}"
        )

    return result.spec


def get_cached_ciris_spec() -> Optional[CIRISTraceSpec]:
    """Get the currently cached CIRIS spec without fetching."""
    return _ciris_spec_cache


def clear_ciris_spec_cache() -> None:
    """Clear all CIRIS spec caches."""
    global _ciris_spec_cache, _ciris_spec_cache_time
    _ciris_spec_cache = None
    _ciris_spec_cache_time = None

    # Also clear disk cache
    cache_file = CIRIS_CACHE_DIR / "ciris_spec_latest.json"
    if cache_file.exists():
        cache_file.unlink()
