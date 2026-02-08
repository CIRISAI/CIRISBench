"""
Authentication dependencies for Engine API.

Dual auth: JWT Bearer (HS256, shared JWT_SECRET with CIRISNode) or
static API key (ENGINE_API_KEYS env var).  GET endpoints stay public;
POST endpoints use ``Depends(require_auth)``.
"""

import logging
from typing import Optional

import jwt
from fastapi import Header, HTTPException

from config.settings import settings

logger = logging.getLogger(__name__)

_ALGORITHM = "HS256"


def _validate_jwt(token: str) -> Optional[str]:
    """Return the *sub* claim if the JWT is valid, else None."""
    if not settings.jwt_secret:
        logger.warning("JWT_SECRET not configured — JWT auth disabled")
        return None
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[_ALGORITHM])
        return payload.get("sub") or payload.get("email") or "authenticated"
    except jwt.ExpiredSignatureError:
        logger.debug("JWT expired")
        return None
    except jwt.PyJWTError as exc:
        logger.debug("JWT validation failed: %s", exc)
        return None


def _validate_api_key(api_key: str) -> Optional[str]:
    """Return an identifier if the key is in ENGINE_API_KEYS, else None."""
    if not settings.api_keys:
        return None
    valid = {k.strip() for k in settings.api_keys.split(",") if k.strip()}
    if api_key in valid:
        return f"apikey:{api_key[:8]}…"
    return None


async def require_auth(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> str:
    """FastAPI dependency — returns actor string or raises 401.

    When ``AUTH_ENABLED=false`` (dev mode), returns ``"anonymous"``.
    """
    if not settings.auth_enabled:
        return "anonymous"

    # 1. Try Bearer token (JWT first, then static API key)
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1]
        actor = _validate_jwt(token)
        if actor:
            return actor
        # Bearer token might be a static API key (e.g. AGENTBEATS_API_KEY)
        actor = _validate_api_key(token)
        if actor:
            return actor

    # 2. Try X-API-Key header
    if x_api_key:
        actor = _validate_api_key(x_api_key)
        if actor:
            return actor

    raise HTTPException(
        status_code=401,
        detail="Valid Authorization (Bearer JWT) or X-API-Key header required",
        headers={"WWW-Authenticate": "Bearer"},
    )
