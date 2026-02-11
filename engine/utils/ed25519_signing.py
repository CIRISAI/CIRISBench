"""
Ed25519 Cryptographic Signing Utilities

Provides Ed25519 signing and verification for CIRIS trace compliance.

Per FSD:
- FR-5: Ed25519 signature verification
- FR-9: Cryptographic audit metadata
"""

import logging
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Try to import cryptography library for Ed25519
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    from cryptography.exceptions import InvalidSignature
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logger.warning("cryptography library not installed. Ed25519 signing disabled.")

# Key storage directory
project_root = Path(__file__).resolve().parents[1]
KEYS_DIR = project_root / "data" / "keys"
KEYS_DIR.mkdir(parents=True, exist_ok=True)


class SignatureResult(BaseModel):
    """Result of signing content."""
    signature: str = Field(..., description="Hex-encoded Ed25519 signature")
    signature_algorithm: str = Field(default="Ed25519")
    content_hash: str = Field(..., description="SHA-256 hash of signed content")
    timestamp: str = Field(..., description="ISO 8601 signing timestamp")
    key_id: str = Field(..., description="Identifier for the signing key")


class VerificationResult(BaseModel):
    """Result of signature verification."""
    valid: bool = Field(..., description="Whether signature is valid")
    signature_algorithm: str = Field(default="Ed25519")
    content_hash: str = Field(..., description="Hash of content that was verified")
    key_id: Optional[str] = Field(None, description="Key ID if known")
    error: Optional[str] = Field(None, description="Error message if verification failed")


class Ed25519Signer:
    """
    Ed25519 signing and verification for CIRIS traces.

    Per FSD FR-5: Provides Ed25519 cryptographic signing.
    """

    def __init__(self, key_id: str = "default"):
        """
        Initialize the signer.

        Args:
            key_id: Identifier for the key pair to use
        """
        self.key_id = key_id
        self._private_key: Optional[Ed25519PrivateKey] = None
        self._public_key: Optional[Ed25519PublicKey] = None
        self._load_or_generate_keys()

    def _load_or_generate_keys(self) -> None:
        """Load existing keys or generate new ones."""
        if not HAS_CRYPTO:
            logger.warning("Cryptography library not available, signing disabled")
            return

        private_key_path = KEYS_DIR / f"{self.key_id}_private.pem"
        public_key_path = KEYS_DIR / f"{self.key_id}_public.pem"

        if private_key_path.exists() and public_key_path.exists():
            # Load existing keys
            try:
                with open(private_key_path, "rb") as f:
                    self._private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None,
                    )
                with open(public_key_path, "rb") as f:
                    self._public_key = serialization.load_pem_public_key(f.read())
                logger.info(f"Loaded Ed25519 key pair: {self.key_id}")
            except Exception as e:
                logger.error(f"Failed to load keys: {e}")
                self._generate_new_keys(private_key_path, public_key_path)
        else:
            self._generate_new_keys(private_key_path, public_key_path)

    def _generate_new_keys(self, private_path: Path, public_path: Path) -> None:
        """Generate a new Ed25519 key pair."""
        if not HAS_CRYPTO:
            return

        try:
            self._private_key = Ed25519PrivateKey.generate()
            self._public_key = self._private_key.public_key()

            # Save keys
            with open(private_path, "wb") as f:
                f.write(self._private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                ))

            with open(public_path, "wb") as f:
                f.write(self._public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                ))

            logger.info(f"Generated new Ed25519 key pair: {self.key_id}")
        except Exception as e:
            logger.error(f"Failed to generate keys: {e}")

    def sign(self, content: Any) -> Optional[SignatureResult]:
        """
        Sign content with Ed25519.

        Args:
            content: Content to sign (will be JSON serialized)

        Returns:
            SignatureResult or None if signing unavailable
        """
        if not HAS_CRYPTO or not self._private_key:
            logger.warning("Signing unavailable: no private key")
            return None

        try:
            # Serialize content deterministically
            if isinstance(content, str):
                content_bytes = content.encode()
            else:
                content_bytes = json.dumps(content, sort_keys=True, default=str).encode()

            # Calculate hash
            content_hash = hashlib.sha256(content_bytes).hexdigest()

            # Sign
            signature_bytes = self._private_key.sign(content_bytes)
            signature_hex = signature_bytes.hex()

            return SignatureResult(
                signature=signature_hex,
                signature_algorithm="Ed25519",
                content_hash=content_hash,
                timestamp=datetime.now(timezone.utc).isoformat(),
                key_id=self.key_id,
            )
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            return None

    def verify(
        self,
        content: Any,
        signature_hex: str,
    ) -> VerificationResult:
        """
        Verify an Ed25519 signature.

        Args:
            content: Original content (will be JSON serialized)
            signature_hex: Hex-encoded signature to verify

        Returns:
            VerificationResult with validity status
        """
        if not HAS_CRYPTO:
            return VerificationResult(
                valid=False,
                content_hash="",
                error="Cryptography library not available",
            )

        if not self._public_key:
            return VerificationResult(
                valid=False,
                content_hash="",
                error="No public key available for verification",
            )

        try:
            # Serialize content
            if isinstance(content, str):
                content_bytes = content.encode()
            else:
                content_bytes = json.dumps(content, sort_keys=True, default=str).encode()

            content_hash = hashlib.sha256(content_bytes).hexdigest()

            # Convert signature from hex
            signature_bytes = bytes.fromhex(signature_hex)

            # Verify
            self._public_key.verify(signature_bytes, content_bytes)

            return VerificationResult(
                valid=True,
                content_hash=content_hash,
                key_id=self.key_id,
            )
        except InvalidSignature:
            return VerificationResult(
                valid=False,
                content_hash=hashlib.sha256(content_bytes).hexdigest() if 'content_bytes' in dir() else "",
                key_id=self.key_id,
                error="Signature verification failed: invalid signature",
            )
        except Exception as e:
            return VerificationResult(
                valid=False,
                content_hash="",
                error=f"Verification error: {e}",
            )

    def get_public_key_pem(self) -> Optional[str]:
        """Get the public key in PEM format."""
        if not self._public_key:
            return None

        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()


# Global signer instance
_default_signer: Optional[Ed25519Signer] = None


def get_signer(key_id: str = "default") -> Ed25519Signer:
    """Get or create a signer instance."""
    global _default_signer

    if _default_signer is None or _default_signer.key_id != key_id:
        _default_signer = Ed25519Signer(key_id)

    return _default_signer


def sign_content(content: Any, key_id: str = "default") -> Optional[SignatureResult]:
    """
    Convenience function to sign content.

    Args:
        content: Content to sign
        key_id: Key ID to use

    Returns:
        SignatureResult or None
    """
    signer = get_signer(key_id)
    return signer.sign(content)


def verify_signature(
    content: Any,
    signature_hex: str,
    key_id: str = "default",
) -> VerificationResult:
    """
    Convenience function to verify a signature.

    Args:
        content: Original content
        signature_hex: Hex-encoded signature
        key_id: Key ID to use

    Returns:
        VerificationResult
    """
    signer = get_signer(key_id)
    return signer.verify(content, signature_hex)


def sign_trace(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sign a trace and add signature metadata.

    Args:
        trace_data: Trace data to sign

    Returns:
        Trace data with added signature in audit section
    """
    # Don't include existing signature in the content to sign
    content_to_sign = {k: v for k, v in trace_data.items() if k != "audit"}

    result = sign_content(content_to_sign)

    if result:
        # Update or create audit section
        audit = trace_data.get("audit", {})
        audit.update({
            "signature": result.signature,
            "signature_algorithm": result.signature_algorithm,
            "content_hash": result.content_hash,
            "signature_timestamp": result.timestamp,
            "signature_key_id": result.key_id,
        })
        trace_data["audit"] = audit

    return trace_data


def verify_trace_signature(trace_data: Dict[str, Any]) -> VerificationResult:
    """
    Verify the signature on a trace.

    Args:
        trace_data: Trace data with signature in audit section

    Returns:
        VerificationResult
    """
    audit = trace_data.get("audit", {})
    signature = audit.get("signature")
    key_id = audit.get("signature_key_id", "default")

    if not signature:
        return VerificationResult(
            valid=False,
            content_hash="",
            error="No signature found in trace audit metadata",
        )

    # Get content without audit section
    content_to_verify = {k: v for k, v in trace_data.items() if k != "audit"}

    return verify_signature(content_to_verify, signature, key_id)


def is_signing_available() -> bool:
    """Check if Ed25519 signing is available."""
    return HAS_CRYPTO


def get_signing_status() -> Dict[str, Any]:
    """Get status of the signing subsystem."""
    signer = get_signer("default")

    return {
        "available": HAS_CRYPTO,
        "key_id": signer.key_id if signer else None,
        "has_private_key": signer._private_key is not None if signer else False,
        "has_public_key": signer._public_key is not None if signer else False,
        "algorithm": "Ed25519",
    }
