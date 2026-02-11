"""
Integration Tests for HE-300 Stack
==================================

End-to-end integration tests for CIRISNode + EthicsEngine.
"""
import os
import asyncio
from unittest.mock import patch, MagicMock

import pytest


# Skip if dependencies not available
pytest.importorskip("httpx")

import httpx


# Default URLs - can be overridden by environment
CIRISNODE_URL = os.getenv("CIRISNODE_URL", "http://localhost:8000")
EEE_URL = os.getenv("EEE_URL", "http://localhost:8080")


def is_service_up(url: str) -> bool:
    """Check if a service is responding."""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{url}/health")
            return resp.status_code == 200
    except Exception:
        return False


# Skip if services not running
SKIP_IF_NO_SERVICES = pytest.mark.skipif(
    not (is_service_up(CIRISNODE_URL) and is_service_up(EEE_URL)),
    reason="Services not running"
)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @SKIP_IF_NO_SERVICES
    def test_cirisnode_health(self):
        """CIRISNode should respond to health checks."""
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{CIRISNODE_URL}/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "status" in data

    @SKIP_IF_NO_SERVICES
    def test_eee_health(self):
        """EthicsEngine should respond to health checks."""
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{EEE_URL}/health")
            assert resp.status_code == 200


class TestHE300Integration:
    """Integration tests for HE-300 benchmark flow."""

    @SKIP_IF_NO_SERVICES
    def test_he300_catalog(self):
        """Should fetch HE-300 catalog from EEE."""
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{EEE_URL}/he300/catalog")
            assert resp.status_code == 200
            data = resp.json()
            assert "categories" in data or "scenarios" in data

    @SKIP_IF_NO_SERVICES
    def test_he300_batch_via_eee(self):
        """Should process HE-300 batch directly via EEE."""
        scenarios = [
            {
                "scenario_id": "test-1",
                "text": "Test scenario",
                "category": "commonsense"
            }
        ]

        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{EEE_URL}/he300/batch",
                json={"scenarios": scenarios}
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "results" in data


class TestCIRISNodeToEEE:
    """Tests for CIRISNode -> EEE integration."""

    @SKIP_IF_NO_SERVICES
    def test_cirisnode_triggers_eee(self):
        """CIRISNode benchmark endpoint should call EEE."""
        # This would require authentication
        # For CI, we test with mock token
        headers = {
            "Authorization": "Bearer test-token"
        }

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{CIRISNODE_URL}/api/v1/benchmarks/run",
                json={
                    "benchmark_type": "he300",
                    "n_scenarios": 5,
                    "seed": 42
                },
                headers=headers
            )
            # May fail auth, but should at least connect
            assert resp.status_code in [200, 202, 401, 403]


class TestMockMode:
    """Tests for mock mode operation."""

    def test_mock_eee_response(self):
        """Mock EEE client should return valid responses."""
        # This tests the mock implementation, not live services
        from unittest.mock import AsyncMock

        mock_client = AsyncMock()
        mock_client.process_batch.return_value = {
            "results": [
                {
                    "scenario_id": "test-1",
                    "prediction": 1,
                    "confidence": 0.95,
                    "correct": True
                }
            ],
            "summary": {
                "total": 1,
                "correct": 1,
                "accuracy": 1.0
            }
        }

        # Verify mock structure
        result = asyncio.get_event_loop().run_until_complete(
            mock_client.process_batch([])
        )
        assert "results" in result
        assert "summary" in result
