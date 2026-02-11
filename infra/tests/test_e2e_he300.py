"""
End-to-End HE-300 Benchmark Tests
=================================

Full pipeline tests from CIRISNode through EEE with live services.
"""
import os
import time

import pytest


# Skip if httpx not available
pytest.importorskip("httpx")

import httpx


CIRISNODE_URL = os.getenv("CIRISNODE_URL", "http://localhost:8000")
EEE_URL = os.getenv("EEE_URL", "http://localhost:8080")
WAIT_TIMEOUT = int(os.getenv("E2E_TIMEOUT", "300"))


def get_auth_token() -> str:
    """Get authentication token for CIRISNode."""
    # Try to get a real token
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                f"{CIRISNODE_URL}/api/v1/auth/token",
                data={"username": "test", "password": "test"}
            )
            if resp.status_code == 200:
                return resp.json().get("access_token", "")
    except Exception:
        pass

    # Return test token
    return "test-token"


def is_stack_up() -> bool:
    """Check if the full stack is running."""
    try:
        with httpx.Client(timeout=5) as client:
            cn = client.get(f"{CIRISNODE_URL}/health")
            eee = client.get(f"{EEE_URL}/health")
            return cn.status_code == 200 and eee.status_code == 200
    except Exception:
        return False


SKIP_IF_STACK_DOWN = pytest.mark.skipif(
    not is_stack_up(),
    reason="Full stack not running"
)


@SKIP_IF_STACK_DOWN
class TestE2EBenchmark:
    """End-to-end benchmark tests."""

    @pytest.fixture
    def client(self):
        """Create HTTP client with auth."""
        token = get_auth_token()
        client = httpx.Client(
            timeout=30,
            headers={"Authorization": f"Bearer {token}"}
        )
        yield client
        client.close()

    def test_small_benchmark_run(self, client):
        """Run a small HE-300 benchmark end-to-end."""
        # Start benchmark
        resp = client.post(
            f"{CIRISNODE_URL}/api/v1/benchmarks/run",
            json={
                "benchmark_type": "he300",
                "n_scenarios": 5,
                "seed": 42
            }
        )

        if resp.status_code in [401, 403]:
            pytest.skip("Authentication required")

        assert resp.status_code in [200, 202]
        data = resp.json()

        if "job_id" not in data:
            # Synchronous response
            assert "results" in data or "result" in data
            return

        job_id = data["job_id"]

        # Poll for completion
        start = time.time()
        while time.time() - start < WAIT_TIMEOUT:
            status_resp = client.get(
                f"{CIRISNODE_URL}/api/v1/benchmarks/status/{job_id}"
            )

            if status_resp.status_code != 200:
                time.sleep(2)
                continue

            status = status_resp.json().get("status")

            if status == "completed":
                break
            elif status in ["failed", "error"]:
                pytest.fail(f"Job failed: {status_resp.json()}")

            time.sleep(2)
        else:
            pytest.fail("Timeout waiting for benchmark")

        # Fetch results
        results_resp = client.get(
            f"{CIRISNODE_URL}/api/v1/benchmarks/results/{job_id}"
        )
        assert results_resp.status_code == 200

        results = results_resp.json()
        assert "result" in results or "results" in results

    def test_eee_direct_batch(self, client):
        """Test direct batch processing via EEE."""
        scenarios = [
            {
                "scenario_id": f"e2e-test-{i}",
                "text": f"Test scenario {i}",
                "category": "commonsense",
                "label": i % 2
            }
            for i in range(3)
        ]

        resp = client.post(
            f"{EEE_URL}/he300/batch",
            json={"scenarios": scenarios}
        )

        assert resp.status_code == 200
        data = resp.json()

        assert "results" in data
        assert len(data["results"]) == 3

    def test_benchmark_categories(self, client):
        """Test benchmark with specific categories."""
        resp = client.get(f"{EEE_URL}/he300/catalog")

        if resp.status_code != 200:
            pytest.skip("Catalog not available")

        catalog = resp.json()
        categories = catalog.get("categories", [])

        if not categories:
            pytest.skip("No categories in catalog")

        # Run benchmark with first category
        category = categories[0] if isinstance(categories[0], str) else categories[0].get("name", "commonsense")

        resp = client.post(
            f"{CIRISNODE_URL}/api/v1/benchmarks/run",
            json={
                "benchmark_type": "he300",
                "n_scenarios": 3,
                "categories": [category],
                "seed": 42
            }
        )

        if resp.status_code in [401, 403]:
            pytest.skip("Authentication required")

        assert resp.status_code in [200, 202]


@SKIP_IF_STACK_DOWN
class TestE2EResilience:
    """Resilience and error handling tests."""

    @pytest.fixture
    def client(self):
        """Create HTTP client."""
        return httpx.Client(timeout=30)

    def test_invalid_benchmark_type(self, client):
        """Should handle invalid benchmark type."""
        resp = client.post(
            f"{CIRISNODE_URL}/api/v1/benchmarks/run",
            json={
                "benchmark_type": "invalid",
                "n_scenarios": 5
            },
            headers={"Authorization": "Bearer test"}
        )

        # Should return error, not crash
        assert resp.status_code in [400, 401, 422]

    def test_empty_batch(self, client):
        """Should handle empty batch gracefully."""
        resp = client.post(
            f"{EEE_URL}/he300/batch",
            json={"scenarios": []}
        )

        # Should return empty results or error
        assert resp.status_code in [200, 400, 422]

    def test_concurrent_requests(self, client):
        """Should handle concurrent requests."""
        import concurrent.futures

        def make_request():
            return client.get(f"{CIRISNODE_URL}/health")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in futures]

        assert all(r.status_code == 200 for r in results)
