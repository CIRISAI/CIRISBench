"""Pytest configuration and fixtures for CIRISBench tests."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def engine_dir(project_root: Path) -> Path:
    """Return the engine directory."""
    return project_root / "engine"


@pytest.fixture
def infra_dir(project_root: Path) -> Path:
    """Return the infra directory."""
    return project_root / "infra"


@pytest.fixture
def sample_benchmark_result() -> dict:
    """Return a sample benchmark result for testing."""
    return {
        "timestamp": "2026-01-28T12:00:00Z",
        "model": "ollama/llama3.2",
        "quantization": "q4_k_m",
        "summary": {
            "total": 50,
            "correct": 42,
            "accuracy": 0.84,
            "elapsed_seconds": 30.5,
        },
        "categories": {
            "commonsense": {"accuracy": 0.86},
            "deontology": {"accuracy": 0.82},
            "justice": {"accuracy": 0.84},
            "virtue": {"accuracy": 0.84},
        },
    }
