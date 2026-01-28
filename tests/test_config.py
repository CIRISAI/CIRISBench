"""Tests for CIRISBench configuration."""

import pytest
from cirisbench.config import Settings, get_settings


def test_settings_defaults():
    """Test that settings have sensible defaults."""
    settings = Settings()

    assert settings.cirisnode_url == "http://localhost:8000"
    assert settings.eee_host == "0.0.0.0"
    assert settings.eee_port == 8080
    assert settings.llm_model == "ollama/llama3.2"
    assert settings.he300_sample_size == 300


def test_settings_paths():
    """Test that path properties resolve correctly."""
    settings = Settings()

    assert settings.root_dir.exists()
    assert settings.engine_dir.name == "engine"
    assert settings.infra_dir.name == "infra"


def test_get_settings_cached():
    """Test that get_settings returns cached instance."""
    s1 = get_settings()
    s2 = get_settings()

    assert s1 is s2


def test_settings_from_env(monkeypatch):
    """Test that settings can be overridden via environment variables."""
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o")
    monkeypatch.setenv("HE300_SAMPLE_SIZE", "100")

    # Clear the cache to get fresh settings
    get_settings.cache_clear()
    settings = Settings()

    assert settings.llm_model == "openai/gpt-4o"
    assert settings.he300_sample_size == 100
