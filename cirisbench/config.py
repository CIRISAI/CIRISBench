"""CIRISBench configuration management."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # CIRISNode
    cirisnode_url: str = "http://localhost:8000"
    cirisnode_api_key: Optional[str] = None

    # EthicsEngine
    eee_host: str = "0.0.0.0"
    eee_port: int = 8080
    eee_debug: bool = False
    eee_log_level: str = "INFO"

    # LLM Configuration
    llm_model: str = "ollama/llama3.2"
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Benchmark Settings
    he300_sample_size: int = 300
    he300_batch_size: int = 50
    he300_seed: int = 42
    he300_timeout: int = 300

    # Feature Flags
    ff_mock_llm: bool = False
    ff_trace_enabled: bool = True

    # Paths
    @property
    def root_dir(self) -> Path:
        """Project root directory."""
        return Path(__file__).parent.parent

    @property
    def engine_dir(self) -> Path:
        """EthicsEngine directory."""
        return self.root_dir / "engine"

    @property
    def infra_dir(self) -> Path:
        """Infrastructure directory."""
        return self.root_dir / "infra"

    @property
    def data_dir(self) -> Path:
        """Data directory."""
        return self.engine_dir / "data"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
