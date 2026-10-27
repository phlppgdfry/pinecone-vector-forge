"""Application Configuration and Environment Resolver."""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for Pinecone Vector Forge."""

    # Pinecone Credentials
    pinecone_api_key: Optional[str] = None
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_index_name: str = "vector-forge-knowledge"
    pinecone_dimension: int = 1536
    pinecone_metric: str = "cosine"  # cosine, euclidean, dotproduct
    pinecone_namespace: str = "default"

    # OpenAI API Key (optional)
    openai_api_key: Optional[str] = None

    # Server Settings
    web_host: str = "0.0.0.0"
    web_port: int = 8000
    debug: bool = False

    # Mock mode auto-detection
    mock_mode: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def is_pinecone_configured(self) -> bool:
        """Check whether a valid Pinecone API key is provided."""
        if not self.pinecone_api_key:
            return False
        key = self.pinecone_api_key.strip()
        if not key or key == "your_pinecone_api_key_here":
            return False
        return True


# Global settings instance
settings = Settings()

# Check for CLI stored token/key if env var is empty or placeholder
if not settings.is_pinecone_configured():
    # 1. Check ~/.config/pinecone/secrets.yaml
    secrets_yaml_path = Path.home() / ".config" / "pinecone" / "secrets.yaml"
    if secrets_yaml_path.exists():
        try:
            for line in secrets_yaml_path.read_text().splitlines():
                if line.startswith("api_key:"):
                    val = line.split("api_key:", 1)[1].strip().strip('"').strip("'")
                    if val and val.startswith("pcsk_"):
                        settings.pinecone_api_key = val
                        break
        except Exception:
            pass

    # 2. Check ~/.pinecone/token
    if not settings.is_pinecone_configured():
        cli_token_path = Path.home() / ".pinecone" / "token"
        if cli_token_path.exists():
            try:
                token = cli_token_path.read_text().strip()
                if token:
                    settings.pinecone_api_key = token
            except Exception:
                pass
