import os
import yaml
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

CONFIG_PATH = Path("config/config.yaml")
ENV_PATH = Path(".env")

class Secrets(BaseSettings):
    """
    Environment-based configuration loaded from .env.
    """

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    cohere_api_key: str = Field(..., env="COHERE_API_KEY")

    qdrant_api_key: Optional[str] = Field(None, env="QDRANT_API_KEY")
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")

    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    elasticsearch_url: str = Field("http://localhost:9200", env="ELASTICSEARCH_URL")
    elasticsearch_api_key: Optional[str] = Field(None, env="ELASTICSEARCH_API_KEY")

    encryption_key: Optional[str] = Field(None, env="ENCRYPTION_KEY")
    jwt_secret_key: Optional[str] = Field(None, env="JWT_SECRET_KEY")

    app_env: str = Field("development", env="APP_ENV")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    class Config:
        env_file = str(ENV_PATH)  
        env_file_encoding = "utf-8"
        case_sensitive = False

class AppConfig:
    """
    YAML config with typed section access.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    @property
    def app(self) -> Dict[str, Any]:
        return self._data["app"]

    @property
    def knowledge_base(self) -> Dict[str, Any]:
        return self._data["knowledge_base"]

    @property
    def ingestion(self) -> Dict[str, Any]:
        return self._data["ingestion"]

    @property
    def chunking(self) -> Dict[str, Any]:
        return self._data["chunking"]

    @property
    def embeddings(self) -> Dict[str, Any]:
        return self._data["embeddings"]

    @property
    def vector_store(self) -> Dict[str, Any]:
        return self._data["vector_store"]

    @property
    def query(self) -> Dict[str, Any]:
        return self._data["query"]

    @property
    def retrieval(self) -> Dict[str, Any]:
        return self._data["retrieval"]

    @property
    def generation(self) -> Dict[str, Any]:
        return self._data["generation"]

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self._data["evaluation"]

    @property
    def operations(self) -> Dict[str, Any]:
        return self._data["operations"]
    
    @property
    def log(self) -> Dict[str, Any]:
        return self._data["log"]

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Dot-path lookup (e.g. 'ingestion.versioning.ttl_days')."""
        keys = key.split(".")
        node = self._data
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
        return node


def load_yaml_config() -> AppConfig:
    """Load YAML configuration file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return AppConfig(raw)


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    """Load and cache environment-based secrets."""
    return Secrets()


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and cache application configuration."""
    return load_yaml_config()