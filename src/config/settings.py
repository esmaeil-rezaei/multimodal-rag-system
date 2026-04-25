

import os
import yaml
from pathlib import Path
from functools import lru_cache     # Cache the loaded settings singleton
from typing import Any, Dict, List, Optional

from pydantic import Field                          # Pydantic field decorators
from pydantic_settings import BaseSettings          # Pydantic v2 settings with env-var support

CONFIG_PATH = Path("config/config.yaml")
ENV_PATH = Path(".env")


# Secrets
class Secrets(BaseSettings):
    """
    Pydantic-settings model that maps environment variables to typed attributes.
    All fields here correspond to entries in .env.example.
    """
    # LLM providers
    open_ai_key: str = Field(..., env="OPEN_AI_KEY")

    # Runtime
    app_env: str = Field("development", env="APP_ENV")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = str(ENV_PATH)                    # Load .env automatically
        env_file_encoding = "utf-8"                 # Encoding for .env file
        case_sensitive = False                      # Allow UPPER or lower env var names



# Config — non-secret options sourced from config/config.yaml
class AppConfig():
    """
    Thin wrapper around the parsed YAML dictionary.
    Provides typed convenience accessors so callers avoid dict key typos.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    # -- Convenience accessors ---

    @property
    def app(self) -> Dict[str, Any]:
        """Top-level application metadata (name, version, …)."""
        return self._data["app"]
    
    @property
    def knowledge_base(self) -> Dict[str, Any]:
        """Knowledge base root directory and file-type settings."""
        return self._data["knowledge_base"]


    @property
    def operations(self) -> Dict[str, Any]:
        """Production ops: cache, ACL, PII, observability."""
        return self._data["operations"]

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Generic dot-notation getter (e.g. 'ingestion.versioning.ttl_days')."""
        keys = key.split(".")                       # Split dot path into components
        node = self._data                           # Start at root
        for k in keys:
            if not isinstance(node, dict):
                return default                      
            node = node.get(k, default)             # Traverse one level
        return node



# Loaderfunctions
def load_yaml_config() -> AppConfig:
    """Parse config/config.yaml and return an AppConfig wrapper."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return AppConfig(raw)                      


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    """
    Return the singleton Secrets object (cached after first call).
    Reads from .env via pydantic-settings.
    """
    return Secrets()                                # pydantic-settings handles .env loading


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return the singleton AppConfig object (cached after first call)."""
    return load_yaml_config()
