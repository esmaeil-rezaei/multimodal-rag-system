"""
tests/conftest.py
──────────────────
Shared pytest fixtures for the unit test suite.

Provides dummy environment-based secrets so that
``src.config.settings.get_secrets()`` (a required-field pydantic-settings
model) can be instantiated without a real ``.env`` file or real API keys.
This is purely for import-time / construction-time validation in tests —
no network calls are made with these values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the project root (containing `src/`, `app/`, `config/`) is importable
# regardless of where pytest is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _dummy_secrets_env(monkeypatch):
    """
    Provide dummy values for required/secret env vars used across the app.

    `Secrets` (src/config/settings.py) requires OPENAI_API_KEY and
    COHERE_API_KEY with no defaults; `app/auth.py` requires JWT_SECRET_KEY.
    These dummy values are never used to make real API calls in unit tests.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("COHERE_API_KEY", "test-cohere-key")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-key-for-unit-tests-only")
    monkeypatch.setenv("APP_ENV", "test")

    # Clear lru_cache'd singletons so they pick up the patched env per-test.
    from src.config.settings import Secrets, get_config, get_secrets

    # `Secrets` reads from a real `.env` file (Config.env_file = ".env") if
    # one is present in the project root. That file may contain real
    # credentials (Neo4j Aura URI/password, etc.) which must never be loaded
    # — let alone surfaced in a test failure's repr/traceback — during the
    # test suite. Disable .env loading so Secrets() only sees the explicit
    # env vars set above (plus its hardcoded field defaults).
    monkeypatch.setitem(Secrets.model_config, "env_file", None)

    get_secrets.cache_clear()
    get_config.cache_clear()
    yield
    get_secrets.cache_clear()
    get_config.cache_clear()
