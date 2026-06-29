from __future__ import annotations

import pytest
from src.config.settings import AppConfig, get_config, get_secrets, load_yaml_config



def test_load_yaml_config_returns_app_config():
    cfg = load_yaml_config()
    assert isinstance(cfg, AppConfig)


@pytest.mark.parametrize(
    "section",
    [
        "app",
        "knowledge_base",
        "ingestion",
        "chunking",
        "embeddings",
        "vector_store",
        "query",
        "retrieval",
        "generation",
        "evaluation",
        "operations",
        "log",
    ],
)
def test_required_sections_present(section):
    cfg = get_config()
    value = getattr(cfg, section)
    assert isinstance(value, dict)
    assert value, f"config section '{section}' should not be empty"


def test_graphrag_section_defaults_to_empty_dict_if_absent():
    cfg = get_config()
    # graphrag uses .get(..., {}) so it must never raise even if missing
    assert isinstance(cfg.graphrag, dict)




def test_chunking_config_has_expected_keys():
    chunking = get_config().chunking
    for key in ("strategy", "chunk_size", "min_chunk_size", "chunk_overlap"):
        assert key in chunking




def test_get_dot_path_returns_nested_value():
    cfg = get_config()
    chunk_size = cfg.get("chunking.chunk_size")
    assert chunk_size == cfg.chunking["chunk_size"]


def test_get_dot_path_missing_key_returns_default():
    cfg = get_config()
    assert cfg.get("does.not.exist") is None
    assert cfg.get("does.not.exist", "fallback") == "fallback"


def test_get_dot_path_partial_path_through_non_dict_returns_default():
    cfg = get_config()
    # 'app.name' is a string, so 'app.name.deeper' must hit the
    # `not isinstance(node, dict)` branch and fall back to default.
    assert cfg.get("app.name.deeper", "fallback") == "fallback"


@pytest.mark.parametrize(
    "dot_path",
    [
        "evaluation.cost_latency_slo",
        "evaluation.multi_turn",
        "evaluation.fairness",
    ],
)
def test_layer_5_6_7_config_sections_exist(dot_path):
    """The Layer 5/6/7 evaluation config sections added for the offline
    evaluation suite must be present and non-empty."""
    cfg = get_config()
    section = cfg.get(dot_path)
    assert isinstance(section, dict)
    assert section, f"config section '{dot_path}' should not be empty"




def test_get_config_is_cached_singleton():
    assert get_config() is get_config()


def test_get_secrets_is_cached_singleton():
    assert get_secrets() is get_secrets()




def test_secrets_reads_required_keys_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("COHERE_API_KEY", "test-cohere")
    get_secrets.cache_clear()

    secrets = get_secrets()
    assert secrets.openai_api_key == "sk-test-openai"
    assert secrets.cohere_api_key == "test-cohere"


def test_secrets_jwt_secret_key_from_env(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "super-secret-test-value")
    get_secrets.cache_clear()

    secrets = get_secrets()
    assert secrets.jwt_secret_key == "super-secret-test-value"


def test_secrets_optional_fields_have_sane_defaults():
    # conftest disables .env loading for Secrets, so these reflect the
    # hardcoded field defaults in src/config/settings.py rather than
    # whatever real values happen to be in the project's .env file.
    secrets = get_secrets()
    assert secrets.qdrant_url == "http://localhost:6333"
    assert secrets.redis_url == "redis://localhost:6379/0"
    assert secrets.neo4j_uri == "neo4j://localhost:7687"
