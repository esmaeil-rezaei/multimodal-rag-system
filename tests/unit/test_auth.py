from __future__ import annotations

import sqlite3

import jwt
import pytest

from app import auth


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Point app.auth at a throwaway SQLite file for the duration of each test."""
    db_path = tmp_path / "test_auth.db"
    monkeypatch.setattr(auth, "DB_PATH", db_path)
    auth.init_db()
    yield db_path


def test_hash_password_produces_salt_and_hash():
    hashed = auth._hash_password("correct horse battery staple")
    assert ":" in hashed
    salt, digest = hashed.split(":", 1)
    assert len(salt) == 32  # 16 bytes hex-encoded
    assert len(digest) == 64  # sha256 hex digest


def test_hash_password_is_salted_and_non_deterministic():
    h1 = auth._hash_password("same-password")
    h2 = auth._hash_password("same-password")
    assert h1 != h2  # different random salts


def test_verify_password_round_trip():
    hashed = auth._hash_password("my-secret-pw")
    assert auth._verify_password("my-secret-pw", hashed) is True
    assert auth._verify_password("wrong-pw", hashed) is False


def test_verify_password_handles_malformed_hash_gracefully():
    assert auth._verify_password("anything", "not-a-valid-hash") is False


def test_init_db_creates_users_table(_isolated_db):
    conn = sqlite3.connect(str(_isolated_db))
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        assert cursor.fetchone() is not None
    finally:
        conn.close()


def test_init_db_is_idempotent(_isolated_db):
    # Calling init_db() again must not raise (CREATE TABLE IF NOT EXISTS).
    auth.init_db()


def test_register_user_returns_token_and_namespace():
    response = auth.register_user("alice", "hunter2-password", "default")
    assert response.token
    assert response.username == "alice"
    assert response.namespace == "default"
    assert response.role == "user"


def test_register_user_validates_username_length():
    with pytest.raises(ValueError):
        auth.register_user("ab", "password123", "default")


def test_register_user_validates_password_length():
    with pytest.raises(ValueError):
        auth.register_user("validname", "short", "default")


def test_register_user_requires_namespace():
    with pytest.raises(ValueError):
        auth.register_user("validname", "password123", "")


def test_register_duplicate_username_raises_value_error():
    auth.register_user("bob", "password123", "default")
    with pytest.raises(ValueError, match="already taken"):
        auth.register_user("bob", "another-password", "default")


def test_login_user_with_correct_credentials():
    auth.register_user("carol", "s3cr3t-pw", "default")
    response = auth.login_user("carol", "s3cr3t-pw")
    assert response.token
    assert response.username == "carol"


def test_login_user_with_wrong_password_raises_value_error():
    auth.register_user("dave", "correct-password", "default")
    with pytest.raises(ValueError, match="Invalid username or password"):
        auth.login_user("dave", "incorrect-password")


def test_login_unknown_user_raises_value_error():
    with pytest.raises(ValueError, match="Invalid username or password"):
        auth.login_user("nonexistent-user", "whatever")


def test_decode_token_round_trip():
    response = auth.register_user("erin", "another-pw", "ns_erin")
    payload = auth.decode_token(response.token)
    assert payload["sub"] == "erin"
    assert payload["namespace"] == "ns_erin"
    assert payload["role"] == "user"
    assert "exp" in payload


def test_decode_token_rejects_garbage_token():
    with pytest.raises(jwt.PyJWTError):
        auth.decode_token("not-a-real-jwt")


def test_get_secret_raises_runtime_error_when_jwt_secret_missing(monkeypatch):
    from src.config.settings import get_secrets

    monkeypatch.setenv("JWT_SECRET_KEY", "")
    get_secrets.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
            auth._get_secret()
    finally:
        get_secrets.cache_clear()
