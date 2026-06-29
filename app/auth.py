from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import jwt
from pydantic import BaseModel

from src.config.settings import get_secrets

DB_PATH = Path("auth.db")
_JWT_ALGORITHM = "HS256"
_TOKEN_EXPIRE_HOURS = 24 * 7  # 7 days


def _get_secret() -> str:
    secret = get_secrets().jwt_secret_key
    if not secret:
        raise RuntimeError(
            "JWT_SECRET_KEY is not configured. "
            "Add it to your .env file:\n"
            "  JWT_SECRET_KEY=<random 32+ character string>"
        )
    return secret


def _hash_password(password: str) -> str:
    salt = os.urandom(16).hex()
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}:{h.hex()}"


def _verify_password(password: str, hashed: str) -> bool:
    try:
        salt, h = hashed.split(":", 1)
        expected = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
        return hmac.compare_digest(expected.hex(), h)
    except ValueError:
        return False


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                namespace     TEXT NOT NULL,
                created_at    TEXT NOT NULL
            )
        """
        )
        conn.commit()


class RegisterRequest(BaseModel):
    username: str
    password: str
    namespace: str


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    username: str
    namespace: str
    role: str = "user"


def register_user(username: str, password: str, namespace: str) -> AuthResponse:
    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    if not namespace:
        raise ValueError("A knowledge base namespace is required.")

    pw_hash = _hash_password(password)
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, namespace, created_at)"
                " VALUES (?, ?, ?, ?)",
                (username, pw_hash, namespace, created_at),
            )
            conn.commit()
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"Username '{username}' is already taken.") from exc

    token = _issue_token(username, namespace, "user")
    return AuthResponse(token=token, username=username, namespace=namespace)


def login_user(username: str, password: str) -> AuthResponse:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    if row is None or not _verify_password(password, row["password_hash"]):
        raise ValueError("Invalid username or password.")

    token = _issue_token(row["username"], row["namespace"], "user")
    return AuthResponse(
        token=token,
        username=row["username"],
        namespace=row["namespace"],
    )


def decode_token(token: str) -> dict[str, Any]:
    """Decode and verify a JWT. Raises jwt.PyJWTError on invalid/expired tokens."""
    return jwt.decode(token, _get_secret(), algorithms=[_JWT_ALGORITHM])


def _issue_token(username: str, namespace: str, role: str) -> str:
    payload = {
        "sub": username,
        "namespace": namespace,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=_TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, _get_secret(), algorithm=_JWT_ALGORITHM)
