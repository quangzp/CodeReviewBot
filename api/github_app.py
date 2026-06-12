"""
GitHub App authentication helpers.

Handles:
- App JWT generation (to call GitHub's App API)
- Installation token retrieval (to call GitHub API on behalf of an installation)
- Webhook signature verification (HMAC-SHA256)
- OAuth token exchange (web UI login)
"""
from __future__ import annotations

import hashlib
import hmac
import time
import os
import requests
from functools import lru_cache
from pathlib import Path


# ---------------------------------------------------------------------------
# Config (loaded from env / .env)
# ---------------------------------------------------------------------------
def _get_app_id() -> str:
    v = os.getenv("GITHUB_APP_ID", "")
    if not v:
        raise ValueError("GITHUB_APP_ID is not set in .env")
    return v


def _get_private_key() -> str:
    """Load RSA private key from file path or inline env var."""
    path = os.getenv("GITHUB_APP_PRIVATE_KEY_PATH", "github-app.pem")
    key_inline = os.getenv("GITHUB_APP_PRIVATE_KEY", "")
    if key_inline:
        return key_inline.replace("\\n", "\n")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"GitHub App private key not found at '{path}'. "
            "Download it from your GitHub App settings and place it here, "
            "or set GITHUB_APP_PRIVATE_KEY in .env."
        )
    return p.read_text()


def _get_webhook_secret() -> str:
    return os.getenv("GITHUB_APP_WEBHOOK_SECRET", "")


def _get_oauth_client_id() -> str:
    return os.getenv("GITHUB_OAUTH_CLIENT_ID", "")


def _get_oauth_client_secret() -> str:
    return os.getenv("GITHUB_OAUTH_CLIENT_SECRET", "")


# ---------------------------------------------------------------------------
# App JWT (short-lived, used to get installation tokens)
# ---------------------------------------------------------------------------
def generate_app_jwt() -> str:
    """
    Generate a signed JWT for the GitHub App.
    Valid for 10 minutes. Used only to get installation tokens.
    """
    try:
        import jwt as pyjwt
    except ImportError:
        raise ImportError("Install PyJWT: pip install PyJWT cryptography")

    app_id = _get_app_id()
    private_key = _get_private_key()
    now = int(time.time())
    payload = {
        "iat": now - 60,   # issued 60s ago (handles clock skew)
        "exp": now + 600,  # valid for 10 minutes
        "iss": app_id,
    }
    return pyjwt.encode(payload, private_key, algorithm="RS256")


# ---------------------------------------------------------------------------
# Installation token (used for all GitHub API calls on a repo)
# ---------------------------------------------------------------------------
_token_cache: dict[int, tuple[str, float]] = {}  # {installation_id: (token, expires_at)}


def get_installation_token(installation_id: int) -> str:
    """
    Get (or refresh) a GitHub App installation access token.
    Tokens are cached and reused until 5 minutes before expiry.
    """
    cached = _token_cache.get(installation_id)
    if cached:
        token, expires_at = cached
        if time.time() < expires_at - 300:  # 5min buffer
            return token

    app_jwt = generate_app_jwt()
    resp = requests.post(
        f"https://api.github.com/app/installations/{installation_id}/access_tokens",
        headers={
            "Authorization": f"Bearer {app_jwt}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    token = data["token"]

    # Parse expiry (e.g. "2025-01-01T00:00:00Z")
    from datetime import datetime, timezone
    expires_str = data.get("expires_at", "")
    try:
        expires_at = datetime.fromisoformat(expires_str.replace("Z", "+00:00")).timestamp()
    except Exception:
        expires_at = time.time() + 3600  # default 1h

    _token_cache[installation_id] = (token, expires_at)
    return token


def get_installation_id_for_repo(repo_full_name: str) -> int:
    """
    Look up the installation ID for a given repo (owner/name).
    Requires App JWT.
    """
    app_jwt = generate_app_jwt()
    resp = requests.get(
        f"https://api.github.com/repos/{repo_full_name}/installation",
        headers={
            "Authorization": f"Bearer {app_jwt}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def get_token_for_repo(repo_full_name: str) -> str | None:
    """
    Get a GitHub access token for a repo, or None if no token is configured.

    Resolution order:
      1. GitHub App installation token (production)
      2. GITHUB_TOKEN env var (personal token)
      3. None — caller should fall back to anonymous API access

    Anonymous access works for public repos but is rate-limited to 60 req/hour
    per IP. Private repos always require a token.
    """
    app_id = os.getenv("GITHUB_APP_ID", "")
    if not app_id:
        token = os.getenv("GITHUB_TOKEN", "").strip()
        # Reject the .env.example placeholder so users get a clear "anonymous mode" message
        if not token or token == "ghp_your_token_here":
            return None
        return token

    try:
        installation_id = get_installation_id_for_repo(repo_full_name)
        return get_installation_token(installation_id)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Webhook signature verification
# ---------------------------------------------------------------------------
def verify_webhook_signature(payload_bytes: bytes, signature_header: str) -> bool:
    """
    Verify the X-Hub-Signature-256 header from GitHub.
    Returns True if valid, False otherwise.
    Skips verification if GITHUB_APP_WEBHOOK_SECRET is not set (dev mode).
    """
    secret = _get_webhook_secret()
    if not secret:
        return True  # dev mode: skip verification

    if not signature_header or not signature_header.startswith("sha256="):
        return False

    expected = hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()

    received = signature_header[len("sha256="):]
    return hmac.compare_digest(expected, received)


# ---------------------------------------------------------------------------
# GitHub OAuth (web UI login)
# ---------------------------------------------------------------------------
def get_oauth_authorize_url(state: str) -> str:
    """Build the GitHub OAuth authorization URL."""
    client_id = _get_oauth_client_id()
    if not client_id:
        raise ValueError("GITHUB_OAUTH_CLIENT_ID is not set in .env")
    return (
        f"https://github.com/login/oauth/authorize"
        f"?client_id={client_id}"
        f"&scope=read:user"
        f"&state={state}"
    )


def exchange_oauth_code(code: str) -> dict:
    """
    Exchange an OAuth authorization code for a user access token.
    Returns the GitHub user info dict.
    """
    client_id = _get_oauth_client_id()
    client_secret = _get_oauth_client_secret()

    # Exchange code for token
    resp = requests.post(
        "https://github.com/login/oauth/access_token",
        json={"client_id": client_id, "client_secret": client_secret, "code": code},
        headers={"Accept": "application/json"},
        timeout=10,
    )
    resp.raise_for_status()
    token_data = resp.json()
    access_token = token_data.get("access_token")
    if not access_token:
        raise ValueError(f"OAuth token exchange failed: {token_data}")

    # Fetch user info
    user_resp = requests.get(
        "https://api.github.com/user",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=10,
    )
    user_resp.raise_for_status()
    user = user_resp.json()
    user["access_token"] = access_token
    return user
