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


def _token_from_system() -> str | None:
    """
    Discover a GitHub token from local dev tooling — same credentials git uses.

    Resolution order:
      1. `gh auth token` — GitHub CLI (if installed anywhere in common paths)
      2. `git credential fill` — reads macOS Keychain / git-credential-manager,
         which is what `git clone` uses. Works even without gh CLI.

    Returns None if neither source has credentials.
    """
    import subprocess
    import shutil

    # 1. Try gh CLI (check common install locations beyond default PATH)
    gh_candidates = [
        "gh",
        "/opt/homebrew/bin/gh",
        "/usr/local/bin/gh",
        os.path.expanduser("~/.local/bin/gh"),
    ]
    for gh_bin in gh_candidates:
        if gh_bin != "gh" and not os.path.isfile(gh_bin):
            continue
        try:
            result = subprocess.run(
                [gh_bin, "auth", "token"],
                capture_output=True, text=True, timeout=5,
            )
            token = result.stdout.strip()
            if result.returncode == 0 and token and len(token) > 10:
                return token
        except Exception:
            continue

    # 2. Fall back to git credential fill — reads macOS Keychain /
    #    git-credential-manager / any configured credential helper.
    #    This is exactly what `git clone https://github.com/...` uses.
    try:
        result = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n",
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("password="):
                    token = line[len("password="):].strip()
                    if token and len(token) > 10:
                        return token
    except Exception:
        pass

    return None


def get_token_for_repo(repo_full_name: str) -> str | None:
    """
    Get a GitHub access token for a repo, or None if no token is configured.

    Resolution order:
      1. GitHub App installation token (production)
      2. GITHUB_TOKEN env var (personal token)
      3. `gh auth token` — GitHub CLI cached credentials (dev convenience)
      4. None — anonymous API access (public repos only, 60 req/hour)

    Resolution 3 lets the system work for private repos when the user has
    `gh` CLI installed and logged in (same credentials git clone uses via
    the gh credential helper), without requiring a manual token in .env.
    """
    app_id = os.getenv("GITHUB_APP_ID", "")
    if not app_id:
        token = os.getenv("GITHUB_TOKEN", "").strip()
        if token and token != "ghp_your_token_here":
            return token
        # Fall back to gh CLI / macOS Keychain (same credentials git uses)
        return _token_from_system()

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


# ---------------------------------------------------------------------------
# PR review comment + fix branch (opt-in via config flags)
# ---------------------------------------------------------------------------

def format_pr_review_comment(
    file_reviews: list,
    pr_number: int,
    repo_name: str,
    patches_generated: int,
) -> str:
    """Format file_reviews as a markdown comment body for a GitHub PR."""
    RISK_LABEL = {"low": "[LOW]", "medium": "[MEDIUM]", "high": "[HIGH]"}

    lines: list[str] = [
        f"## Code Review Bot — PR #{pr_number}",
        "",
        f"**{patches_generated}** patch(es) generated for `{repo_name}`.",
        "",
    ]

    for fr in file_reviews:
        risk = RISK_LABEL.get(fr.risk_level, "[UNKNOWN]")
        lines.append(f"### `{fr.file_path}` {risk}")

        if fr.phase2_fault:
            lines.append(f"**Fault:** {fr.phase2_fault}")

        if fr.eval_score is not None:
            lines.append(f"**Quality score:** {fr.eval_score}/5 — {fr.eval_reason[:120]}")

        if fr.patch and fr.applies_cleanly:
            lines.extend(["", "```diff", fr.patch[:3000], "```"])
        elif fr.patch:
            lines.append("")
            lines.append("> Patch generated but did not apply cleanly to the base commit.")

        lines.append("")

    lines.extend([
        "---",
        "*Generated by CodeReviewBot — GraphRAG-powered code analysis*",
    ])
    return "\n".join(lines)


def post_pr_review_comment(repo, pr, comment_body: str) -> bool:
    """
    Post `comment_body` as an issue comment on the given PR.
    Returns True on success, False on failure.
    `repo` and `pr` are PyGitHub objects (github.Repository.Repository,
    github.PullRequest.PullRequest).
    """
    import logging
    _log = logging.getLogger(__name__)
    try:
        pr.create_issue_comment(comment_body)
        _log.info("Posted review comment on %s#%s", repo.full_name, pr.number)
        return True
    except Exception as e:
        _log.warning("Failed to post PR review comment: %s", e)
        return False


def create_fix_branch_pr(
    repo,
    pr,
    file_reviews: list,
    workdir,
    github_token: str,
) -> "str | None":
    """
    Create a fix branch from the PR's base SHA, apply clean patches, push, and
    open a new PR against the PR's base branch.

    Returns the URL of the created PR, or None if nothing was pushed.

    Requires `github_token` to have write/push access to the repo (repo scope
    for personal tokens, or push permission for GitHub App tokens).
    """
    import logging
    import subprocess
    import tempfile
    import textwrap
    from pathlib import Path

    _log = logging.getLogger(__name__)

    clean_patches = [fr for fr in file_reviews if fr.patch and fr.applies_cleanly]
    if not clean_patches:
        _log.info("create_fix_branch_pr: no clean patches, skipping")
        return None

    workdir = Path(workdir)
    branch_name = f"bot/fix-pr{pr.number}"

    # Configure a minimal git identity inside the workdir so the commit works
    subprocess.run(
        ["git", "config", "user.name", "CodeReviewBot"],
        cwd=workdir, capture_output=True, timeout=10,
    )
    subprocess.run(
        ["git", "config", "user.email", "bot@codereviewbot.local"],
        cwd=workdir, capture_output=True, timeout=10,
    )

    # Create the fix branch at the current HEAD (= PR base SHA)
    result = subprocess.run(
        ["git", "checkout", "-b", branch_name],
        cwd=workdir, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        _log.warning("create_fix_branch_pr: branch creation failed: %s", result.stderr)
        return None

    applied_files: list[str] = []
    for fr in clean_patches:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".patch", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(fr.patch)
            tmp_path = tmp.name

        apply_result = subprocess.run(
            ["git", "apply", "--index", tmp_path],
            cwd=workdir, capture_output=True, text=True, timeout=60,
        )
        Path(tmp_path).unlink(missing_ok=True)

        if apply_result.returncode == 0:
            applied_files.append(fr.file_path)
        else:
            _log.warning(
                "create_fix_branch_pr: patch for %s did not apply: %s",
                fr.file_path, apply_result.stderr[:200],
            )

    if not applied_files:
        _log.warning("create_fix_branch_pr: no patches applied cleanly")
        return None

    file_list = ", ".join(f"`{f}`" for f in applied_files)
    commit_msg = textwrap.dedent(f"""\
        fix: automated patches for PR #{pr.number} [{", ".join(applied_files)}]

        Generated by CodeReviewBot. Patches apply cleanly to base commit {pr.base.sha[:7]}.
        Reviewed files: {file_list}
    """)

    commit_result = subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=workdir, capture_output=True, text=True, timeout=30,
    )
    if commit_result.returncode != 0:
        _log.warning("create_fix_branch_pr: commit failed: %s", commit_result.stderr)
        return None

    # Push with embedded token so git doesn't prompt for credentials
    auth_remote = f"https://x-access-token:{github_token}@github.com/{repo.full_name}.git"
    push_result = subprocess.run(
        ["git", "push", auth_remote, f"{branch_name}:{branch_name}"],
        cwd=workdir, capture_output=True, text=True, timeout=120,
    )
    if push_result.returncode != 0:
        _log.warning(
            "create_fix_branch_pr: push failed (token may lack write access): %s",
            push_result.stderr[:300],
        )
        return None

    # Create the PR on GitHub
    pr_title = f"fix: automated patches from code review of PR #{pr.number}"
    pr_body = (
        f"Automated fix branch generated by CodeReviewBot.\n\n"
        f"**Source PR:** #{pr.number} — {pr.title}\n"
        f"**Applied patches:** {file_list}\n\n"
        f"These patches were generated by the GraphRAG-powered review pipeline "
        f"and apply cleanly to the base commit `{pr.base.sha[:7]}`.\n\n"
        f"Please review the changes carefully before merging."
    )
    try:
        fix_pr = repo.create_pull(
            title=pr_title,
            body=pr_body,
            head=branch_name,
            base=pr.base.ref,
        )
        _log.info("Created fix PR: %s", fix_pr.html_url)
        return fix_pr.html_url
    except Exception as e:
        _log.warning("create_fix_branch_pr: PR creation failed: %s", e)
        return None


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
