#!/usr/bin/env python3
"""
SWE-bench Lite v2 — 3-Phase GraphRAG Pipeline + Reflexion.

Phase 1: File Localization   (BM25 + LLM → which file to edit)
Phase 2: Fault Localization  (file content + graph → which function/line is buggy)
Phase 3: Patch Generation    (focused context → unified diff)
Reflexion: evaluate patch → reflect on failure → retry Phase 3 (up to N times)

Target: 30%+ resolved with R2EGym-32B + GraphRAG + Reflexion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neo4j import GraphDatabase
from src_bot.llm.router import get_llm
from src_bot.config.config import configs
from src_bot.reflexion.reflector import generate_reflection
from swebench.neo4j_ingest import (
    ingest_repo_to_neo4j,
    bm25_search,
    hybrid_search,
    graph_expand_simple,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHED_TASKS_PATH = "D:/temp/swebench_lite_tasks.json"
REPOS_DIR = Path("D:/temp/swebench_repos")
INGESTION_CACHE = Path("D:/temp/swebench_ingestion_cache.json")
# INGESTION_CACHE = Path("D:/temp/matplotlib_matplotlib_cache.json")

NEO4J_URI = configs.APP_NEO4J_URL
NEO4J_USER = configs.APP_NEO4J_USER
NEO4J_PASSWORD = configs.APP_NEO4J_PASSWORD

PHASE1_BM25_TOP_K = int(os.getenv("SWEBENCH_PHASE1_BM25_TOP_K", "150"))
PHASE1_CANDIDATE_FILES = int(os.getenv("SWEBENCH_PHASE1_CANDIDATE_FILES", "30"))
PHASE1_FILE_TREE_LIMIT = int(os.getenv("SWEBENCH_PHASE1_FILE_TREE_LIMIT", "500"))
PHASE1_RETURN_FILES = int(os.getenv("SWEBENCH_PHASE1_RETURN_FILES", "3"))
ALT_FILE_MAX_RETRIES = int(os.getenv("SWEBENCH_ALT_FILE_MAX_RETRIES", "1"))


# ---------------------------------------------------------------------------
# Ingestion cache
# ---------------------------------------------------------------------------
def load_ingestion_cache() -> dict:
    if INGESTION_CACHE.exists():
        return json.loads(INGESTION_CACHE.read_text())
    return {}

def save_ingestion_cache(cache: dict):
    INGESTION_CACHE.write_text(json.dumps(cache, indent=2))


def make_project_id(repo_name: str, base_commit: str) -> str:
    """Neo4j graph id for one SWE-bench repository snapshot."""
    repo_key = repo_name.replace("/", "_")
    return f"{repo_key}__{base_commit[:12]}"


def make_ingestion_cache_key(repo_name: str, base_commit: str) -> str:
    """Cache key must include the commit so graph context matches checkout."""
    return f"{repo_name}@{base_commit}"


def to_posix_path(path: str | Path) -> str:
    """Normalize repo-relative paths to POSIX separators for SWE-bench diffs."""
    return str(path).replace("\\", "/")


def make_prediction(
    *,
    instance_id: str,
    repo_name: str,
    base_commit: str,
    project_id: str,
    model_patch: str = "",
    selected_file: str = "",
    attempted_files: list[str] | None = None,
    fault_description: str = "",
    attempts: int = 0,
    failure_stage: str = "",
    failure_reason: str = "",
    apply_error: str = "",
    ingestion_error: str = "",
) -> dict:
    """Build one JSONL prediction row with benchmark-compatible core fields."""
    return {
        "instance_id": instance_id,
        "model_patch": model_patch,
        "model_name_or_path": f"graphrag-reflexion-{configs.LLM_MODEL}",
        "repo": repo_name,
        "base_commit": base_commit,
        "project_id": project_id,
        "selected_file": to_posix_path(selected_file) if selected_file else "",
        "attempted_files": [to_posix_path(f) for f in (attempted_files or [])],
        "fault_description": fault_description[:500],
        "attempts": attempts,
        "status": "patched" if model_patch else "failed",
        "failure_stage": failure_stage,
        "failure_reason": failure_reason[:1000],
        "apply_error": apply_error[:1000],
        "ingestion_error": ingestion_error[:1000],
    }


# ---------------------------------------------------------------------------
# Repo management
# ---------------------------------------------------------------------------
SWEBENCH_REPOS = {
    "django/django": "https://github.com/django/django.git",
    "scikit-learn/scikit-learn": "https://github.com/scikit-learn/scikit-learn.git",
    "sympy/sympy": "https://github.com/sympy/sympy.git",
    "matplotlib/matplotlib": "https://github.com/matplotlib/matplotlib.git",
    "pytest-dev/pytest": "https://github.com/pytest-dev/pytest.git",
    "pallets/flask": "https://github.com/pallets/flask.git",
    "psf/requests": "https://github.com/psf/requests.git",
    "astropy/astropy": "https://github.com/astropy/astropy.git",
    "pydata/xarray": "https://github.com/pydata/xarray.git",
    "sphinx-doc/sphinx": "https://github.com/sphinx-doc/sphinx.git",
    "mwaskom/seaborn": "https://github.com/mwaskom/seaborn.git",
    "pylint-dev/pylint": "https://github.com/pylint-dev/pylint.git",
}

def clone_repo(repo_name: str) -> Path:
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    safe = repo_name.replace("/", "__")
    repo_dir = REPOS_DIR / safe
    if not repo_dir.exists():
        url = SWEBENCH_REPOS.get(repo_name, f"https://github.com/{repo_name}.git")
        print(f"  Cloning {repo_name}...")
        subprocess.run(["git", "clone", "--depth", "1", url, str(repo_dir)],
                       capture_output=True, text=True, timeout=600)
    return repo_dir

def checkout_commit(repo_dir: Path, commit: str):
    subprocess.run(["git", "fetch", "--unshallow"], cwd=repo_dir, capture_output=True, timeout=600)
    subprocess.run(["git", "fetch", "origin", commit], cwd=repo_dir, capture_output=True, timeout=120)
    subprocess.run(["git", "checkout", commit], cwd=repo_dir, capture_output=True, timeout=60)


# ---------------------------------------------------------------------------
# Phase 1: File Localization (BM25 + LLM)
# ---------------------------------------------------------------------------
PHASE1_PROMPT = """You are an expert Python developer. Given a GitHub issue, identify which file(s) need to be modified to fix it.

## Examples

Example 1:
Issue: "BoundWidget.id_for_label ignores id set by ChoiceWidget.options"
Repository: django/django
Answer: django/forms/boundfield.py

Example 2:
Issue: "Modeling's separability_matrix does not compute separability correctly for nested CompoundModels"
Repository: astropy/astropy
Answer: astropy/modeling/separable.py

Example 3:
Issue: "linear_model.RidgeClassifierCV's store_cv_values parameter broken"
Repository: scikit-learn/scikit-learn
Answer: sklearn/linear_model/ridge.py

## Your Task

Issue: {issue_text}

Repository: {repo_name}

## Candidate files from code search:
{candidate_files}

## Repository file structure (source files only):
{file_tree}

Based on the issue description and code search results, which files are most likely to need modification?
Return up to {return_files} file paths, one per line, ranked by likelihood.
Return ONLY file paths, no explanation. Example:
django/forms/widgets.py
django/forms/fields.py"""


def phase1_localize_files(
    llm,
    issue_text: str,
    repo_name: str,
    repo_dir: Path,
    neo4j_session,
    project_id: str,
) -> list[str]:
    """Phase 1: Find likely files to edit, ranked by likelihood."""

    # Hybrid search: BM25 + vector (RRF merge). Falls back to BM25-only if no embeddings.
    seeds = hybrid_search(
        neo4j_session,
        issue_text[:2000],
        project_id,
        top_k=PHASE1_BM25_TOP_K,
        model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )
    file_scores: dict[str, float] = {}
    file_hits: dict[str, list[str]] = {}
    for s in seeds:
        fp = to_posix_path(s.get("file_path", ""))
        if not fp:
            continue
        score = float(s.get("score") or 0.0)
        file_scores[fp] = file_scores.get(fp, 0.0) + score
        hit_name = s.get("qualified_name") or s.get("name") or ""
        hit_type = s.get("type") or "CodeNode"
        if hit_name:
            file_hits.setdefault(fp, []).append(f"{hit_type}: {hit_name}")

    ranked_files = sorted(file_scores, key=file_scores.get, reverse=True)
    candidate_files = []
    for fp in ranked_files[:PHASE1_CANDIDATE_FILES]:
        hits = "; ".join(file_hits.get(fp, [])[:3])
        if hits:
            candidate_files.append(f"  - {fp} (score={file_scores[fp]:.2f}; hits: {hits})")
        else:
            candidate_files.append(f"  - {fp} (score={file_scores[fp]:.2f})")

    # Get repo file tree (source files only, skip tests)
    file_tree = []
    for f in sorted(repo_dir.rglob("*.py")):
        if "__pycache__" in str(f) or "/.git/" in str(f):
            continue
        rel = f.relative_to(repo_dir).as_posix()
        if not rel.startswith("test") and "/tests/" not in rel and not f.name.startswith("test_"):
            file_tree.append(f"  {rel}")

    tree_str = "\n".join(file_tree[:PHASE1_FILE_TREE_LIMIT])
    if len(file_tree) > PHASE1_FILE_TREE_LIMIT:
        tree_str += f"\n  ... ({len(file_tree) - PHASE1_FILE_TREE_LIMIT} more files)"

    prompt = PHASE1_PROMPT.format(
        issue_text=issue_text[:2000],
        repo_name=repo_name,
        candidate_files="\n".join(candidate_files) if candidate_files else "  (no candidates found)",
        file_tree=tree_str,
        return_files=PHASE1_RETURN_FILES,
    )

    try:
        response = llm.invoke(prompt)
        file_paths = _extract_file_paths(response.content, repo_dir)
        if file_paths:
            return file_paths[:PHASE1_RETURN_FILES]
    except Exception as e:
        print(f"    Phase 1 error: {e}")
    # Fallback: use top ranked files from BM25 aggregation.
    return ranked_files[:PHASE1_RETURN_FILES]


def _extract_file_paths(response: str, repo_dir: Path) -> list[str]:
    """Extract valid repo-relative Python paths from an LLM response."""
    response = response.strip()
    paths: list[str] = []
    seen: set[str] = set()

    def _add_path(raw_path: str) -> None:
        path = to_posix_path(raw_path.strip().strip("`").strip("*").strip())
        if not path.endswith(".py"):
            return
        if path in seen:
            return
        if (repo_dir / path).exists():
            seen.add(path)
            paths.append(path)

    # Try each line
    for line in response.split("\n"):
        line = line.strip().strip("`").strip("*").strip()
        if line.endswith(".py"):
            # Remove common prefixes
            line = re.sub(r'^(Answer:|File:|Path:|>\s*)', '', line).strip()
            _add_path(line)
    # Try to find any .py path in the response
    for p in re.findall(r'[\w/\\.-]+\.py', response):
        _add_path(p)
    return paths


# ---------------------------------------------------------------------------
# Phase 2: Fault Localization (file content + graph context)
# ---------------------------------------------------------------------------
PHASE2_PROMPT = """You are an expert Python debugger. Given a GitHub issue and the source file, identify the exact function or class that needs to be fixed.

## Examples

Example 1:
Issue: "BoundWidget.id_for_label ignores id set by ChoiceWidget.options"
File: django/forms/boundfield.py
Answer: BoundWidget.id_for_label (line 277) — the method ignores the 'id' attribute passed via widget options

Example 2:
Issue: "separability_matrix does not compute separability correctly for nested CompoundModels"
File: astropy/modeling/separable.py
Answer: _separable (line 116) — recursive call doesn't handle nested CompoundModels, returns wrong matrix shape

## Your Task

Issue: {issue_text}

File: {file_path}

```python
{file_content}
```

## Related code from knowledge graph:
{graph_context}

Identify the EXACT function/method that has the bug and explain what's wrong in ONE sentence.
Format: function_name (line N) — brief explanation"""


def phase2_localize_fault(
    llm,
    issue_text: str,
    file_path: str,
    repo_dir: Path,
    neo4j_session,
    project_id: str,
) -> tuple[str, str]:
    """Phase 2: Find the buggy function/line within the file. Returns (fault_description, file_content)."""
    file_path = to_posix_path(file_path)

    # Read file content
    full_path = repo_dir / file_path
    try:
        file_content = full_path.read_text(errors="ignore")
    except Exception:
        return "", ""

    # Get graph context for this file's functions
    graph_context = ""
    try:
        results = neo4j_session.run("""
            MATCH (n:CodeNode {file_path: $fp, project_id: $pid})
            OPTIONAL MATCH (caller:CodeNode {project_id: $pid})-[:CALLS]->(n)
            OPTIONAL MATCH (n)-[:CALLS]->(callee:CodeNode {project_id: $pid})
            RETURN n.name AS name, n.type AS type, n.lineno AS lineno,
                   collect(DISTINCT caller.name) AS called_by,
                   collect(DISTINCT callee.name) AS calls
            LIMIT 20
        """, fp=file_path, pid=project_id)

        parts = []
        for r in results:
            line = f"  {r['type']}: {r['name']} (line {r['lineno']})"
            if r["called_by"]:
                line += f" — called by: {', '.join(r['called_by'][:5])}"
            if r["calls"]:
                line += f" — calls: {', '.join(r['calls'][:5])}"
            parts.append(line)
        graph_context = "\n".join(parts) if parts else "  (no graph data)"
    except Exception:
        graph_context = "  (graph unavailable)"

    # ACI: add line numbers so LLM can reference exact lines in its patch
    def _add_line_numbers(code: str, max_chars: int = 8000) -> str:
        lines = code.split("\n")
        numbered = [f"{i+1:4d} | {l}" for i, l in enumerate(lines)]
        result = "\n".join(numbered)
        if len(result) > max_chars:
            # Truncate but keep line numbers intact
            result = result[:max_chars] + "\n ... (truncated)"
        return result

    content_for_prompt = _add_line_numbers(file_content)

    prompt = PHASE2_PROMPT.format(
        issue_text=issue_text[:2000],
        file_path=file_path,
        file_content=content_for_prompt,
        graph_context=graph_context,
    )

    try:
        response = llm.invoke(prompt)
        return response.content.strip(), file_content
    except Exception as e:
        print(f"    Phase 2 error: {e}")
        return "", file_content


# ---------------------------------------------------------------------------
# Phase 3: Patch Generation (focused context)
# ---------------------------------------------------------------------------
PHASE3_PROMPT = """You are a senior Python developer generating a minimal bug fix.

## Examples

Example 1:
Bug: datetime.now() returns wrong timezone in get_current_time()
File: utils/time.py
Fix:
```diff
--- a/utils/time.py
+++ b/utils/time.py
@@ -5,7 +5,7 @@
 from datetime import datetime, timezone

 def get_current_time():
-    return datetime.now()
+    return datetime.now(tz=timezone.utc)
```

Example 2:
Bug: Missing null check in send_notification() causes AttributeError
File: services/notification.py
Fix:
```diff
--- a/services/notification.py
+++ b/services/notification.py
@@ -12,6 +12,8 @@
 def send_notification(user, message):
     email = user.get_email()
+    if email is None:
+        return False
     send_email(email, message)
+    return True
```

## Your Task

Issue: {issue_text}

File to modify: {file_path}

Bug location: {fault_description}

Source code:
```python
{file_content}
```

Generate a MINIMAL unified diff to fix this bug. Change ONLY what is necessary.
Rules:
- The file path MUST be exactly: {file_path}
- Include the full diff header (--- a/... +++ b/...)
- Include correct @@ line numbers
- Include 3 lines of context around each change
- The source code above may be a focused excerpt; generate the diff against the full target file
- Do not copy artificial excerpt comments or line-number prefixes into the patch

Respond ONLY with a ```diff ... ``` block. No explanation.
"""

PHASE3_RETRY_PROMPT = """You are a senior Python developer. Your previous patch FAILED — read the reflection carefully and generate a corrected fix.

## Original Issue
{issue_text}

## File to modify: {file_path}

## Bug location: {fault_description}

## Your Previous Patch (FAILED):
```diff
{failed_patch}
```

## Self-Reflection (why it failed and what to fix):
{reflection}

## Source Code:
```python
{file_content}
```

Generate a corrected MINIMAL unified diff. Address the specific issues identified in the reflection.
Rules:
- The file path MUST be exactly: {file_path}
- Include the full diff header (--- a/... +++ b/...)
- Include correct @@ line numbers
- Include 3 lines of context around each change
- The source code above may be a focused excerpt; generate the diff against the full target file
- Do not copy artificial excerpt comments or line-number prefixes into the patch

Respond ONLY with a ```diff ... ``` block. No explanation.
"""


def _extract_fault_line(fault_description: str) -> int | None:
    match = re.search(r"\bline\s+(\d+)\b", fault_description, re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _focused_file_context(
    file_content: str,
    fault_description: str,
    full_file_limit: int = 12000,
    window: int = 150,
) -> str:
    if len(file_content) <= full_file_limit:
        return file_content

    fault_line = _extract_fault_line(fault_description)
    if not fault_line:
        return file_content[:full_file_limit] + "\n ... (truncated)"

    lines = file_content.splitlines()
    if not lines:
        return file_content

    center = max(1, min(fault_line, len(lines)))
    start = max(1, center - window)
    end = min(len(lines), center + window)
    snippet = "\n".join(lines[start - 1:end])
    return (
        f"# Focused excerpt from full file: lines {start}-{end} of {len(lines)} "
        f"(suspected bug line {center}).\n"
        f"# Generate the unified diff against the full file path, not a new excerpt file.\n"
        f"{snippet}"
    )


def phase3_generate_patch(
    llm,
    issue_text: str,
    file_path: str,
    fault_description: str,
    file_content: str,
    reflection: str = "",
    failed_patch: str = "",
) -> str:
    """Phase 3: Generate the actual patch. Accepts optional reflection for retry attempts."""
    file_path = to_posix_path(file_path)

    content_for_prompt = _focused_file_context(file_content, fault_description)

    if reflection and failed_patch:
        # Retry with reflection context
        prompt = PHASE3_RETRY_PROMPT.format(
            issue_text=issue_text[:2000],
            file_path=file_path,
            fault_description=fault_description[:500],
            failed_patch=failed_patch[:2000],
            reflection=reflection[:1000],
            file_content=content_for_prompt,
        )
    else:
        prompt = PHASE3_PROMPT.format(
            issue_text=issue_text[:2000],
            file_path=file_path,
            fault_description=fault_description[:500],
            file_content=content_for_prompt,
        )

    try:
        response = llm.invoke(prompt)
        patch = _extract_and_clean_diff(response.content, file_path)
        return patch
    except Exception as e:
        print(f"    Phase 3 error: {e}")
        return ""


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def _extract_and_clean_diff(text: str, expected_file: str) -> str:
    """Extract diff from LLM response and fix common issues."""
    expected_file = to_posix_path(expected_file)
    # Extract diff block
    diff = ""
    if "```diff" in text:
        s = text.index("```diff") + 7
        e = text.index("```", s) if "```" in text[s:] else len(text)
        diff = text[s:s+e-s].strip() if s < len(text) else ""
        # Recalculate — find the closing ``` after the opening
        remaining = text[s:]
        if "```" in remaining:
            e2 = remaining.index("```")
            diff = remaining[:e2].strip()
        else:
            diff = remaining.strip()
    elif "--- a/" in text:
        lines = text.split("\n")
        dl, in_diff = [], False
        for line in lines:
            if line.startswith("--- a/") or line.startswith("+++ b/"):
                in_diff = True
            if in_diff:
                dl.append(line)
        diff = "\n".join(dl)

    if not diff:
        return ""

    # Post-processing: fix common issues
    lines = diff.split("\n")
    fixed = []
    in_hunk = False

    for line in lines:
        # Fix file path if LLM used wrong path
        if line.startswith("--- a/") or line.startswith("--- /dev/null"):
            line = f"--- a/{expected_file}"
        elif line.startswith("+++ b/") or line.startswith("+++ /dev/null"):
            line = f"+++ b/{expected_file}"
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk and line == "":
            # Blank lines inside a hunk MUST be a single space (context line).
            # LLMs frequently omit the leading space — git apply rejects these as corrupt.
            line = " "

        # Remove explanation text that snuck outside diff structure
        if line and not line.startswith(("---", "+++", "@@", "+", "-", " ", "\\", "diff", "index", "new file", "old mode", "new mode")):
            if any(c.isalpha() for c in line) and not line.startswith("No newline"):
                continue

        fixed.append(line)

    result = "\n".join(fixed).strip()

    # Ensure diff has proper file header
    if result and not result.startswith("---"):
        result = f"--- a/{expected_file}\n+++ b/{expected_file}\n{result}"

    # Ensure diff ends with newline (git apply requires it)
    if result and not result.endswith("\n"):
        result += "\n"

    return result


def _is_comment_like(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return True

    lowered = stripped.lower()
    docstring_prefixes = (
        '"""', "'''",
        'r"""', "r'''",
        'u"""', "u'''",
        'f"""', "f'''",
        'b"""', "b'''",
        'fr"""', "fr'''",
        'rf"""', "rf'''",
    )
    return any(lowered.startswith(prefix) for prefix in docstring_prefixes)


def _assess_patch_quality(patch: str, expected_file: str) -> tuple[bool, str]:
    """Reject obvious low-quality diffs before spending an attempt on apply."""
    expected_file = to_posix_path(expected_file)
    if not patch or not patch.strip():
        return False, "Empty patch"

    lines = patch.splitlines()
    old_headers = [
        to_posix_path(line[6:].strip()) for line in lines if line.startswith("--- a/")
    ]
    new_headers = [
        to_posix_path(line[6:].strip()) for line in lines if line.startswith("+++ b/")
    ]
    hunk_count = sum(1 for line in lines if line.startswith("@@"))

    if len(old_headers) != 1 or len(new_headers) != 1:
        return False, "Patch must modify exactly one file"
    if old_headers[0] != expected_file or new_headers[0] != expected_file:
        touched_file = new_headers[0] if new_headers else old_headers[0]
        return False, f"Patch modifies {touched_file}, expected {expected_file}"
    if hunk_count == 0:
        return False, "Patch has no unified-diff hunk"

    deleted: list[str] = []
    added: list[str] = []
    in_hunk = False
    for line in lines:
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("-") and not line.startswith("---"):
            deleted.append(line[1:])
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])

    if not deleted and not added:
        return False, "Patch has no changed lines"

    changed = deleted + added
    if any(re.match(r"^\s*\d+\s*\|", line) for line in changed):
        return False, "Patch contains prompt line-number prefixes"

    old_semantic = [line.strip() for line in deleted if line.strip()]
    new_semantic = [line.strip() for line in added if line.strip()]
    if old_semantic == new_semantic:
        return False, "Patch is a no-op after whitespace normalization"

    changed_nonblank = [line for line in changed if line.strip()]
    if changed_nonblank and all(_is_comment_like(line) for line in changed_nonblank):
        return False, "Patch only changes comments or docstrings"

    statement_start = re.compile(
        r"^(def|class|if|elif|for|while|except|with|return|raise|from|import|"
        r"pass|break|continue|yield|assert)\b|^(else|try|finally):"
    )
    for line in added:
        if not line.startswith(" "):
            continue
        stripped = line.lstrip(" ")
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(stripped)
        if indent % 4 != 0 and statement_start.match(stripped):
            return False, f"Suspicious indentation on added line: {stripped[:80]}"

    return True, ""


# ---------------------------------------------------------------------------
# Patch normalization — rebuild correct line numbers from actual file
# ---------------------------------------------------------------------------
def _normalize_patch(patch: str, repo_dir: Path) -> str:
    """
    LLMs often generate correct logic but wrong context / line numbers.
    This function:
      1. Parses each hunk to extract the deletion lines (what to replace)
      2. Fuzzy-searches the actual file for those lines
      3. Rebuilds a clean unified diff with correct @@ numbers and 3 lines of context
    Returns the normalized patch string, or the original if normalization fails.
    """
    import difflib

    # Parse the file path from the patch
    file_path = None
    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            file_path = to_posix_path(line[6:].strip())
            break
    if not file_path:
        return patch

    actual_file = repo_dir / file_path
    if not actual_file.exists():
        return patch

    try:
        actual_lines = actual_file.read_text(errors="replace").split("\n")
    except Exception:
        return patch

    # Parse hunks: collect (del_lines, add_lines) per hunk
    hunks = []
    current_del, current_add = [], []
    in_hunk = False
    for line in patch.split("\n"):
        if line.startswith("@@"):
            if in_hunk and (current_del or current_add):
                hunks.append((current_del[:], current_add[:]))
            current_del, current_add = [], []
            in_hunk = True
        elif in_hunk:
            if line.startswith("-"):
                current_del.append(line[1:])
            elif line.startswith("+"):
                current_add.append(line[1:])
            # context lines ignored for matching
    if in_hunk and (current_del or current_add):
        hunks.append((current_del, current_add))

    if not hunks:
        return patch

    # For each hunk, find the best match in the actual file
    new_content = actual_lines[:]
    offset = 0  # track line shift from previous hunks
    rebuilt_hunks = []

    for del_lines, add_lines in hunks:
        if not del_lines:
            continue  # pure insertion — skip normalization for now

        # Find best match using SequenceMatcher
        del_text = "\n".join(del_lines)
        best_score, best_start = 0.0, -1
        for i in range(len(new_content) - len(del_lines) + 1):
            window = "\n".join(new_content[i:i + len(del_lines)])
            score = difflib.SequenceMatcher(None, del_text, window).ratio()
            if score > best_score:
                best_score, best_start = score, i

        if best_score < 0.6 or best_start < 0:
            continue  # can't find a good match

        # Rebuild unified diff hunk with correct numbers
        ctx = 3
        hunk_start = max(0, best_start - ctx)
        hunk_end = min(len(new_content), best_start + len(del_lines) + ctx)

        before = new_content[hunk_start:best_start]
        # Use ACTUAL file lines for deletions (not the LLM's possibly-wrong indentation)
        actual_del_lines = new_content[best_start:best_start + len(del_lines)]
        after = new_content[best_start + len(del_lines):hunk_end]

        old_count = len(before) + len(actual_del_lines) + len(after)
        new_count = len(before) + len(add_lines) + len(after)
        hunk_header = f"@@ -{hunk_start+1},{old_count} +{hunk_start+1},{new_count} @@"

        hunk_lines = [hunk_header]
        for l in before:
            hunk_lines.append(f" {l}")
        for l in actual_del_lines:
            hunk_lines.append(f"-{l}")
        for l in add_lines:
            hunk_lines.append(f"+{l}")
        for l in after:
            hunk_lines.append(f" {l}")

        rebuilt_hunks.append("\n".join(hunk_lines))

        # Apply the change to new_content for subsequent hunks
        new_content[best_start:best_start + len(del_lines)] = add_lines

    if not rebuilt_hunks:
        return patch

    normalized = f"--- a/{file_path}\n+++ b/{file_path}\n" + "\n".join(rebuilt_hunks) + "\n"
    return normalized


# ---------------------------------------------------------------------------
# Reflexion helpers
# ---------------------------------------------------------------------------
def _verify_patch_applies(patch: str, repo_dir: Path) -> tuple[bool, str]:
    """
    Quick check: does the patch apply cleanly to the repo?
    Uses `git apply --check` — no side effects.
    Returns (applies_ok, error_message).
    """
    if not patch or not patch.strip():
        return False, "Empty patch"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(patch)
        patch_file = f.name

    try:
        # Try git apply first (strict)
        result = subprocess.run(
            ["git", "apply", "--check", "--recount", patch_file],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True, ""

        # Fallback: patch -p1 with fuzzy matching if GNU patch is available.
        # On Windows, missing `patch` raises WinError 2 and hides git's real error.
        if shutil.which("patch"):
            result2 = subprocess.run(
                ["patch", "-p1", "--dry-run", "--fuzz=3", "-i", patch_file],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result2.returncode == 0:
                return True, ""

        # Return the git apply error (more descriptive)
        return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)
    finally:
        Path(patch_file).unlink(missing_ok=True)


def _get_graph_context_for_file(neo4j_session, file_path: str, project_id: str) -> str:
    """Get a concise graph context string for the reflector."""
    file_path = to_posix_path(file_path)
    try:
        results = neo4j_session.run("""
            MATCH (n:CodeNode {file_path: $fp, project_id: $pid})
            OPTIONAL MATCH (caller:CodeNode {project_id: $pid})-[:CALLS]->(n)
            OPTIONAL MATCH (n)-[:CALLS]->(callee:CodeNode {project_id: $pid})
            RETURN n.name AS name, n.type AS type, n.lineno AS lineno,
                   collect(DISTINCT caller.name) AS called_by,
                   collect(DISTINCT callee.name) AS calls
            LIMIT 15
        """, fp=file_path, pid=project_id)
        parts = []
        for r in results:
            line = f"  {r['type']}: {r['name']} (line {r['lineno']})"
            if r["called_by"]:
                line += f" ← called by: {', '.join(r['called_by'][:4])}"
            if r["calls"]:
                line += f" → calls: {', '.join(r['calls'][:4])}"
            parts.append(line)
        return "\n".join(parts) if parts else "No graph data available."
    except Exception:
        return "Graph unavailable."


def _try_patch_for_file(
    *,
    llm,
    issue_text: str,
    file_path: str,
    repo_dir: Path,
    neo4j_session,
    neo4j_driver,
    project_id: str,
    reflexion: bool,
    max_retries: int,
    file_attempt_index: int,
) -> dict:
    file_path = to_posix_path(file_path)
    max_retries = max(1, max_retries)
    print(f"  Trying file #{file_attempt_index}: {file_path} (max {max_retries} attempts)")

    fault_desc, file_content = phase2_localize_fault(
        llm, issue_text, file_path, repo_dir, neo4j_session, project_id
    )
    print(f"  P2 Fault: {fault_desc[:80] if fault_desc else '✗'}")

    if not file_content:
        return {
            "model_patch": "",
            "selected_file": file_path,
            "fault_description": fault_desc,
            "attempts": 0,
            "failure_stage": "phase2_no_file_content",
            "failure_reason": "Phase 2 could not read selected file content",
            "apply_error": "",
        }

    graph_context = _get_graph_context_for_file(neo4j_session, file_path, project_id)
    reflections: list[str] = []
    patch = ""
    last_patch = ""
    last_error = ""
    failure_stage = ""
    failure_reason = ""
    apply_error_message = ""
    attempts = 0

    for attempt in range(1, max_retries + 1):
        attempts = attempt
        if attempt == 1:
            patch = phase3_generate_patch(
                llm, issue_text, file_path, fault_desc, file_content
            )
        else:
            print(f"  Reflexion attempt {attempt}/{max_retries}...")
            try:
                reflection = generate_reflection(
                    llm=llm,
                    bug_description=f"{issue_text[:500]}\n\nFault: {fault_desc}",
                    failed_patch=last_patch,
                    test_output=last_error,
                    graph_context=graph_context,
                    previous_reflections=reflections,
                )
                reflections.append(reflection)
                print(f"    Reflection: {reflection[:100]}...")
            except Exception as e:
                print(f"    Reflection error: {e}")
                reflection = f"Previous patch failed to apply: {last_error}"
                reflections.append(reflection)

            patch = phase3_generate_patch(
                llm, issue_text, file_path, fault_desc, file_content,
                reflection=reflection,
                failed_patch=last_patch,
            )

        if not patch:
            print(f"  P3 Patch: ✗ (empty, attempt {attempt})")
            last_error = "Patch generation produced empty output"
            failure_stage = "patch_empty"
            failure_reason = last_error
            last_patch = ""
            continue

        normalized = _normalize_patch(patch, repo_dir)
        if normalized != patch:
            print("  Patch normalized (line numbers corrected)")

        quality_ok, quality_error = _assess_patch_quality(normalized, file_path)
        if not quality_ok:
            print(f"  P3 Patch: ✗ quality failed (attempt {attempt}): {quality_error}")
            last_patch = normalized
            last_error = f"Patch quality failed: {quality_error}"
            failure_stage = "patch_quality_failed"
            failure_reason = quality_error
            patch = ""
            continue

        applies_ok, apply_error = _verify_patch_applies(normalized, repo_dir)

        if applies_ok:
            patch = normalized

            if reflexion:
                try:
                    from api.agent.evaluator import evaluate_patch
                    eval_score, eval_reason = evaluate_patch(
                        bug_description=f"{issue_text[:300]}\nFault: {fault_desc}",
                        patch=patch,
                    )
                    print(f"  Evaluator: {eval_score}/5 — {eval_reason}")
                    if eval_score < 3:
                        print(f"  Evaluator rejected patch (score {eval_score})")
                        last_patch = patch
                        last_error = f"Evaluator feedback: {eval_reason}"
                        failure_stage = "evaluator_rejected"
                        failure_reason = eval_reason
                        patch = ""
                        if attempt < max_retries:
                            continue
                        break
                except Exception:
                    pass

            print(f"  P3 Patch: ✓ {len(patch)} chars (attempt {attempt}, applies cleanly)")
            return {
                "model_patch": patch,
                "selected_file": file_path,
                "fault_description": fault_desc,
                "attempts": attempts,
                "failure_stage": "",
                "failure_reason": "",
                "apply_error": "",
            }

        print(f"  P3 Patch: ✗ apply failed (attempt {attempt}): {apply_error[:80]}")
        patch_preview = "\n".join(normalized.split("\n")[:10])
        print(f"  Patch preview:\n{patch_preview}")

        if last_patch and len(patch) < len(last_patch) * 0.6:
            print("  Frustration detected (patch shrinking) — widening context")
            try:
                with neo4j_driver.session() as s:
                    extra = _get_graph_context_for_file(s, file_path, project_id)
                graph_context = extra + "\n" + graph_context
            except Exception:
                pass

        last_patch = normalized
        last_error = f"git apply --check failed: {apply_error}"
        failure_stage = "patch_apply_failed"
        failure_reason = apply_error
        apply_error_message = apply_error
        patch = ""

    if not failure_stage:
        failure_stage = "patch_not_generated"
        failure_reason = last_error or "Patch loop ended without an accepted patch"

    return {
        "model_patch": "",
        "selected_file": file_path,
        "fault_description": fault_desc,
        "attempts": attempts,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "apply_error": apply_error_message,
    }


# ---------------------------------------------------------------------------
# Main task runner
# ---------------------------------------------------------------------------
def run_single_task(
    task: dict,
    llm,
    neo4j_driver,
    ingestion_cache: dict,
    reflexion: bool = True,
) -> dict:
    instance_id = task["instance_id"]
    repo_name = task["repo"]
    base_commit = task["base_commit"]
    issue_text = task["problem_statement"]
    hints = task.get("hints_text", "")
    if hints:
        issue_text += f"\n\nHints:\n{hints}"

    project_id = make_project_id(repo_name, base_commit)
    cache_key = make_ingestion_cache_key(repo_name, base_commit)
    ingestion_error = ""
    max_retries = configs.REFLEXION_MAX_RETRIES if reflexion else 1  # 1 = no retries

    # Clone + checkout
    repo_dir = clone_repo(repo_name)
    checkout_commit(repo_dir, base_commit)

    # Ingest if needed
    if not ingestion_cache.get(cache_key, {}).get("ingested"):
        print(f"  Ingesting {repo_name}@{base_commit[:12]}...")
        try:
            stats = ingest_repo_to_neo4j(
                repo_dir, project_id,
                NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
                max_files=1000,
            )
            ingestion_cache[cache_key] = {
                "ingested": True,
                "repo": repo_name,
                "base_commit": base_commit,
                "project_id": project_id,
                **stats,
            }
            save_ingestion_cache(ingestion_cache)
        except Exception as e:
            ingestion_error = str(e)
            print(f"  Ingestion failed: {e}")

    with neo4j_driver.session() as session:
        file_candidates = phase1_localize_files(
            llm, issue_text, repo_name, repo_dir, session, project_id
        )
        file_candidates = [to_posix_path(f) for f in file_candidates if f]
        print(f"  P1 Files: {', '.join(file_candidates) if file_candidates else '✗'}")

        if not file_candidates:
            return make_prediction(
                instance_id=instance_id,
                repo_name=repo_name,
                base_commit=base_commit,
                project_id=project_id,
                attempted_files=[],
                failure_stage="phase1_no_file",
                failure_reason="Phase 1 did not return an existing Python file",
                ingestion_error=ingestion_error,
            )

        attempted_files: list[str] = []
        total_attempts = 0
        last_result: dict | None = None

        for file_index, file_path in enumerate(file_candidates, 1):
            attempted_files.append(file_path)
            file_max_retries = max_retries if file_index == 1 else ALT_FILE_MAX_RETRIES
            if not reflexion:
                file_max_retries = 1

            result = _try_patch_for_file(
                llm=llm,
                issue_text=issue_text,
                file_path=file_path,
                repo_dir=repo_dir,
                neo4j_session=session,
                neo4j_driver=neo4j_driver,
                project_id=project_id,
                reflexion=reflexion,
                max_retries=file_max_retries,
                file_attempt_index=file_index,
            )
            last_result = result
            total_attempts += int(result.get("attempts") or 0)

            if result.get("model_patch"):
                return make_prediction(
                    instance_id=instance_id,
                    repo_name=repo_name,
                    base_commit=base_commit,
                    project_id=project_id,
                    model_patch=result["model_patch"],
                    selected_file=result.get("selected_file", file_path),
                    attempted_files=attempted_files,
                    fault_description=result.get("fault_description", ""),
                    attempts=total_attempts,
                    ingestion_error=ingestion_error,
                )

            print(
                f"  File #{file_index} failed: "
                f"{result.get('failure_stage') or 'unknown'} — "
                f"{str(result.get('failure_reason') or '')[:120]}"
            )

        last_result = last_result or {}
        return make_prediction(
            instance_id=instance_id,
            repo_name=repo_name,
            base_commit=base_commit,
            project_id=project_id,
            selected_file=last_result.get("selected_file", attempted_files[-1]),
            attempted_files=attempted_files,
            fault_description=last_result.get("fault_description", ""),
            attempts=total_attempts,
            failure_stage=last_result.get("failure_stage") or "patch_not_generated",
            failure_reason=last_result.get("failure_reason") or (
                "All candidate files failed to produce an accepted patch"
            ),
            apply_error=last_result.get("apply_error", ""),
            ingestion_error=ingestion_error,
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SWE-bench v2 — 3-Phase GraphRAG")
    parser.add_argument("--output", "-o", default="predictions_v2.jsonl")
    parser.add_argument("--limit", "-n", type=int, default=None)
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-reflexion", action="store_true", help="Single attempt only — saves ~3x tokens")
    args = parser.parse_args()

    if not os.path.exists(CACHED_TASKS_PATH):
        print("SWE-bench Lite cache not found — downloading from HuggingFace...")
        try:
            from datasets import load_dataset
            ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
            tasks_raw = [dict(r) for r in ds]
            with open(CACHED_TASKS_PATH, "w") as f:
                json.dump(tasks_raw, f)
            print(f"Downloaded and cached {len(tasks_raw)} SWE-bench Lite tasks → {CACHED_TASKS_PATH}")
        except Exception as e:
            print(f"ERROR: Could not download SWE-bench Lite: {e}")
            print("Install with: python3.11 -m pip install datasets")
            sys.exit(1)

    tasks = json.loads(Path(CACHED_TASKS_PATH).read_text())
    print(f"Loaded {len(tasks)} SWE-bench Lite tasks")

    if args.instance_id:
        tasks = [t for t in tasks if t["instance_id"] == args.instance_id]
    if args.limit:
        tasks = tasks[:args.limit]

    # Resume
    done_ids, existing = set(), []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    done_ids.add(p["instance_id"])
                    existing.append(p)
        tasks = [t for t in tasks if t["instance_id"] not in done_ids]
        print(f"Resuming: {len(done_ids)} done, {len(tasks)} remaining")

    use_reflexion = not args.no_reflexion
    mode = "3-phase, no reflexion" if not use_reflexion else f"3-phase + reflexion (max {configs.REFLEXION_MAX_RETRIES} retries)"
    print(f"Running {len(tasks)} tasks ({mode})")

    llm = get_llm(provider=configs.LLM_PROVIDER, model=configs.LLM_MODEL, temperature=0)
    print(f"LLM: {configs.LLM_PROVIDER}/{configs.LLM_MODEL}")

    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    print(f"Neo4j: connected")

    ingestion_cache = load_ingestion_cache()
    predictions = list(existing)
    start = time.time()
    failure_counts = Counter(
        p.get("failure_stage") or "unknown_failure"
        for p in predictions
        if not p.get("model_patch")
    )

    for i, task in enumerate(tasks):
        t0 = time.time()
        print(f"\n[{i+1+len(done_ids)}/{len(tasks)+len(done_ids)}] {task['instance_id']}")

        try:
            pred = run_single_task(task, llm, neo4j_driver, ingestion_cache, reflexion=use_reflexion)
            predictions.append(pred)
            if not pred["model_patch"]:
                failure_counts[pred.get("failure_stage") or "unknown_failure"] += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            pred = make_prediction(
                instance_id=task["instance_id"],
                repo_name=task["repo"],
                base_commit=task["base_commit"],
                project_id=make_project_id(task["repo"], task["base_commit"]),
                failure_stage="exception",
                failure_reason=str(e),
            )
            predictions.append(pred)
            failure_counts["exception"] += 1

        print(f"  ({time.time()-t0:.1f}s)")

        # Save after each
        with open(args.output, "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            patched = sum(1 for p in predictions if p["model_patch"])
            failures = sum(failure_counts.values())
            top_failures = ", ".join(
                f"{stage}:{count}" for stage, count in failure_counts.most_common(3)
            ) or "none"
            print(
                f"\n  ── {patched}/{i+1+len(done_ids)} patched | "
                f"{failures} failed | {top_failures} ──"
            )

    neo4j_driver.close()
    elapsed = time.time() - start
    patched = sum(1 for p in predictions if p["model_patch"])
    total = len(predictions)

    print(f"\n{'='*60}")
    print(f"DONE: {total} tasks in {elapsed/60:.1f}min")
    print(f"Patches: {patched}/{total} ({100*patched/total:.1f}%)")
    failures = total - patched
    if failures:
        print("Failure stages:")
        for stage, count in failure_counts.most_common():
            print(f"  {stage}: {count}")

    # Compare file targeting with gold
    tasks_map = {t["instance_id"]: t for t in json.loads(Path(CACHED_TASKS_PATH).read_text())}
    right_file = 0
    for p in predictions:
        if not p["model_patch"]: continue
        gold = tasks_map.get(p["instance_id"], {}).get("patch", "")
        gold_files = {l[6:].strip() for l in gold.split("\n") if l.startswith("--- a/")}
        pred_files = {l[6:].strip() for l in p["model_patch"].split("\n") if l.startswith("--- a/")}
        if gold_files & pred_files:
            right_file += 1
    print(f"Right file: {right_file}/{patched} ({100*right_file//max(patched,1)}%)")


if __name__ == "__main__":
    main()
