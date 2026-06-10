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
import json
import os
import re
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
    graph_expand_simple,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHED_TASKS_PATH = "/tmp/swebench_lite_tasks.json"
REPOS_DIR = Path("/tmp/swebench_repos")
INGESTION_CACHE = Path("/tmp/swebench_ingestion_cache.json")

NEO4J_URI = configs.APP_NEO4J_URL
NEO4J_USER = configs.APP_NEO4J_USER
NEO4J_PASSWORD = configs.APP_NEO4J_PASSWORD


# ---------------------------------------------------------------------------
# Ingestion cache
# ---------------------------------------------------------------------------
def load_ingestion_cache() -> dict:
    if INGESTION_CACHE.exists():
        return json.loads(INGESTION_CACHE.read_text())
    return {}

def save_ingestion_cache(cache: dict):
    INGESTION_CACHE.write_text(json.dumps(cache, indent=2))


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

Based on the issue description and code search results, which single file needs to be modified?
Return ONLY the file path, nothing else. Example: django/forms/widgets.py"""


# Module-level CodeBERT encoder — lazy-loaded once per process (~500MB)
_weaviate_encoder = None


def _get_weaviate_encoder():
    global _weaviate_encoder
    if _weaviate_encoder is None:
        from sentence_transformers import SentenceTransformer
        _weaviate_encoder = SentenceTransformer(
            "microsoft/codebert-base", device="cpu"
        )
    return _weaviate_encoder


def _weaviate_semantic_search(query_text: str, repo_dir: Path, top_k: int = 10) -> list:
    """
    Semantic search via Weaviate, post-filtered to this repo only.
    Returns [{file_path, name, node_type}] or [] if Weaviate is unavailable.
    """
    try:
        import weaviate, os
        coll_name = os.getenv("WEAVIATE_COLLECTION_NAME", "")
        if not coll_name:
            return []

        encoder = _get_weaviate_encoder()
        query_vec = encoder.encode(query_text[:500], normalize_embeddings=True).tolist()

        client = weaviate.connect_to_local()
        col = client.collections.get(coll_name)
        response = col.query.hybrid(
            query=query_text[:500],
            vector=query_vec,
            alpha=0.5,
            limit=top_k * 3,  # over-fetch before repo filter
            return_properties=["name", "file_path", "node_type"],
        )
        client.close()

        results, seen = [], set()
        for obj in response.objects:
            fp = obj.properties.get("file_path", "")
            if fp and fp not in seen and (repo_dir / fp).exists():
                seen.add(fp)
                results.append({
                    "file_path": fp,
                    "name": obj.properties.get("name", ""),
                    "node_type": obj.properties.get("node_type", ""),
                })
            if len(results) >= top_k:
                break
        return results
    except Exception as e:
        print(f"  [weaviate] Semantic search failed: {e}")
        return []


_LOW_VALUE_PATTERNS = frozenset({
    "test_", "_test.", "/test/", "/tests/", "/__tests__/",
    "spec_", "_spec.", "/spec/", "/specs/",
    "mock_", "_mock.", "/mocks/", "/fixtures/",
    "conftest.py", "/migrations/", "migrations/", "__tests__/",
})


def _is_low_value_file(file_path: str) -> bool:
    """Return True for test/mock/spec/migration files (codegraph isLowValueFile pattern)."""
    p = file_path.lower()
    return any(pat in p for pat in _LOW_VALUE_PATTERNS)


def phase1_localize_file(
    llm,
    issue_text: str,
    repo_name: str,
    repo_dir: Path,
    neo4j_session,
    project_id: str,
) -> str:
    """Phase 1: Find which file to edit."""

    # BM25 search for candidate files
    seeds = bm25_search(neo4j_session, issue_text[:1000], project_id, top_k=20)
    # Deprioritize test/mock/spec files — preserve BM25 score order within each group
    seeds.sort(key=lambda s: _is_low_value_file(s.get("file_path", "")))
    candidate_files = []
    seen_files = set()
    for s in seeds:
        fp = s.get("file_path", "")
        if fp and fp not in seen_files:
            seen_files.add(fp)
            candidate_files.append(f"  - {fp} ({s.get('type','')}: {s.get('name','')})")

    # Weaviate semantic search — supplements BM25 with vector similarity
    for hit in _weaviate_semantic_search(issue_text, repo_dir, top_k=10):
        fp = hit["file_path"]
        if fp not in seen_files:
            seen_files.add(fp)
            candidate_files.append(
                f"  - {fp} (semantic: {hit.get('name','')} [{hit.get('node_type','')}])"
            )

    # Get repo file tree (source files only, skip tests)
    file_tree = []
    for f in sorted(repo_dir.rglob("*.py")):
        if "__pycache__" in str(f) or "/.git/" in str(f):
            continue
        rel = str(f.relative_to(repo_dir))
        if not rel.startswith("test") and "/tests/" not in rel and not f.name.startswith("test_"):
            file_tree.append(f"  {rel}")

    # Cap file tree at 200 lines
    tree_str = "\n".join(file_tree[:200])
    if len(file_tree) > 200:
        tree_str += f"\n  ... ({len(file_tree) - 200} more files)"

    prompt = PHASE1_PROMPT.format(
        issue_text=issue_text[:2000],
        repo_name=repo_name,
        candidate_files="\n".join(candidate_files[:15]) if candidate_files else "  (no candidates found)",
        file_tree=tree_str,
    )

    try:
        response = llm.invoke(prompt)
        file_path = _extract_file_path(response.content, repo_dir)
        return file_path
    except Exception as e:
        print(f"    Phase 1 error: {e}")
        # Fallback: use top BM25 result
        if seeds:
            return seeds[0].get("file_path", "")
        return ""


def _extract_file_path(response: str, repo_dir: Path) -> str:
    """Extract a valid file path from LLM response."""
    response = response.strip()
    # Try each line
    for line in response.split("\n"):
        line = line.strip().strip("`").strip("*").strip()
        if line.endswith(".py"):
            # Remove common prefixes
            line = re.sub(r'^(Answer:|File:|Path:|>\s*)', '', line).strip()
            # Verify it exists
            if (repo_dir / line).exists():
                return line
    # Try to find any .py path in the response
    paths = re.findall(r'[\w/]+\.py', response)
    for p in paths:
        if (repo_dir / p).exists():
            return p
    return ""


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

Respond ONLY with a ```diff ... ``` block. No explanation.
"""


REFACTOR_PROMPT = """You are a senior Python developer performing a code refactoring.

## Refactoring Request:
{refactor_description}

## File: {file_path}

## Source Code:
```python
{file_content}
```

Generate a MINIMAL unified diff to refactor this code.
Rules:
- Preserve ALL existing behavior — no semantic changes whatsoever
- Address only what is described in the refactoring request
- The file path MUST be exactly: {file_path}
- Include the full diff header (--- a/... +++ b/...)
- Include correct @@ line numbers
- Include 3 lines of context around each change

Respond ONLY with a ```diff ... ``` block. No explanation."""

REFACTOR_RETRY_PROMPT = """You are a senior Python developer. Your previous refactoring patch FAILED — read the reflection carefully and generate a corrected version.

## Refactoring Request:
{refactor_description}

## File to modify: {file_path}

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
- Preserve ALL existing behavior
- The file path MUST be exactly: {file_path}
- Include the full diff header (--- a/... +++ b/...)
- Include correct @@ line numbers with 3 lines of context

Respond ONLY with a ```diff ... ``` block. No explanation."""


def phase3_refactor_patch(
    llm,
    refactor_description: str,
    file_path: str,
    file_content: str,
    reflection: str = "",
    failed_patch: str = "",
) -> str:
    """Generate a refactoring patch. Accepts optional reflection for retry attempts."""
    lines = file_content.split("\n")
    numbered_lines = [f"{i+1:4d} | {l}" for i, l in enumerate(lines)]
    content_for_prompt = "\n".join(numbered_lines)
    if len(content_for_prompt) > 10000:
        content_for_prompt = content_for_prompt[:10000] + "\n ... (truncated)"

    if reflection and failed_patch:
        prompt = REFACTOR_RETRY_PROMPT.format(
            refactor_description=refactor_description[:2000],
            file_path=file_path,
            failed_patch=failed_patch[:2000],
            reflection=reflection[:1000],
            file_content=content_for_prompt,
        )
    else:
        prompt = REFACTOR_PROMPT.format(
            refactor_description=refactor_description[:2000],
            file_path=file_path,
            file_content=content_for_prompt,
        )

    try:
        response = llm.invoke(prompt)
        patch = _extract_and_clean_diff(response.content, file_path)
        return patch
    except Exception as e:
        print(f"    Refactor phase 3 error: {e}")
        return ""


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

    # ACI: add line numbers so patch @@ line numbers are accurate
    lines = file_content.split("\n")
    numbered_lines = [f"{i+1:4d} | {l}" for i, l in enumerate(lines)]
    content_for_prompt = "\n".join(numbered_lines)
    if len(content_for_prompt) > 10000:
        content_for_prompt = content_for_prompt[:10000] + "\n ... (truncated)"

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

    # Strip line-number prefixes: LLMs sometimes keep "   17 | content" or "+ 20 | content"
    # from the numbered source format (e.g. "   4 | def foo():").
    # Detect by checking if a majority of hunk lines match the pattern.
    _NUM_PREFIX = re.compile(r'^([+ -]?)\s*\d+\s*\|[ \t]?(.*)', re.DOTALL)
    _raw_lines = diff.split("\n")
    hunk_lines = [l for l in _raw_lines if l and l[0] in ' +-' and not l.startswith(('---', '+++'))]
    numbered_count = sum(1 for l in hunk_lines if _NUM_PREFIX.match(l))
    if hunk_lines and numbered_count / len(hunk_lines) > 0.4:
        stripped = []
        for l in _raw_lines:
            if l.startswith(('---', '+++', '@@', 'diff', 'index')):
                stripped.append(l)
            else:
                m = _NUM_PREFIX.match(l)
                if m:
                    marker = m.group(1) or ' '
                    stripped.append(marker + m.group(2))
                else:
                    stripped.append(l)
        diff = "\n".join(stripped)

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
            file_path = line[6:].strip()
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

        # Fallback: patch -p1 with fuzzy matching (tolerates minor context mismatches)
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


def _get_graph_context_for_file(
    neo4j_session, file_path: str, project_id: str, top_k: int = 15
) -> str:
    """Get a concise graph context string for the reflector."""
    try:
        # Node-level caller/callee context
        results = neo4j_session.run("""
            MATCH (n:CodeNode {file_path: $fp, project_id: $pid})
            OPTIONAL MATCH (caller:CodeNode {project_id: $pid})-[:CALLS]->(n)
            OPTIONAL MATCH (n)-[:CALLS]->(callee:CodeNode {project_id: $pid})
            RETURN n.name AS name, n.type AS type, n.lineno AS lineno,
                   collect(DISTINCT caller.name) AS called_by,
                   collect(DISTINCT callee.name) AS calls
            LIMIT $top_k
        """, fp=file_path, pid=project_id, top_k=top_k)
        parts = []
        for r in results:
            line = f"  {r['type']}: {r['name']} (line {r['lineno']})"
            if r["called_by"]:
                line += f" ← called by: {', '.join(r['called_by'][:4])}"
            if r["calls"]:
                line += f" → calls: {', '.join(r['calls'][:4])}"
            parts.append(line)

        # ── Harness: Blast radius (codegraph getImpactRadius pattern) ───────
        # Show which OTHER files call into this file (up to 2 hops).
        # Helps LLM avoid breaking callers when modifying interfaces.
        blast_header = ""
        try:
            impact = neo4j_session.run("""
                MATCH (n:CodeNode {file_path: $fp, project_id: $pid})
                MATCH (direct:CodeNode {project_id: $pid})-[:CALLS]->(n)
                WHERE direct.file_path <> $fp
                OPTIONAL MATCH (indirect:CodeNode {project_id: $pid})-[:CALLS]->(direct)
                WHERE indirect.file_path <> $fp
                  AND indirect.file_path <> direct.file_path
                RETURN DISTINCT direct.file_path AS direct_file,
                       collect(DISTINCT indirect.file_path)[0..3] AS indirect_files
                LIMIT 10
            """, fp=file_path, pid=project_id)
            direct_files, indirect_files = set(), set()
            for row in impact:
                if row["direct_file"]:
                    direct_files.add(row["direct_file"])
                for f in (row["indirect_files"] or []):
                    if f:
                        indirect_files.add(f)
            indirect_files -= direct_files  # don't repeat
            if direct_files or indirect_files:
                blast_lines = ["Blast radius (files that call into this file):"]
                if direct_files:
                    blast_lines.append(f"  Direct: {', '.join(sorted(direct_files)[:5])}")
                if indirect_files:
                    blast_lines.append(f"  Indirect: {', '.join(sorted(indirect_files)[:3])}")
                blast_header = "\n".join(blast_lines) + "\n"
        except Exception:
            pass
        # ── End Blast radius ─────────────────────────────────────────────────

        body = "\n".join(parts) if parts else "No graph data available."
        return blast_header + body
    except Exception:
        return "Graph unavailable."


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

    project_id = repo_name.replace("/", "_")
    max_retries = configs.REFLEXION_MAX_RETRIES if reflexion else 1  # 1 = no retries

    # Clone + checkout
    repo_dir = clone_repo(repo_name)
    checkout_commit(repo_dir, base_commit)

    # Ingest if needed
    if not ingestion_cache.get(project_id, {}).get("ingested"):
        print(f"  Ingesting {repo_name}...")
        try:
            stats = ingest_repo_to_neo4j(
                repo_dir, project_id,
                NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
                max_files=500,
            )
            ingestion_cache[project_id] = {"ingested": True, **stats}
            save_ingestion_cache(ingestion_cache)
        except Exception as e:
            print(f"  Ingestion failed: {e}")

    patch = ""
    with neo4j_driver.session() as session:
        # Phase 1: File Localization (run once)
        file_path = phase1_localize_file(
            llm, issue_text, repo_name, repo_dir, session, project_id
        )
        print(f"  P1 File: {file_path or '✗'}")

        if not file_path:
            return {
                "instance_id": instance_id,
                "model_patch": "",
                "model_name_or_path": f"graphrag-reflexion-{configs.LLM_MODEL}",
            }

        # Phase 2: Fault Localization (run once)
        fault_desc, file_content = phase2_localize_fault(
            llm, issue_text, file_path, repo_dir, session, project_id
        )
        print(f"  P2 Fault: {fault_desc[:80] if fault_desc else '✗'}")

        if not file_content:
            return {
                "instance_id": instance_id,
                "model_patch": "",
                "model_name_or_path": f"graphrag-reflexion-{configs.LLM_MODEL}",
            }

        # Graph context for reflector (reuse across retries)
        graph_context = _get_graph_context_for_file(session, file_path, project_id)

        # Phase 3 + Reflexion loop
        reflections: list[str] = []
        last_patch = ""
        last_error = ""

        for attempt in range(1, max_retries + 1):
            # Generate patch (first attempt = fresh, retries = with reflection)
            if attempt == 1:
                patch = phase3_generate_patch(
                    llm, issue_text, file_path, fault_desc, file_content
                )
            else:
                # Generate reflection on the previous failure
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
                last_patch = ""
                continue

            # Normalize patch: rebuild correct line numbers from actual file
            normalized = _normalize_patch(patch, repo_dir)
            if normalized != patch:
                print(f"  Patch normalized (line numbers corrected)")

            # Verify the patch applies cleanly
            applies_ok, apply_error = _verify_patch_applies(normalized, repo_dir)

            if applies_ok:
                patch = normalized  # save the clean version

                # ── Harness: Evaluator ──────────────────────────────────────
                # A second LLM independently scores patch quality (1-5).
                # Score < 3 → reject and trigger another reflexion attempt.
                # This implements Planner → Generator → Evaluator pipeline.
                if reflexion:
                    try:
                        from api.agent.evaluator import evaluate_patch
                        eval_score, eval_reason = evaluate_patch(
                            bug_description=f"{issue_text[:300]}\nFault: {fault_desc}",
                            patch=patch,
                        )
                        print(f"  Evaluator: {eval_score}/5 — {eval_reason}")
                        if eval_score < 3 and attempt < max_retries:
                            # Frustration signal: patch applies but evaluator rejects it
                            print(f"  Evaluator rejected patch (score {eval_score}) — retrying")
                            last_patch = patch
                            last_error = f"Evaluator feedback: {eval_reason}"
                            patch = ""
                            continue
                    except Exception:
                        pass  # evaluator is optional — never block on it
                # ── End Evaluator ───────────────────────────────────────────

                print(f"  P3 Patch: ✓ {len(patch)} chars (attempt {attempt}, applies cleanly)")
                break  # ← success, exit reflexion loop
            else:
                print(f"  P3 Patch: ✗ apply failed (attempt {attempt}): {apply_error[:80]}")
                patch_preview = "\n".join(normalized.split("\n")[:10])
                print(f"  Patch preview:\n{patch_preview}")

                # ── Harness: Frustration detection ─────────────────────────
                # If patches are shrinking across attempts, the agent is confused.
                # Widen the graph context to give it more signal.
                if last_patch and len(patch) < len(last_patch) * 0.6:
                    print(f"  Frustration detected (patch shrinking) — widening context")
                    try:
                        with neo4j_driver.session() as s:
                            extra = _get_graph_context_for_file(s, file_path, project_id)
                        graph_context = extra + "\n" + graph_context
                    except Exception:
                        pass
                # ── End Frustration detection ───────────────────────────────

                last_patch = patch
                last_error = f"git apply --check failed: {apply_error}"
                patch = ""  # don't save a broken patch

    return {
        "instance_id": instance_id,
        "model_patch": patch,
        "model_name_or_path": f"graphrag-reflexion-{configs.LLM_MODEL}",
    }


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
    errors = 0

    for i, task in enumerate(tasks):
        t0 = time.time()
        print(f"\n[{i+1+len(done_ids)}/{len(tasks)+len(done_ids)}] {task['instance_id']}")

        try:
            pred = run_single_task(task, llm, neo4j_driver, ingestion_cache, reflexion=use_reflexion)
            predictions.append(pred)
            if not pred["model_patch"]:
                errors += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            predictions.append({
                "instance_id": task["instance_id"],
                "model_patch": "",
                "model_name_or_path": f"graphrag-reflexion-{configs.LLM_MODEL}",
            })
            errors += 1

        print(f"  ({time.time()-t0:.1f}s)")

        # Save after each
        with open(args.output, "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            patched = sum(1 for p in predictions if p["model_patch"])
            print(f"\n  ── {patched}/{i+1+len(done_ids)} patched | {errors} errors ──")

    neo4j_driver.close()
    elapsed = time.time() - start
    patched = sum(1 for p in predictions if p["model_patch"])
    total = len(predictions)

    print(f"\n{'='*60}")
    print(f"DONE: {total} tasks in {elapsed/60:.1f}min")
    print(f"Patches: {patched}/{total} ({100*patched/total:.1f}%)")

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
