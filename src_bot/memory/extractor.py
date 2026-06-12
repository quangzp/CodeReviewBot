"""
Fact extractor — runs after each review to convert raw review output
into structured facts (bug patterns, files touched, etc.).

This is what turns "the bot did a review" into "the bot LEARNED something."
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

from src_bot.memory.schema import ReviewFact


EXTRACTION_PROMPT = """You analyze a completed code review and extract structured facts about it.

## Review Context
Repository: {repo_name}
PR #: {pr_number}
PR URL: {pr_url}
Author (GitHub login): {author}
Files modified: {files}

## Review Output
Issue/PR description: {issue_text}

Files reviewed and any patches generated:
{review_output}

## Your Task
Extract these facts as JSON:

1. **bug_patterns**: A list of canonical bug pattern IDs (kebab-case, 2-5 words) that were detected.
   Use existing pattern names when possible. Examples of good pattern IDs:
   - "missing-null-check"
   - "missing-await-on-async"
   - "incorrect-error-handling"
   - "race-condition"
   - "off-by-one-error"
   - "unhandled-exception"
   - "type-mismatch"
   - "memory-leak"
   - "sql-injection-risk"
   - "missing-input-validation"
   - "incorrect-timezone-handling"

   If NO bugs were found, return an empty list [].
   Only include patterns you have strong evidence for. Do NOT speculate.

2. **summary**: One sentence (under 25 words) describing what this review found/changed.

Respond with ONLY valid JSON in this exact format:
{{
  "bug_patterns": ["pattern-id-1", "pattern-id-2"],
  "summary": "Brief description here."
}}
"""


def extract_facts_from_review(
    llm: BaseChatModel,
    *,
    repo_name: str,
    pr_number: int,
    pr_url: str,
    author_login: str,
    files_touched: list[str],
    issue_text: str,
    review_output: str,
) -> ReviewFact:
    """
    Extract structured facts from a completed review using an LLM.
    Always returns a ReviewFact (with empty patterns if nothing detected).
    """
    files_str = "\n".join(f"  - {f}" for f in files_touched[:20])
    review_str = (review_output or "")[:6000]

    prompt = EXTRACTION_PROMPT.format(
        repo_name=repo_name,
        pr_number=pr_number,
        pr_url=pr_url,
        author=author_login,
        files=files_str or "  (none)",
        issue_text=issue_text[:1500],
        review_output=review_str,
    )

    patterns: list[str] = []
    summary = f"Review of PR #{pr_number}"

    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        # Extract first JSON object from the response
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            patterns = [_normalize_pattern(p) for p in data.get("bug_patterns", []) if p]
            patterns = [p for p in patterns if p]  # drop empty
            summary = (data.get("summary") or summary)[:300]
    except Exception as e:
        # Extraction failure should never break the review
        logger.warning("[memory] Fact extraction failed: %s", e)

    return ReviewFact(
        developer_login=author_login,
        repo_name=repo_name,
        pr_number=pr_number,
        pr_url=pr_url,
        files_touched=files_touched,
        bug_patterns=patterns,
        summary=summary,
        reviewed_at=datetime.now(timezone.utc),
    )


def _normalize_pattern(p: str) -> str:
    """Normalize a pattern ID to canonical kebab-case."""
    p = p.lower().strip()
    p = re.sub(r"[^a-z0-9\s-]", "", p)
    p = re.sub(r"\s+", "-", p)
    p = re.sub(r"-+", "-", p)
    p = p.strip("-")
    # Cap at 5 words
    parts = p.split("-")[:5]
    return "-".join(parts)
