from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SearchReplaceEdit:
    file_path: str
    search: str
    replace: str


def _to_posix(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def _clean_file_header(line: str) -> str:
    text = line.strip()
    text = re.sub(r"^#+\s*", "", text)
    text = re.sub(r"^File:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip("` ")
    match = re.search(r"[\w./\\-]+\.py", text)
    return _to_posix(match.group(0)) if match else ""


def _trim_block_newlines(lines: list[str]) -> str:
    text = "\n".join(lines)
    return text.strip("\n")


def parse_search_replace_edits(
    text: str,
    default_file: str = "",
) -> list[SearchReplaceEdit]:
    """Parse KGCompass/Aider-style SEARCH/REPLACE edits from an LLM response."""
    edits: list[SearchReplaceEdit] = []
    current_file = _to_posix(default_file) if default_file else ""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    index = 0

    while index < len(lines):
        line = lines[index]
        possible_file = _clean_file_header(line)
        if possible_file:
            current_file = possible_file
            index += 1
            continue

        if line.strip() != "<<<<<<< SEARCH":
            index += 1
            continue

        index += 1
        search_lines: list[str] = []
        while index < len(lines) and lines[index].strip() != "=======":
            search_lines.append(lines[index])
            index += 1

        if index >= len(lines):
            break
        index += 1

        replace_lines: list[str] = []
        while index < len(lines) and lines[index].strip() != ">>>>>>> REPLACE":
            replace_lines.append(lines[index])
            index += 1

        if index >= len(lines):
            break
        index += 1

        if current_file:
            edits.append(SearchReplaceEdit(
                file_path=current_file,
                search=_trim_block_newlines(search_lines),
                replace=_trim_block_newlines(replace_lines),
            ))

    return edits


def _adjust_indent(text: str, spaces: int) -> str:
    adjusted = []
    for line in text.split("\n"):
        if not line:
            adjusted.append(line)
        elif spaces < 0:
            width = abs(spaces)
            adjusted.append(line[width:] if line.startswith(" " * width) else line)
        else:
            adjusted.append(" " * spaces + line)
    return "\n".join(adjusted)


def _apply_edits_to_content(
    original: str,
    edits: list[SearchReplaceEdit],
) -> tuple[str, str]:
    content = original
    for edit in edits:
        if not edit.search:
            return original, "Empty SEARCH block"
        if edit.search == edit.replace:
            return original, "SEARCH and REPLACE blocks are identical"
        if edit.search not in content:
            preview = edit.search.splitlines()[0] if edit.search.splitlines() else edit.search
            return original, f"SEARCH block not found: {preview[:120]}"
        content = content.replace(edit.search, edit.replace, 1)
    if content == original:
        return original, "SEARCH/REPLACE produced no changes"
    return content, ""


def _python_syntax_ok(path: str, content: str) -> tuple[bool, str]:
    if not path.endswith(".py"):
        return True, ""
    try:
        ast.parse(content)
        return True, ""
    except SyntaxError as exc:
        return False, f"Python syntax error after edit: {exc}"


def _candidate_edit_sets(edits: list[SearchReplaceEdit]) -> list[list[SearchReplaceEdit]]:
    candidates = [edits]
    for shift in (-4, 4, -8, 8):
        candidates.append([
            SearchReplaceEdit(
                file_path=edit.file_path,
                search=_adjust_indent(edit.search, shift),
                replace=_adjust_indent(edit.replace, shift),
            )
            for edit in edits
        ])
    if len(edits) > 1:
        candidates.append(edits[:1])
    return candidates


def search_replace_to_unified_diff(
    llm_output: str,
    repo_dir: Path,
    expected_file: str,
) -> tuple[str, str]:
    """
    Convert SEARCH/REPLACE edits into a unified diff against repo_dir.

    Returns (patch, error). No files are modified.
    """
    expected_file = _to_posix(expected_file)
    edits = parse_search_replace_edits(llm_output, default_file=expected_file)
    if not edits:
        return "", "No SEARCH/REPLACE edit blocks found"

    unexpected = sorted({_to_posix(edit.file_path) for edit in edits if _to_posix(edit.file_path) != expected_file})
    if unexpected:
        return "", f"SEARCH/REPLACE edits target {unexpected[0]}, expected {expected_file}"

    file_path = repo_dir / expected_file
    if not file_path.exists():
        return "", f"Target file does not exist: {expected_file}"

    original = file_path.read_text(errors="ignore")
    last_error = ""
    for candidate_edits in _candidate_edit_sets(edits):
        new_content, error = _apply_edits_to_content(original, candidate_edits)
        if error:
            last_error = error
            continue

        syntax_ok, syntax_error = _python_syntax_ok(expected_file, new_content)
        if not syntax_ok:
            last_error = syntax_error
            continue

        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{expected_file}",
            tofile=f"b/{expected_file}",
        )
        patch = "".join(diff)
        if patch and not patch.endswith("\n"):
            patch += "\n"
        return patch, ""

    return "", last_error or "Could not apply SEARCH/REPLACE edits"
