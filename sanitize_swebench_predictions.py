#!/usr/bin/env python3
"""Fix malformed unified-diff hunk counts inside SWE-bench prediction JSONL.

The SWE-bench harness can reject patches whose @@ header line counts are wrong,
even if `git apply --recount` accepted them during generation. This script reads
predictions JSONL and writes a sanitized copy with corrected hunk headers in
each `model_patch`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<suffix>.*)$"
)


def _split_patch_lines(patch: str) -> tuple[list[str], bool]:
    has_trailing_newline = patch.endswith("\n")
    lines = patch.split("\n")
    if has_trailing_newline:
        lines = lines[:-1]
    return lines, has_trailing_newline


def _is_next_file_header(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and lines[index].startswith("--- ")
        and lines[index + 1].startswith("+++ ")
    )


def _count_hunk_lines(hunk_lines: list[str]) -> tuple[int, int]:
    old_count = 0
    new_count = 0

    for line in hunk_lines:
        if line.startswith("\\ No newline at end of file"):
            continue
        if line.startswith(" "):
            old_count += 1
            new_count += 1
        elif line.startswith("-"):
            old_count += 1
        elif line.startswith("+"):
            new_count += 1

    return old_count, new_count


def fix_patch_hunk_headers(patch: str) -> tuple[str, int, int]:
    """Return (fixed_patch, hunk_count, changed_hunk_count)."""
    if not patch:
        return patch, 0, 0

    lines, has_trailing_newline = _split_patch_lines(patch)
    fixed: list[str] = []
    index = 0
    hunk_count = 0
    changed_hunks = 0

    while index < len(lines):
        line = lines[index]
        match = HUNK_RE.match(line)
        if not match:
            fixed.append(line)
            index += 1
            continue

        hunk_count += 1
        hunk_start = index + 1
        hunk_end = hunk_start

        while hunk_end < len(lines):
            if HUNK_RE.match(lines[hunk_end]):
                break
            if _is_next_file_header(lines, hunk_end):
                break
            hunk_end += 1

        hunk_lines = lines[hunk_start:hunk_end]
        old_count, new_count = _count_hunk_lines(hunk_lines)
        fixed_header = (
            f"@@ -{match.group('old_start')},{old_count} "
            f"+{match.group('new_start')},{new_count} @@{match.group('suffix')}"
        )
        if fixed_header != line:
            changed_hunks += 1

        fixed.append(fixed_header)
        fixed.extend(hunk_lines)
        index = hunk_end

    fixed_patch = "\n".join(fixed)
    if has_trailing_newline:
        fixed_patch += "\n"

    return fixed_patch, hunk_count, changed_hunks


def sanitize_predictions(input_path: Path, output_path: Path) -> dict[str, int]:
    stats = {
        "rows": 0,
        "patch_rows": 0,
        "changed_rows": 0,
        "hunks": 0,
        "changed_hunks": 0,
    }

    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue

            try:
                pred = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_number}: invalid JSON: {exc}") from exc

            stats["rows"] += 1
            patch = pred.get("model_patch") or ""
            if patch:
                stats["patch_rows"] += 1
                fixed_patch, hunks, changed_hunks = fix_patch_hunk_headers(patch)
                stats["hunks"] += hunks
                stats["changed_hunks"] += changed_hunks
                if fixed_patch != patch:
                    stats["changed_rows"] += 1
                    pred["model_patch"] = fixed_patch

            dst.write(json.dumps(pred, ensure_ascii=False) + "\n")

    return stats


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_sanitized{input_path.suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sanitize SWE-bench prediction JSONL patch hunk headers."
    )
    parser.add_argument("input", type=Path, help="Input predictions JSONL")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to INPUT_stem_sanitized.jsonl.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or default_output_path(input_path)

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        return 2
    if input_path.resolve() == output_path.resolve():
        print("ERROR: output must be different from input", file=sys.stderr)
        return 2
    if output_path.exists() and not args.force:
        print(f"ERROR: output exists, use --force: {output_path}", file=sys.stderr)
        return 2

    stats = sanitize_predictions(input_path, output_path)
    print(f"Wrote: {output_path}")
    print(
        "Rows: {rows}, patch rows: {patch_rows}, changed rows: {changed_rows}, "
        "hunks: {hunks}, changed hunks: {changed_hunks}".format(**stats)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
