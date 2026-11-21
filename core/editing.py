"""Resilient text-replacement and closest-match feedback helpers for file editing."""

from __future__ import annotations

import ast
import difflib
import textwrap
import warnings
from collections.abc import Callable, Sequence


def _exact_match_lines(content: str, target: str) -> list[int]:
    """Returns 1-indexed starting line numbers for all non-overlapping occurrences."""
    lines: list[int] = []
    start = 0
    while (idx := content.find(target, start)) != -1:
        lines.append(content.count("\n", 0, idx) + 1)
        start = idx + len(target)
    return lines


def _common_leading_indent(lines: Sequence[str]) -> str:
    """Computes the longest common whitespace prefix across non-empty lines."""
    indents = [line[: len(line) - len(line.lstrip(" \t"))] for line in lines if line.strip()]
    if not indents:
        return ""
    prefix = indents[0]
    for indent in indents[1:]:
        while not indent.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def _dedent_lines(lines: Sequence[str]) -> list[str]:
    """Strips trailing whitespace and the common leading indent across lines."""
    stripped = [line.rstrip() for line in lines]
    indent = _common_leading_indent(stripped)
    if not indent:
        return [line if line.strip() else "" for line in stripped]
    prefix_len = len(indent)
    return [line[prefix_len:] if line.strip() else "" for line in stripped]


def _reindent_block(text: str, target_indent: str) -> str:
    """Re-indents a replacement block to ``target_indent`` while preserving blank lines."""
    dedented = textwrap.dedent(text)
    if not target_indent:
        return dedented
    return "".join(
        f"{target_indent}{line}" if line.strip() else line
        for line in dedented.splitlines(keepends=True)
    )


def _find_normalized_block_matches(
    file_lines: Sequence[str],
    old_lines: Sequence[str],
    normalizer: Callable[[Sequence[str]], list[str]],
) -> list[int]:
    """Finds 0-indexed start line indices where ``normalizer(window) == normalizer(old_lines)``."""
    window_len = len(old_lines)
    if window_len == 0 or window_len > len(file_lines):
        return []
    norm_old = normalizer(old_lines)
    norm_file = normalizer(file_lines)
    return [
        idx
        for idx in range(len(norm_file) - window_len + 1)
        if norm_file[idx : idx + window_len] == norm_old
    ]


def _apply_line_block_replacement(
    file_lines: list[str],
    *,
    start_idx: int,
    line_count: int,
    old_lines: list[str],
    new_str: str,
    allow_reindent: bool,
) -> str:
    """Replaces ``file_lines[start_idx : start_idx + line_count]`` with ``new_str``."""
    end_idx = start_idx + line_count
    matched_block = file_lines[start_idx:end_idx]
    tail_line = matched_block[-1]
    tail_eol = tail_line[len(tail_line.rstrip("\r\n")) :]

    replacement = new_str.replace("\r\n", "\n")
    if allow_reindent and replacement:
        file_indent = _common_leading_indent([line.rstrip("\r\n") for line in matched_block])
        old_indent = _common_leading_indent(old_lines)
        if file_indent != old_indent:
            replacement = _reindent_block(replacement, file_indent)

    if replacement and tail_eol and not replacement.endswith(("\n", "\r")):
        replacement = f"{replacement}{tail_eol}"

    return "".join(file_lines[:start_idx]) + replacement + "".join(file_lines[end_idx:])


def _format_closest_match_feedback(content: str, old_str: str, path_label: str) -> str:
    """Formats an actionable error message with the closest difflib block and line numbers."""
    base = (
        f"Target string not found in '{path_label}'. "
        "Make sure old_str matches, including surrounding context."
    )
    file_lines = content.splitlines()
    if not file_lines:
        return f"{base} (File is empty.)"

    old_lines = old_str.splitlines() or [old_str]
    window_size = min(len(old_lines), len(file_lines))
    norm_old = "\n".join(line.strip() for line in old_lines)

    best_start = 0
    best_ratio = -1.0
    matcher = difflib.SequenceMatcher(None, "", norm_old)
    for idx in range(len(file_lines) - window_size + 1):
        candidate = "\n".join(line.strip() for line in file_lines[idx : idx + window_size])
        matcher.set_seq1(candidate)
        if matcher.real_quick_ratio() <= best_ratio or matcher.quick_ratio() <= best_ratio:
            continue
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = idx

    if best_ratio < 0.2:
        return f"{base} No close match found (file has {len(file_lines)} lines)."

    context_lines = 2
    match_start_line = best_start + 1
    match_end_line = best_start + window_size
    ctx_start = max(0, best_start - context_lines)
    ctx_end = min(len(file_lines), best_start + window_size + context_lines)
    snippet = "\n".join(
        f"{ln:4d} | {file_lines[ln - 1]}" for ln in range(ctx_start + 1, ctx_end + 1)
    )
    line_desc = (
        f"line {match_start_line}"
        if match_start_line == match_end_line
        else f"lines {match_start_line}-{match_end_line}"
    )
    return (
        f"{base}\n"
        f"Closest match near {line_desc} (showing lines {ctx_start + 1}-{ctx_end}):\n"
        f"{snippet}"
    )


def apply_text_edit(
    content: str,
    old_str: str,
    new_str: str,
    *,
    path_label: str = "file",
) -> tuple[str, str]:
    """Replaces a unique occurrence of ``old_str`` with ``new_str`` in ``content``.

    Tries exact substring matching first, then falls back to unique
    whitespace/indentation-normalized line-block matching. Raises ``ValueError``
    with line numbers or closest-match context when the target is missing or ambiguous.
    """
    if not old_str:
        raise ValueError("old_str cannot be empty.")

    occurrences = content.count(old_str)
    if occurrences > 1:
        line_list = ", ".join(str(ln) for ln in _exact_match_lines(content, old_str))
        raise ValueError(
            f"Target string found {occurrences} times in '{path_label}' (at lines: {line_list}). "
            "The target string must be unique. Provide more surrounding context lines to disambiguate."
        )

    if occurrences == 1:
        return content.replace(
            old_str, new_str, 1
        ), f"Successfully replaced 1 occurrence in '{path_label}'."

    if old_str.strip():
        file_lines = content.splitlines(keepends=True)
        old_lines = old_str.splitlines()
        normalizers: tuple[tuple[Callable[[Sequence[str]], list[str]], bool], ...] = (
            (lambda lines: [line.rstrip() for line in lines], False),
            (_dedent_lines, True),
            (lambda lines: [line.strip() for line in lines], True),
        )
        for normalizer, allow_reindent in normalizers:
            matches = _find_normalized_block_matches(file_lines, old_lines, normalizer)
            if len(matches) > 1:
                line_list = ", ".join(str(idx + 1) for idx in matches)
                raise ValueError(
                    f"Target string matched {len(matches)} times in '{path_label}' after whitespace "
                    f"normalization (at lines: {line_list}). The target string must be unique. "
                    "Provide more surrounding context lines to disambiguate."
                )
            if len(matches) == 1:
                updated = _apply_line_block_replacement(
                    file_lines,
                    start_idx=matches[0],
                    line_count=len(old_lines),
                    old_lines=old_lines,
                    new_str=new_str,
                    allow_reindent=allow_reindent,
                )
                return updated, (
                    f"Successfully replaced 1 occurrence in '{path_label}' "
                    "(matched with whitespace normalization)."
                )

    raise ValueError(_format_closest_match_feedback(content, old_str, path_label))


def check_python_syntax(content: str, path_label: str = "file") -> str | None:
    """Checks Python syntax with ``ast.parse`` and returns a concise line/column diagnostic, or ``None``."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(content, filename=path_label)
        return None
    except SyntaxError as exc:
        msg = exc.msg or "invalid syntax"
        if exc.lineno is not None and exc.offset is not None:
            header = (
                f"Syntax error in '{path_label}' at line {exc.lineno}, column {exc.offset}: {msg}"
            )
        elif exc.lineno is not None:
            header = f"Syntax error in '{path_label}' at line {exc.lineno}: {msg}"
        else:
            header = f"Syntax error in '{path_label}': {msg}"

        if exc.text:
            raw_line = exc.text.rstrip("\r\n")
            if exc.offset is not None and 1 <= exc.offset <= len(raw_line) + 1:
                caret = " " * (exc.offset - 1) + "^"
                return f"{header}\n  {raw_line}\n  {caret}"
            return f"{header}\n  {raw_line}"
        return header
    except ValueError as exc:
        return f"Syntax error in '{path_label}': {exc}"
