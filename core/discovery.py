from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class DiscoveredResource(Generic[T]):
    """Represents a discovered capability or configuration resource."""

    id: str
    item: T
    scope: str  # 'global' or 'project'
    source_path: Path


def resolve_search_directories(
    resource_name: str,
    workspace_dir: Path | str | None = None,
    global_dir: Path | str | None = None,
) -> list[tuple[Path, str]]:
    """Resolves hierarchical search directories for a resource type.

    Precedence order:
    1. Global directories (scope='global'): `global_dir / resource_name`, `global_dir / f".{resource_name}"`
    2. Project directories (scope='project'): `workspace / ".microharness" / resource_name` (and for
       non-tools resources, legacy `workspace / resource_name`, `workspace / f".{resource_name}"`).
       Tools are strictly restricted to `workspace / ".microharness" / "tools"` to avoid generic search paths.

    If both workspace_dir and global_dir are None, global_dir defaults to `~/.microharness`.
    """
    dirs: list[tuple[Path, str]] = []

    # 1. Global directories
    resolved_global = global_dir
    if resolved_global is None and workspace_dir is None:
        resolved_global = Path.home() / ".microharness"

    if resolved_global is not None:
        g_base = Path(resolved_global).resolve()
        for cand in [g_base / resource_name, g_base / f".{resource_name}"]:
            if cand.is_dir():
                dirs.append((cand, "global"))

    # 2. Project directories (ordered so .microharness takes priority over legacy project folders)
    if workspace_dir is not None:
        ws_base = Path(workspace_dir).resolve()
        if resource_name == "tools":
            cand = ws_base / ".microharness" / "tools"
            if cand.is_dir():
                dirs.append((cand, "project"))
        else:
            for cand in [
                ws_base / resource_name,
                ws_base / f".{resource_name}",
                ws_base / ".microharness" / resource_name,
            ]:
                if cand.is_dir():
                    dirs.append((cand, "project"))

    return dirs


def discover_resources(
    resource_name: str,
    loader: Callable[[Path, str], Iterable[tuple[str, T]] | tuple[str, T] | None],
    workspace_dir: Path | str | None = None,
    global_dir: Path | str | None = None,
    file_pattern: str | None = None,
) -> dict[str, DiscoveredResource[T]]:
    """Discovers resources across global and project directories with automatic shadowing.

    The loader callback receives `(path, scope)` and may return:
    - An iterable/generator of `(id, item)` tuples
    - A single `(id, item)` tuple
    - `None` (if the file/directory does not contain a valid resource)

    Returns a dict mapping `resource_id -> DiscoveredResource[T]`.
    Project-level resources cleanly shadow global resources sharing the same identifier.
    """
    discovered: dict[str, DiscoveredResource[T]] = {}
    search_dirs = resolve_search_directories(
        resource_name=resource_name,
        workspace_dir=workspace_dir,
        global_dir=global_dir,
    )

    for search_dir, scope in search_dirs:
        entries: list[Path] = (
            sorted(search_dir.glob(file_pattern))
            if file_pattern
            else sorted(search_dir.iterdir())
        )

        for entry in entries:
            name_lower = entry.name.lower()
            if entry.name.startswith(("_", ".")) or name_lower == "readme.md" or name_lower == "__pycache__":
                continue

            try:
                result: Any = loader(entry, scope)
            except Exception:
                continue

            if result is None:
                continue

            # Normalize loader result into list of (item_id, item)
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], str):
                items = [result]
            else:
                try:
                    items = list(result)
                except TypeError:
                    continue

            for item_id, item in items:
                if not isinstance(item_id, str) or not item_id:
                    continue
                discovered[item_id] = DiscoveredResource(
                    id=item_id,
                    item=item,
                    scope=scope,
                    source_path=entry,
                )

    return discovered


IGNORED_WORKSPACE_DIRS: set[str] = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    ".cache",
    "build",
    "dist",
}

IGNORED_WORKSPACE_EXTENSIONS: set[str] = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".webp",
    ".svg",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".7z",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".mp3",
    ".mp4",
    ".wav",
    ".mov",
}


def discover_workspace_files(
    workspace_dir: Path | str,
    max_files: int = 2000,
) -> list[str]:
    """Finds relative file paths within the workspace for @ file-tagging and autocomplete."""
    ws = Path(workspace_dir).resolve()
    if not ws.is_dir():
        return []

    collected: list[str] = []
    for root, dirnames, filenames in os.walk(ws, followlinks=False):
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in IGNORED_WORKSPACE_DIRS
        ]
        dirnames.sort()

        for f in sorted(filenames):
            if f.startswith(".") and f not in (".gitignore", ".env.example"):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in IGNORED_WORKSPACE_EXTENSIONS:
                continue

            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, ws).replace("\\", "/")
            collected.append(rel_path)
            if len(collected) >= max_files:
                break
        if len(collected) >= max_files:
            break

    return sorted(collected)


def expand_tagged_files(
    user_text: str,
    workspace_dir: Path | str,
    max_file_chars: int = 50_000,
) -> tuple[str, list[str]]:
    """Detects @file tags in user_text, reads referenced workspace files, and appends context.

    Returns:
        (effective_prompt, attached_files)
    """
    ws = Path(workspace_dir).resolve()
    matches = re.finditer(r"(?:^|\s)@([a-zA-Z0-9_\-\./\\]+)", user_text)
    seen: set[str] = set()
    attached: list[str] = []
    file_sections: list[str] = []

    for m in matches:
        raw_rel = m.group(1).strip().rstrip(".,;:!?)")
        clean_rel = raw_rel.replace("\\", "/").rstrip("/")
        if not clean_rel or clean_rel in seen:
            continue
        seen.add(clean_rel)

        file_path = (ws / clean_rel).resolve()
        try:
            file_path.relative_to(ws)
        except ValueError:
            continue

        if not file_path.is_file():
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            if len(content) > max_file_chars:
                content = (
                    content[:max_file_chars]
                    + f"\n\n[... Truncated: File exceeds {max_file_chars:,} characters ...]"
                )
            attached.append(clean_rel)
            lang = _guess_file_lang(clean_rel)
            file_sections.append(f"[Attached File: @{clean_rel}]\n```{lang}\n{content}\n```")
        except Exception:
            continue

    if not file_sections:
        return user_text, []

    effective_prompt = user_text + "\n\n" + "\n\n".join(file_sections)
    return effective_prompt, attached


def _guess_file_lang(path_str: str) -> str:
    ext = os.path.splitext(path_str)[1].lower()
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".json": "json",
        ".md": "markdown",
        ".html": "html",
        ".css": "css",
        ".sh": "bash",
        ".zsh": "bash",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".sql": "sql",
        ".rs": "rust",
        ".go": "go",
    }
    return mapping.get(ext, "")
