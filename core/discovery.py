from __future__ import annotations

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
    2. Project directories (scope='project'): `workspace / resource_name`, `workspace / f".{resource_name}"`, `workspace / ".microharness" / resource_name`

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
