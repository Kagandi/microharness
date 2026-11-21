"""Lightweight pre-turn workspace checkpoints and rollback (/undo).

In git repositories, snapshots use ``git write-tree`` with an isolated
``GIT_INDEX_FILE`` so the user's real ``HEAD``, branches, and staging index are
never modified. In non-git directories, scoped file backups are stored under a
checkpoint directory (skipping build/cache/VCS directories).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from core.defaults import load_defaults
from core.memory import Memory

_DEFAULTS = load_defaults()
_SKIP_DIRS: frozenset[str] = frozenset({".git", ".logs", "__pycache__", ".venv", "node_modules"})
_EMPTY_GIT_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
DEFAULT_MAX_SNAPSHOT_FILES: int = int(_DEFAULTS["max_snapshot_files"])
DEFAULT_MAX_SNAPSHOT_BYTES: int = int(_DEFAULTS["max_snapshot_bytes"])


@dataclass(frozen=True)
class Checkpoint:
    """Immutable snapshot metadata for one pre-turn workspace state."""

    id: int
    label: str
    backend: str  # "git" | "files"
    tree_sha: str | None = None
    snapshot_dir: Path | None = None
    manifest: tuple[str, ...] = ()
    memory_len: int | None = None
    modified_files: frozenset[str] = frozenset()
    dirty_before: bool = False


@dataclass(frozen=True)
class UndoResult:
    """Outcome of restoring a workspace checkpoint."""

    ok: bool
    message: str
    checkpoint_id: int | None = None
    restored_files: tuple[str, ...] = ()
    removed_files: tuple[str, ...] = ()


def _run_git(
    args: list[str],
    *,
    cwd: Path,
    index_file: Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    if index_file is not None:
        env["GIT_INDEX_FILE"] = str(index_file)
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _prune_empty_parents(path: Path, stop_at: Path) -> None:
    """Removes empty parent directories created during a turn up to ``stop_at``."""
    parent = path.parent
    while parent != stop_at and parent.is_relative_to(stop_at):
        if parent.name == ".git":
            break
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


class CheckpointManager:
    """Manages a LIFO stack of lightweight pre-turn workspace checkpoints."""

    def __init__(
        self,
        workspace_dir: Path | str,
        storage_dir: Path | str | None = None,
        *,
        max_files: int = DEFAULT_MAX_SNAPSHOT_FILES,
        max_bytes: int = DEFAULT_MAX_SNAPSHOT_BYTES,
    ):
        self.workspace_dir = Path(workspace_dir).resolve()
        self._explicit_storage_dir = Path(storage_dir).resolve() if storage_dir else None
        self._temp_storage_dir: tempfile.TemporaryDirectory[str] | None = None
        self.max_files = max_files
        self.max_bytes = max_bytes
        self._stack: list[Checkpoint] = []
        self._next_id = 1
        self.last_warning: str | None = None

    @property
    def count(self) -> int:
        return len(self._stack)

    @property
    def has_checkpoints(self) -> bool:
        return bool(self._stack)

    @property
    def latest(self) -> Checkpoint | None:
        return self._stack[-1] if self._stack else None

    def _storage_root(self) -> Path:
        if self._explicit_storage_dir is not None:
            self._explicit_storage_dir.mkdir(parents=True, exist_ok=True)
            return self._explicit_storage_dir
        if self._temp_storage_dir is None:
            self._temp_storage_dir = tempfile.TemporaryDirectory(prefix="mh_checkpoints_")
        return Path(self._temp_storage_dir.name)

    def _git_toplevel(self) -> Path | None:
        if not self.workspace_dir.is_dir() or shutil.which("git") is None:
            return None
        try:
            res = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=self.workspace_dir)
        except OSError:
            return None
        if res.returncode != 0 or res.stdout.decode("utf-8", errors="replace").strip() != "true":
            return None
        try:
            top_res = _run_git(["rev-parse", "--show-toplevel"], cwd=self.workspace_dir)
        except OSError:
            return None
        if top_res.returncode != 0:
            return None
        top_str = top_res.stdout.decode("utf-8", errors="replace").strip()
        return Path(top_str).resolve() if top_str else None

    def _git_real_index_path(self) -> Path | None:
        res = _run_git(["rev-parse", "--git-path", "index"], cwd=self.workspace_dir)
        if res.returncode != 0:
            return None
        raw = res.stdout.decode("utf-8", errors="replace").strip()
        if not raw:
            return None
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (self.workspace_dir / candidate).resolve()
        return candidate if candidate.is_file() else None

    def _capture_git_tree(self, toplevel: Path) -> tuple[str | None, bool, str | None]:
        """Captures workspace working-tree state into a dangling git tree object."""
        with tempfile.TemporaryDirectory(prefix="mh_git_idx_") as tmp:
            tmp_index = Path(tmp) / "index"
            if self.workspace_dir == toplevel:
                real_index = self._git_real_index_path()
                if real_index is not None:
                    shutil.copy2(real_index, tmp_index)

            add_res = _run_git(
                ["add", "-A", "--", "."], cwd=self.workspace_dir, index_file=tmp_index
            )
            if add_res.returncode != 0:
                err = add_res.stderr.decode("utf-8", errors="replace").strip()
                return None, False, err or "git add failed"

            tree_res = _run_git(["write-tree"], cwd=self.workspace_dir, index_file=tmp_index)
            if tree_res.returncode != 0:
                err = tree_res.stderr.decode("utf-8", errors="replace").strip()
                return None, False, err or "git write-tree failed"

            tree_sha = tree_res.stdout.decode("utf-8", errors="replace").strip()
            status_res = _run_git(["status", "--porcelain", "--", "."], cwd=self.workspace_dir)
            dirty_before = status_res.returncode == 0 and bool(status_res.stdout.strip())
            return tree_sha, dirty_before, None

    def _collect_workspace_files(self) -> tuple[list[str], int, bool]:
        """Collects non-ignored workspace-relative file paths for non-git snapshots."""
        files: list[str] = []
        total_bytes = 0
        storage_root = (
            self._explicit_storage_dir if self._explicit_storage_dir is not None else None
        )

        for root_str, dirnames, filenames in os.walk(self.workspace_dir):
            root_path = Path(root_str).resolve()
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in _SKIP_DIRS
                and not (
                    storage_root is not None
                    and (root_path / d).resolve().is_relative_to(storage_root)
                )
            )
            for fname in sorted(filenames):
                fpath = root_path / fname
                if fpath.is_symlink() or not fpath.is_file():
                    continue
                try:
                    size = fpath.stat().st_size
                except OSError:
                    continue
                files.append(fpath.relative_to(self.workspace_dir).as_posix())
                total_bytes += size
                if len(files) > self.max_files or total_bytes > self.max_bytes:
                    return files, total_bytes, True

        return files, total_bytes, False

    def create(
        self,
        label: str = "",
        *,
        memory_len: int | None = None,
        modified_files: frozenset[str] | set[str] | None = None,
    ) -> Checkpoint | None:
        """Records a lightweight pre-turn checkpoint. Returns None if skipped."""
        self.last_warning = None
        if not self.workspace_dir.is_dir():
            return None

        cp_id = self._next_id
        frozen_modified = frozenset(modified_files or ())
        toplevel = self._git_toplevel()

        if toplevel is not None:
            tree_sha, dirty_before, err = self._capture_git_tree(toplevel)
            if tree_sha is not None:
                cp = Checkpoint(
                    id=cp_id,
                    label=label,
                    backend="git",
                    tree_sha=tree_sha,
                    memory_len=memory_len,
                    modified_files=frozen_modified,
                    dirty_before=dirty_before,
                )
                self._stack.append(cp)
                self._next_id += 1
                return cp
            self.last_warning = f"Git checkpoint failed ({err}); falling back to file backup."

        rel_files, total_bytes, exceeded = self._collect_workspace_files()
        if exceeded:
            self.last_warning = (
                f"Skipped checkpoint: non-git workspace exceeds safety limit "
                f"({len(rel_files)} files / {total_bytes} bytes)."
            )
            return None

        cp_dir = self._storage_root() / f"cp_{cp_id:04d}"
        cp_dir.mkdir(parents=True, exist_ok=True)
        for rel in rel_files:
            src = self.workspace_dir / rel
            dst = cp_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        cp = Checkpoint(
            id=cp_id,
            label=label,
            backend="files",
            snapshot_dir=cp_dir,
            manifest=tuple(rel_files),
            memory_len=memory_len,
            modified_files=frozen_modified,
            dirty_before=False,
        )
        self._stack.append(cp)
        self._next_id += 1
        return cp

    def _restore_git(self, cp: Checkpoint, toplevel: Path) -> UndoResult:
        assert cp.tree_sha is not None
        with tempfile.TemporaryDirectory(prefix="mh_git_restore_") as tmp:
            tmp_index_now = Path(tmp) / "index_now"
            add_res = _run_git(
                ["add", "-A", "--", "."], cwd=self.workspace_dir, index_file=tmp_index_now
            )
            if add_res.returncode != 0:
                err = add_res.stderr.decode("utf-8", errors="replace").strip()
                return UndoResult(
                    ok=False,
                    checkpoint_id=cp.id,
                    message=f"Failed to inspect workspace for undo: {err}",
                )

            cur_tree_res = _run_git(
                ["write-tree"], cwd=self.workspace_dir, index_file=tmp_index_now
            )
            cur_tree = (
                cur_tree_res.stdout.decode("utf-8", errors="replace").strip()
                if cur_tree_res.returncode == 0
                else ""
            )

            ls_now_res = _run_git(
                ["ls-files", "-z", "--", "."], cwd=self.workspace_dir, index_file=tmp_index_now
            )
            current_paths = {
                p for p in ls_now_res.stdout.decode("utf-8", errors="replace").split("\0") if p
            }

            cp_paths: set[str] = set()
            if cp.tree_sha != _EMPTY_GIT_TREE:
                ls_cp_res = _run_git(
                    ["ls-tree", "-r", "--name-only", "-z", cp.tree_sha, "--", "."],
                    cwd=self.workspace_dir,
                )
                if ls_cp_res.returncode != 0:
                    err = ls_cp_res.stderr.decode("utf-8", errors="replace").strip()
                    return UndoResult(
                        ok=False,
                        checkpoint_id=cp.id,
                        message=f"Failed to read checkpoint tree: {err}",
                    )
                cp_paths = {
                    p for p in ls_cp_res.stdout.decode("utf-8", errors="replace").split("\0") if p
                }

            removed: list[str] = []
            for rel in sorted(current_paths - cp_paths):
                candidate = self.workspace_dir / rel
                if candidate.parent.resolve().is_relative_to(self.workspace_dir) and (
                    candidate.is_symlink() or candidate.is_file()
                ):
                    candidate.unlink(missing_ok=True)
                    _prune_empty_parents(candidate, self.workspace_dir)
                    removed.append(rel)

            restored: list[str] = []
            if cur_tree and cur_tree != cp.tree_sha:
                diff_res = _run_git(
                    [
                        "diff-tree",
                        "-r",
                        "--relative",
                        "--name-status",
                        "--no-renames",
                        "-z",
                        cur_tree,
                        cp.tree_sha,
                        "--",
                        ".",
                    ],
                    cwd=self.workspace_dir,
                )
                if diff_res.returncode == 0:
                    tokens = diff_res.stdout.decode("utf-8", errors="replace").split("\0")
                    i = 0
                    while i + 1 < len(tokens):
                        status, rel = tokens[i], tokens[i + 1]
                        i += 2
                        if status and not status.startswith("D") and rel:
                            restored.append(rel)

            if cp_paths:
                tmp_index_restore = Path(tmp) / "index_restore"
                read_res = _run_git(
                    ["read-tree", cp.tree_sha], cwd=toplevel, index_file=tmp_index_restore
                )
                if read_res.returncode != 0:
                    err = read_res.stderr.decode("utf-8", errors="replace").strip()
                    return UndoResult(
                        ok=False,
                        checkpoint_id=cp.id,
                        message=f"Failed to stage checkpoint tree for restore: {err}",
                    )
                co_res = _run_git(
                    ["checkout-index", "-a", "-f"],
                    cwd=toplevel,
                    index_file=tmp_index_restore,
                )
                if co_res.returncode != 0:
                    err = co_res.stderr.decode("utf-8", errors="replace").strip()
                    return UndoResult(
                        ok=False,
                        checkpoint_id=cp.id,
                        message=f"Failed to checkout checkpoint files: {err}",
                    )

        return UndoResult(
            ok=True,
            checkpoint_id=cp.id,
            restored_files=tuple(sorted(restored)),
            removed_files=tuple(removed),
            message=self._format_undo_message(cp, restored, removed),
        )

    def _restore_files(self, cp: Checkpoint) -> UndoResult:
        if cp.snapshot_dir is None or not cp.snapshot_dir.is_dir():
            return UndoResult(
                ok=False,
                checkpoint_id=cp.id,
                message=f"Checkpoint #{cp.id} backup directory is missing.",
            )

        current_files, _, _ = self._collect_workspace_files()
        cp_set = set(cp.manifest)
        removed: list[str] = []
        for rel in sorted(set(current_files) - cp_set):
            target = (self.workspace_dir / rel).resolve()
            if target.is_relative_to(self.workspace_dir) and target.is_file():
                target.unlink(missing_ok=True)
                _prune_empty_parents(target, self.workspace_dir)
                removed.append(rel)

        restored: list[str] = []
        for rel in cp.manifest:
            src = cp.snapshot_dir / rel
            dst = self.workspace_dir / rel
            if not src.is_file():
                continue
            changed = not dst.is_file() or dst.read_bytes() != src.read_bytes()
            if changed:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                restored.append(rel)

        shutil.rmtree(cp.snapshot_dir, ignore_errors=True)
        return UndoResult(
            ok=True,
            checkpoint_id=cp.id,
            restored_files=tuple(restored),
            removed_files=tuple(removed),
            message=self._format_undo_message(cp, restored, removed),
        )

    @staticmethod
    def _format_undo_message(cp: Checkpoint, restored: list[str], removed: list[str]) -> str:
        if not restored and not removed:
            return f"Reverted turn #{cp.id}: workspace was already unchanged."
        parts: list[str] = []
        if restored:
            parts.append(f"{len(restored)} file(s) restored")
        if removed:
            parts.append(f"{len(removed)} file(s) removed")
        return f"Reverted turn #{cp.id}: {', '.join(parts)}."

    def undo(self, memory: Memory | None = None) -> UndoResult:
        """Pops and restores the latest pre-turn workspace checkpoint."""
        if not self._stack:
            return UndoResult(
                ok=False,
                message="Nothing to undo: no workspace checkpoints available.",
            )

        cp = self._stack.pop()
        if cp.backend == "git":
            toplevel = self._git_toplevel()
            if toplevel is None:
                return UndoResult(
                    ok=False,
                    checkpoint_id=cp.id,
                    message="Cannot restore git checkpoint: workspace is no longer a git repository.",
                )
            result = self._restore_git(cp, toplevel)
        else:
            result = self._restore_files(cp)

        if result.ok and memory is not None and cp.memory_len is not None:
            if 0 <= cp.memory_len <= len(memory.messages):
                memory.messages = memory.messages[: cp.memory_len]
                memory.modified_files = set(cp.modified_files)

        return result

    def cleanup(self) -> None:
        """Releases any temporary checkpoint backup directory."""
        if self._temp_storage_dir is not None:
            self._temp_storage_dir.cleanup()
            self._temp_storage_dir = None
