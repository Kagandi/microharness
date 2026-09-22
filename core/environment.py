from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class Environment:
    """Provides workspace execution, security boundaries, and sandboxing."""

    def __init__(
        self,
        workspace_dir: Path | str | None = None,
        logs_dir: Path | str | None = None,
        allowed_read_dirs: list[Path | str] | None = None,
        cleanup_on_teardown: bool = False,
    ):
        if workspace_dir:
            self.workspace_dir = Path(workspace_dir).resolve()
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            self._is_temp = False
        else:
            self.workspace_dir = Path(tempfile.mkdtemp(prefix="mini_harness_")).resolve()
            self._is_temp = True

        self.cleanup_on_teardown = cleanup_on_teardown

        # Read permissions: workspace_dir + any allowed read directories (e.g. global user config/skills)
        self.allowed_read_dirs: list[Path] = [
            Path(d).resolve() for d in (allowed_read_dirs or []) if d is not None
        ]

        # Logs directory
        if logs_dir:
            self.logs_dir = Path(logs_dir).resolve()
        else:
            self.logs_dir = self.workspace_dir / ".logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def populate_files(self, files: dict[str, str]) -> None:
        """Populate initial workspace files."""
        for rel_path, content in files.items():
            self.write_file(rel_path, content)

    def resolve_path(self, rel_path: str) -> Path:
        """Resolve a path strictly inside workspace (used for write operations)."""
        expanded = os.path.expanduser(rel_path)
        target = Path(expanded)
        if not target.is_absolute():
            target = (self.workspace_dir / target).resolve()
        else:
            target = target.resolve()

        if not target.is_relative_to(self.workspace_dir):
            raise PermissionError(
                f"Access denied: path '{rel_path}' resolves outside workspace sandbox."
            )
        return target

    def resolve_read_path(self, path_str: str) -> Path:
        """Resolve a path for reading, permitting workspace and authorized global directories."""
        expanded = os.path.expanduser(path_str)
        target = Path(expanded)
        if not target.is_absolute():
            target = (self.workspace_dir / target).resolve()
        else:
            target = target.resolve()

        # Check workspace
        if target.is_relative_to(self.workspace_dir):
            return target

        # Check allowed external read directories (e.g. ~/.mini-harness/)
        for allowed in self.allowed_read_dirs:
            if target.is_relative_to(allowed):
                return target

        raise PermissionError(
            f"Access denied: path '{path_str}' resolves outside workspace and allowed read directories."
        )

    def read_file(self, path_str: str, start_line: int = 1, end_line: int = -1) -> str:
        """Read file contents with optional line slice (1-indexed, inclusive)."""
        path = self.resolve_read_path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path_str}")
        if not path.is_file():
            raise IsADirectoryError(f"Path is a directory: {path_str}")

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        total_lines = len(lines)

        start = max(1, start_line)
        end = total_lines if end_line == -1 else min(total_lines, end_line)

        if start > total_lines:
            return f"[File has {total_lines} lines; start_line {start} is beyond end of file]"

        selected_lines = lines[start - 1 : end]
        numbered = [f"{start + i:4d} | {line}" for i, line in enumerate(selected_lines)]
        return "".join(numbered)

    def write_file(self, rel_path: str, content: str) -> None:
        """Write or overwrite a file within the workspace (strictly workspace-scoped)."""
        path = self.resolve_path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def execute_bash(self, command: str, timeout: float = 30.0) -> tuple[int, str]:
        """Execute a shell command inside the workspace directory."""
        try:
            process = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
            return process.returncode, process.stdout
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + f"\n[Command timed out after {timeout} seconds]"
            return 124, output
        except Exception as exc:
            return 1, f"[Execution error: {exc}]"

    def teardown(self) -> None:
        """Clean up workspace if configured to do so."""
        if self.cleanup_on_teardown and self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir, ignore_errors=True)
