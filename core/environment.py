from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ContainerEnvironment(Protocol):
    """Protocol for sandboxed environments executing inside an isolated container."""

    container_name: str
    is_running: bool

    def execute_bash(self, command: str, timeout: float = 30.0) -> tuple[int, str]:
        """Executes a bash command inside the container and returns (exit_code, output)."""
        ...


def _decode_timeout_output(exc: subprocess.TimeoutExpired, timeout: float) -> tuple[int, str]:
    """Decodes subprocess timeout output gracefully, handling bytes or string."""
    stdout_str = (
        exc.stdout.decode("utf-8", errors="replace")
        if isinstance(exc.stdout, bytes)
        else (exc.stdout or "")
    )
    return 124, f"{stdout_str}\n[Command timed out after {timeout} seconds]"


def _run_command_process(
    cmd: str | list[str],
    *,
    cwd: Path | str | None = None,
    timeout: float = 30.0,
    shell: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Runs a subprocess with standardized text-mode streaming and unified stdout/stderr."""
    return subprocess.run(
        cmd,
        shell=shell,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )


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

    def _resolve_target(self, path_str: str) -> Path:
        """Expands and resolves a path string relative to the workspace directory."""
        expanded = os.path.expanduser(path_str)
        target = Path(expanded)
        if not target.is_absolute():
            return (self.workspace_dir / target).resolve()
        return target.resolve()

    def resolve_path(self, rel_path: str) -> Path:
        """Resolve a path strictly inside workspace (used for write operations)."""
        target = self._resolve_target(rel_path)
        if not target.is_relative_to(self.workspace_dir):
            raise PermissionError(
                f"Access denied: path '{rel_path}' resolves outside workspace sandbox."
            )
        return target

    def resolve_read_path(self, path_str: str) -> Path:
        """Resolve a path for reading, permitting workspace and authorized global directories."""
        target = self._resolve_target(path_str)

        # Check workspace
        if target.is_relative_to(self.workspace_dir):
            return target

        # Check allowed external read directories (e.g. ~/.microharness/)
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
        if path.is_dir():
            raise IsADirectoryError(f"Target path is a directory: {rel_path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def edit_file(self, rel_path: str, old_str: str, new_str: str) -> tuple[int, str]:
        """Replaces a unique exact occurrence of old_str with new_str in a workspace file."""
        if not old_str:
            raise ValueError("old_str cannot be empty.")
        path = self.resolve_path(rel_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {rel_path}")
        if not path.is_file():
            raise IsADirectoryError(f"Path is a directory: {rel_path}")

        content = path.read_text(encoding="utf-8", errors="replace")
        occurrences = content.count(old_str)
        if occurrences == 0:
            raise ValueError(
                f"Target string not found in '{rel_path}'. Make sure old_str matches exactly, including indentation."
            )
        if occurrences > 1:
            raise ValueError(
                f"Target string found {occurrences} times in '{rel_path}'. "
                "The target string must be unique. Provide more surrounding context lines to disambiguate."
            )

        updated_content = content.replace(old_str, new_str, 1)
        path.write_text(updated_content, encoding="utf-8")
        return occurrences, f"Successfully replaced 1 occurrence in '{rel_path}'."

    @property
    def is_running(self) -> bool:
        """Returns True if the environment is ready for command execution."""
        return True

    def get_git_patch(self) -> str | None:
        """Captures git diff of the workspace if it is a git repository.

        Safely represents untracked files using intent-to-add without destroying
        pre-existing staged changes.
        """
        code, _ = self.execute_bash("git rev-parse --is-inside-work-tree")
        if code != 0:
            return None

        # Find untracked files to add with intent-to-add
        code, untracked_out = self.execute_bash("git ls-files --others --exclude-standard")
        untracked = [line.strip() for line in untracked_out.splitlines() if line.strip()]

        if untracked:
            quoted_untracked = " ".join(shlex.quote(f) for f in untracked)
            self.execute_bash(f"git add -N -- {quoted_untracked}")

        code, diff_out = self.execute_bash("git diff HEAD")
        if code != 0 or not diff_out.strip():
            code, diff_out = self.execute_bash("git diff")

        # Only reset the specific untracked files we added with intent-to-add
        if untracked:
            quoted_untracked = " ".join(shlex.quote(f) for f in untracked)
            self.execute_bash(f"git reset -q -- {quoted_untracked}")

        return diff_out.strip() if diff_out.strip() else None

    def execute_bash(self, command: str, timeout: float = 30.0) -> tuple[int, str]:
        """Execute a shell command inside the workspace directory."""
        try:
            process = _run_command_process(
                command,
                cwd=self.workspace_dir,
                timeout=timeout,
                shell=True,
            )
            return process.returncode, process.stdout
        except subprocess.TimeoutExpired as exc:
            return _decode_timeout_output(exc, timeout)
        except Exception as exc:
            return 1, f"[Execution error: {exc}]"

    def teardown(self) -> None:
        """Clean up workspace if configured to do so."""
        if self.cleanup_on_teardown and self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir, ignore_errors=True)

    def __enter__(self) -> Environment:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.teardown()


def is_docker_available() -> bool:
    """Checks if the docker CLI is installed and the daemon is responsive."""
    if not shutil.which("docker"):
        return False
    try:
        res = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        )
        return res.returncode == 0
    except Exception:
        return False


class DockerEnvironment(Environment):
    """Provides execution sandboxing inside an isolated Docker container.

    - Mounts the host workspace directory into the container at `/workspace`.
    - File read/write/edit operations happen on the host mounted workspace with instant I/O.
    - Bash commands (`execute_bash`) execute inside the isolated container via `docker exec`.
    - Guarantees complete operating system, process, filesystem, and network isolation.
    """

    container_name: str = ""

    def __init__(
        self,
        image: str = "python:3.11",
        workspace_dir: Path | str | None = None,
        logs_dir: Path | str | None = None,
        allowed_read_dirs: list[Path | str] | None = None,
        container_name: str | None = None,
        cleanup_on_teardown: bool = True,
        docker_args: list[str] | None = None,
        init_workdir: str | None = None,
    ):
        super().__init__(
            workspace_dir=workspace_dir,
            logs_dir=logs_dir,
            allowed_read_dirs=allowed_read_dirs,
            cleanup_on_teardown=cleanup_on_teardown,
        )
        self.image = image
        self.container_name = container_name or f"microharness_{uuid.uuid4().hex[:10]}"
        self.docker_args = docker_args or []
        self.init_workdir = init_workdir
        self._container_started = False
        try:
            self._start_container()
        except Exception:
            self.teardown()
            raise

    def _start_container(self) -> None:
        """Starts a persistent background container mounting workspace at /workspace."""
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            self.container_name,
            "-v",
            f"{self.workspace_dir}:/workspace",
            "-w",
            "/workspace",
            *self.docker_args,
            self.image,
            "tail",
            "-f",
            "/dev/null",
        ]
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"Failed to start Docker container '{self.container_name}' with image '{self.image}':\n{res.stderr.strip()}"
            )
        self._container_started = True

        if self.init_workdir:
            # Sync pre-baked files from container's internal workdir (e.g. /app) into host workspace
            cp_res = subprocess.run(
                [
                    "docker",
                    "cp",
                    f"{self.container_name}:{self.init_workdir}/.",
                    str(self.workspace_dir),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if cp_res.returncode == 0:
                # Symlink internal workdir to /workspace inside container so paths match
                quoted_workdir = shlex.quote(self.init_workdir)
                subprocess.run(
                    [
                        "docker",
                        "exec",
                        self.container_name,
                        "bash",
                        "-c",
                        f"rm -rf {quoted_workdir} && ln -s /workspace {quoted_workdir}",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )

    @property
    def is_running(self) -> bool:
        """Returns True if the Docker container is currently active."""
        return self._container_started

    def execute_bash(self, command: str, timeout: float = 30.0) -> tuple[int, str]:
        """Execute a shell command inside the Docker container at /workspace."""
        if not self._container_started:
            return 1, "[Error: Docker container is not running]"

        cmd = [
            "docker",
            "exec",
            "-w",
            "/workspace",
            self.container_name,
            "/bin/bash",
            "-c",
            command,
        ]
        try:
            process = _run_command_process(cmd, timeout=timeout)
            # If bash doesn't exist in the target image (e.g. alpine), fall back to sh (index 5)
            if process.returncode == 127 and "bash: not found" in (process.stdout or ""):
                cmd[5] = "/bin/sh"
                process = _run_command_process(cmd, timeout=timeout)
            return process.returncode, process.stdout
        except subprocess.TimeoutExpired as exc:
            return _decode_timeout_output(exc, timeout)
        except Exception as exc:
            return 1, f"[Docker execution error: {exc}]"

    def teardown(self) -> None:
        """Stops and removes the Docker container, then cleans up host files if requested."""
        if self._container_started:
            subprocess.run(
                ["docker", "rm", "-f", self.container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            self._container_started = False
        super().teardown()
