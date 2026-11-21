from __future__ import annotations

import difflib
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import textwrap
import uuid
from collections.abc import Callable, Sequence
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


def _decode_timeout_output(
    exc: subprocess.TimeoutExpired,
    timeout: float,
    note: str | None = None,
    stdout_override: str | None = None,
) -> tuple[int, str]:
    """Decodes subprocess timeout output gracefully, handling bytes or string."""
    if stdout_override is not None:
        stdout_str = stdout_override
    else:
        stdout_str = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
    suffix = f"[Command timed out after {timeout} seconds]"
    if note:
        suffix = f"[Command timed out after {timeout} seconds; {note}]"
    return 124, f"{stdout_str}\n{suffix}"


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


def _find_exact_match_lines(content: str, target: str) -> list[int]:
    """Returns 1-indexed starting line numbers for all non-overlapping occurrences."""
    lines: list[int] = []
    start = 0
    while True:
        idx = content.find(target, start)
        if idx == -1:
            break
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


def _rstrip_lines(lines: Sequence[str]) -> list[str]:
    return [line.rstrip() for line in lines]


def _dedent_lines(lines: Sequence[str]) -> list[str]:
    stripped = _rstrip_lines(lines)
    indent = _common_leading_indent(stripped)
    if not indent:
        return [line if line.strip() else "" for line in stripped]
    prefix_len = len(indent)
    return [line[prefix_len:] if line.strip() else "" for line in stripped]


def _strip_lines(lines: Sequence[str]) -> list[str]:
    return [line.strip() for line in lines]


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


def _format_closest_match_feedback(content: str, old_str: str, rel_path: str) -> str:
    """Formats an actionable error message with the closest difflib block and line numbers."""
    base = (
        f"Target string not found in '{rel_path}'. "
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


class Environment:
    """Provides workspace execution, security boundaries, and sandboxing."""

    def __init__(
        self,
        workspace_dir: Path | str | None = None,
        logs_dir: Path | str | None = None,
        allowed_read_dirs: Sequence[Path | str] | None = None,
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
        """Replaces a unique occurrence of old_str with new_str in a workspace file."""
        if not old_str:
            raise ValueError("old_str cannot be empty.")
        path = self.resolve_path(rel_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {rel_path}")
        if not path.is_file():
            raise IsADirectoryError(f"Path is a directory: {rel_path}")

        content = path.read_text(encoding="utf-8", errors="replace")
        occurrences = content.count(old_str)
        if occurrences > 1:
            line_list = ", ".join(str(ln) for ln in _find_exact_match_lines(content, old_str))
            raise ValueError(
                f"Target string found {occurrences} times in '{rel_path}' (at lines: {line_list}). "
                "The target string must be unique. Provide more surrounding context lines to disambiguate."
            )

        if occurrences == 1:
            updated_content = content.replace(old_str, new_str, 1)
            path.write_text(updated_content, encoding="utf-8")
            return occurrences, f"Successfully replaced 1 occurrence in '{rel_path}'."

        updated_content = self._try_normalized_edit(content, rel_path, old_str, new_str)
        if updated_content is None:
            raise ValueError(_format_closest_match_feedback(content, old_str, rel_path))

        path.write_text(updated_content, encoding="utf-8")
        return 1, (
            f"Successfully replaced 1 occurrence in '{rel_path}' "
            "(matched with whitespace normalization)."
        )

    def _try_normalized_edit(
        self, content: str, rel_path: str, old_str: str, new_str: str
    ) -> str | None:
        """Attempts a unique whitespace/indentation-normalized replacement."""
        if not old_str.strip():
            return None

        file_lines = content.splitlines(keepends=True)
        old_lines = old_str.splitlines()
        normalizers: tuple[tuple[Callable[[Sequence[str]], list[str]], bool], ...] = (
            (_rstrip_lines, False),
            (_dedent_lines, True),
            (_strip_lines, True),
        )
        for normalizer, allow_reindent in normalizers:
            matches = _find_normalized_block_matches(file_lines, old_lines, normalizer)
            if len(matches) > 1:
                line_list = ", ".join(str(idx + 1) for idx in matches)
                raise ValueError(
                    f"Target string matched {len(matches)} times in '{rel_path}' after whitespace "
                    f"normalization (at lines: {line_list}). The target string must be unique. "
                    "Provide more surrounding context lines to disambiguate."
                )
            if len(matches) == 1:
                return self._apply_line_block_replacement(
                    file_lines,
                    start_idx=matches[0],
                    line_count=len(old_lines),
                    old_lines=old_lines,
                    new_str=new_str,
                    allow_reindent=allow_reindent,
                )
        return None

    @staticmethod
    def _apply_line_block_replacement(
        file_lines: list[str],
        *,
        start_idx: int,
        line_count: int,
        old_lines: list[str],
        new_str: str,
        allow_reindent: bool,
    ) -> str:
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

        if not diff_out.strip():
            return None
        return diff_out if diff_out.endswith("\n") else f"{diff_out}\n"

    def execute_bash(self, command: str, timeout: float = 30.0) -> tuple[int, str]:
        """Execute a shell command inside the workspace directory."""
        if os.name != "posix":
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

        proc: subprocess.Popen[str] | None = None
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=str(self.workspace_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )
            stdout, _ = proc.communicate(timeout=timeout)
            return proc.returncode, stdout
        except subprocess.TimeoutExpired as exc:
            stdout_override: str | None = None
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
                try:
                    tail, _ = proc.communicate(timeout=2.0)
                except Exception:
                    tail = ""
                    try:
                        proc.kill()
                        proc.wait(timeout=0.5)
                    except Exception:
                        pass
                    if proc.stdout is not None:
                        try:
                            proc.stdout.close()
                        except Exception:
                            pass
                if tail:
                    existing = exc.stdout or ""
                    if isinstance(existing, bytes):
                        existing = existing.decode("utf-8", errors="replace")
                    stdout_override = f"{existing}{tail}" if existing else tail
            return _decode_timeout_output(exc, timeout, stdout_override=stdout_override)
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
    network: str | None = None

    def __init__(
        self,
        image: str = "python:3.11",
        workspace_dir: Path | str | None = None,
        logs_dir: Path | str | None = None,
        allowed_read_dirs: Sequence[Path | str] | None = None,
        container_name: str | None = None,
        cleanup_on_teardown: bool = True,
        docker_args: list[str] | None = None,
        init_workdir: str | None = None,
        network: str | None = None,
    ):
        super().__init__(
            workspace_dir=workspace_dir,
            logs_dir=logs_dir,
            allowed_read_dirs=allowed_read_dirs,
            cleanup_on_teardown=cleanup_on_teardown,
        )
        self.image = image
        self.container_name = container_name or f"microharness_{uuid.uuid4().hex[:10]}"
        self.docker_args = list(docker_args or [])
        self.init_workdir = init_workdir
        self.network = network
        self._container_started = False
        try:
            self._start_container()
        except Exception:
            self.teardown()
            raise

    def _effective_docker_args(self) -> list[str]:
        """Builds docker run arguments, applying network isolation unless already set."""
        args = list(self.docker_args)
        if self.network:
            has_network_flag = any(
                a in ("--network", "--net") or a.startswith(("--network=", "--net=")) for a in args
            )
            if not has_network_flag:
                args = ["--network", self.network, *args]
        return args

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
            *self._effective_docker_args(),
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

    def _cleanup_timed_out_exec(self, exec_id: str) -> str | None:
        """Terminates in-container processes tagged with exec_id after a timeout."""
        target = shlex.quote(f"_MH_EXEC_ID={exec_id}")
        cleanup_script = (
            f"target={target}; found=0; "
            "for f in /proc/[0-9]*/environ; do "
            "p=${f#/proc/}; p=${p%/environ}; "
            '[ "$p" = "$$" ] || [ "$p" = "1" ] && continue; '
            'if tr "\\0" "\\n" < "$f" 2>/dev/null | grep -qx "$target"; then '
            'found=1; kill -TERM "$p" 2>/dev/null || true; '
            "fi; "
            "done; "
            '[ "$found" -eq 0 ] && exit 0; '
            "sleep 0.1 2>/dev/null || sleep 1; "
            "for f in /proc/[0-9]*/environ; do "
            "p=${f#/proc/}; p=${p%/environ}; "
            '[ "$p" = "$$" ] || [ "$p" = "1" ] && continue; '
            'if tr "\\0" "\\n" < "$f" 2>/dev/null | grep -qx "$target"; then '
            'kill -KILL "$p" 2>/dev/null || true; '
            "fi; "
            "done"
        )
        try:
            subprocess.run(
                ["docker", "exec", self.container_name, "/bin/sh", "-c", cleanup_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
                check=False,
            )
            return None
        except subprocess.TimeoutExpired:
            try:
                restart_res = subprocess.run(
                    ["docker", "restart", "-t", "2", self.container_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15.0,
                    check=False,
                )
                if restart_res.returncode == 0:
                    return "container became unresponsive and was restarted"
            except Exception:
                pass
            self._container_started = False
            return "container became unresponsive and restart failed"
        except Exception:
            return None

    def execute_bash(self, command: str, timeout: float = 30.0) -> tuple[int, str]:
        """Execute a shell command inside the Docker container at /workspace."""
        if not self._container_started:
            return 1, "[Error: Docker container is not running]"

        exec_id = f"mh_{uuid.uuid4().hex[:12]}"
        cmd = [
            "docker",
            "exec",
            "-e",
            f"_MH_EXEC_ID={exec_id}",
            "-w",
            "/workspace",
            self.container_name,
            "/bin/bash",
            "-c",
            command,
        ]
        try:
            process = _run_command_process(cmd, timeout=timeout)
            # If bash doesn't exist in the target image (e.g. alpine), fall back to sh
            if process.returncode == 127 and "bash: not found" in (process.stdout or ""):
                cmd[cmd.index("/bin/bash")] = "/bin/sh"
                process = _run_command_process(cmd, timeout=timeout)
            return process.returncode, process.stdout
        except subprocess.TimeoutExpired as exc:
            note = self._cleanup_timed_out_exec(exec_id)
            return _decode_timeout_output(exc, timeout, note=note)
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
