from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.editing import apply_text_edit, apply_text_edits
from core.types import normalize_permission_mode


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

        # Check logs directory (spilled tool outputs)
        if target.is_relative_to(self.logs_dir):
            return target

        # Check allowed external read directories (e.g. ~/.microharness/)
        for allowed in self.allowed_read_dirs:
            if target.is_relative_to(allowed):
                return target

        raise PermissionError(
            f"Access denied: path '{path_str}' resolves outside workspace and allowed read directories."
        )

    @property
    def agent_logs_dir(self) -> str:
        """Directory path where spilled logs are accessible to tools (bash/read_file)."""
        return str(self.logs_dir)

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

    def edit_file(
        self,
        rel_path: str,
        old_str: str = "",
        new_str: str = "",
        *,
        edits: Sequence[tuple[str, str]] | None = None,
    ) -> tuple[int, str]:
        """Replaces unique target block(s) in a workspace file atomically."""
        if edits is None and not old_str:
            raise ValueError("old_str cannot be empty.")
        path = self.resolve_path(rel_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {rel_path}")
        if not path.is_file():
            raise IsADirectoryError(f"Path is a directory: {rel_path}")

        content = path.read_text(encoding="utf-8", errors="replace")
        if edits is not None:
            updated_content, message = apply_text_edits(content, edits, path_label=rel_path)
            edit_count = len(edits)
        else:
            updated_content, message = apply_text_edit(
                content, old_str, new_str, path_label=rel_path
            )
            edit_count = 1
        path.write_text(updated_content, encoding="utf-8")
        return edit_count, message

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


_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_SAFE_REDIRECTIONS = ("2>&1", "1>&2", "2>/dev/null", "1>/dev/null", "&>/dev/null", ">/dev/null")
_READ_ONLY_CMDS = frozenset(
    "ack ag basename cat cmp comm cut date diff dir dirname echo egrep false fgrep file grep "
    "head id less ls more nl printenv printf pwd readlink realpath rg stat tail test tr tree "
    "true type uname uniq wc which whoami [".split()
)
_FIND_BLOCKED = frozenset(
    "-delete -exec -execdir -ok -okdir -fprint -fprint0 -fprintf -fls".split()
)
_GIT_READ_SUBS = frozenset(
    "blame cat-file check-ignore describe diff diff-files diff-index diff-tree for-each-ref "
    "grep help log ls-files ls-tree merge-base name-rev rev-list rev-parse shortlog show "
    "show-ref status var version".split()
)
_GIT_GLOBAL_VALUE_FLAGS = frozenset(("-C", "--git-dir", "--work-tree"))
_GIT_GLOBAL_BOOL_FLAGS = frozenset(
    "-p --paginate --no-pager --bare --no-replace-objects --literal-pathspecs "
    "--glob-pathspecs --noglob-pathspecs --icase-pathspecs".split()
)
_GIT_REF_MUTATING_FLAGS = {
    "branch": frozenset(
        "-d -D -m -M -c -C -u -f --delete --move --copy --edit-description "
        "--set-upstream-to --unset-upstream --force".split()
    ),
    "tag": frozenset(
        "-d -a -s -f -m -F -u --delete --annotate --sign --force --local-user".split()
    ),
}
_GIT_REF_READ_FLAGS = frozenset(
    "-l --list -v --verify --contains --no-contains --merged --no-merged".split()
)


def _split_shell_segments(command: str) -> tuple[bool, list[str], str]:
    """Splits a shell command into pipeline/chain segments while rejecting unsafe syntax."""
    segments: list[str] = []
    cur: list[str] = []
    in_single = in_double = False
    i, n = 0, len(command)

    def _flush() -> bool:
        seg = "".join(cur).strip()
        cur.clear()
        if seg:
            segments.append(seg)
            return True
        return False

    while i < n:
        ch = command[i]
        if in_single:
            if ch == "'":
                in_single = False
            cur.append(ch)
            i += 1
            continue

        if ch == "\\" and i + 1 < n:
            cur.extend((ch, command[i + 1]))
            i += 2
            continue
        if ch == "`" or command.startswith("$(", i):
            return False, [], "Command substitution is not allowed in read-only mode."
        if in_double:
            if ch == '"':
                in_double = False
            cur.append(ch)
            i += 1
            continue

        if ch == "'":
            in_single = True
        elif ch == '"':
            in_double = True
        elif command.startswith(("<(", ">("), i):
            return False, [], "Process substitution is not allowed in read-only mode."
        elif ch in ("(", ")"):
            return False, [], "Subshells are not allowed in read-only mode."
        elif command.startswith("<<", i):
            return False, [], "Here-documents and here-strings are not allowed in read-only mode."
        elif ch == "#" and (not cur or cur[-1].isspace()):
            i = command.find("\n", i)
            if i == -1:
                break
            continue
        elif command.startswith(("&&", "||"), i):
            if not _flush():
                return False, [], "Empty command segment before operator."
            i += 2
            continue
        elif (not cur or cur[-1].isspace()) and ch in (">", "1", "2", "&"):
            matched = next(
                (
                    pat
                    for pat in _SAFE_REDIRECTIONS
                    if command.startswith(pat, i)
                    and (i + len(pat) == n or command[i + len(pat)] in " \t\r\n;|&")
                ),
                None,
            )
            if matched:
                cur.append(" ")
                i += len(matched)
                continue
            if ch == ">":
                return False, [], "Output redirection is not allowed in read-only mode."
            if ch == "&":
                return False, [], "Background execution `&` is not allowed in read-only mode."
        elif ch == ">":
            return False, [], "Output redirection is not allowed in read-only mode."
        elif command.startswith(";;", i):
            return False, [], "Shell `;;` operator is not allowed in read-only mode."
        elif ch in ("|", ";", "\n"):
            if not _flush() and ch == "|":
                return False, [], "Empty command segment before pipe."
            i += 1
            continue
        elif ch == "&":
            return False, [], "Background execution `&` is not allowed in read-only mode."

        cur.append(ch)
        i += 1

    if in_single or in_double:
        return False, [], "Unterminated quote in command."
    _flush()
    return (True, segments, "") if segments else (False, [], "Empty command.")


def _validate_git_args(args: list[str]) -> tuple[bool, str]:
    """Validates that a `git` invocation uses only read-only subcommands and flags."""
    idx, n = 0, len(args)
    while idx < n:
        tok = args[idx]
        if tok in _GIT_GLOBAL_VALUE_FLAGS:
            if idx + 1 >= n:
                return False, f"Git option '{tok}' requires an argument."
            idx += 2
        elif tok.startswith(("--git-dir=", "--work-tree=")) or tok in _GIT_GLOBAL_BOOL_FLAGS:
            idx += 1
        elif tok in ("-v", "--version", "-h", "--help"):
            return True, ""
        elif tok.startswith("-"):
            return False, f"Git global option '{tok}' is not allowed in read-only mode."
        else:
            break

    if idx >= n:
        return False, "Bare `git` command without a read-only subcommand is not allowed."

    sub, sub_args = args[idx], args[idx + 1 :]
    if any(a == "--output" or a.startswith("--output=") or a == "--ext-diff" for a in sub_args):
        return False, "Git output/ext-diff flags are not allowed in read-only mode."
    if sub in _GIT_READ_SUBS:
        return True, ""
    if sub in _GIT_REF_MUTATING_FLAGS:
        blocked = _GIT_REF_MUTATING_FLAGS[sub]
        has_filter = False
        has_pos = False
        for arg in sub_args:
            if arg in blocked or arg.startswith(
                ("--set-upstream-to=", "--local-user=", "--file=", "--message=")
            ):
                return False, f"Git {sub} flag '{arg}' is not allowed in read-only mode."
            if arg in _GIT_REF_READ_FLAGS or arg.startswith(
                ("--contains=", "--no-contains=", "--merged=", "--no-merged=")
            ):
                has_filter = True
            elif not arg.startswith("-"):
                has_pos = True
        if has_pos and not has_filter:
            return False, f"Creating or modifying git {sub}s is not allowed in read-only mode."
        return True, ""
    return False, f"Git subcommand '{sub}' is not allowed in read-only mode."


def validate_read_only_git_args(args: list[str]) -> tuple[bool, str]:
    """Validates that a ``git`` argument list uses only read-only subcommands and flags."""
    return _validate_git_args(args)


def _validate_read_only_segment(segment: str) -> tuple[bool, str]:
    """Validates one simple command segment against the read-only command allowlist."""
    try:
        tokens = shlex.split(segment)
    except ValueError as exc:
        return False, f"Invalid shell syntax: {exc}"

    while tokens and _ENV_ASSIGN_RE.match(tokens[0]):
        tokens.pop(0)
    if not tokens:
        return False, "Bare variable assignments without a read-only command are not allowed."

    cmd, args = tokens[0], tokens[1:]
    if "/" in cmd:
        return False, f"Path-qualified executable '{cmd}' is not allowed in read-only mode."
    if cmd in _READ_ONLY_CMDS:
        if cmd == "rg" and any(a == "--pre" or a.startswith("--pre=") for a in args):
            return False, "`rg --pre` is not allowed in read-only mode."
        return True, ""
    if cmd == "sort" and any(
        a in ("-o", "--output") or a.startswith("--output=") or (a.startswith("-o") and len(a) > 2)
        for a in args
    ):
        return False, "`sort -o/--output` is not allowed in read-only mode."
    if cmd == "sort":
        return True, ""
    if cmd == "find":
        bad = next(
            (
                a
                for a in args
                if a in _FIND_BLOCKED or a.startswith(("-exec", "-ok", "-fprint", "-fls"))
            ),
            None,
        )
        return (False, f"`find {bad}` is not allowed in read-only mode.") if bad else (True, "")
    if cmd == "fd":
        bad = next(
            (
                a
                for a in args
                if a in ("-x", "--exec", "-X", "--exec-batch")
                or a.startswith(("--exec=", "--exec-batch="))
            ),
            None,
        )
        return (False, f"`fd {bad}` is not allowed in read-only mode.") if bad else (True, "")
    if cmd == "git":
        return _validate_git_args(args)
    return False, f"Command '{cmd}' is not allowed in read-only mode."


def validate_read_only_bash(command: str) -> tuple[bool, str]:
    """Checks whether ``command`` consists exclusively of non-mutating read-only commands."""
    if not isinstance(command, str) or not command.strip():
        return False, "Empty command."
    ok, segments, err = _split_shell_segments(command)
    if not ok:
        return False, err
    for segment in segments:
        seg_ok, seg_err = _validate_read_only_segment(segment)
        if not seg_ok:
            return False, seg_err
    return True, ""


class ReadOnlyEnvironment(Environment):
    """Read-only Environment wrapper that blocks file writes and mutating shell commands."""

    def __init__(self, inner: Environment):
        self._inner = inner
        self.workspace_dir = inner.workspace_dir
        self.logs_dir = inner.logs_dir
        self.allowed_read_dirs = list(inner.allowed_read_dirs)
        self.cleanup_on_teardown = False
        self._is_temp = False

    @property
    def is_running(self) -> bool:
        return self._inner.is_running

    @property
    def agent_logs_dir(self) -> str:
        return self._inner.agent_logs_dir

    def populate_files(self, files: dict[str, str]) -> None:
        raise PermissionError("Workspace is read-only; write operations are not permitted.")

    def resolve_path(self, rel_path: str) -> Path:
        raise PermissionError("Workspace is read-only; write operations are not permitted.")

    def resolve_read_path(self, path_str: str) -> Path:
        return self._inner.resolve_read_path(path_str)

    def read_file(self, path_str: str, start_line: int = 1, end_line: int = -1) -> str:
        return self._inner.read_file(path_str, start_line=start_line, end_line=end_line)

    def write_file(self, rel_path: str, content: str) -> None:
        raise PermissionError("Workspace is read-only; write operations are not permitted.")

    def edit_file(
        self,
        rel_path: str,
        old_str: str = "",
        new_str: str = "",
        *,
        edits: Sequence[tuple[str, str]] | None = None,
    ) -> tuple[int, str]:
        raise PermissionError("Workspace is read-only; write operations are not permitted.")

    def get_git_patch(self) -> str | None:
        return None

    def execute_bash(self, command: str, timeout: float = 30.0) -> tuple[int, str]:
        allowed, reason = validate_read_only_bash(command)
        if not allowed:
            return 1, f"Error: Read-only environment blocked bash command: {reason}"
        return self._inner.execute_bash(command, timeout=timeout)

    def teardown(self) -> None:
        return None


class GuardedEnvironment(Environment):
    """Environment wrapper enforcing configurable host permission modes ('auto', 'ask', 'read-only').

    Delegates read-only mode to :class:`ReadOnlyEnvironment` and prompts via
    ``confirm_handler`` before mutating files or executing bash commands in ``'ask'`` mode.
    """

    def __init__(
        self,
        inner: Environment,
        mode: str = "auto",
        confirm_handler: Callable[[str, str], bool] | None = None,
    ):
        self._inner = inner
        self._ro = ReadOnlyEnvironment(inner)
        self.workspace_dir = inner.workspace_dir
        self.logs_dir = inner.logs_dir
        self.allowed_read_dirs = list(inner.allowed_read_dirs)
        self.cleanup_on_teardown = False
        self._is_temp = False
        self.confirm_handler = confirm_handler
        self._mode = self._normalize_mode(mode)
        if isinstance(inner, ContainerEnvironment):
            self.container_name = inner.container_name

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError(
                f"Unsupported permission_mode: '{mode}'. Valid options: 'auto', 'ask', 'read-only'."
            )
        return normalize_permission_mode(mode)

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = self._normalize_mode(value)

    @property
    def permission_mode(self) -> str:
        return self._mode

    @property
    def is_running(self) -> bool:
        return self._inner.is_running

    @property
    def agent_logs_dir(self) -> str:
        return self._inner.agent_logs_dir

    def confirm_action(self, action: str, detail: str) -> tuple[bool, str]:
        """Checks whether ``action`` may execute under the active permission mode."""
        if self._mode == "read-only":
            return False, f"Workspace is read-only; '{action}' is not permitted."
        if self._mode == "auto":
            return True, ""
        if self.confirm_handler is None:
            return (
                False,
                f"Action '{action}' requires user confirmation (permission_mode='ask'), "
                "but no interactive confirmation handler is available.",
            )
        if not self.confirm_handler(action, detail):
            return False, f"Action '{action}' denied by user."
        return True, ""

    def populate_files(self, files: dict[str, str]) -> None:
        if self._mode == "read-only":
            self._ro.populate_files(files)
            return
        self._inner.populate_files(files)

    def resolve_path(self, rel_path: str) -> Path:
        if self._mode == "read-only":
            return self._ro.resolve_path(rel_path)
        return self._inner.resolve_path(rel_path)

    def resolve_read_path(self, path_str: str) -> Path:
        return self._inner.resolve_read_path(path_str)

    def read_file(self, path_str: str, start_line: int = 1, end_line: int = -1) -> str:
        return self._inner.read_file(path_str, start_line=start_line, end_line=end_line)

    def write_file(self, rel_path: str, content: str) -> None:
        if self._mode == "read-only":
            self._ro.write_file(rel_path, content)
            return
        ok, reason = self.confirm_action("write_file", f"Write '{rel_path}'")
        if not ok:
            raise PermissionError(reason)
        self._inner.write_file(rel_path, content)

    def edit_file(
        self,
        rel_path: str,
        old_str: str = "",
        new_str: str = "",
        *,
        edits: Sequence[tuple[str, str]] | None = None,
    ) -> tuple[int, str]:
        if self._mode == "read-only":
            return self._ro.edit_file(rel_path, old_str, new_str, edits=edits)
        detail = (
            f"Edit '{rel_path}' ({len(edits)} edits)" if edits is not None else f"Edit '{rel_path}'"
        )
        ok, reason = self.confirm_action("edit_file", detail)
        if not ok:
            raise PermissionError(reason)
        return self._inner.edit_file(rel_path, old_str, new_str, edits=edits)

    def get_git_patch(self) -> str | None:
        if self._mode == "read-only":
            return None
        return self._inner.get_git_patch()

    def execute_bash(self, command: str, timeout: float = 30.0) -> tuple[int, str]:
        if self._mode == "read-only":
            return self._ro.execute_bash(command, timeout=timeout)
        ok, reason = self.confirm_action("bash", command)
        if not ok:
            return 1, f"Error: {reason}"
        return self._inner.execute_bash(command, timeout=timeout)

    def teardown(self) -> None:
        self._inner.teardown()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


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

    CONTAINER_LOGS_DIR = "/var/log/microharness"
    container_name: str = ""
    network: str | None = None
    memory_limit: str | None = None
    cpu_limit: str | float | None = None
    pids_limit: int | None = None

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
        memory_limit: str | None = None,
        cpu_limit: str | float | None = None,
        pids_limit: int | None = None,
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
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit
        self._container_started = False
        try:
            self._start_container()
        except Exception:
            self.teardown()
            raise

    def _effective_docker_args(self) -> list[str]:
        """Builds docker run arguments, applying network and resource limits unless already set."""
        args = list(self.docker_args)
        prefix: list[str] = []
        if self.network:
            has_network_flag = any(
                a in ("--network", "--net") or a.startswith(("--network=", "--net=")) for a in args
            )
            if not has_network_flag:
                prefix.extend(["--network", self.network])
        if self.memory_limit:
            has_mem_flag = any(
                a in ("-m", "--memory") or a.startswith(("-m=", "--memory=")) for a in args
            )
            if not has_mem_flag:
                prefix.extend(["--memory", str(self.memory_limit)])
        if self.cpu_limit is not None:
            has_cpu_flag = any(a == "--cpus" or a.startswith("--cpus=") for a in args)
            if not has_cpu_flag:
                prefix.extend(["--cpus", str(self.cpu_limit)])
        if self.pids_limit is not None:
            has_pids_flag = any(a == "--pids-limit" or a.startswith("--pids-limit=") for a in args)
            if not has_pids_flag:
                prefix.extend(["--pids-limit", str(self.pids_limit)])
        return [*prefix, *args]

    def _start_container(self) -> None:
        """Starts a persistent background container mounting workspace at /workspace and logs read-only."""
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            self.container_name,
            "-v",
            f"{self.workspace_dir}:/workspace",
            "-v",
            f"{self.logs_dir}:{self.CONTAINER_LOGS_DIR}:ro",
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
    def agent_logs_dir(self) -> str:
        """Spilled logs are mounted read-only at CONTAINER_LOGS_DIR inside the container."""
        return self.CONTAINER_LOGS_DIR

    def _map_container_workspace_path(self, path_str: str) -> str:
        """Maps container workspace paths (`/workspace/...`, `init_workdir/...`) to workspace-relative paths."""
        prefixes = ["/workspace"]
        init_workdir = getattr(self, "init_workdir", None)
        if isinstance(init_workdir, str):
            normalized = init_workdir.rstrip("/")
            if normalized and normalized not in prefixes:
                prefixes.append(normalized)
        for prefix in prefixes:
            if path_str == prefix or path_str.startswith(f"{prefix}/"):
                rel = path_str[len(prefix) :].lstrip("/")
                return rel or "."
        return path_str

    def resolve_path(self, rel_path: str) -> Path:
        """Resolves host or container workspace paths safely inside `workspace_dir`."""
        return super().resolve_path(self._map_container_workspace_path(rel_path))

    def resolve_read_path(self, path_str: str) -> Path:
        """Resolves container log or workspace paths in addition to normal read paths."""
        prefix = self.CONTAINER_LOGS_DIR
        if path_str == prefix or path_str.startswith(f"{prefix}/"):
            rel = path_str[len(prefix) :].lstrip("/")
            target = (self.logs_dir / rel).resolve()
            if not target.is_relative_to(self.logs_dir):
                raise PermissionError(
                    f"Access denied: path '{path_str}' resolves outside logs directory."
                )
            return target
        return super().resolve_read_path(self._map_container_workspace_path(path_str))

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
