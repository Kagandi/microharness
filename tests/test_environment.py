import tempfile
import unittest
from pathlib import Path

from core.environment import (
    DockerEnvironment,
    Environment,
    GuardedEnvironment,
    ReadOnlyEnvironment,
    is_docker_available,
    validate_read_only_bash,
)


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.env = Environment(workspace_dir=self.temp_dir, cleanup_on_teardown=True)

    def tearDown(self):
        self.env.teardown()

    def test_populate_and_read_files(self):
        self.env.populate_files(
            {
                "a.txt": "Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
                "nested/b.py": "x = 42",
            }
        )

        content = self.env.read_file("a.txt")
        self.assertIn("Line 1", content)
        self.assertIn("Line 5", content)

        # Slice lines 2 to 4
        slice_content = self.env.read_file("a.txt", start_line=2, end_line=4)
        self.assertNotIn("Line 1", slice_content)
        self.assertIn("Line 2", slice_content)
        self.assertIn("Line 4", slice_content)
        self.assertNotIn("Line 5", slice_content)

        # Slice beyond file length returns informative message
        out_of_bounds = self.env.read_file("a.txt", start_line=10)
        self.assertIn("beyond end of file", out_of_bounds)

    def test_read_missing_file_raises_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.env.read_file("does_not_exist.txt")

    def test_sandbox_path_traversal_prevention(self):
        # 1. Relative traversal
        with self.assertRaises(PermissionError):
            self.env.resolve_path("../../etc/passwd")

        # 2. Absolute path outside sandbox
        with self.assertRaises(PermissionError):
            self.env.resolve_path("/etc/passwd")

        # 3. Hidden trick traversal
        with self.assertRaises(PermissionError):
            self.env.resolve_path("sub/../../../../etc/shadow")

    def test_bash_nonzero_exit_and_stderr(self):
        # Verify stderr is captured and non-zero exit codes are preserved
        exit_code, output = self.env.execute_bash(
            "python3 -c 'import sys; sys.stderr.write(\"CRASH_TRACE\"); sys.exit(42)'"
        )
        self.assertEqual(exit_code, 42)
        self.assertIn("CRASH_TRACE", output)

    def test_bash_timeout_kill(self):
        exit_code, output = self.env.execute_bash(
            "python3 -c 'import time; time.sleep(5)'", timeout=0.1
        )
        self.assertEqual(exit_code, 124)
        self.assertIn("timed out after 0.1", output)

    def test_bash_timeout_kills_process_group_and_captures_partial_output(self):
        """Invariant: Host POSIX timeout sends SIGKILL to the entire process group and preserves flushed output."""
        import os
        import signal
        import time

        pid_file = Path(self.temp_dir) / "grandchild.pid"
        cmd = (
            "python3 -c "
            f'"import os, subprocess, sys, time; '
            f"p = subprocess.Popen(['sleep', '60']); "
            f"open({str(pid_file)!r}, 'w').write(str(p.pid)); "
            "print('PARTIAL_BEFORE_TIMEOUT', flush=True); "
            'time.sleep(60)"'
        )
        exit_code, output = self.env.execute_bash(cmd, timeout=0.35)
        self.assertEqual(exit_code, 124)
        self.assertIn("timed out after 0.35", output)
        self.assertIn("PARTIAL_BEFORE_TIMEOUT", output)

        if pid_file.exists():
            grandchild_pid = int(pid_file.read_text(encoding="utf-8").strip())
            # Give the kernel a brief moment to reap the killed child process
            alive = True
            for _ in range(20):
                try:
                    os.kill(grandchild_pid, 0)
                    time.sleep(0.05)
                except ProcessLookupError:
                    alive = False
                    break
            if alive:
                try:
                    os.kill(grandchild_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.assertFalse(alive, f"Grandchild pid {grandchild_pid} survived timeout killpg")

    def test_allowed_read_dirs_permits_external_read(self):
        allowed_dir = Path(tempfile.mkdtemp(prefix="allowed_dir_"))
        external_file = allowed_dir / "global_skill.md"
        external_file.write_text("global skill body", encoding="utf-8")

        env = Environment(workspace_dir=self.temp_dir, allowed_read_dirs=[allowed_dir])
        content = env.read_file(str(external_file))
        self.assertIn("global skill body", content)

    def test_write_outside_workspace_strictly_denied_even_in_allowed_read_dirs(self):
        allowed_dir = Path(tempfile.mkdtemp(prefix="allowed_dir_"))
        env = Environment(workspace_dir=self.temp_dir, allowed_read_dirs=[allowed_dir])

        with self.assertRaises(PermissionError):
            env.write_file(str(allowed_dir / "overwrite.txt"), "malicious write")

    def test_read_outside_workspace_and_allowed_read_dirs_raises_permission_error(self):
        unauthorized_dir = Path(tempfile.mkdtemp(prefix="unauthorized_"))
        unauth_file = unauthorized_dir / "private.txt"
        unauth_file.write_text("private", encoding="utf-8")

        env = Environment(workspace_dir=self.temp_dir, allowed_read_dirs=[])
        with self.assertRaises(PermissionError):
            env.read_file(str(unauth_file))

    def test_sibling_directory_prefix_traversal_prevention(self):
        # Sibling directory starting with same prefix must be denied
        sibling_dir = Path(f"{self.temp_dir}_sibling")
        sibling_dir.mkdir(exist_ok=True)
        sibling_file = sibling_dir / "secret.txt"
        sibling_file.write_text("secret", encoding="utf-8")

        with self.assertRaises(PermissionError):
            self.env.resolve_path(str(sibling_file))
        with self.assertRaises(PermissionError):
            self.env.resolve_read_path(str(sibling_file))

    def test_edit_file_exact_replacement_success(self):
        self.env.write_file("code.py", "def compute():\n    return 41\n")
        count, msg = self.env.edit_file("code.py", "return 41", "return 42")
        self.assertEqual(count, 1)
        self.assertIn("Successfully replaced", msg)
        self.assertEqual(
            self.env.read_file("code.py"), "   1 | def compute():\n   2 |     return 42\n"
        )

    def test_edit_file_not_found_raises_value_error(self):
        self.env.write_file("code.py", "x = 1\n")
        with self.assertRaises(ValueError) as ctx:
            self.env.edit_file("code.py", "x = 99", "x = 2")
        self.assertIn("Target string not found", str(ctx.exception))

    def test_edit_file_ambiguous_multiple_matches_raises_value_error(self):
        self.env.write_file("code.py", "alpha\nline = 10\nbeta\nline = 10\ngamma\n")
        with self.assertRaises(ValueError) as ctx:
            self.env.edit_file("code.py", "line = 10", "line = 20")
        msg = str(ctx.exception)
        self.assertIn("found 2 times", msg)
        self.assertIn("at lines: 2, 4", msg)
        self.assertIn("must be unique", msg)
        self.assertEqual(
            Path(self.temp_dir, "code.py").read_text(encoding="utf-8"),
            "alpha\nline = 10\nbeta\nline = 10\ngamma\n",
        )

    def test_edit_file_trailing_whitespace_and_crlf_fallback(self):
        self.env.write_file("code.py", "def greet():\n    msg = 'hi'   \n    return msg\n")
        count, msg = self.env.edit_file(
            "code.py",
            "    msg = 'hi'\r\n    return msg",
            "    msg = 'hello'\r\n    return msg",
        )
        self.assertEqual(count, 1)
        self.assertIn("Successfully replaced", msg)
        self.assertIn("whitespace normalization", msg)
        self.assertEqual(
            Path(self.temp_dir, "code.py").read_text(encoding="utf-8"),
            "def greet():\n    msg = 'hello'\n    return msg\n",
        )

    def test_edit_file_dedent_reindent_fallback(self):
        original = (
            "class Calc:\n    def add(self, a, b):\n        res = a + b\n        return res\n"
        )
        self.env.write_file("calc.py", original)
        count, msg = self.env.edit_file(
            "calc.py",
            "def add(self, a, b):\n    res = a + b\n    return res",
            "def add(self, a, b):\n    res = a + b + 1\n    return res",
        )
        self.assertEqual(count, 1)
        self.assertIn("Successfully replaced", msg)
        self.assertEqual(
            Path(self.temp_dir, "calc.py").read_text(encoding="utf-8"),
            (
                "class Calc:\n"
                "    def add(self, a, b):\n"
                "        res = a + b + 1\n"
                "        return res\n"
            ),
        )

    def test_edit_file_stripped_line_fallback_mixed_indent(self):
        self.env.write_file("mixed.py", "def f():\n    if True:\n        return 1\n")
        count, _ = self.env.edit_file(
            "mixed.py",
            "  if True:\n    return 1",
            "  if True:\n      return 2",
        )
        self.assertEqual(count, 1)
        self.assertEqual(
            Path(self.temp_dir, "mixed.py").read_text(encoding="utf-8"),
            "def f():\n    if True:\n        return 2\n",
        )

    def test_edit_file_normalized_ambiguous_rejected(self):
        original = "def a():\n    x = 1\n    return x\n\ndef b():\n    x = 1\n    return x   \n"
        self.env.write_file("dup.py", original)
        with self.assertRaises(ValueError) as ctx:
            self.env.edit_file("dup.py", "x = 1\nreturn x", "x = 2\nreturn x")
        msg = str(ctx.exception)
        self.assertIn("matched 2 times", msg)
        self.assertIn("at lines: 2, 6", msg)
        self.assertIn("must be unique", msg)
        self.assertEqual(Path(self.temp_dir, "dup.py").read_text(encoding="utf-8"), original)

    def test_edit_file_zero_matches_includes_closest_block_and_lines(self):
        self.env.write_file(
            "orders.py",
            (
                "def subtotal(price, qty):\n"
                "    if qty <= 0:\n"
                "        return 0\n"
                "    # compute\n"
                "    total = price * qty\n"
                "    return total\n"
                "\n"
                "def done():\n"
                "    pass\n"
            ),
        )
        with self.assertRaises(ValueError) as ctx:
            self.env.edit_file(
                "orders.py",
                "    total = price * quantity\n    return total",
                "    total = price * quantity\n    return total + 1",
            )
        msg = str(ctx.exception)
        self.assertIn("Target string not found", msg)
        self.assertIn("lines 5-6", msg)
        self.assertIn("   5 |     total = price * qty", msg)
        self.assertIn("   6 |     return total", msg)

    def test_edit_file_zero_matches_negative_cases(self):
        # Empty file
        self.env.write_file("empty.py", "")
        with self.assertRaises(ValueError) as ctx_empty:
            self.env.edit_file("empty.py", "x = 1", "x = 2")
        self.assertIn("File is empty", str(ctx_empty.exception))

        # Completely unrelated target string
        self.env.write_file("small.py", "x = 1\n")
        with self.assertRaises(ValueError) as ctx_unrel:
            self.env.edit_file("small.py", "zzzzzzzzzzzz", "y = 2")
        self.assertIn("No close match found", str(ctx_unrel.exception))

        # Whitespace-only target must never match blank lines via normalization
        self.env.write_file("blanks.py", "a = 1\n\nb = 2\n")
        with self.assertRaises(ValueError) as ctx_ws:
            self.env.edit_file("blanks.py", "   \n", "c = 3\n")
        self.assertIn("Target string not found", str(ctx_ws.exception))

    def test_edit_file_normalized_deletion_and_no_trailing_newline(self):
        self.env.write_file("no_eol.py", "a = 1\n    b = 2  ")
        count, _ = self.env.edit_file("no_eol.py", "      b = 2", "      b = 3")
        self.assertEqual(count, 1)
        self.assertEqual(
            Path(self.temp_dir, "no_eol.py").read_text(encoding="utf-8"), "a = 1\n    b = 3"
        )

        self.env.write_file("del.py", "a = 1\n    b = 2\nc = 3\n")
        count_del, _ = self.env.edit_file("del.py", "      b = 2", "")
        self.assertEqual(count_del, 1)
        self.assertEqual(
            Path(self.temp_dir, "del.py").read_text(encoding="utf-8"), "a = 1\nc = 3\n"
        )

    def test_edit_file_empty_target_raises_value_error(self):
        self.env.write_file("code.py", "data = 'hello'\n")
        with self.assertRaises(ValueError):
            self.env.edit_file("code.py", "", "new")

    def test_edit_file_path_traversal_denied(self):
        with self.assertRaises(PermissionError):
            self.env.edit_file("../../etc/passwd", "root", "toor")

    def test_write_file_on_directory_raises_error(self):
        # Create a directory inside the workspace
        (Path(self.temp_dir) / "somedir").mkdir()
        with self.assertRaises(IsADirectoryError):
            self.env.write_file("somedir", "data")

    def test_is_docker_available(self):
        result = is_docker_available()
        self.assertIsInstance(result, bool)

    def test_docker_environment_lifecycle_and_execution(self):
        if not is_docker_available():
            self.skipTest("Docker is not available on this host.")

        temp_workspace = tempfile.mkdtemp(prefix="test_docker_ws_")
        docker_env = DockerEnvironment(
            image="python:3.11",
            workspace_dir=temp_workspace,
            cleanup_on_teardown=True,
        )

        try:
            # 1. Write file on host, execute in container
            docker_env.write_file("script.py", "import sys\nprint('HELLO_DOCKER')\n")
            code, out = docker_env.execute_bash("python3 script.py")
            self.assertEqual(code, 0)
            self.assertEqual(out.strip(), "HELLO_DOCKER")

            # 2. Strict OS isolation check: Host is Darwin (macOS), container must be Linux
            code_os, out_os = docker_env.execute_bash("uname -s")
            self.assertEqual(code_os, 0)
            self.assertEqual(out_os.strip(), "Linux")

            # 3. Container writes file in /workspace, host sees it
            code_write, _ = docker_env.execute_bash("echo 'FROM_CONTAINER' > out.txt")
            self.assertEqual(code_write, 0)
            content = docker_env.read_file("out.txt")
            self.assertIn("FROM_CONTAINER", content)

            # 4. Timeout handling in container
            code_t, out_t = docker_env.execute_bash("sleep 5", timeout=0.5)
            self.assertEqual(code_t, 124)
            self.assertIn("timed out", out_t)
        finally:
            docker_env.teardown()

    def test_docker_environment_context_manager(self):
        if not is_docker_available():
            self.skipTest("Docker is not available on this host.")

        temp_workspace = tempfile.mkdtemp(prefix="test_docker_ctx_")
        with DockerEnvironment(image="python:3.11", workspace_dir=temp_workspace) as d_env:
            code, out = d_env.execute_bash("echo 'CTX_TEST'")
            self.assertEqual(code, 0)
            self.assertEqual(out.strip(), "CTX_TEST")

    def test_get_git_patch_preserves_staged_changes(self):
        # Initialize real git repository
        self.env.execute_bash("git init")
        self.env.execute_bash("git config user.name 'Test'")
        self.env.execute_bash("git config user.email 'test@test.com'")
        self.env.write_file("committed.txt", "v1\n")
        self.env.execute_bash("git add committed.txt && git commit -m 'init'")

        # 1. User pre-stages staged.txt
        self.env.write_file("staged.txt", "staged_content\n")
        self.env.execute_bash("git add staged.txt")

        # 2. Untracked file
        self.env.write_file("untracked.txt", "untracked_content\n")

        # 3. Capture patch
        patch = self.env.get_git_patch()
        self.assertIsNotNone(patch)
        assert patch is not None
        self.assertTrue(patch.endswith("\n"))
        self.assertIn("staged_content", patch)
        self.assertIn("untracked_content", patch)

        # Patch must be directly applicable by git apply without corrupt-patch errors
        patch_file = Path(self.temp_dir) / ".tmp_check.patch"
        patch_file.write_text(patch, encoding="utf-8")
        try:
            stat_code, stat_out = self.env.execute_bash(f"git apply --stat {patch_file.name}")
            self.assertEqual(stat_code, 0, f"git apply --stat failed: {stat_out}")
        finally:
            patch_file.unlink(missing_ok=True)

        # 4. CRITICAL: staged.txt must STILL be staged in git index
        _, status_out = self.env.execute_bash("git status --porcelain")
        status_lines = [line.strip() for line in status_out.splitlines()]
        self.assertIn("A  staged.txt", status_lines)
        self.assertIn("?? untracked.txt", status_lines)

    def test_docker_sh_fallback_argv_index(self):
        from unittest.mock import MagicMock, patch

        d_env = DockerEnvironment.__new__(DockerEnvironment)
        d_env.container_name = "test_container_xyz"
        d_env._container_started = True

        first_res = MagicMock(returncode=127, stdout="bash: not found")
        second_res = MagicMock(returncode=0, stdout="success")

        with patch("subprocess.run", side_effect=[first_res, second_res]) as mock_run:
            code, out = d_env.execute_bash("echo hi")
            self.assertEqual(code, 0)
            self.assertEqual(mock_run.call_count, 2)
            second_cmd = mock_run.call_args_list[1][0][0]
            self.assertEqual(second_cmd[2], "-e")
            self.assertTrue(second_cmd[3].startswith("_MH_EXEC_ID=mh_"))
            container_idx = second_cmd.index("test_container_xyz")
            self.assertEqual(container_idx, 6)
            self.assertEqual(second_cmd[container_idx + 1], "/bin/sh")
            self.assertEqual(second_cmd[container_idx + 2 :], ["-c", "echo hi"])

    def test_is_running_property(self):
        self.assertTrue(self.env.is_running)
        d_env = DockerEnvironment.__new__(DockerEnvironment)
        d_env._container_started = False
        self.assertFalse(d_env.is_running)
        d_env._container_started = True
        self.assertTrue(d_env.is_running)

    def test_docker_env_teardown_on_start_failure(self):
        from unittest.mock import patch

        temp_ws = tempfile.mkdtemp()
        temp_logs = tempfile.mkdtemp()
        self.assertTrue(Path(temp_ws).exists())

        with patch.object(
            DockerEnvironment,
            "_start_container",
            side_effect=RuntimeError("Container startup error"),
        ):
            with self.assertRaises(RuntimeError):
                DockerEnvironment(
                    workspace_dir=temp_ws,
                    logs_dir=temp_logs,
                    cleanup_on_teardown=True,
                )
        # Verify teardown cleaned up the directory
        self.assertFalse(Path(temp_ws).exists())

    def test_container_environment_protocol(self):
        """Invariant: ContainerEnvironment protocol is @runtime_checkable and identifies container backends."""
        from core.environment import ContainerEnvironment

        d_env = DockerEnvironment.__new__(DockerEnvironment)
        d_env.container_name = "test_cont"
        d_env._container_started = True

        self.assertTrue(isinstance(d_env, ContainerEnvironment))

        # Plain host Environment lacks container_name and is not a ContainerEnvironment
        self.assertFalse(isinstance(self.env, ContainerEnvironment))

        # Custom container backend conforming to protocol without inheriting DockerEnvironment
        class CustomContainerEnv:
            container_name = "podman_123"
            is_running = True

            def execute_bash(self, command: str, timeout: float = 30.0) -> tuple[int, str]:
                return 0, "podman execution"

        self.assertTrue(isinstance(CustomContainerEnv(), ContainerEnvironment))

    def test_docker_environment_network_arg_and_dedup(self):
        """Invariant: DockerEnvironment applies --network when configured unless docker_args already sets it."""
        d_env = DockerEnvironment.__new__(DockerEnvironment)
        d_env.network = "none"
        d_env.docker_args = ["--cpus", "2"]
        self.assertEqual(d_env._effective_docker_args(), ["--network", "none", "--cpus", "2"])

        d_env_dup = DockerEnvironment.__new__(DockerEnvironment)
        d_env_dup.network = "none"
        d_env_dup.docker_args = ["--network=bridge", "--cpus", "2"]
        self.assertEqual(d_env_dup._effective_docker_args(), ["--network=bridge", "--cpus", "2"])

        d_env_short = DockerEnvironment.__new__(DockerEnvironment)
        d_env_short.network = "none"
        d_env_short.docker_args = ["--net", "host"]
        self.assertEqual(d_env_short._effective_docker_args(), ["--net", "host"])

    def test_docker_execute_bash_timeout_cleans_up_container_process(self):
        """Invariant: On TimeoutExpired, DockerEnvironment cleans up matching _MH_EXEC_ID processes inside the container."""
        import subprocess
        from unittest.mock import MagicMock, patch

        d_env = DockerEnvironment.__new__(DockerEnvironment)
        d_env.container_name = "test_container_timeout"
        d_env._container_started = True

        timeout_exc = subprocess.TimeoutExpired(
            cmd=["docker", "exec"], timeout=1.0, output="partial"
        )
        cleanup_res = MagicMock(returncode=0, stdout="")

        with patch("subprocess.run", side_effect=[timeout_exc, cleanup_res]) as mock_run:
            code, out = d_env.execute_bash("sleep 10", timeout=1.0)

        self.assertEqual(code, 124)
        self.assertIn("timed out after 1.0 seconds", out)
        self.assertIn("partial", out)
        self.assertEqual(mock_run.call_count, 2)

        exec_cmd = mock_run.call_args_list[0][0][0]
        cleanup_cmd = mock_run.call_args_list[1][0][0]
        self.assertEqual(
            cleanup_cmd[:6],
            ["docker", "exec", "test_container_timeout", "/bin/sh", "-c", cleanup_cmd[5]],
        )
        # Extract exec_id from "-e _MH_EXEC_ID=mh_..." and confirm cleanup targets the exact ID
        self.assertEqual(exec_cmd[2], "-e")
        env_pair = exec_cmd[3]
        self.assertTrue(env_pair.startswith("_MH_EXEC_ID=mh_"))
        exec_id = env_pair.split("=", 1)[1]
        self.assertIn(f"_MH_EXEC_ID={exec_id}", cleanup_cmd[5])

    def test_docker_execute_bash_timeout_restarts_unresponsive_container(self):
        """Invariant: If cleanup exec itself times out, DockerEnvironment restarts the container and reports it."""
        import subprocess
        from unittest.mock import MagicMock, patch

        d_env = DockerEnvironment.__new__(DockerEnvironment)
        d_env.container_name = "wedged_container"
        d_env._container_started = True

        exec_timeout = subprocess.TimeoutExpired(cmd=["docker", "exec"], timeout=1.0)
        cleanup_timeout = subprocess.TimeoutExpired(cmd=["docker", "exec"], timeout=5.0)
        restart_ok = MagicMock(returncode=0)

        with patch(
            "subprocess.run", side_effect=[exec_timeout, cleanup_timeout, restart_ok]
        ) as mock_run:
            code, out = d_env.execute_bash("echo hi", timeout=1.0)

        self.assertEqual(code, 124)
        self.assertIn("container became unresponsive and was restarted", out)
        self.assertTrue(d_env.is_running)
        self.assertEqual(
            mock_run.call_args_list[2][0][0],
            ["docker", "restart", "-t", "2", "wedged_container"],
        )

    def test_docker_execute_bash_timeout_marks_stopped_when_restart_fails(self):
        """Invariant: If container restart fails after unresponsive timeout, is_running becomes False."""
        import subprocess
        from unittest.mock import MagicMock, patch

        d_env = DockerEnvironment.__new__(DockerEnvironment)
        d_env.container_name = "dead_container"
        d_env._container_started = True

        exec_timeout = subprocess.TimeoutExpired(cmd=["docker", "exec"], timeout=1.0)
        cleanup_timeout = subprocess.TimeoutExpired(cmd=["docker", "exec"], timeout=5.0)
        restart_fail = MagicMock(returncode=1)

        with patch("subprocess.run", side_effect=[exec_timeout, cleanup_timeout, restart_fail]):
            code, out = d_env.execute_bash("echo hi", timeout=1.0)

        self.assertEqual(code, 124)
        self.assertIn("restart failed", out)
        self.assertFalse(d_env.is_running)

    def test_docker_environment_network_none_and_timeout_process_kill(self):
        """Integration invariant: --network none blocks outbound HTTP and timeout kills child & pure-shell processes."""
        if not is_docker_available():
            self.skipTest("Docker is not available on this host.")

        temp_workspace = tempfile.mkdtemp(prefix="test_docker_net_none_")
        with DockerEnvironment(
            image="python:3.11",
            workspace_dir=temp_workspace,
            cleanup_on_teardown=True,
            network="none",
        ) as d_env:
            # 1. Verify outbound network is blocked inside container
            code_net, _ = d_env.execute_bash(
                "python3 -c \"import urllib.request; urllib.request.urlopen('https://example.com', timeout=2)\""
            )
            self.assertNotEqual(code_net, 0)

            # 2. Verify timed-out child process is terminated and not orphaned
            code_t, out_t = d_env.execute_bash(
                'python3 -c "import time; time.sleep(60)" # ORPHAN_CANARY_MARKER',
                timeout=0.5,
            )
            self.assertEqual(code_t, 124)
            self.assertIn("timed out", out_t)

            code_ps, out_ps = d_env.execute_bash(
                "ps -ef | grep -v grep | grep ORPHAN_CANARY_MARKER || echo NO_ORPHAN"
            )
            self.assertEqual(code_ps, 0)
            self.assertEqual(out_ps.strip(), "NO_ORPHAN")

            # 3. Verify pure-shell infinite loop (no child exec) is also terminated
            code_sh, out_sh = d_env.execute_bash(
                "while true; do :; done # PURE_SHELL_CANARY",
                timeout=0.5,
            )
            self.assertEqual(code_sh, 124)
            self.assertIn("timed out", out_sh)

            code_ps_sh, out_ps_sh = d_env.execute_bash(
                "ps -ef | grep -v grep | grep PURE_SHELL_CANARY || echo NO_SHELL_ORPHAN"
            )
            self.assertEqual(code_ps_sh, 0)
            self.assertEqual(out_ps_sh.strip(), "NO_SHELL_ORPHAN")

    def test_docker_mounts_logs_dir_read_only_and_resolves_container_log_paths(self):
        """Invariant: DockerEnvironment mounts logs_dir at /logs:ro and resolves /logs/... reads safely."""
        from unittest.mock import MagicMock, patch

        temp_ws = tempfile.mkdtemp(prefix="docker_logs_ws_")
        temp_logs = tempfile.mkdtemp(prefix="docker_logs_dir_")
        log_file = Path(temp_logs) / "step_01_bash_abc123.log"
        log_file.write_text("line1\nMATCH_TOKEN\nline3\n", encoding="utf-8")

        with patch(
            "subprocess.run", return_value=MagicMock(returncode=0, stdout="cid", stderr="")
        ) as mock_run:
            d_env = DockerEnvironment(
                image="python:3.11",
                workspace_dir=temp_ws,
                logs_dir=temp_logs,
                cleanup_on_teardown=True,
            )
            try:
                docker_run_cmd = mock_run.call_args_list[0][0][0]
                self.assertIn(f"{Path(temp_logs).resolve()}:/logs:ro", docker_run_cmd)
                self.assertEqual(d_env.agent_logs_dir, "/logs")

                # read_file works on /logs/<file>
                content = d_env.read_file("/logs/step_01_bash_abc123.log")
                self.assertEqual(content, "   1 | line1\n   2 | MATCH_TOKEN\n   3 | line3\n")

                # Negative cases: traversal and writes into /logs are denied
                with self.assertRaises(PermissionError):
                    d_env.read_file("/logs/../../etc/passwd")
                with self.assertRaises(PermissionError):
                    d_env.read_file("/logs_evil/secret.txt")
                with self.assertRaises(PermissionError):
                    d_env.write_file("/logs/evil.log", "nope")

                # ReadOnlyEnvironment wrapping DockerEnvironment delegates /logs access & grep
                from core.environment import ReadOnlyEnvironment

                ro_env = ReadOnlyEnvironment(d_env)
                self.assertEqual(ro_env.agent_logs_dir, "/logs")
                self.assertEqual(
                    ro_env.read_file("/logs/step_01_bash_abc123.log"),
                    "   1 | line1\n   2 | MATCH_TOKEN\n   3 | line3\n",
                )
                with self.assertRaises(PermissionError):
                    ro_env.write_file("/logs/evil.log", "nope")
                code_rm, out_rm = ro_env.execute_bash("rm /logs/step_01_bash_abc123.log")
                self.assertEqual(code_rm, 1)
                self.assertIn("blocked bash command", out_rm)
            finally:
                d_env.teardown()

    def test_docker_spilled_logs_accessible_via_grep_and_read_file(self):
        """Integration invariant: in-container grep and host read_file both access spilled logs at /logs."""
        if not is_docker_available():
            self.skipTest("Docker is not available on this host.")

        from core.tools import ReadFileTool, ToolRegistry

        temp_ws = tempfile.mkdtemp(prefix="docker_spill_ws_")
        temp_logs = tempfile.mkdtemp(prefix="docker_spill_logs_")
        with DockerEnvironment(
            image="python:3.11",
            workspace_dir=temp_ws,
            logs_dir=temp_logs,
            cleanup_on_teardown=True,
        ) as d_env:

            class BigTool:
                name = "big_tool"
                description = "Emits big output"
                parameters = {"type": "object", "properties": {}}

                def execute(self):
                    return 0, (
                        "HEAD\n" + ("noise line\n" * 40) + "NEEDLE_LINE\n" + ("tail line\n" * 40)
                    )

            reg = ToolRegistry(env=d_env, max_tool_tokens=30)
            reg.register(BigTool())
            reg.register(ReadFileTool(d_env))

            res = reg.dispatch(
                type("TC", (), {"id": "call123456", "name": "big_tool", "arguments": {}})(),
                step_index=1,
            )
            self.assertTrue(res.was_truncated)
            self.assertEqual(res.raw_log_path, "/logs/step_01_big_tool_call12.log")
            self.assertIn("/logs/step_01_big_tool_call12.log", res.output)

            # In-container grep works on res.raw_log_path
            code, grep_out = d_env.execute_bash(f"grep NEEDLE_LINE {res.raw_log_path}")
            self.assertEqual(code, 0)
            self.assertEqual(grep_out.strip(), "NEEDLE_LINE")

            # read_file works on the exact same path
            read_out = d_env.read_file(res.raw_log_path)
            self.assertIn("NEEDLE_LINE", read_out)

            # Mount is read-only inside container
            ro_code, _ = d_env.execute_bash("touch /logs/should_fail.log")
            self.assertNotEqual(ro_code, 0)


class TestReadOnlyEnvironment(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.inner = Environment(workspace_dir=self.temp_dir, cleanup_on_teardown=True)
        self.inner.populate_files({"sample.txt": "alpha\nbeta\ngamma\n"})
        self.ro = ReadOnlyEnvironment(self.inner)

    def tearDown(self):
        self.inner.teardown()

    def test_reads_succeed_and_writes_are_rejected(self):
        """Invariant: ReadOnlyEnvironment allows reads and blocks write_file/edit_file/populate_files/teardown."""
        self.assertEqual(self.ro.workspace_dir, self.inner.workspace_dir)
        self.assertEqual(self.ro.logs_dir, self.inner.logs_dir)
        self.assertTrue(self.ro.is_running)
        self.assertEqual(
            self.ro.read_file("sample.txt"),
            "   1 | alpha\n   2 | beta\n   3 | gamma\n",
        )

        with self.assertRaises(PermissionError):
            self.ro.resolve_path("new.txt")
        with self.assertRaises(PermissionError):
            self.ro.write_file("new.txt", "blocked")
        with self.assertRaises(PermissionError):
            self.ro.edit_file("sample.txt", "alpha", "omega")
        with self.assertRaises(PermissionError):
            self.ro.populate_files({"a.txt": "1"})

        self.assertFalse((self.inner.workspace_dir / "new.txt").exists())
        self.assertEqual(
            (self.inner.workspace_dir / "sample.txt").read_text(encoding="utf-8"),
            "alpha\nbeta\ngamma\n",
        )
        self.assertIsNone(self.ro.get_git_patch())

        # teardown on wrapper must not destroy parent's workspace
        self.ro.teardown()
        self.assertTrue(self.inner.workspace_dir.exists())

    def test_validate_read_only_bash_allows_safe_commands_and_blocks_mutations(self):
        """Invariant: read-only bash allows inspection commands/pipelines and blocks mutating or arbitrary commands."""
        for cmd in [
            "ls -la",
            "cat sample.txt | head -n 2",
            'grep -n "alpha > beta" sample.txt',
            "find . -name '*.txt' 2>&1",
            "find . \\( -name '*.txt' -o -name '*.py' \\)",
            "git status",
            "git -C . log -n 5 --oneline",
            "git diff HEAD",
            "git branch -a",
            "git branch --merged main",
            "git tag -l",
            "sort sample.txt >/dev/null",
        ]:
            ok, err = validate_read_only_bash(cmd)
            self.assertTrue(ok, f"expected allowed for {cmd!r}, got error: {err}")

        for bad_cmd in [
            "",
            "   ",
            "rm sample.txt",
            "touch new.txt",
            "sed -i 's/alpha/omega/' sample.txt",
            "echo hacked > hacked.txt",
            "echo hacked >> sample.txt",
            "cat sample.txt | tee copy.txt",
            "ls && rm sample.txt",
            "echo $(rm sample.txt)",
            'echo "$(touch x)"',
            "echo `touch x`",
            "(rm sample.txt)",
            "cat <<EOF\nhi\nEOF",
            "ls &",
            "find . -delete",
            "find . -exec rm {} +",
            "fd -x rm",
            "sort -o sample.txt sample.txt",
            "rg --pre=sh alpha",
            "python3 -c \"open('x','w').write('y')\"",
            "./local_script.sh",
            "git checkout -- sample.txt",
            "git commit -m test",
            "git branch new-branch",
            "git branch -D main",
            "git tag v1.0",
            "git diff --output=patch.diff",
            "git -c core.fsmonitor=evil status",
            "FOO=bar",
            "ls 'unterminated",
        ]:
            ok, reason = validate_read_only_bash(bad_cmd)
            self.assertFalse(ok, f"expected blocked for {bad_cmd!r}")
            self.assertNotEqual(reason, "")

    def test_execute_bash_in_read_only_environment(self):
        """Invariant: ReadOnlyEnvironment executes allowed bash commands and blocks mutating ones without touching disk."""
        code_ok, out_ok = self.ro.execute_bash("cat sample.txt | wc -l")
        self.assertEqual(code_ok, 0)
        self.assertEqual(out_ok.strip(), "3")

        code_bad, out_bad = self.ro.execute_bash("echo pwn > pwn.txt")
        self.assertEqual(code_bad, 1)
        self.assertIn("Read-only environment blocked bash command", out_bad)
        self.assertFalse((self.inner.workspace_dir / "pwn.txt").exists())


class TestDockerResourceLimitsAndGuardedEnvironment(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.inner = Environment(workspace_dir=self.temp_dir, cleanup_on_teardown=True)
        self.inner.populate_files({"sample.txt": "alpha\nbeta\n"})

    def tearDown(self):
        self.inner.teardown()

    def test_docker_resource_limits_and_dedup(self):
        """Invariant: DockerEnvironment injects --memory/--cpus/--pids-limit unless already present in docker_args."""
        d_env = DockerEnvironment.__new__(DockerEnvironment)
        d_env.network = "none"
        d_env.memory_limit = "2g"
        d_env.cpu_limit = 1.5
        d_env.pids_limit = 128
        d_env.docker_args = []
        self.assertEqual(
            d_env._effective_docker_args(),
            ["--network", "none", "--memory", "2g", "--cpus", "1.5", "--pids-limit", "128"],
        )

        # Explicit docker_args flags take precedence without duplicating flags
        d_env.docker_args = ["-m", "4g", "--cpus=2", "--pids-limit=64"]
        self.assertEqual(
            d_env._effective_docker_args(),
            ["--network", "none", "-m", "4g", "--cpus=2", "--pids-limit=64"],
        )

    def test_guarded_environment_modes_and_ask_confirmation(self):
        """Invariant: GuardedEnvironment enforces auto, read-only, and ask modes cleanly."""
        prompts: list[tuple[str, str]] = []
        allow = False

        def confirm(action: str, detail: str) -> bool:
            prompts.append((action, detail))
            return allow

        guarded = GuardedEnvironment(self.inner, mode="ask", confirm_handler=confirm)
        # Reads are never prompted
        self.assertEqual(guarded.read_file("sample.txt"), "   1 | alpha\n   2 | beta\n")
        self.assertEqual(prompts, [])

        # Denied write/edit/bash leave workspace untouched
        with self.assertRaises(PermissionError):
            guarded.write_file("new.txt", "nope")
        with self.assertRaises(PermissionError):
            guarded.edit_file("sample.txt", "alpha", "omega")
        code_denied, out_denied = guarded.execute_bash("echo hi")
        self.assertEqual(code_denied, 1)
        self.assertIn("denied by user", out_denied)
        self.assertEqual([p[0] for p in prompts], ["write_file", "edit_file", "bash"])
        self.assertFalse((self.inner.workspace_dir / "new.txt").exists())

        # Approved actions execute normally
        allow = True
        guarded.write_file("new.txt", "yes\n")
        self.assertEqual(
            (self.inner.workspace_dir / "new.txt").read_text(encoding="utf-8"), "yes\n"
        )
        code_ok, out_ok = guarded.execute_bash("echo allowed")
        self.assertEqual(code_ok, 0)
        self.assertEqual(out_ok.strip(), "allowed")

        # Switching to read-only reuses ReadOnlyEnvironment
        guarded.mode = "read-only"
        self.assertEqual(guarded.mode, "read-only")
        with self.assertRaises(PermissionError):
            guarded.write_file("blocked.txt", "x")
        code_ro, out_ro = guarded.execute_bash("rm new.txt")
        self.assertEqual(code_ro, 1)
        self.assertIn("Read-only environment blocked bash command", out_ro)
        self.assertIsNone(guarded.get_git_patch())

        # Ask mode without confirm_handler fails closed
        no_handler = GuardedEnvironment(self.inner, mode="ask", confirm_handler=None)
        code_no_cb, out_no_cb = no_handler.execute_bash("echo hi")
        self.assertEqual(code_no_cb, 1)
        self.assertIn("no interactive confirmation handler", out_no_cb)

        with self.assertRaises(ValueError):
            GuardedEnvironment(self.inner, mode="invalid")

    def test_guarded_environment_preserves_container_protocol(self):
        """Invariant: GuardedEnvironment wrapping ContainerEnvironment preserves ContainerEnvironment protocol."""
        from core.environment import ContainerEnvironment

        d_env = DockerEnvironment.__new__(DockerEnvironment)
        d_env.container_name = "c1"
        d_env._container_started = True
        d_env.workspace_dir = self.inner.workspace_dir
        d_env.logs_dir = self.inner.logs_dir
        d_env.allowed_read_dirs = []

        self.assertTrue(isinstance(GuardedEnvironment(d_env, mode="auto"), ContainerEnvironment))
        self.assertFalse(
            isinstance(GuardedEnvironment(self.inner, mode="auto"), ContainerEnvironment)
        )


if __name__ == "__main__":
    unittest.main()
