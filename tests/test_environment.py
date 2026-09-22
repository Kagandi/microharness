import tempfile
import unittest
from pathlib import Path

from core.environment import Environment


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
        self.env.write_file("code.py", "line = 10\nline = 10\n")
        with self.assertRaises(ValueError) as ctx:
            self.env.edit_file("code.py", "line = 10", "line = 20")
        self.assertIn("must be unique", str(ctx.exception))

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


if __name__ == "__main__":
    unittest.main()
