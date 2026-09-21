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
        self.env.populate_files({
            "a.txt": "Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
            "nested/b.py": "x = 42",
        })

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
        exit_code, output = self.env.execute_bash("python3 -c 'import sys; sys.stderr.write(\"CRASH_TRACE\"); sys.exit(42)'")
        self.assertEqual(exit_code, 42)
        self.assertIn("CRASH_TRACE", output)

    def test_bash_timeout_kill(self):
        exit_code, output = self.env.execute_bash("python3 -c 'import time; time.sleep(5)'", timeout=0.1)
        self.assertEqual(exit_code, 124)
        self.assertIn("timed out after 0.1", output)


if __name__ == "__main__":
    unittest.main()
