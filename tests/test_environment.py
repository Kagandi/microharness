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

        # Test whole read
        content = self.env.read_file("a.txt")
        self.assertIn("Line 1", content)
        self.assertIn("Line 5", content)

        # Test line slice (lines 2 to 4)
        slice_content = self.env.read_file("a.txt", start_line=2, end_line=4)
        self.assertNotIn("Line 1", slice_content)
        self.assertIn("Line 2", slice_content)
        self.assertIn("Line 4", slice_content)
        self.assertNotIn("Line 5", slice_content)

        # Test nested file
        nested_content = self.env.read_file("nested/b.py")
        self.assertIn("x = 42", nested_content)

    def test_sandbox_path_traversal_prevention(self):
        with self.assertRaises(PermissionError):
            self.env.resolve_path("../../etc/passwd")

    def test_bash_execution_and_timeout(self):
        exit_code, output = self.env.execute_bash("echo 'hello sandbox'")
        self.assertEqual(exit_code, 0)
        self.assertIn("hello sandbox", output)

        # Test timeout
        exit_code, output = self.env.execute_bash("python3 -c 'import time; time.sleep(2)'", timeout=0.2)
        self.assertEqual(exit_code, 124)
        self.assertIn("timed out", output)


if __name__ == "__main__":
    unittest.main()
