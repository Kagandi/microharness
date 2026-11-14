import unittest

import microharness
from core.harness import Harness
from core.types import RunResult, Task, ToolCall


class TestPackageExports(unittest.TestCase):
    def test_package_metadata(self):
        self.assertEqual(microharness.__version__, "0.1.0")
        self.assertEqual(
            microharness.__all__, ["Harness", "Task", "ToolCall", "RunResult", "__version__"]
        )

    def test_package_exports(self):
        self.assertIs(microharness.Harness, Harness)
        self.assertIs(microharness.Task, Task)
        self.assertIs(microharness.ToolCall, ToolCall)
        self.assertIs(microharness.RunResult, RunResult)

    def test_missing_attribute_raises(self):
        with self.assertRaises(AttributeError):
            _ = microharness.NonExistentClass


if __name__ == "__main__":
    unittest.main()
