import tempfile
import unittest

from core.environment import Environment
from core.evaluator import CommandTestEvaluator, FileContentEvaluator


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.env = Environment(workspace_dir=self.temp_dir, cleanup_on_teardown=True)

    def tearDown(self):
        self.env.teardown()

    def test_file_content_evaluator_all_assertions(self):
        self.env.write_file("result.txt", "SUCCESS: All 10 jobs finished cleanly.")

        # 1. Happy path: required present, forbidden absent
        evaluator = FileContentEvaluator(
            file_path="result.txt",
            must_contain=["SUCCESS", "10 jobs"],
            must_not_contain=["ERROR", "FAILED"],
        )
        res = evaluator.evaluate(self.env, [])
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 1.0)

        # 2. Strict Negative: Missing required string
        failing_missing = FileContentEvaluator(file_path="result.txt", must_contain=["NOT_PRESENT"])
        res_missing = failing_missing.evaluate(self.env, [])
        self.assertFalse(res_missing.passed)
        self.assertEqual(res_missing.score, 0.0)
        self.assertIn("missing required string", res_missing.feedback)

        # 3. Strict Negative: Forbidden string present
        failing_forbidden = FileContentEvaluator(file_path="result.txt", must_not_contain=["SUCCESS"])
        res_forbidden = failing_forbidden.evaluate(self.env, [])
        self.assertFalse(res_forbidden.passed)
        self.assertEqual(res_forbidden.score, 0.0)
        self.assertIn("forbidden string", res_forbidden.feedback)

        # 4. Strict Negative: File does not exist
        failing_nofile = FileContentEvaluator(file_path="nonexistent.txt")
        res_nofile = failing_nofile.evaluate(self.env, [])
        self.assertFalse(res_nofile.passed)
        self.assertEqual(res_nofile.score, 0.0)
        self.assertIn("does not exist", res_nofile.feedback)

    def test_command_test_evaluator_strict_exit_codes(self):
        # Pass on exit 0
        evaluator_pass = CommandTestEvaluator(command="python3 -c 'exit(0)'")
        res_pass = evaluator_pass.evaluate(self.env, [])
        self.assertTrue(res_pass.passed)
        self.assertEqual(res_pass.score, 1.0)

        # Fail on exit 1
        evaluator_fail = CommandTestEvaluator(command="python3 -c 'exit(1)'")
        res_fail = evaluator_fail.evaluate(self.env, [])
        self.assertFalse(res_fail.passed)
        self.assertEqual(res_fail.score, 0.0)

        # Fail on command failure
        evaluator_crash = CommandTestEvaluator(command="invalid_command_that_does_not_exist_xyz")
        res_crash = evaluator_crash.evaluate(self.env, [])
        self.assertFalse(res_crash.passed)
        self.assertEqual(res_crash.score, 0.0)


if __name__ == "__main__":
    unittest.main()
