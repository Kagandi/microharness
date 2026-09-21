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

    def test_file_content_evaluator(self):
        self.env.write_file("result.txt", "SUCCESS: All 10 jobs finished.")

        evaluator = FileContentEvaluator(
            file_path="result.txt",
            must_contain=["SUCCESS", "10 jobs"],
            must_not_contain=["ERROR", "FAILED"],
        )
        res = evaluator.evaluate(self.env, [])
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 1.0)

        # Negative test (missing required)
        failing_eval = FileContentEvaluator(file_path="result.txt", must_contain=["MISSING_KEYWORD"])
        res_fail = failing_eval.evaluate(self.env, [])
        self.assertFalse(res_fail.passed)
        self.assertEqual(res_fail.score, 0.0)

    def test_command_test_evaluator(self):
        evaluator_pass = CommandTestEvaluator(command="python3 -c 'exit(0)'")
        res_pass = evaluator_pass.evaluate(self.env, [])
        self.assertTrue(res_pass.passed)

        evaluator_fail = CommandTestEvaluator(command="python3 -c 'exit(1)'")
        res_fail = evaluator_fail.evaluate(self.env, [])
        self.assertFalse(res_fail.passed)


if __name__ == "__main__":
    unittest.main()
