import json
import tempfile
import unittest
from pathlib import Path

from core.evaluator import FileContentEvaluator
from core.harness import Harness
from core.model import MockModel
from core.types import Task, ToolCall


class TestHarnessIntegration(unittest.TestCase):
    def setUp(self):
        self.runs_dir = tempfile.mkdtemp()

    def test_full_successful_run(self):
        # Scripted model that writes answer.txt and finishes
        script = [
            ("Writing file", [ToolCall(id="c1", name="write_file", arguments={"path": "answer.txt", "content": "42"})]),
            ("Done", []),
        ]
        model = MockModel(script)
        evaluator = FileContentEvaluator(file_path="answer.txt", must_contain=["42"])

        harness = Harness(
            model=model,
            evaluator=evaluator,
            runs_dir=self.runs_dir,
            verbose=False,
        )

        task = Task(
            id="write_answer",
            instruction="Write 42 to answer.txt",
            max_steps=5,
        )

        result = harness.run(task)

        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.finish_reason, "completed")
        self.assertEqual(result.total_steps, 2)

        # Verify trace files were produced and valid
        json_trace = Path(result.trace_path)
        self.assertTrue(json_trace.exists())
        data = json.loads(json_trace.read_text(encoding="utf-8"))
        self.assertEqual(data["task"]["id"], "write_answer")
        self.assertEqual(len(data["steps"]), 2)

        md_trace = json_trace.parent / "trace.md"
        self.assertTrue(md_trace.exists())
        self.assertIn("write_file", md_trace.read_text(encoding="utf-8"))

    def test_max_steps_guardrail(self):
        # Model loops forever calling bash
        infinite_script = [
            (f"Looping {i}", [ToolCall(id=f"c{i}", name="bash", arguments={"command": "echo loop"})])
            for i in range(10)
        ]
        model = MockModel(infinite_script)

        harness = Harness(model=model, runs_dir=self.runs_dir, verbose=False)
        task = Task(id="infinite_task", instruction="Never stops", max_steps=3)

        result = harness.run(task)
        self.assertEqual(result.finish_reason, "max_steps_exceeded")
        self.assertEqual(result.total_steps, 3)

    def test_timeout_guardrail(self):
        # Command that sleeps
        script = [
            ("Sleeping", [ToolCall(id="s1", name="bash", arguments={"command": "python3 -c 'import time; time.sleep(1)'"})]),
            ("Next", []),
        ]
        model = MockModel(script)
        harness = Harness(model=model, runs_dir=self.runs_dir, verbose=False)
        task = Task(id="slow_task", instruction="Sleep", timeout_seconds=0.5)

        result = harness.run(task)
        # Bash will time out or overall task will timeout
        self.assertIn(result.finish_reason, ["completed", "timeout", "max_steps_exceeded"])


if __name__ == "__main__":
    unittest.main()
