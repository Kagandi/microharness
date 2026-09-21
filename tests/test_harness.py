import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from core.evaluator import FileContentEvaluator
from core.harness import Harness
from core.model import MockModel
from core.types import Message, Task, ToolCall


class TestHarnessIntegration(unittest.TestCase):
    def setUp(self):
        self.runs_dir = tempfile.mkdtemp()

    def test_full_successful_run_and_trace_integrity(self):
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

        # Strict outcome invariants
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.finish_reason, "completed")
        self.assertEqual(result.total_steps, 2)

        # Deep trace integrity check
        json_trace = Path(result.trace_path)
        self.assertTrue(json_trace.exists())
        data = json.loads(json_trace.read_text(encoding="utf-8"))

        self.assertEqual(data["task"]["id"], "write_answer")
        self.assertEqual(data["finish_reason"], "completed")
        self.assertEqual(data["eval_result"]["passed"], True)
        self.assertEqual(len(data["steps"]), 2)

        # Step 1 must have the tool call and observation
        step1 = data["steps"][0]
        self.assertEqual(step1["step_index"], 1)
        self.assertEqual(step1["model_response"]["tool_calls"][0]["name"], "write_file")
        self.assertEqual(step1["tool_results"][0]["call_id"], "c1")
        self.assertEqual(step1["tool_results"][0]["exit_code"], 0)

        # Markdown trace check
        md_trace = json_trace.parent / "trace.md"
        self.assertTrue(md_trace.exists())
        md_content = md_trace.read_text(encoding="utf-8")
        self.assertIn("# Run Trace: Task write_answer", md_content)
        self.assertIn("write_file", md_content)

    def test_max_steps_strictly_halts_execution(self):
        # Model that would loop 10 times
        infinite_script = [
            (f"Looping step {i}", [ToolCall(id=f"c{i}", name="bash", arguments={"command": "echo loop"})])
            for i in range(10)
        ]
        model = MockModel(infinite_script)

        # Hard guardrail at 3 steps
        harness = Harness(model=model, runs_dir=self.runs_dir, verbose=False)
        task = Task(id="infinite_task", instruction="Never stops", max_steps=3)

        result = harness.run(task)

        # Strict guardrail assertions
        self.assertEqual(result.finish_reason, "max_steps_exceeded")
        self.assertEqual(result.total_steps, 3)
        self.assertFalse(result.passed)

    def test_timeout_guardrail_strictly_halts_execution(self):
        # A model that causes time to elapse beyond timeout_seconds
        class SlowModel:
            def generate(self, messages, tools):
                time.sleep(0.15)
                return "Still working", [ToolCall(id="s1", name="bash", arguments={"command": "echo tick"})]

        harness = Harness(model=SlowModel(), runs_dir=self.runs_dir, verbose=False)
        # Timeout is 0.1s; step 1 takes 0.15s, so step 2 must trigger timeout
        task = Task(id="slow_task", instruction="Slow execution", timeout_seconds=0.1, max_steps=10)

        result = harness.run(task)

        # Strict: must be timeout, NOT completed or max_steps_exceeded
        self.assertEqual(result.finish_reason, "timeout")
        self.assertFalse(result.passed)

    def test_unhandled_model_crash_is_trapped_safely(self):
        class ExplodingModel:
            def generate(self, messages, tools):
                raise ConnectionResetError("Remote model socket terminated abruptly")

        harness = Harness(model=ExplodingModel(), runs_dir=self.runs_dir, verbose=False)
        task = Task(id="crash_task", instruction="Will crash", max_steps=5)

        # Invariant: Harness does not raise unhandled exception; records error cleanly
        result = harness.run(task)
        self.assertEqual(result.finish_reason, "error")
        self.assertFalse(result.passed)
        self.assertTrue(Path(result.trace_path).exists())

    def test_interactive_multi_turn_session(self):
        # Scripted model that handles 2 interactive turns:
        # Turn 1: writes step1.txt, then yields text
        # Turn 2: writes step2.txt, then yields text
        script = [
            ("Writing step 1", [ToolCall(id="t1", name="write_file", arguments={"path": "step1.txt", "content": "hello 1"})]),
            ("Finished turn 1.", []),
            ("Writing step 2", [ToolCall(id="t2", name="write_file", arguments={"path": "step2.txt", "content": "hello 2"})]),
            ("Finished turn 2.", []),
        ]
        model = MockModel(script)
        harness = Harness(model=model, runs_dir=self.runs_dir, verbose=False)

        # Mock user typing follow-up and then 'exit'
        user_inputs = iter(["Now do step 2", "exit"])
        with patch("builtins.input", lambda *args: next(user_inputs)):
            harness.run_interactive(initial_instruction="Start step 1")

        # Invariant: Files from both turns exist in the same workspace!
        latest_run = max(Path(self.runs_dir).glob("interactive_session_*"))
        workspace = latest_run / "workspace"
        self.assertTrue((workspace / "step1.txt").exists())
        self.assertTrue((workspace / "step2.txt").exists())
        self.assertEqual((workspace / "step1.txt").read_text(), "hello 1")
        self.assertEqual((workspace / "step2.txt").read_text(), "hello 2")

        # Invariant: Trace captured all turns across the session
        trace_json = latest_run / "trace.json"
        self.assertTrue(trace_json.exists())
        trace_data = json.loads(trace_json.read_text(encoding="utf-8"))
        self.assertEqual(trace_data["finish_reason"], "session_ended")
        self.assertEqual(len(trace_data["steps"]), 4)


if __name__ == "__main__":
    unittest.main()
