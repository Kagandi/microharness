import tempfile
import unittest
from pathlib import Path

from core.environment import Environment
from core.tools import BashTool, ReadFileTool, ToolRegistry, WriteFileTool, format_tool_output
from core.types import ToolCall


class TestTools(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.env = Environment(workspace_dir=self.temp_dir, cleanup_on_teardown=True)
        self.registry = ToolRegistry(env=self.env, max_tool_chars=120)
        self.registry.register(BashTool(self.env))
        self.registry.register(ReadFileTool(self.env))
        self.registry.register(WriteFileTool(self.env))

    def tearDown(self):
        self.env.teardown()

    def test_output_below_threshold_does_not_truncate_or_spill(self):
        short_text = "Short clean output within budget."
        formatted, was_truncated, log_path = format_tool_output(
            output=short_text,
            call_id="call_short",
            tool_name="bash",
            step_index=1,
            logs_dir=self.env.logs_dir,
            max_chars=120,
        )

        self.assertFalse(was_truncated)
        self.assertIsNone(log_path)
        self.assertEqual(formatted, short_text)

        # Confirm zero spill files created in logs directory
        log_files = list(self.env.logs_dir.glob("*.log"))
        self.assertEqual(len(log_files), 0)

    def test_output_above_threshold_strictly_preserves_head_and_tail(self):
        head_part = "HEADER_START" + ("." * 40)  # > 40 chars (head_size)
        middle_part = "SHOULD_BE_PURGED" * 20
        tail_part = ("." * 90) + "TRAILER_FINISH"  # > 80 chars (tail_size)
        full_text = head_part + middle_part + tail_part

        max_chars = 120
        formatted, was_truncated, log_path = format_tool_output(
            output=full_text,
            call_id="call_long",
            tool_name="bash",
            step_index=1,
            logs_dir=self.env.logs_dir,
            max_chars=max_chars,
        )

        self.assertTrue(was_truncated)
        self.assertIsNotNone(log_path)

        # Invariant: Head and tail markers are preserved
        self.assertTrue(formatted.startswith("HEADER_START"))
        self.assertTrue(formatted.endswith("TRAILER_FINISH"))

        # Invariant: Middle bloated section is removed
        self.assertNotIn("SHOULD_BE_PURGED", formatted)

        # Invariant: Full original text is 100% preserved on disk
        saved = Path(log_path).read_text(encoding="utf-8")
        self.assertEqual(saved, full_text)

    def test_read_missing_file_returns_error_exit_code(self):
        call_read = ToolCall(id="r_err", name="read_file", arguments={"path": "non_existent.txt"})
        res = self.registry.dispatch(call_read, step_index=1)
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Error reading file", res.output)

    def test_write_outside_sandbox_returns_error_exit_code(self):
        call_write = ToolCall(
            id="w_err",
            name="write_file",
            arguments={"path": "../../forbidden.txt", "content": "hack"},
        )
        res = self.registry.dispatch(call_write, step_index=1)
        self.assertEqual(res.exit_code, 1)
        self.assertIn("outside workspace sandbox", res.output)

    def test_tool_crashing_internally_is_trapped_gracefully(self):
        class CrashingTool:
            name = "crash_tool"
            description = "Explodes on purpose"
            parameters = {"type": "object", "properties": {}}

            def execute(self, **kwargs):
                raise ZeroDivisionError("Simulated internal tool crash")

        self.registry.register(CrashingTool())
        res = self.registry.dispatch(
            ToolCall(id="c1", name="crash_tool", arguments={}), step_index=1
        )
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Simulated internal tool crash", res.output)


if __name__ == "__main__":
    unittest.main()
