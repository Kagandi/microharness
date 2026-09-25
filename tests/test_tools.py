import tempfile
import unittest
from pathlib import Path

from core.environment import Environment
from core.tools import (
    BashTool,
    EditFileTool,
    ReadFileTool,
    ToolRegistry,
    WriteFileTool,
    format_tool_output,
)
from core.types import ToolCall


class TestTools(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.env = Environment(workspace_dir=self.temp_dir, cleanup_on_teardown=True)
        self.registry = ToolRegistry(env=self.env, max_tool_chars=120)
        self.registry.register(BashTool(self.env))
        self.registry.register(ReadFileTool(self.env))
        self.registry.register(WriteFileTool(self.env))
        self.registry.register(EditFileTool(self.env))

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

    def test_edit_file_tool_dispatch_success(self):
        self.env.write_file("greeting.py", "msg = 'hello world'\n")
        call = ToolCall(
            id="e1",
            name="edit_file",
            arguments={"path": "greeting.py", "old_str": "world", "new_str": "universe"},
        )
        res = self.registry.dispatch(call, step_index=1)
        self.assertEqual(res.exit_code, 0)
        self.assertIn("Successfully replaced", res.output)
        self.assertIn("hello universe", self.env.read_file("greeting.py"))

    def test_edit_file_tool_dispatch_error(self):
        self.env.write_file("greeting.py", "msg = 'hello world'\n")
        call = ToolCall(
            id="e2",
            name="edit_file",
            arguments={"path": "greeting.py", "old_str": "non_existent", "new_str": "foo"},
        )
        res = self.registry.dispatch(call, step_index=1)
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Error editing file", res.output)

    def test_tool_registry_to_system_prompt_snippet(self):
        snippet = self.registry.to_system_prompt_snippet()
        self.assertIn("You have access to tools to interact with the environment:", snippet)
        self.assertIn("bash(command: string):", snippet)
        self.assertIn("read_file(path: string, start_line: integer, end_line: integer):", snippet)
        self.assertIn("write_file(path: string, content: string):", snippet)
        self.assertIn("edit_file(path: string, old_str: string, new_str: string):", snippet)

    def test_tool_dispatch_with_non_dict_arguments(self):
        # When an LLM sends a string or list instead of a dict for arguments
        call = ToolCall(id="e3", name="bash", arguments="echo hello")  # type: ignore
        res = self.registry.dispatch(call, step_index=1)
        self.assertEqual(res.exit_code, 1)
        self.assertIn("must be a dictionary/object", res.output)

    def test_format_tool_output_handles_oserror(self):
        from unittest.mock import patch

        from core.tools import format_tool_output

        with patch.object(Path, "write_text", side_effect=OSError("Disk full")):
            formatted, was_truncated, raw_log_path = format_tool_output(
                output="x" * 5000,
                call_id="call_test",
                tool_name="bash",
                step_index=1,
                logs_dir=self.env.logs_dir,
                max_chars=1000,
            )
            self.assertTrue(was_truncated)
            self.assertIsNone(raw_log_path)
            self.assertIn("Truncated", formatted)

    def test_tool_registry_modifies_files(self):
        self.assertTrue(self.registry.modifies_files("write_file"))
        self.assertTrue(self.registry.modifies_files("edit_file"))
        self.assertFalse(self.registry.modifies_files("read_file"))
        self.assertFalse(self.registry.modifies_files("bash"))


if __name__ == "__main__":
    unittest.main()
