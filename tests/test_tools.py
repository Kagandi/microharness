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
        self.registry = ToolRegistry(env=self.env, max_tool_chars=100)
        self.registry.register(BashTool(self.env))
        self.registry.register(ReadFileTool(self.env))
        self.registry.register(WriteFileTool(self.env))

    def tearDown(self):
        self.env.teardown()

    def test_write_and_read_tools(self):
        # Write file
        call_write = ToolCall(id="w1", name="write_file", arguments={"path": "msg.txt", "content": "hi there"})
        res_write = self.registry.dispatch(call_write, step_index=1)
        self.assertEqual(res_write.exit_code, 0)
        self.assertIn("Successfully wrote", res_write.output)

        # Read file
        call_read = ToolCall(id="r1", name="read_file", arguments={"path": "msg.txt"})
        res_read = self.registry.dispatch(call_read, step_index=2)
        self.assertEqual(res_read.exit_code, 0)
        self.assertIn("hi there", res_read.output)

    def test_output_truncation_and_disk_spillover(self):
        # 500 characters output with max_chars=100
        huge_text = "A" * 200 + "MIDDLE_DATA" + "Z" * 200
        formatted, was_truncated, log_path = format_tool_output(
            output=huge_text,
            call_id="call_trunc",
            tool_name="bash",
            step_index=1,
            logs_dir=self.env.logs_dir,
            max_chars=100,
        )

        self.assertTrue(was_truncated)
        self.assertIn("[... Truncated", formatted)
        self.assertNotIn("MIDDLE_DATA", formatted)
        self.assertIsNotNone(log_path)

        # Verify raw untruncated file exists on disk
        saved_file = Path(log_path)
        self.assertTrue(saved_file.exists())
        self.assertEqual(saved_file.read_text(encoding="utf-8"), huge_text)

    def test_unknown_tool_dispatch(self):
        call_unknown = ToolCall(id="u1", name="non_existent", arguments={})
        res = self.registry.dispatch(call_unknown, step_index=1)
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Unknown tool", res.output)


if __name__ == "__main__":
    unittest.main()
