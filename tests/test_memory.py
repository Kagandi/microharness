import unittest

from core.memory import Memory
from core.types import ToolCall, ToolResult


class TestMemory(unittest.TestCase):
    def test_initial_messages(self):
        mem = Memory(system_prompt="SYS", user_prompt="TASK")
        msgs = mem.get_messages()
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].role, "system")
        self.assertEqual(msgs[0].content, "SYS")
        self.assertEqual(msgs[1].role, "user")
        self.assertEqual(msgs[1].content, "TASK")

    def test_compaction_preserves_boundaries_and_prunes_middle(self):
        # max_context_chars is small (300) so compaction triggers
        mem = Memory(
            system_prompt="SYSTEM_INSTRUCTION",
            user_prompt="USER_TASK_GOAL",
            max_context_chars=300,
            keep_recent_turns=2,
        )

        # Step 1: Assistant + Tool (long output in the middle)
        mem.add_assistant("Thinking about step 1", [ToolCall(id="c1", name="bash", arguments={})])
        mem.add_tool_result(ToolResult(call_id="c1", output="A" * 300))

        # Step 2: Recent turns (should be kept intact)
        mem.add_assistant("Recent thought", [])
        mem.add_tool_result(ToolResult(call_id="c2", output="Recent observation"))

        self.assertGreater(mem.total_chars(), 300)

        # Run compaction
        compacted = mem.compact_if_needed()
        self.assertTrue(compacted)

        msgs = mem.get_messages()
        # System and Task preserved
        self.assertEqual(msgs[0].content, "SYSTEM_INSTRUCTION")
        self.assertEqual(msgs[1].content, "USER_TASK_GOAL")

        # Middle tool observation collapsed
        middle_tool_msg = msgs[3]
        self.assertEqual(middle_tool_msg.role, "tool")
        self.assertIn("[Tool output collapsed", middle_tool_msg.content)
        self.assertNotIn("A" * 300, middle_tool_msg.content)

        # Recent turns preserved
        self.assertEqual(msgs[-2].content, "Recent thought")
        self.assertEqual(msgs[-1].content, "Recent observation")


if __name__ == "__main__":
    unittest.main()
