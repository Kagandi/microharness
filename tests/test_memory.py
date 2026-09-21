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

    def test_compaction_inert_when_under_budget(self):
        mem = Memory(
            system_prompt="System instructions",
            user_prompt="Task goal",
            max_context_chars=1000,
        )
        mem.add_assistant("Short response")
        mem.add_tool_result(ToolResult(call_id="c1", output="Short output"))

        # Invariant: Compaction does nothing when below max_context_chars
        compacted = mem.compact_if_needed()
        self.assertFalse(compacted)
        self.assertEqual(mem.messages[-1].content, "Short output")

    def test_compaction_never_corrupts_system_or_task_prompts(self):
        huge_system = "SYSTEM_CRITICAL_INVARIANT_" * 50
        huge_task = "TASK_CRITICAL_GOAL_" * 50

        # Budget is lower than initial prompts
        mem = Memory(
            system_prompt=huge_system,
            user_prompt=huge_task,
            max_context_chars=100,
            keep_recent_turns=2,
        )
        mem.add_assistant("Step 1 thought")
        mem.add_tool_result(ToolResult(call_id="c1", output="Bloated log " * 50))
        mem.add_assistant("Recent thought")
        mem.add_tool_result(ToolResult(call_id="c2", output="Recent observation"))

        compacted = mem.compact_if_needed()
        self.assertTrue(compacted)

        # Invariant: System and Task prompts are 100% byte-for-byte identical
        self.assertEqual(mem.messages[0].content, huge_system)
        self.assertEqual(mem.messages[1].content, huge_task)

    def test_compaction_preserves_recent_turns_and_assistant_thoughts(self):
        mem = Memory(
            system_prompt="SYS",
            user_prompt="GOAL",
            max_context_chars=150,
            keep_recent_turns=2,
        )

        # Middle turn
        mem.add_assistant("My crucial plan for step 1")
        mem.add_tool_result(ToolResult(call_id="c1", output="X" * 300))

        # Recent turns
        mem.add_assistant("Recent turn thought")
        mem.add_tool_result(ToolResult(call_id="c2", output="Recent observation"))

        compacted = mem.compact_if_needed()
        self.assertTrue(compacted)

        # Invariant: Assistant plan thought in middle is NOT pruned
        self.assertEqual(mem.messages[2].content, "My crucial plan for step 1")

        # Invariant: Middle tool result IS pruned to tombstone
        self.assertIn("[Tool output collapsed", mem.messages[3].content)
        self.assertNotIn("X" * 300, mem.messages[3].content)

        # Invariant: Recent turns are NOT pruned
        self.assertEqual(mem.messages[-2].content, "Recent turn thought")
        self.assertEqual(mem.messages[-1].content, "Recent observation")


if __name__ == "__main__":
    unittest.main()
