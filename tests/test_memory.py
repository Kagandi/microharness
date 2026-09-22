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

    def test_stats_metrics(self):
        mem = Memory(
            system_prompt="SYS",
            user_prompt="TASK",
            max_context_chars=1000,
            trigger_ratio=0.80,
        )
        mem.add_assistant("reply", [])
        mem.add_tool_result(ToolResult(call_id="c1", output="tool_out"))

        s = mem.stats()
        self.assertEqual(s["total_messages"], 4)
        self.assertEqual(s["by_role"]["system"], 1)
        self.assertEqual(s["by_role"]["user"], 1)
        self.assertEqual(s["by_role"]["assistant"], 1)
        self.assertEqual(s["by_role"]["tool"], 1)
        self.assertEqual(s["trigger_ratio"], 0.80)
        self.assertEqual(s["trigger_threshold_chars"], 800)
        self.assertGreater(s["total_chars"], 0)
        self.assertGreater(s["usage_pct"], 0.0)

    def test_compaction_triggers_at_high_water_mark_ratio(self):
        # max_context_chars = 1000, trigger_ratio = 0.80 -> threshold is 800 chars
        mem = Memory(
            system_prompt="SYS",
            user_prompt="GOAL",
            max_context_chars=1000,
            trigger_ratio=0.80,
            keep_recent_turns=1,
        )
        # Add 750 characters of middle tool output -> total chars ~ 760 (< 800)
        mem.add_assistant("Plan")
        mem.add_tool_result(ToolResult(call_id="c1", output="X" * 750))
        mem.add_assistant("Recent")

        # 760 chars <= 800 chars -> should not trigger yet
        self.assertFalse(mem.compact_if_needed(force=False))

        # Add another 100 chars -> total chars ~ 860 (> 800 chars threshold, but < 1000 chars max)
        mem.add_tool_result(ToolResult(call_id="c2", output="Y" * 100))
        mem.add_assistant("Latest")

        # Invariant: Compaction triggers proactively at 80% (800 chars), well before hitting 100% (1000 chars)
        self.assertTrue(mem.compact_if_needed(force=False))
        # Verify middle output was collapsed
        self.assertIn("[Tool output collapsed", mem.messages[3].content)

    def test_reset_and_force_compact(self):
        mem = Memory(
            system_prompt="SYS_PROMPT",
            user_prompt="INITIAL",
            max_context_chars=5000,
            keep_recent_turns=2,
        )
        mem.add_assistant("Plan", [ToolCall(id="c1", name="bash", arguments={})])
        mem.add_tool_result(ToolResult(call_id="c1", output="A" * 200))
        mem.add_assistant("Recent", [])
        mem.add_tool_result(ToolResult(call_id="c2", output="Recent obs"))

        # Under budget: standard compact does nothing
        self.assertFalse(mem.compact_if_needed(force=False))

        # Force compact: collapses even under budget
        self.assertTrue(mem.compact_if_needed(force=True))
        self.assertIn("[Tool output collapsed", mem.messages[3].content)

        # Reset: preserves system prompt, clears history
        mem.reset(new_instruction="FRESH_START")
        msgs = mem.get_messages()
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].content, "SYS_PROMPT")
        self.assertEqual(msgs[1].content, "FRESH_START")

    def test_add_user(self):
        mem = Memory(system_prompt="SYS", user_prompt="Initial")
        mem.add_user("Next turn from user")
        msgs = mem.get_messages()
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[-1].role, "user")
        self.assertEqual(msgs[-1].content, "Next turn from user")

    def test_compaction_with_no_system_prompt(self):
        # When system_prompt is empty, only initial user prompt is at index 0
        mem = Memory(
            system_prompt="", user_prompt="TASK", max_context_chars=100, keep_recent_turns=1
        )
        mem.add_assistant("Step 1")
        mem.add_tool_result(ToolResult(call_id="c1", output="A" * 300))
        mem.add_assistant("Recent")

        self.assertTrue(mem.compact_if_needed(force=True))
        # Initial user task is preserved
        self.assertEqual(mem.messages[0].role, "user")
        self.assertEqual(mem.messages[0].content, "TASK")
        # Middle tool output is collapsed
        self.assertIn("[Tool output collapsed", mem.messages[2].content)
        # Recent turn is preserved
        self.assertEqual(mem.messages[-1].content, "Recent")

    def test_compaction_prunes_thinking_tags(self):
        mem = Memory(
            system_prompt="SYS",
            user_prompt="TASK",
            max_context_chars=100,
            keep_recent_turns=2,
        )
        # Turn 1: thinking with actual response and tool call
        mem.add_assistant(
            "<think>Analyzing dependencies and planning changes...</think>I will now inspect the project.",
            [ToolCall(id="c1", name="bash", arguments={"command": "ls"})],
        )
        mem.add_tool_result(ToolResult(call_id="c1", output="output1"))

        # Turn 2: thinking-only response
        mem.add_assistant(
            "<thinking>\nInternal step-by-step reasoning that should be discarded.\n</thinking>"
        )
        mem.add_tool_result(ToolResult(call_id="c2", output="output2"))

        # Recent turns: should keep thinking intact
        mem.add_assistant("<think>Recent reasoning stream</think>Recent answer")
        mem.add_tool_result(ToolResult(call_id="c3", output="recent output"))

        self.assertTrue(mem.compact_if_needed(force=True))

        messages = mem.get_messages()
        # Middle turn 1: <think> pruned, visible answer preserved, tool call preserved
        self.assertEqual(messages[2].content, "I will now inspect the project.")
        self.assertIsNotNone(messages[2].tool_calls)
        self.assertEqual(len(messages[2].tool_calls), 1)
        self.assertEqual(messages[2].tool_calls[0].id, "c1")

        # Middle turn 2: pure thinking tag collapsed to fallback tombstone
        self.assertEqual(
            messages[4].content,
            "[Reasoning stream collapsed to conserve context window]",
        )

        # Recent turn: not compacted, <think> is preserved
        self.assertEqual(
            messages[-2].content,
            "<think>Recent reasoning stream</think>Recent answer",
        )

    def test_compaction_enriches_tombstone_with_log_path(self):
        mem = Memory(
            system_prompt="SYS",
            user_prompt="TASK",
            max_context_chars=100,
            keep_recent_turns=1,
        )
        mem.add_assistant("Step 1")
        mem.add_tool_result(
            ToolResult(
                call_id="c1",
                output="Log line\n" * 30,  # > 120 chars
                raw_log_path="/tmp/workspace/runs/logs/step_01_bash_c1.log",
            )
        )
        mem.add_assistant("Step 2")
        mem.add_tool_result(
            ToolResult(
                call_id="c2",
                output="Another line\n" * 30,
                raw_log_path=None,
            )
        )
        mem.add_assistant("Recent")

        self.assertTrue(mem.compact_if_needed(force=True))

        msgs = mem.get_messages()
        len1 = len("Log line\n" * 30)
        len2 = len("Another line\n" * 30)
        # Message with raw_log_path includes exact log path and tip
        self.assertEqual(
            msgs[3].content,
            f"[Tool output collapsed ({len1} chars). "
            f"Full log: `/tmp/workspace/runs/logs/step_01_bash_c1.log`. Use `read_file` to inspect if needed.]",
        )
        # Message without raw_log_path uses standard collapse message
        self.assertEqual(
            msgs[5].content,
            f"[Tool output collapsed ({len2} chars) to conserve context window]",
        )

    def test_compaction_injects_and_updates_modified_files_ledger(self):
        mem = Memory(
            system_prompt="SYS",
            user_prompt="TASK",
            max_context_chars=100,
            keep_recent_turns=1,
        )
        mem.record_modified_file("src/main.py")
        mem.record_modified_file("src/utils.py")

        mem.add_assistant("Step 1")
        mem.add_tool_result(ToolResult(call_id="c1", output="A" * 300))
        mem.add_assistant("Recent")

        # First compaction: injects ledger at index 2 (after system & user)
        self.assertTrue(mem.compact_if_needed(force=True))

        msgs = mem.get_messages()
        self.assertEqual(msgs[0].role, "system")
        self.assertEqual(msgs[0].content, "SYS")
        self.assertEqual(msgs[1].role, "user")
        self.assertEqual(msgs[1].content, "TASK")
        self.assertEqual(msgs[2].role, "system")
        self.assertEqual(
            msgs[2].content,
            "[Session Files Modified: src/main.py, src/utils.py]",
        )

        # Record another modified file and run compaction again
        mem.record_modified_file("tests/test_main.py")
        mem.add_tool_result(ToolResult(call_id="c2", output="B" * 300))
        mem.add_assistant("Latest")

        self.assertTrue(mem.compact_if_needed(force=True))

        msgs = mem.get_messages()
        # Ledger was updated in-place at index 2, not duplicated
        self.assertEqual(msgs[2].role, "system")
        self.assertEqual(
            msgs[2].content,
            "[Session Files Modified: src/main.py, src/utils.py, tests/test_main.py]",
        )
        self.assertNotEqual(msgs[3].content, msgs[2].content)

        # Stats reflect modified files
        stats = mem.stats()
        self.assertEqual(
            stats["modified_files"],
            ["src/main.py", "src/utils.py", "tests/test_main.py"],
        )

        # Reset clears modified files
        mem.reset(new_instruction="NEW_TASK")
        self.assertEqual(mem.stats()["modified_files"], [])


if __name__ == "__main__":
    unittest.main()
