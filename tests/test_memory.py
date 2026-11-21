import unittest

from core.loop import AgentLoop
from core.memory import COMPACTION_SUMMARY_HEADER, Memory, format_compact_num
from core.tools import ToolRegistry
from core.types import Message, ToolCall, ToolResult


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
            max_context_tokens=250,
        )
        mem.add_assistant("Short response")
        mem.add_tool_result(ToolResult(call_id="c1", output="Short output"))

        # Invariant: Compaction does nothing when below max_context_tokens
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
            max_context_tokens=25,
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

    def test_compaction_preserves_recent_turns_and_summarizes_middle(self):
        mem = Memory(
            system_prompt="SYS",
            user_prompt="GOAL",
            max_context_tokens=40,
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

        # Invariant: Prefix stays immutable and middle is collapsed into one structured summary
        self.assertEqual(mem.messages[0].content, "SYS")
        self.assertEqual(mem.messages[1].content, "GOAL")
        self.assertEqual(len(mem.messages), 5)
        self.assertTrue(mem.messages[2].content.startswith(COMPACTION_SUMMARY_HEADER))
        self.assertIn("My crucial plan for step 1", mem.messages[2].content)
        self.assertNotIn("X" * 300, mem.messages[2].content)

        # Invariant: Recent turns are NOT pruned
        self.assertEqual(mem.messages[-2].content, "Recent turn thought")
        self.assertEqual(mem.messages[-1].content, "Recent observation")

    def test_stats_metrics(self):
        mem = Memory(
            system_prompt="SYS",
            user_prompt="TASK",
            max_context_tokens=250,
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
        self.assertEqual(s["trigger_threshold_tokens"], 200)
        self.assertEqual(s["max_context_tokens"], 250)
        self.assertGreater(s["estimated_tokens"], 0)
        self.assertGreater(s["usage_pct"], 0.0)

    def test_compaction_triggers_at_high_water_mark_ratio(self):
        # max_context_tokens = 250, trigger_ratio = 0.80 -> threshold is 200 tokens (800 chars)
        mem = Memory(
            system_prompt="SYS",
            user_prompt="GOAL",
            max_context_tokens=250,
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

        # Invariant: Compaction triggers proactively at 80% (200 tokens / 800 chars), well before hitting 100%
        self.assertTrue(mem.compact_if_needed(force=False))
        # Verify middle turns were replaced by structured summary
        self.assertTrue(mem.messages[2].content.startswith(COMPACTION_SUMMARY_HEADER))
        self.assertNotIn("X" * 750, mem.messages[2].content)
        self.assertEqual(mem.messages[-1].content, "Latest")

    def test_drop_incomplete_tool_turn_removes_dangling_tool_calls(self):
        """Invariant: an interrupted tool turn is trimmed so the next request stays valid."""
        mem = Memory(system_prompt="SYS", user_prompt="TASK")
        mem.add_assistant("working", [ToolCall(id="c1", name="bash", arguments={"command": "ls"})])
        self.assertEqual([m.role for m in mem.get_messages()], ["system", "user", "assistant"])

        removed = mem.drop_incomplete_tool_turn()

        self.assertEqual(removed, 1)
        self.assertEqual(
            [m.role for m in mem.get_messages()],
            ["system", "user"],
            "the unmatched assistant tool_call must not survive an interruption",
        )

    def test_drop_incomplete_tool_turn_keeps_complete_turns(self):
        """Invariant: a fully answered tool turn is never discarded."""
        mem = Memory(system_prompt="SYS", user_prompt="TASK")
        mem.add_assistant("working", [ToolCall(id="c1", name="bash", arguments={"command": "ls"})])
        mem.add_tool_result(ToolResult(call_id="c1", output="files"))

        self.assertEqual(mem.drop_incomplete_tool_turn(), 0)
        self.assertEqual(
            [m.role for m in mem.get_messages()], ["system", "user", "assistant", "tool"]
        )

    def test_drop_incomplete_tool_turn_noop_without_tool_calls(self):
        """Invariant: a plain assistant reply (no tool calls) is left intact."""
        mem = Memory(system_prompt="SYS", user_prompt="TASK")
        mem.add_assistant("all done", [])
        self.assertEqual(mem.drop_incomplete_tool_turn(), 0)
        self.assertEqual(len(mem.get_messages()), 3)

    def test_reset_and_force_compact(self):
        mem = Memory(
            system_prompt="SYS_PROMPT",
            user_prompt="INITIAL",
            max_context_tokens=1250,
            keep_recent_turns=2,
        )
        mem.add_assistant("Plan", [ToolCall(id="c1", name="bash", arguments={})])
        mem.add_tool_result(ToolResult(call_id="c1", output="A" * 200))
        mem.add_assistant("Recent", [])
        mem.add_tool_result(ToolResult(call_id="c2", output="Recent obs"))

        # Under budget: standard compact does nothing
        self.assertFalse(mem.compact_if_needed(force=False))

        # Force compact: summarizes middle even under budget
        self.assertTrue(mem.compact_if_needed(force=True))
        self.assertTrue(mem.messages[2].content.startswith(COMPACTION_SUMMARY_HEADER))
        self.assertNotIn("A" * 200, mem.messages[2].content)

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
            system_prompt="", user_prompt="TASK", max_context_tokens=25, keep_recent_turns=1
        )
        mem.add_assistant("Step 1")
        mem.add_tool_result(ToolResult(call_id="c1", output="A" * 300))
        mem.add_assistant("Recent")

        self.assertTrue(mem.compact_if_needed(force=True))
        # Initial user task is preserved
        self.assertEqual(mem.messages[0].role, "user")
        self.assertEqual(mem.messages[0].content, "TASK")
        # Middle turns replaced by structured compaction summary
        self.assertTrue(mem.messages[1].content.startswith(COMPACTION_SUMMARY_HEADER))
        self.assertNotIn("A" * 300, mem.messages[1].content)
        # Recent turn is preserved
        self.assertEqual(mem.messages[-1].content, "Recent")

    def test_compaction_prunes_thinking_tags_and_extra_fields(self):
        mem = Memory(
            system_prompt="SYS",
            user_prompt="TASK",
            max_context_tokens=25,
            keep_recent_turns=4,
        )
        # Turn 1 (compacted): thinking with actual response and tool call
        mem.add_assistant(
            "<think>Analyzing dependencies and planning changes...</think>I will now inspect the project.",
            [ToolCall(id="c1", name="bash", arguments={"command": "ls"})],
            extra_fields={"reasoning_content": "secret chain 1"},
        )
        mem.add_tool_result(ToolResult(call_id="c1", output="output1"))

        # Turn 2 (retained older assistant turn): thinking + extra_fields reasoning
        mem.add_assistant(
            "<thinking>\nInternal step-by-step reasoning that should be discarded.\n</thinking>Checked status.",
            [ToolCall(id="c2", name="bash", arguments={"command": "pwd"})],
            extra_fields={
                "reasoning_content": "stale reasoning text",
                "reasoning": "stale summary",
                "reasoning_details": [{"type": "reasoning.text", "text": "detail"}],
                "provider_tag": "keep_me",
            },
        )
        mem.add_tool_result(ToolResult(call_id="c2", output="output2"))

        # Turn 3 (latest active assistant turn): should keep thinking and extra_fields intact
        mem.add_assistant(
            "<think>Recent reasoning stream</think>Recent answer",
            [ToolCall(id="c3", name="bash", arguments={"command": "echo ok"})],
            extra_fields={"reasoning_content": "active reasoning"},
        )
        mem.add_tool_result(ToolResult(call_id="c3", output="recent output"))

        self.assertTrue(mem.compact_if_needed(force=True))

        messages = mem.get_messages()
        # Compacted summary preserves visible finding and drops <think> content
        summary = messages[2].content
        self.assertTrue(summary.startswith(COMPACTION_SUMMARY_HEADER))
        self.assertIn("I will now inspect the project.", summary)
        self.assertNotIn("Analyzing dependencies", summary)

        # Older retained assistant turn: <thinking> and reasoning extra_fields pruned, non-reasoning kept
        older_asst = messages[3]
        self.assertEqual(older_asst.content, "Checked status.")
        self.assertEqual(older_asst.extra_fields, {"provider_tag": "keep_me"})

        # Latest active assistant turn: reasoning intact
        latest_asst = messages[5]
        self.assertEqual(latest_asst.content, "<think>Recent reasoning stream</think>Recent answer")
        self.assertEqual(latest_asst.extra_fields, {"reasoning_content": "active reasoning"})

    def test_compaction_retains_structured_findings_errors_and_log_paths(self):
        mem = Memory(
            system_prompt="SYS",
            user_prompt="TASK",
            max_context_tokens=2000,
            keep_recent_turns=1,
        )
        mem.add_assistant(
            "Running tests",
            [ToolCall(id="c1", name="bash", arguments={"command": "pytest -q"})],
        )
        mem.add_tool_result(
            ToolResult(
                call_id="c1",
                output="Log line\n" * 30,
                raw_log_path="/logs/step_01_bash_c1.log",
            )
        )
        mem.add_assistant(
            "Inspecting failure",
            [ToolCall(id="c2", name="read_file", arguments={"path": "core/app.py"})],
        )
        mem.add_tool_result(
            ToolResult(
                call_id="c2",
                output="def broken():\n    raise ValueError('bad config')\n",
                raw_log_path=None,
            )
        )
        mem.add_assistant("Recent")

        self.assertTrue(mem.compact_if_needed(force=True))

        msgs = mem.get_messages()
        self.assertEqual(len(msgs), 4)
        summary = msgs[2].content
        self.assertTrue(summary.startswith(COMPACTION_SUMMARY_HEADER))
        self.assertIn("bash(pytest -q)", summary)
        self.assertIn("/logs/step_01_bash_c1.log", summary)
        self.assertIn("read_file(core/app.py)", summary)
        self.assertIn("bad config", summary)
        self.assertIn("Errors / warnings:", summary)

    def test_prefix_immutability_across_normal_turns_and_compaction_boundaries(self):
        mem = Memory(
            system_prompt="SYS",
            user_prompt="TASK",
            max_context_tokens=500,
            keep_recent_turns=1,
        )
        mem.record_modified_file("src/main.py")
        mem.add_assistant("Step 1")
        mem.add_tool_result(ToolResult(call_id="c1", output="ok1"))

        # Normal turns under budget must NOT insert index-2 ledger or mutate prefix
        self.assertFalse(mem.compact_if_needed(force=False))
        self.assertEqual(len(mem.messages), 4)
        self.assertEqual(mem.messages[0].content, "SYS")
        self.assertEqual(mem.messages[1].content, "TASK")
        self.assertEqual(mem.messages[2].content, "Step 1")
        self.assertEqual(mem.messages[3].content, "ok1")

        # Recording another modified file during normal turns leaves messages untouched
        mem.record_modified_file("src/utils.py")
        self.assertFalse(mem.compact_if_needed(force=False))
        self.assertEqual([m.content for m in mem.messages], ["SYS", "TASK", "Step 1", "ok1"])

        # Add recent turn and force compaction boundary
        mem.add_assistant("Recent")
        self.assertTrue(mem.compact_if_needed(force=True))

        msgs = mem.get_messages()
        self.assertEqual(msgs[0].role, "system")
        self.assertEqual(msgs[0].content, "SYS")
        self.assertEqual(msgs[1].role, "user")
        self.assertEqual(msgs[1].content, "TASK")
        summary1 = msgs[2].content
        self.assertTrue(summary1.startswith(COMPACTION_SUMMARY_HEADER))
        self.assertIn("Modified files: src/main.py, src/utils.py", summary1)

        # Normal turns after compaction must NOT mutate summary1 in-place
        mem.record_modified_file("tests/test_main.py")
        mem.add_tool_result(ToolResult(call_id="c2", output="ok2"))
        self.assertFalse(mem.compact_if_needed(force=False))
        self.assertEqual(mem.messages[2].content, summary1)

        # Calling compact_if_needed(force=True) with no new middle turns is a no-op (keeps summary1 immutable)
        mem2_msgs_before = [m.content for m in mem.messages]
        # Note: here recent is [assistant("Recent"), tool("ok2")] with keep_recent_turns=1;
        # walking back avoids orphaning tool("ok2") so split_idx lands at summary1 -> no new middle turns.
        self.assertFalse(mem.compact_if_needed(force=True))
        self.assertEqual([m.content for m in mem.messages], mem2_msgs_before)

        # Adding another turn moves [Recent, ok2] into middle and updates summary at new boundary
        mem.add_assistant("Latest")
        self.assertTrue(mem.compact_if_needed(force=True))
        msgs_after = mem.get_messages()
        self.assertEqual(msgs_after[0].content, "SYS")
        self.assertEqual(msgs_after[1].content, "TASK")
        self.assertIn(
            "Modified files: src/main.py, src/utils.py, tests/test_main.py",
            msgs_after[2].content,
        )
        self.assertEqual(msgs_after[-1].content, "Latest")

        # Stats reflect modified files and reset clears them
        self.assertEqual(
            mem.stats()["modified_files"],
            ["src/main.py", "src/utils.py", "tests/test_main.py"],
        )
        mem.reset(new_instruction="NEW_TASK")
        self.assertEqual(mem.stats()["modified_files"], [])
        self.assertIsNone(mem.stats()["calibrated_prompt_tokens"])

    def test_message_from_dict_deserialization(self):
        """Invariant: Message.from_dict deserializes all message shapes, tool calls, and handles malformed args."""
        # 1. Plain user message
        m1 = Message.from_dict({"role": "user", "content": "hello world"})
        self.assertEqual(m1.role, "user")
        self.assertEqual(m1.content, "hello world")
        self.assertIsNone(m1.tool_calls)
        self.assertIsNone(m1.tool_call_id)

        # 2. Assistant message with function tool call (JSON string arguments)
        m2 = Message.from_dict(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": "pwd"}'},
                        "extra_content": "some_extra",
                    }
                ],
            }
        )
        self.assertEqual(m2.role, "assistant")
        self.assertIsNotNone(m2.tool_calls)
        self.assertEqual(len(m2.tool_calls), 1)
        tc1 = m2.tool_calls[0]
        self.assertEqual(tc1.id, "call_1")
        self.assertEqual(tc1.name, "bash")
        self.assertEqual(tc1.arguments, {"command": "pwd"})
        self.assertEqual(tc1.extra_content, "some_extra")

        # 3. Direct dictionary arguments (non-standard / alternative format)
        m3 = Message.from_dict(
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_2", "name": "read_file", "arguments": {"path": "a.txt"}}
                ],
            }
        )
        self.assertEqual(m3.tool_calls[0].arguments, {"path": "a.txt"})

        # 4. Malformed JSON string arguments fallback
        m4 = Message.from_dict(
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_3", "name": "custom", "arguments": "not valid json"}],
            }
        )
        self.assertEqual(m4.tool_calls[0].arguments, {"raw": "not valid json"})

        # 5. Tool result message with tool_call_id and raw_log_path
        m5 = Message.from_dict(
            {
                "role": "tool",
                "content": "output text",
                "tool_call_id": "call_1",
                "raw_log_path": "/tmp/raw.log",
            }
        )
        self.assertEqual(m5.role, "tool")
        self.assertEqual(m5.content, "output text")
        self.assertEqual(m5.tool_call_id, "call_1")
        self.assertEqual(m5.raw_log_path, "/tmp/raw.log")

    def test_memory_restore_messages(self):
        """Invariant: restore_messages reconstructs history and properly handles system prompt refresh."""
        mem = Memory(system_prompt="OLD_SYSTEM", user_prompt="OLD_USER")

        # 1. Restoring empty list is a no-op
        mem.restore_messages([])
        self.assertEqual(len(mem.get_messages()), 2)

        # 2. Restoring with refresh_system_prompt when messages start with system message
        restored = [
            Message(role="system", content="OLD_PERSISTED_SYSTEM"),
            Message(role="user", content="Turn 1 prompt"),
            Message(role="assistant", content="Turn 1 reply"),
        ]
        mem.restore_messages(restored, refresh_system_prompt="NEW_UPDATED_SYSTEM")
        msgs = mem.get_messages()
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[0].role, "system")
        self.assertEqual(msgs[0].content, "NEW_UPDATED_SYSTEM")
        self.assertEqual(msgs[1].content, "Turn 1 prompt")
        self.assertEqual(msgs[2].content, "Turn 1 reply")

        # 3. Restoring when messages lack system message at index 0 prepends refreshed system message
        no_sys = [
            Message(role="user", content="Direct prompt"),
            Message(role="assistant", content="Direct reply"),
        ]
        mem.restore_messages(no_sys, refresh_system_prompt="PREPENDED_SYSTEM")
        msgs_prepended = mem.get_messages()
        self.assertEqual(len(msgs_prepended), 3)
        self.assertEqual(msgs_prepended[0].role, "system")
        self.assertEqual(msgs_prepended[0].content, "PREPENDED_SYSTEM")
        self.assertEqual(msgs_prepended[1].content, "Direct prompt")

    def test_token_estimation_counts_tool_calls_and_extra_fields(self):
        """Invariant: estimated_tokens accounts for tool_calls arguments and extra_fields."""
        mem = Memory(system_prompt="", user_prompt="")
        mem.add_assistant(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="write_file",
                    arguments={"path": "big.py", "content": "A" * 400},
                )
            ],
            extra_fields={"reasoning_content": "B" * 200},
        )
        tokens = mem.estimated_tokens()
        # content is empty, yet tool_calls + extra_fields contribute > 150 tokens
        self.assertGreaterEqual(tokens, 150)

    def test_token_calibration_and_agent_loop_telemetry(self):
        """Invariant: AgentLoop calibrates Memory from actual prompt_tokens and scales across compaction."""
        from unittest.mock import MagicMock

        class TelemetryModel:
            def __init__(self):
                self.last_usage = {}
                self.step = 0

            def generate(self, messages, tools):
                self.step += 1
                if self.step == 1:
                    self.last_usage = {"prompt_tokens": 1200, "completion_tokens": 50}
                    return "Step 1", [ToolCall(id="c1", name="echo", arguments={"text": "hi"})]
                self.last_usage = {"input_tokens": 2400, "output_tokens": 40}
                return "Done", []

            def get_last_step_usage(self):
                return dict(self.last_usage)

        class EchoTool:
            name = "echo"
            description = "Echo"
            parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

            def execute(self, text: str = "") -> tuple[int, str]:
                return 0, "R" * 400  # 100 raw tokens

        mem = Memory(
            system_prompt="SYS",
            user_prompt="TASK",
            max_context_tokens=5000,
            keep_recent_turns=1,
        )
        registry = ToolRegistry(env=MagicMock())
        registry.register(EchoTool())
        loop = AgentLoop(model=TelemetryModel(), memory=mem, tools=registry)

        raw_before_step1 = mem.raw_estimated_tokens()
        finished1, _ = loop.step(1)
        self.assertFalse(finished1)
        self.assertEqual(mem.stats()["calibrated_prompt_tokens"], 1200)
        # Estimate reflects actual 1200 prompt tokens + delta from step 1 assistant & tool result
        expected_delta = mem.raw_estimated_tokens() - raw_before_step1
        self.assertEqual(mem.estimated_tokens(), 1200 + expected_delta)

        # Negative test: invalid / zero calibration does not overwrite valid calibration
        mem.calibrate_tokens(0)
        self.assertEqual(mem.stats()["calibrated_prompt_tokens"], 1200)

        # Second step reports Anthropic-style input_tokens=2400
        finished2, _ = loop.step(2)
        self.assertTrue(finished2)
        self.assertEqual(mem.stats()["calibrated_prompt_tokens"], 2400)

        # Compaction shrinks raw tokens and calibrated estimate scales down proportionally
        before_compact = mem.estimated_tokens()
        self.assertTrue(mem.compact_if_needed(force=True))
        after_compact = mem.estimated_tokens()
        self.assertLess(after_compact, before_compact)

        # restore_messages clears calibration
        mem.restore_messages([Message(role="user", content="12345678")])
        self.assertIsNone(mem.stats()["calibrated_prompt_tokens"])
        self.assertEqual(mem.estimated_tokens(), 2)

    def test_compaction_handles_assistant_tool_call_dominated_sessions(self):
        """Invariant: compaction shrinks sessions dominated by assistant/tool-call payloads without orphaning tool results."""
        mem = Memory(
            system_prompt="SYS",
            user_prompt="GOAL",
            max_context_tokens=150,
            trigger_ratio=0.80,
            keep_recent_turns=1,
        )
        for idx in range(5):
            mem.add_assistant(
                f"Applying edit {idx}",
                [
                    ToolCall(
                        id=f"c{idx}",
                        name="write_file",
                        arguments={
                            "path": f"file_{idx}.py",
                            "content": f"x_{idx} = " + ("1" * 200),
                        },
                    )
                ],
                extra_fields={"reasoning_content": "R" * 160},
            )
            mem.add_tool_result(ToolResult(call_id=f"c{idx}", output="ok"))

        before_tokens = mem.estimated_tokens()
        self.assertTrue(mem.compact_if_needed())
        after_tokens = mem.estimated_tokens()
        self.assertLess(after_tokens, before_tokens)

        msgs = mem.get_messages()
        # Prefix preserved, summary emitted, recent [assistant, tool] pair kept intact (not orphaned!)
        self.assertEqual(msgs[0].content, "SYS")
        self.assertEqual(msgs[1].content, "GOAL")
        self.assertTrue(msgs[2].content.startswith(COMPACTION_SUMMARY_HEADER))
        self.assertEqual(msgs[3].role, "assistant")
        self.assertEqual(msgs[4].role, "tool")

    def test_compaction_prunes_assistant_reasoning_followed_by_user_message(self):
        """Invariant: an assistant turn followed by a user turn is stale and gets reasoning pruned."""
        mem = Memory(
            system_prompt="SYS",
            user_prompt="GOAL",
            max_context_tokens=500,
            keep_recent_turns=4,
        )
        mem.add_assistant(
            "Done part 1",
            extra_fields={"reasoning_content": "stale thinking", "reasoning_details": [{"x": 1}]},
        )
        mem.add_user("Now do part 2")

        self.assertTrue(mem.compact_if_needed(force=True))
        msgs = mem.get_messages()
        self.assertEqual(msgs[2].content, "Done part 1")
        self.assertIsNone(msgs[2].extra_fields)


class TestFormatCompactNum(unittest.TestCase):
    def test_small_numbers(self):
        """Invariant: numbers below 1000 format as exact integers."""
        self.assertEqual(format_compact_num(0), "0")
        self.assertEqual(format_compact_num(1), "1")
        self.assertEqual(format_compact_num(500), "500")
        self.assertEqual(format_compact_num(999), "999")

    def test_thousands_k_suffix(self):
        """Invariant: numbers >= 1000 use 'k' suffix with single decimal when non-round."""
        self.assertEqual(format_compact_num(1000), "1k")
        self.assertEqual(format_compact_num(1420), "1.4k")
        self.assertEqual(format_compact_num(8000), "8k")
        self.assertEqual(format_compact_num(8500), "8.5k")
        self.assertEqual(format_compact_num(128000), "128k")
        self.assertEqual(format_compact_num(999900), "999.9k")

    def test_millions_m_suffix(self):
        """Invariant: numbers >= 1,000,000 use 'M' suffix."""
        self.assertEqual(format_compact_num(1000000), "1M")
        self.assertEqual(format_compact_num(1500000), "1.5M")
        self.assertEqual(format_compact_num(2000000), "2M")

    def test_edge_cases_and_non_numbers(self):
        """Invariant: None or invalid values degrade safely without exceptions."""
        self.assertEqual(format_compact_num(None), "0")
        self.assertEqual(format_compact_num("non-numeric"), "non-numeric")


if __name__ == "__main__":
    unittest.main()
