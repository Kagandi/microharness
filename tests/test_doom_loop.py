import json
import tempfile
import unittest
from pathlib import Path

from core.environment import Environment
from core.harness import Harness
from core.hooks import DOOM_LOOP_HOOK_NAME, POST_TOOL, Hook, HookRegistry, make_doom_loop_hook
from core.loop import AgentLoop
from core.memory import Memory
from core.model import MockModel
from core.subagents import BUILTIN_SUBAGENTS, SubagentTool
from core.tools import BashTool, EditFileTool, ReadFileTool, ToolRegistry, WriteFileTool
from core.tracer import Tracer
from core.tui import TUIState
from core.tui_session import TuiTurnRunner
from core.types import DOOM_LOOP, Task, ToolCall


class _DummyTui:
    stdscr = None

    def check_cancel_requested(self) -> bool:
        return False

    def request_redraw(self) -> None:
        return None


class TestDoomLoopHook(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(tempfile.mkdtemp(prefix="test_doom_loop_ws_"))
        self.runs_dir = Path(tempfile.mkdtemp(prefix="test_doom_loop_runs_"))
        self.env = Environment(workspace_dir=self.workspace, cleanup_on_teardown=True)
        self.registry = ToolRegistry(env=self.env)
        self.registry.register(BashTool(self.env))
        self.registry.register(ReadFileTool(self.env))
        self.registry.register(WriteFileTool(self.env))
        self.registry.register(EditFileTool(self.env))

    def tearDown(self):
        self.env.teardown()

    def test_repeated_identical_failing_calls_nudge_once_and_recover(self):
        """Invariant: 3 identical failing calls inject one nudge; changing approach completes cleanly."""
        self.env.write_file("app.py", "value = 1\n")
        fail_call = ToolCall(
            id="e_fail",
            name="edit_file",
            arguments={"path": "app.py", "old_str": "missing", "new_str": "value = 2"},
        )
        script = [
            ("Attempt 1", [fail_call]),
            ("Attempt 2", [ToolCall(id="e2", name="edit_file", arguments=fail_call.arguments)]),
            ("Attempt 3", [ToolCall(id="e3", name="edit_file", arguments=fail_call.arguments)]),
            (
                "Inspecting after nudge",
                [ToolCall(id="r1", name="read_file", arguments={"path": "app.py"})],
            ),
            (
                "Applying valid edit",
                [
                    ToolCall(
                        id="e4",
                        name="edit_file",
                        arguments={
                            "path": "app.py",
                            "old_str": "value = 1",
                            "new_str": "value = 2",
                        },
                    )
                ],
            ),
            ("Done.", []),
        ]
        harness = Harness(
            model=MockModel(script),
            workspace_dir=self.workspace,
            runs_dir=self.runs_dir,
            sandbox=False,
            verbose=False,
        )
        self.assertIn(DOOM_LOOP_HOOK_NAME, harness.hooks)
        result = harness.run(Task(id="recover", instruction="Update app.py", max_steps=10))

        self.assertTrue(result.passed)
        self.assertEqual(result.finish_reason, "completed")
        self.assertEqual(result.total_steps, 6)
        self.assertEqual((self.workspace / "app.py").read_text(encoding="utf-8"), "value = 2\n")

        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        step4_user_msgs = [
            m["content"] for m in trace["steps"][3]["prompt_messages"] if m["role"] == "user"
        ]
        self.assertTrue(any("[Loop guard]" in msg for msg in step4_user_msgs))

    def test_repeated_identical_failing_calls_halt_when_persisting(self):
        """Invariant: persisting in identical failing calls after nudge stops with finish_reason='doom_loop'."""
        self.env.write_file("app.py", "value = 1\n")
        script = [
            (
                f"Attempt {i}",
                [
                    ToolCall(
                        id=f"e{i}",
                        name="edit_file",
                        arguments={"path": "app.py", "old_str": "missing", "new_str": "value = 2"},
                    )
                ],
            )
            for i in range(1, 8)
        ]
        harness = Harness(
            model=MockModel(script),
            workspace_dir=self.workspace,
            runs_dir=self.runs_dir,
            sandbox=False,
            verbose=False,
        )
        result = harness.run(Task(id="halt_loop", instruction="Edit app.py", max_steps=10))

        self.assertFalse(result.passed)
        self.assertEqual(result.finish_reason, DOOM_LOOP)
        self.assertEqual(result.total_steps, 4)

        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        self.assertEqual(trace["finish_reason"], DOOM_LOOP)
        self.assertEqual(trace["steps"][-1]["stop_reason"], DOOM_LOOP)
        self.assertIn("edit_file", trace["steps"][-1]["stop_detail"])

    def test_zero_nudges_halts_immediately_and_disabled_hook_turns_it_off(self):
        """Invariant: max_nudges=0 halts at threshold, and disabling doom_loop_detection turns it off."""
        script = [
            (f"Run {i}", [ToolCall(id=f"b{i}", name="bash", arguments={"command": "exit 2"})])
            for i in range(1, 6)
        ]
        harness = Harness(
            model=MockModel(script),
            workspace_dir=self.workspace,
            runs_dir=self.runs_dir,
            sandbox=False,
            verbose=False,
            hooks=[make_doom_loop_hook(threshold=3, max_nudges=0)],
        )
        result = harness.run(Task(id="no_nudge", instruction="Run", max_steps=10))
        self.assertFalse(result.passed)
        self.assertEqual(result.finish_reason, DOOM_LOOP)
        self.assertEqual(result.total_steps, 3)

        # Disabled via disabled_hooks
        harness_off = Harness(
            model=MockModel(script),
            workspace_dir=self.workspace,
            runs_dir=self.runs_dir,
            sandbox=False,
            verbose=False,
            disabled_hooks=[DOOM_LOOP_HOOK_NAME],
        )
        self.assertNotIn(DOOM_LOOP_HOOK_NAME, harness_off.hooks)
        res_off = harness_off.run(Task(id="disabled", instruction="Run", max_steps=5))
        self.assertFalse(res_off.passed)
        self.assertEqual(res_off.finish_reason, "max_steps_exceeded")
        self.assertEqual(res_off.total_steps, 5)

    def test_intervening_file_edit_resets_failing_bash_streak(self):
        """Invariant: alternating bash failure -> file edit -> bash failure does not false-trigger."""
        self.env.write_file("counter.txt", "0\n")
        script = [
            ("Test 1", [ToolCall(id="b1", name="bash", arguments={"command": "exit 1"})]),
            (
                "Fix 1",
                [
                    ToolCall(
                        id="w1",
                        name="write_file",
                        arguments={"path": "counter.txt", "content": "1\n"},
                    )
                ],
            ),
            ("Test 2", [ToolCall(id="b2", name="bash", arguments={"command": "exit 1"})]),
            (
                "Fix 2",
                [
                    ToolCall(
                        id="w2",
                        name="write_file",
                        arguments={"path": "counter.txt", "content": "2\n"},
                    )
                ],
            ),
            ("Test 3", [ToolCall(id="b3", name="bash", arguments={"command": "exit 1"})]),
            (
                "Fix 3",
                [
                    ToolCall(
                        id="w3",
                        name="write_file",
                        arguments={"path": "counter.txt", "content": "3\n"},
                    )
                ],
            ),
            ("All done.", []),
        ]
        harness = Harness(
            model=MockModel(script),
            workspace_dir=self.workspace,
            runs_dir=self.runs_dir,
            sandbox=False,
            verbose=False,
        )
        result = harness.run(Task(id="interleaved", instruction="Iterate", max_steps=10))
        self.assertTrue(result.passed)
        self.assertEqual(result.finish_reason, "completed")
        self.assertEqual(result.total_steps, 7)

    def test_edit_oscillation_a_b_a_nudges_once_and_stops(self):
        """Invariant: single revert does not trigger, while sustained A->B->A oscillation nudges then stops."""
        self.env.write_file("mod.py", "state = 'A'\n")
        # 1. Single revert: A -> B -> A -> finish (2 edits only) must succeed
        single_revert = [
            (
                "to B",
                [
                    ToolCall(
                        id="1",
                        name="edit_file",
                        arguments={"path": "mod.py", "old_str": "'A'", "new_str": "'B'"},
                    )
                ],
            ),
            (
                "revert to A",
                [
                    ToolCall(
                        id="2",
                        name="edit_file",
                        arguments={"path": "mod.py", "old_str": "'B'", "new_str": "'A'"},
                    )
                ],
            ),
            ("done", []),
        ]
        ok_harness = Harness(
            model=MockModel(single_revert),
            workspace_dir=self.workspace,
            runs_dir=self.runs_dir,
            sandbox=False,
            verbose=False,
        )
        ok_res = ok_harness.run(Task(id="single_revert", instruction="Edit mod.py", max_steps=5))
        self.assertEqual(ok_res.finish_reason, "completed")

        # 2. Oscillation: B -> A -> B (nudge) -> A (stop)
        oscillate = [
            (
                "to B",
                [
                    ToolCall(
                        id="1",
                        name="edit_file",
                        arguments={"path": "mod.py", "old_str": "'A'", "new_str": "'B'"},
                    )
                ],
            ),
            (
                "to A",
                [
                    ToolCall(
                        id="2",
                        name="edit_file",
                        arguments={"path": "mod.py", "old_str": "'B'", "new_str": "'A'"},
                    )
                ],
            ),
            (
                "to B again",
                [
                    ToolCall(
                        id="3",
                        name="edit_file",
                        arguments={"path": "mod.py", "old_str": "'A'", "new_str": "'B'"},
                    )
                ],
            ),
            (
                "to A again",
                [
                    ToolCall(
                        id="4",
                        name="edit_file",
                        arguments={"path": "mod.py", "old_str": "'B'", "new_str": "'A'"},
                    )
                ],
            ),
            ("should not run", []),
        ]
        halt_harness = Harness(
            model=MockModel(oscillate),
            workspace_dir=self.workspace,
            runs_dir=self.runs_dir,
            sandbox=False,
            verbose=False,
        )
        halt_res = halt_harness.run(Task(id="oscillate", instruction="Edit mod.py", max_steps=10))
        self.assertFalse(halt_res.passed)
        self.assertEqual(halt_res.finish_reason, DOOM_LOOP)
        self.assertEqual(halt_res.total_steps, 4)

    def test_tui_turn_runner_surfaces_doom_loop_nudge_and_stop(self):
        """Invariant: TuiTurnRunner surfaces post_step hook nudges and stops cleanly on DOOM_LOOP."""
        script = [
            (f"Fail {i}", [ToolCall(id=f"b{i}", name="bash", arguments={"command": "exit 1"})])
            for i in range(1, 6)
        ]
        hooks = HookRegistry.with_defaults()
        hooks.register(make_doom_loop_hook(threshold=2, max_nudges=1))
        reg = ToolRegistry(env=self.env, hooks=hooks)
        reg.register(BashTool(self.env))

        memory = Memory(system_prompt="sys", user_prompt="go")
        loop = AgentLoop(model=MockModel(script), memory=memory, tools=reg)
        tracer = Tracer(
            task=Task(id="tui_doom", instruction="go"), output_dir=self.runs_dir, verbose=False
        )
        state = TUIState(title="test")
        runner = TuiTurnRunner(
            model=loop.model,
            loop=loop,
            memory=memory,
            tracer=tracer,
            state=state,
            tui=_DummyTui(),
            cache_stats=lambda: {},
        )
        runner.run_turn(max_steps=10)

        self.assertEqual(len(tracer.steps), 3)
        sys_entries = [line.text for line in state.lines if line.tag == "system"]
        self.assertTrue(any("Loop guard:" in msg for msg in sys_entries))
        self.assertTrue(any(f"Stopped turn ({DOOM_LOOP})" in msg for msg in sys_entries))

    def test_subagent_and_config_hooks_end_to_end(self):
        """Invariant: SubagentTool shares hooks with isolated per-loop state and config overrides doom_loop params."""
        (self.workspace / ".microharness").mkdir()
        (self.workspace / ".microharness" / "config.json").write_text(
            json.dumps(
                {
                    "hooks": {
                        "post_step": [
                            {"name": DOOM_LOOP_HOOK_NAME, "threshold": 2, "max_nudges": 0}
                        ],
                        "post_tool": [
                            {
                                "name": "echo_hook",
                                "matcher": "write_file",
                                "command": "python -c \"import sys; print('HOOK:' + sys.argv[1]); sys.exit(1)\" {path}",
                            }
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )
        fail_script = [
            (f"Fail {i}", [ToolCall(id=f"b{i}", name="bash", arguments={"command": "exit 1"})])
            for i in range(1, 5)
        ]
        harness = Harness(
            model=MockModel(fail_script),
            workspace_dir=self.workspace,
            runs_dir=self.runs_dir,
            sandbox=False,
            verbose=False,
            trust_project_tools=True,
        )
        res = harness.run(Task(id="cfg_doom", instruction="Loop", max_steps=10))
        self.assertEqual(res.finish_reason, DOOM_LOOP)
        self.assertEqual(res.total_steps, 2)

        # Subagent shares parent HookRegistry
        hooks = HookRegistry.with_defaults()
        hooks.register(Hook(name="child_note", event=POST_TOOL, fn=lambda _ctx: "from_parent_hook"))
        parent_reg = ToolRegistry(env=self.env, hooks=hooks)
        parent_reg.register(WriteFileTool(self.env))
        sub_tool = SubagentTool(
            env=self.env,
            parent_registry=parent_reg,
            subagents=BUILTIN_SUBAGENTS,
            model_getter=lambda: MockModel([]),
        )
        child_env = Environment(workspace_dir=self.workspace / "child", cleanup_on_teardown=True)
        try:
            child_reg = sub_tool._build_child_registry(BUILTIN_SUBAGENTS["worker"], child_env)
            self.assertIs(child_reg.hooks, hooks)
            out = child_reg.dispatch(
                ToolCall(
                    id="cw1", name="write_file", arguments={"path": "x.py", "content": "x = 1\n"}
                ),
                step_index=1,
            )
            self.assertIn("from_parent_hook", out.output)
        finally:
            child_env.teardown()


if __name__ == "__main__":
    unittest.main()
