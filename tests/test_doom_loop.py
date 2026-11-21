import json
import tempfile
import unittest
from pathlib import Path

from core.environment import Environment
from core.harness import Harness
from core.loop import DOOM_LOOP, AgentLoop, DoomLoopDetector
from core.memory import Memory
from core.model import MockModel
from core.tools import BashTool, EditFileTool, ReadFileTool, ToolRegistry, WriteFileTool
from core.tracer import Tracer
from core.tui import TUIState
from core.tui_session import TuiTurnRunner
from core.types import Task, ToolCall


class _DummyTui:
    stdscr = None

    def check_cancel_requested(self) -> bool:
        return False

    def request_redraw(self) -> None:
        return None


class TestDoomLoopDetection(unittest.TestCase):
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
                "Inspecting file after nudge",
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
        result = harness.run(Task(id="recover", instruction="Update app.py", max_steps=10))

        self.assertTrue(result.passed)
        self.assertEqual(result.finish_reason, "completed")
        self.assertEqual(result.total_steps, 6)
        self.assertEqual((self.workspace / "app.py").read_text(encoding="utf-8"), "value = 2\n")

        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        # Step 4 prompt must include the injected loop-guard nudge
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
            doom_loop_threshold=3,
            doom_loop_max_nudges=1,
        )
        result = harness.run(Task(id="halt_loop", instruction="Edit app.py", max_steps=10))

        self.assertFalse(result.passed)
        self.assertEqual(result.finish_reason, DOOM_LOOP)
        self.assertEqual(result.total_steps, 4)

        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        self.assertEqual(trace["finish_reason"], DOOM_LOOP)
        self.assertEqual(trace["steps"][-1]["stop_reason"], DOOM_LOOP)
        self.assertIn("edit_file", trace["steps"][-1]["stop_detail"])

    def test_zero_nudges_halts_immediately_at_threshold(self):
        """Invariant: doom_loop_max_nudges=0 halts on the threshold step without nudging."""
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
            doom_loop_threshold=3,
            doom_loop_max_nudges=0,
        )
        result = harness.run(Task(id="no_nudge", instruction="Run", max_steps=10))

        self.assertFalse(result.passed)
        self.assertEqual(result.finish_reason, DOOM_LOOP)
        self.assertEqual(result.total_steps, 3)

    def test_threshold_zero_disables_doom_loop_detection(self):
        """Invariant: doom_loop_threshold=0 disables doom-loop detection completely."""
        script = [
            (f"Run {i}", [ToolCall(id=f"b{i}", name="bash", arguments={"command": "exit 1"})])
            for i in range(1, 6)
        ]
        harness = Harness(
            model=MockModel(script),
            workspace_dir=self.workspace,
            runs_dir=self.runs_dir,
            sandbox=False,
            verbose=False,
            doom_loop_threshold=0,
        )
        result = harness.run(Task(id="disabled", instruction="Run", max_steps=5))

        self.assertFalse(result.passed)
        self.assertEqual(result.finish_reason, "max_steps_exceeded")
        self.assertEqual(result.total_steps, 5)

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
            doom_loop_threshold=3,
        )
        result = harness.run(Task(id="interleaved", instruction="Iterate", max_steps=10))

        self.assertTrue(result.passed)
        self.assertEqual(result.finish_reason, "completed")
        self.assertEqual(result.total_steps, 7)

    def test_edit_oscillation_a_b_a_nudges_once_and_stops_on_b(self):
        """Invariant: A->B->A edit oscillation nudges once and halts on continued oscillation, while a single revert does not trigger."""
        self.env.write_file("mod.py", "state = 'A'\n")

        # 1. Single revert (A -> B -> A in 2 edits) must NOT trigger
        detector = DoomLoopDetector(threshold=3, max_nudges=1, env=self.env)
        self.env.write_file("mod.py", "state = 'B'\n")
        c1 = detector.observe_step(
            [
                ToolCall(
                    id="1",
                    name="edit_file",
                    arguments={"path": "mod.py", "old_str": "A", "new_str": "B"},
                )
            ],
            [{"call_id": "1", "exit_code": 0, "output": "ok"}],
        )
        self.assertFalse(c1.triggered)

        self.env.write_file("mod.py", "state = 'A'\n")
        c2 = detector.observe_step(
            [
                ToolCall(
                    id="2",
                    name="edit_file",
                    arguments={"path": "mod.py", "old_str": "B", "new_str": "A"},
                )
            ],
            [{"call_id": "2", "exit_code": 0, "output": "ok"}],
        )
        self.assertFalse(c2.triggered)

        # 2. Full A -> B -> A -> B oscillation in AgentLoop/Harness:
        #    Step 1: B, Step 2: A, Step 3: B (nudge), Step 4: A (stop)
        script = [
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
        harness = Harness(
            model=MockModel(script),
            workspace_dir=self.workspace,
            runs_dir=self.runs_dir,
            sandbox=False,
            verbose=False,
            doom_loop_threshold=3,
            doom_loop_max_nudges=1,
        )
        result = harness.run(Task(id="oscillate", instruction="Edit mod.py", max_steps=10))

        self.assertFalse(result.passed)
        self.assertEqual(result.finish_reason, DOOM_LOOP)
        self.assertEqual(result.total_steps, 4)

    def test_tui_turn_runner_surfaces_doom_loop_nudge_and_stop(self):
        """Invariant: TuiTurnRunner surfaces loop-guard nudges and stops cleanly on DOOM_LOOP."""
        script = [
            (f"Fail {i}", [ToolCall(id=f"b{i}", name="bash", arguments={"command": "exit 1"})])
            for i in range(1, 6)
        ]
        memory = Memory(system_prompt="sys", user_prompt="go")
        loop = AgentLoop(
            model=MockModel(script),
            memory=memory,
            tools=self.registry,
            doom_loop_threshold=2,
            doom_loop_max_nudges=1,
        )
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

    def test_subagent_inherits_hooks_and_doom_loop(self):
        """Invariant: SubagentTool shares parent HookRegistry and doom-loop limits."""
        from core.hooks import POST_TOOL, FunctionHook, HookRegistry
        from core.subagents import BUILTIN_SUBAGENTS, SubagentTool

        parent_model = MockModel(
            [
                (
                    "Delegate",
                    [
                        ToolCall(
                            id="s1", name="subagent", arguments={"agent": "worker", "task": "edit"}
                        )
                    ],
                ),
                ("Done", []),
            ]
        )
        child_script = [
            (f"Fail {i}", [ToolCall(id=f"b{i}", name="bash", arguments={"command": "exit 1"})])
            for i in range(1, 5)
        ]

        hooks = HookRegistry.with_defaults()
        hooks.register(
            FunctionHook(name="child_note", event=POST_TOOL, fn=lambda _ctx: "from_parent_hook")
        )
        parent_reg = ToolRegistry(env=self.env, hooks=hooks)
        parent_reg.register(BashTool(self.env))
        parent_reg.register(ReadFileTool(self.env))
        parent_reg.register(WriteFileTool(self.env))
        parent_reg.register(EditFileTool(self.env))

        sub_tool = SubagentTool(
            env=self.env,
            parent_registry=parent_reg,
            subagents=BUILTIN_SUBAGENTS,
            model_getter=lambda: parent_model,
        )
        child_env = Environment(workspace_dir=self.workspace / "child", cleanup_on_teardown=True)
        try:
            child_reg = sub_tool._build_child_registry(BUILTIN_SUBAGENTS["worker"], child_env)
            self.assertIs(child_reg.hooks, hooks)
            res = child_reg.dispatch(
                ToolCall(
                    id="cw1", name="write_file", arguments={"path": "x.py", "content": "x = 1\n"}
                ),
                step_index=1,
            )
            self.assertIn("from_parent_hook", res.output)
        finally:
            child_env.teardown()

        # Harness propagates doom_loop_threshold / doom_loop_max_nudges
        harness = Harness(
            model=MockModel(child_script),
            workspace_dir=self.workspace,
            runs_dir=self.runs_dir,
            sandbox=False,
            verbose=False,
            doom_loop_threshold=2,
            doom_loop_max_nudges=0,
        )
        res = harness.run(Task(id="child_cfg", instruction="Loop", max_steps=10))
        self.assertEqual(res.finish_reason, DOOM_LOOP)
        self.assertEqual(res.total_steps, 2)

    def test_harness_config_hooks_end_to_end(self):
        """Invariant: trusted project config hooks run automatically on edits in Harness.run."""
        (self.workspace / ".microharness").mkdir()
        (self.workspace / ".microharness" / "config.json").write_text(
            json.dumps(
                {
                    "hooks": {
                        "post_tool": [
                            {
                                "name": "echo_hook",
                                "matcher": "write_file|edit_file",
                                "path_glob": "*.py",
                                "only_on_success": True,
                                "command": "python -c \"import sys; print('HOOK:' + sys.argv[1]); sys.exit(1)\" {path}",
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        script = [
            (
                "Write file",
                [
                    ToolCall(
                        id="w1",
                        name="write_file",
                        arguments={"path": "demo.py", "content": "x = 1\n"},
                    )
                ],
            ),
            ("Done", []),
        ]
        harness = Harness(
            model=MockModel(script),
            workspace_dir=self.workspace,
            runs_dir=self.runs_dir,
            sandbox=False,
            verbose=False,
            trust_project_tools=True,
        )
        result = harness.run(Task(id="hook_e2e", instruction="Write demo.py", max_steps=5))

        self.assertTrue(result.passed)
        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        tool_out = trace["steps"][0]["tool_results"][0]["output"]
        self.assertIn("HOOK:demo.py", tool_out)


if __name__ == "__main__":
    unittest.main()
