import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from core.environment import Environment
from core.harness import Harness
from core.hooks import (
    POST_STEP,
    POST_TURN,
    PRE_STEP,
    PRE_TOOL,
    PRE_TURN,
    PYTHON_SYNTAX_HOOK_NAME,
    CommandHook,
    FunctionHook,
    HookDecision,
    HookRegistry,
)
from core.loop import AgentLoop
from core.memory import Memory
from core.model import MockModel
from core.session import run_steps
from core.tools import BashTool, ReadFileTool, ToolRegistry, WriteFileTool
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


class TestGenericHooks(unittest.TestCase):
    def setUp(self):
        self.global_dir = Path(tempfile.mkdtemp(prefix="test_hooks_global_"))
        self.workspace = Path(tempfile.mkdtemp(prefix="test_hooks_ws_"))
        self.runs_dir = Path(tempfile.mkdtemp(prefix="test_hooks_runs_"))
        self.env = Environment(workspace_dir=self.workspace, cleanup_on_teardown=True)

    def tearDown(self):
        self.env.teardown()

    def test_default_hooks_registered_and_removable_via_api_or_config(self):
        """Invariant: default hooks register automatically and can be removed by name."""
        reg = HookRegistry.with_defaults()
        self.assertIn(PYTHON_SYNTAX_HOOK_NAME, reg)
        self.assertTrue(reg.unregister(PYTHON_SYNTAX_HOOK_NAME))
        self.assertNotIn(PYTHON_SYNTAX_HOOK_NAME, reg)
        self.assertFalse(reg.unregister(PYTHON_SYNTAX_HOOK_NAME))

        # Disable via project config disabled_hooks (even without trust_project_tools)
        (self.workspace / ".microharness").mkdir()
        (self.workspace / ".microharness" / "config.json").write_text(
            json.dumps({"disabled_hooks": [PYTHON_SYNTAX_HOOK_NAME]}),
            encoding="utf-8",
        )
        script = [
            (
                "Write bad python",
                [
                    ToolCall(
                        id="w1",
                        name="write_file",
                        arguments={"path": "bad.py", "content": "def broken(\n"},
                    )
                ],
            ),
            ("Done", []),
        ]
        harness = Harness(
            model=MockModel(script),
            workspace_dir=self.workspace,
            global_dir=self.global_dir,
            runs_dir=self.runs_dir,
            sandbox=False,
            verbose=False,
        )
        self.assertNotIn(PYTHON_SYNTAX_HOOK_NAME, harness.hooks)
        result = harness.run(Task(id="no_syntax_hook", instruction="Write bad.py", max_steps=5))
        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        self.assertNotIn("[Post-edit diagnostics]", trace["steps"][0]["tool_results"][0]["output"])

        # Also disable via {"enabled": false} inside hooks config
        (self.workspace / ".microharness" / "config.json").write_text(
            json.dumps(
                {"hooks": {"post_tool": [{"name": PYTHON_SYNTAX_HOOK_NAME, "enabled": False}]}}
            ),
            encoding="utf-8",
        )
        harness2 = Harness(
            model=MockModel([]),
            workspace_dir=self.workspace,
            global_dir=self.global_dir,
            runs_dir=self.runs_dir,
            sandbox=False,
            verbose=False,
        )
        self.assertNotIn(PYTHON_SYNTAX_HOOK_NAME, harness2.hooks)

        # Harness(..., include_default_hooks=False) starts without default hooks
        harness3 = Harness(
            model=MockModel([]),
            workspace_dir=self.workspace,
            global_dir=self.global_dir,
            runs_dir=self.runs_dir,
            sandbox=False,
            verbose=False,
            include_default_hooks=False,
        )
        self.assertEqual(harness3.hooks.list_hooks(), [])

    def test_pre_tool_hook_blocks_execution_via_function_or_command(self):
        """Invariant: pre_tool hooks returning block=True or non-zero CommandHook prevent tool execution."""
        hooks = HookRegistry.empty()
        hooks.register(
            FunctionHook(
                name="block_rm",
                event=PRE_TOOL,
                matcher="bash",
                fn=lambda ctx: (
                    HookDecision(message="Blocked dangerous rm command", block=True)
                    if "rm -rf" in str((ctx.tool_args or {}).get("command", ""))
                    else None
                ),
            )
        )
        reg = ToolRegistry(env=self.env, hooks=hooks)
        reg.register(BashTool(self.env))

        blocked = reg.dispatch(
            ToolCall(id="b1", name="bash", arguments={"command": "rm -rf /tmp/nope"}),
            step_index=1,
        )
        self.assertEqual(blocked.exit_code, 1)
        self.assertEqual(blocked.output, "Blocked dangerous rm command")

        allowed = reg.dispatch(
            ToolCall(id="b2", name="bash", arguments={"command": "echo safe"}),
            step_index=2,
        )
        self.assertEqual(allowed.exit_code, 0)
        self.assertEqual(allowed.output.strip(), "safe")

        # CommandHook on pre_tool blocks on non-zero exit
        cmd_hooks = HookRegistry.empty()
        cmd_hooks.register(
            CommandHook(
                name="guard_cmd",
                event=PRE_TOOL,
                matcher="write_file",
                path_glob="*.secret",
                command="python -c \"import sys; print('Refused:' + sys.argv[1]); sys.exit(2)\" {path}",
            )
        )
        reg2 = ToolRegistry(env=self.env, hooks=cmd_hooks)
        reg2.register(WriteFileTool(self.env))
        secret_res = reg2.dispatch(
            ToolCall(
                id="w1", name="write_file", arguments={"path": "api.secret", "content": "leak"}
            ),
            step_index=1,
        )
        self.assertEqual(secret_res.exit_code, 1)
        self.assertEqual(secret_res.output, "Refused:api.secret")
        self.assertFalse((self.workspace / "api.secret").exists())

    def test_turn_and_step_lifecycle_hooks_and_pre_turn_block(self):
        """Invariant: pre_turn, pre_step, post_step, and post_turn fire in order and support blocking."""
        events: list[str] = []
        hooks = HookRegistry.empty()
        hooks.register(FunctionHook("h1", PRE_TURN, lambda _c: events.append("pre_turn") and None))
        hooks.register(
            FunctionHook(
                "h2", PRE_STEP, lambda c: events.append(f"pre_step:{c.step_index}") and None
            )
        )
        hooks.register(
            FunctionHook(
                "h3", POST_STEP, lambda c: events.append(f"post_step:{c.step_index}") and None
            )
        )
        hooks.register(
            FunctionHook("h4", POST_TURN, lambda _c: events.append("post_turn") and None)
        )

        reg = ToolRegistry(env=self.env, hooks=hooks)
        reg.register(ReadFileTool(self.env))
        self.env.write_file("f.txt", "ok\n")
        model = MockModel(
            [
                ("Read", [ToolCall(id="r1", name="read_file", arguments={"path": "f.txt"})]),
                ("Done", []),
            ]
        )
        mem = Memory(system_prompt="sys", user_prompt="go")
        loop = AgentLoop(model=model, memory=mem, tools=reg)
        tracer = Tracer(
            task=Task(id="hooks_order", instruction="go"), output_dir=self.runs_dir, verbose=False
        )
        outcome = run_steps(model, loop, tracer, start_step=1, max_steps=5)

        self.assertTrue(outcome.finished)
        self.assertEqual(
            events,
            ["pre_turn", "pre_step:1", "post_step:1", "pre_step:2", "post_step:2", "post_turn"],
        )

        # pre_turn block halts run_steps immediately before any model call
        block_hooks = HookRegistry.empty()
        block_hooks.register(
            FunctionHook(
                "stop_turn",
                PRE_TURN,
                lambda _c: HookDecision(message="Workspace dirty; commit first", block=True),
            )
        )
        reg_b = ToolRegistry(env=self.env, hooks=block_hooks)
        loop_b = AgentLoop(model=MockModel([("Unreachable", [])]), memory=mem, tools=reg_b)
        out_b = run_steps(loop_b.model, loop_b, tracer, start_step=1, max_steps=5)
        self.assertFalse(out_b.finished)
        self.assertEqual(out_b.reason, "hook_blocked")
        self.assertEqual(out_b.detail, "Workspace dirty; commit first")

        # TuiTurnRunner also enforces pre_turn block
        state = TUIState(title="test")
        runner = TuiTurnRunner(
            model=loop_b.model,
            loop=loop_b,
            memory=mem,
            tracer=tracer,
            state=state,
            tui=_DummyTui(),
            cache_stats=lambda: {},
        )
        runner.run_turn(max_steps=5)
        self.assertTrue(any("Workspace dirty" in line.text for line in state.lines))

    def test_project_config_command_hook_requires_trust_while_global_runs(self):
        """Invariant: project config command hooks are ignored unless trust_project_tools=True; global hooks are trusted."""
        (self.global_dir / "config.json").write_text(
            json.dumps(
                {
                    "hooks": {
                        "post_tool": [
                            {
                                "name": "global_echo",
                                "matcher": "write_file",
                                "command": "python -c \"print('GLOBAL_HOOK'); raise SystemExit(1)\"",
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        (self.workspace / ".microharness").mkdir()
        (self.workspace / ".microharness" / "config.json").write_text(
            json.dumps(
                {
                    "hooks": {
                        "post_tool": [
                            {
                                "name": "proj_echo",
                                "matcher": "write_file",
                                "command": "python -c \"print('PROJECT_HOOK'); raise SystemExit(1)\"",
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        script = [
            (
                "Write",
                [
                    ToolCall(
                        id="w1", name="write_file", arguments={"path": "a.txt", "content": "hi"}
                    )
                ],
            ),
            ("Done", []),
        ]

        # Untrusted: global runs, project skipped with warning
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            untrusted = Harness(
                model=MockModel(script),
                workspace_dir=self.workspace,
                global_dir=self.global_dir,
                runs_dir=self.runs_dir,
                sandbox=False,
                verbose=False,
            )
        self.assertIn("global_echo", untrusted.hooks)
        self.assertNotIn("proj_echo", untrusted.hooks)
        self.assertIn("Skipping untrusted project hook", stderr.getvalue())

        res_untrusted = untrusted.run(Task(id="untrusted", instruction="Write a.txt", max_steps=4))
        trace_u = json.loads(Path(res_untrusted.trace_path).read_text(encoding="utf-8"))
        out_u = trace_u["steps"][0]["tool_results"][0]["output"]
        self.assertIn("GLOBAL_HOOK", out_u)
        self.assertNotIn("PROJECT_HOOK", out_u)

        # Trusted: both global and project run
        trusted = Harness(
            model=MockModel(script),
            workspace_dir=self.workspace,
            global_dir=self.global_dir,
            runs_dir=self.runs_dir,
            sandbox=False,
            verbose=False,
            trust_project_tools=True,
        )
        self.assertIn("proj_echo", trusted.hooks)
        res_trusted = trusted.run(Task(id="trusted", instruction="Write a.txt", max_steps=4))
        trace_t = json.loads(Path(res_trusted.trace_path).read_text(encoding="utf-8"))
        out_t = trace_t["steps"][0]["tool_results"][0]["output"]
        self.assertIn("GLOBAL_HOOK", out_t)
        self.assertIn("PROJECT_HOOK", out_t)


if __name__ == "__main__":
    unittest.main()
