from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.environment import Environment
from core.harness import Harness
from core.interactive import EXIT, SKIP, TURN, CommandContext, CommandRouter, TurnState
from core.loop import AgentLoop
from core.memory import Memory
from core.model import HttpChatModel, MockModel
from core.protocols import ModelCancelledError
from core.subagents import (
    DEFAULT_SUBAGENT_MAX_STEPS,
    Subagent,
    SubagentTool,
    discover_subagents,
    format_subagent_desc,
    parse_subagent_markdown,
    render_subagents_index,
    resolve_subagent_command,
)
from core.tools import BashTool, EditFileTool, FinishTool, ReadFileTool, ToolRegistry, WriteFileTool
from core.tracer import _format_action_summary
from core.tui import CursesTUI, TUIState
from core.types import Message, Task, ToolCall


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class RecordingUI:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.successes: list[str] = []
        self.errors: list[str] = []
        self.statuses: list[str] = []
        self.echoes: list[str] = []
        self.refreshes = 0

    def info(self, text: str) -> None:
        self.infos.append(text)

    def success(self, text: str) -> None:
        self.successes.append(text)

    def error(self, text: str) -> None:
        self.errors.append(text)

    def status(self, text: str) -> None:
        self.statuses.append(text)

    def echo(self, text: str) -> None:
        self.echoes.append(text)

    def refresh(self) -> None:
        self.refreshes += 1

    def render_context(self, stats, reasoning, cache):
        return []

    @property
    def all_text(self) -> str:
        return "\n".join(self.infos + self.successes + self.errors + self.statuses)


class TestSubagentParsingAndDiscovery(unittest.TestCase):
    def test_parse_frontmatter_variants_and_aliases(self):
        """Invariant: frontmatter supports comma lists, bracket lists, YAML lists, effort/thinking and max_steps/max_turns aliases."""
        doc1 = (
            "---\n"
            "name: Security-Auditor\n"
            "description: >\n"
            "  Audits auth and crypto\n"
            "  for vulnerabilities.\n"
            "tools: read, bash, subagent, finish\n"
            "model: fast\n"
            "thinking: high # Tau-style alias\n"
            "max_turns: 8\n"
            "---\n"
            "Audit every endpoint carefully.\n"
        )
        a1 = parse_subagent_markdown(doc1, default_name="fallback", scope="project", path="a1.md")
        self.assertEqual(a1.name, "security-auditor")
        self.assertEqual(a1.description, "Audits auth and crypto for vulnerabilities.")
        self.assertEqual(a1.tools, ("read_file", "bash"))
        self.assertEqual(a1.model, "fast")
        self.assertEqual(a1.effort, "high")
        self.assertEqual(a1.max_steps, 8)
        self.assertEqual(a1.system_prompt, "Audit every endpoint carefully.")

        doc2 = (
            "---\n"
            "tools:\n"
            "  - read\n"
            "  - edit\n"
            "  - write\n"
            "model: inherit\n"
            "effort: off\n"
            "---\n"
            "# Minimal Heading\n\nBody prompt."
        )
        a2 = parse_subagent_markdown(doc2, default_name="Editor", scope="global", path="a2.md")
        self.assertEqual(a2.name, "editor")
        self.assertEqual(a2.description, "Minimal Heading")
        self.assertEqual(a2.tools, ("read_file", "edit_file", "write_file"))
        self.assertIsNone(a2.model)
        self.assertEqual(a2.effort, "off")
        self.assertIsNone(a2.max_steps)

        doc3 = "---\ntools: [read, bash] # inline comment\nreasoning_effort: medium\nmax_steps: 12\n---\nPrompt"
        a3 = parse_subagent_markdown(doc3, default_name="bracket")
        self.assertEqual(a3.tools, ("read_file", "bash"))
        self.assertEqual(a3.effort, "medium")
        self.assertEqual(a3.max_steps, 12)

        doc_wildcard = "---\ntools: '*'\n---\nPrompt"
        a_wild = parse_subagent_markdown(doc_wildcard, default_name="wild")
        self.assertIsNone(a_wild.tools)

        doc_empty_tools = "---\ntools: []\n---\nPrompt"
        a_empty = parse_subagent_markdown(doc_empty_tools, default_name="empty")
        self.assertEqual(a_empty.tools, ())

        doc_only_blocked = "---\ntools: [subagent]\n---\nPrompt"
        a_blocked = parse_subagent_markdown(doc_only_blocked, default_name="blocked")
        self.assertEqual(a_blocked.tools, ())

    def test_parse_invalid_effort_and_max_steps_warn_and_fallback_to_none(self):
        """Invariant: invalid effort or max_steps in an .md file warns on stderr and defaults to None."""
        doc = "---\neffort: bogus_level\nmax_steps: -5\n---\nBody"
        stderr_buf = io.StringIO()
        with contextlib.redirect_stderr(stderr_buf):
            agent = parse_subagent_markdown(doc, default_name="bad", path="bad.md")
        self.assertIsNone(agent.effort)
        self.assertIsNone(agent.max_steps)
        self.assertIn("Ignoring invalid reasoning effort", stderr_buf.getvalue())
        self.assertIn("Ignoring non-positive max_steps", stderr_buf.getvalue())

    def test_discover_subagents_builtins_global_project_and_shadowing(self):
        """Invariant: project agents shadow global agents, and both shadow built-in presets."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            global_dir = root / "global"
            workspace = root / "workspace"

            _write(
                global_dir / "agents" / "reviewer.md",
                "---\ndescription: Global reviewer\ntools: read\nmodel: fast\n---\nGlobal prompt",
            )
            _write(
                global_dir / "agents" / "deploy-checker.md",
                "---\ndescription: Global deploy checker\n---\nCheck deploy",
            )
            _write(
                workspace / ".microharness" / "agents" / "reviewer.md",
                "---\ndescription: Project reviewer\ntools: [read, bash]\neffort: high\n---\nProject prompt",
            )
            _write(
                workspace / ".microharness" / "agents" / "custom-dir" / "AGENT.md",
                "---\ndescription: Dir-based agent\nmax_steps: 4\n---\nDir prompt",
            )
            _write(workspace / ".microharness" / "agents" / "README.md", "# Ignore me")

            agents = discover_subagents(workspace, global_dir)
            self.assertEqual(
                set(agents.keys()),
                {"explore", "reviewer", "worker", "deploy-checker", "custom-dir"},
            )
            self.assertEqual(agents["explore"].scope, "builtin")
            self.assertEqual(agents["worker"].scope, "builtin")
            self.assertEqual(agents["deploy-checker"].scope, "global")

            # Project reviewer shadows global and builtin
            rev = agents["reviewer"]
            self.assertEqual(rev.scope, "project")
            self.assertEqual(rev.description, "Project reviewer")
            self.assertEqual(rev.tools, ("read_file", "bash"))
            self.assertEqual(rev.effort, "high")
            self.assertEqual(rev.system_prompt, "Project prompt")

            # Directory-based AGENT.md works
            cdir = agents["custom-dir"]
            self.assertEqual(cdir.scope, "project")
            self.assertEqual(cdir.max_steps, 4)

            # include_builtins=False omits builtins
            custom_only = discover_subagents(workspace, global_dir, include_builtins=False)
            self.assertEqual(set(custom_only.keys()), {"reviewer", "deploy-checker", "custom-dir"})


class TestSubagentToolExecution(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ws = self.tmp / "ws"
        self.runs = self.tmp / "runs"
        self.ws.mkdir()
        self.runs.mkdir()
        self.env = Environment(workspace_dir=self.ws, logs_dir=self.runs / "logs")
        self.registry = ToolRegistry(env=self.env)
        self.registry.register(BashTool(self.env))
        self.registry.register(ReadFileTool(self.env))
        self.registry.register(WriteFileTool(self.env))
        self.registry.register(EditFileTool(self.env))
        self.registry.register(FinishTool())

    def tearDown(self):
        self.env.teardown()

    def test_clean_context_isolation_and_read_only_scoping(self):
        """Invariant: subagent starts with empty Memory (no parent history) and scoped ToolRegistry."""
        (self.ws / "secret.py").write_text("TOKEN = 'ok'\n", encoding="utf-8")

        captured_child_messages: list[list[Message]] = []
        captured_child_tools: list[list[str]] = []

        class InspectingChildModel:
            def generate(self, messages, tools):
                captured_child_messages.append(list(messages))
                captured_child_tools.append([t["function"]["name"] for t in tools])
                if len(captured_child_messages) == 1:
                    # Attempt to call write_file (which reviewer must NOT have) and read_file
                    return "Checking", [
                        ToolCall(id="r1", name="read_file", arguments={"path": "secret.py"}),
                        ToolCall(
                            id="w1",
                            name="write_file",
                            arguments={"path": "hacked.txt", "content": "nope"},
                        ),
                    ]
                return "Reviewed secret.py: TOKEN='ok'; write_file denied.", []

        model = InspectingChildModel()
        subagents = discover_subagents(self.ws, None)
        tool = SubagentTool(
            env=self.env,
            parent_registry=self.registry,
            subagents=subagents,
            model_getter=lambda: model,
            run_dir_getter=lambda: self.runs / "parent_run",
        )
        self.registry.register(tool)

        code, output = tool.execute(agent="reviewer", task="Review secret.py")
        self.assertEqual(code, 0)
        self.assertIn("[Subagent: reviewer] (completed in 2 step(s))", output)
        self.assertIn("Reviewed secret.py", output)

        # Clean context: step 1 had only [system, user] — zero parent messages
        self.assertEqual([m.role for m in captured_child_messages[0]], ["system", "user"])
        self.assertEqual(captured_child_messages[0][1].content, "Review secret.py")

        # Tool allow-list: reviewer only gets read_file and bash (no write_file, edit_file, finish, subagent)
        self.assertEqual(set(captured_child_tools[0]), {"read_file", "bash"})
        self.assertFalse((self.ws / "hacked.txt").exists())

        # Child trace persisted under <run_dir>/subagents/
        sub_traces = list((self.runs / "parent_run" / "subagents").glob("reviewer_*/trace.json"))
        self.assertEqual(len(sub_traces), 1)
        trace_data = json.loads(sub_traces[0].read_text(encoding="utf-8"))
        self.assertEqual(trace_data["finish_reason"], "completed")
        self.assertEqual(trace_data["steps"][0]["tool_results"][1]["exit_code"], 1)
        self.assertIn(
            "Unknown tool 'write_file'", trace_data["steps"][0]["tool_results"][1]["output"]
        )

    def test_worker_excludes_subagent_and_finish_and_records_modified_files(self):
        """Invariant: worker receives file tools (excluding subagent/finish) and propagates modified_files to parent."""
        seen_tools: list[list[str]] = []
        modified_by_child: list[str] = []

        class WorkerModel:
            def generate(self, messages, tools):
                seen_tools.append([t["function"]["name"] for t in tools])
                if len(seen_tools) == 1:
                    return "Writing", [
                        ToolCall(
                            id="w1",
                            name="write_file",
                            arguments={"path": "created.txt", "content": "hello"},
                        )
                    ]
                return "Created created.txt", []

        model = WorkerModel()
        subagents = discover_subagents(self.ws, None)
        tool = SubagentTool(
            env=self.env,
            parent_registry=self.registry,
            subagents=subagents,
            model_getter=lambda: model,
            run_dir_getter=lambda: self.runs / "parent_run",
            on_file_modified=modified_by_child.append,
        )
        self.registry.register(tool)

        code, output = tool.execute(agent="worker", task="Create created.txt")
        self.assertEqual(code, 0)
        self.assertEqual(set(seen_tools[0]), {"bash", "read_file", "write_file", "edit_file"})
        self.assertNotIn("subagent", seen_tools[0])
        self.assertNotIn("finish", seen_tools[0])
        self.assertEqual((self.ws / "created.txt").read_text(encoding="utf-8"), "hello")
        self.assertEqual(modified_by_child, ["created.txt"])

    def test_effort_override_and_parent_effort_restoration(self):
        """Invariant: subagent effort overrides parent effort for child steps and restores parent effort afterwards."""
        seen_efforts: list[str | None] = []

        class EffortRecordingMock(MockModel):
            def generate(self, messages, tools):
                seen_efforts.append(self.get_reasoning_effort())
                return super().generate(messages, tools)

        parent_model = EffortRecordingMock([("Subagent finished", [])], reasoning_effort="low")
        subagents = {
            "deep": Subagent(
                name="deep",
                description="Deep thinker",
                system_prompt="Think deeply",
                scope="project",
                effort="high",
            ),
            "no_think": Subagent(
                name="no_think",
                description="Fast no-think",
                system_prompt="Answer fast",
                scope="project",
                effort="off",
            ),
        }
        tool = SubagentTool(
            env=self.env,
            parent_registry=self.registry,
            subagents=subagents,
            model_getter=lambda: parent_model,
            effort_getter=parent_model.get_reasoning_effort,
        )

        # Preset effort="high" applies during child run, then restores "low"
        code1, _ = tool.execute(agent="deep", task="Analyze")
        self.assertEqual(code1, 0)
        self.assertEqual(seen_efforts[-1], "high")
        self.assertEqual(parent_model.get_reasoning_effort(), "low")

        # Tool call effort="medium" overrides preset effort="high"
        parent_model.reset()
        code2, _ = tool.execute(agent="deep", task="Analyze", effort="medium")
        self.assertEqual(code2, 0)
        self.assertEqual(seen_efforts[-1], "medium")
        self.assertEqual(parent_model.get_reasoning_effort(), "low")

        # Preset effort="off" disables reasoning for child (None), then restores "low"
        parent_model.reset()
        code3, _ = tool.execute(agent="no_think", task="Quick answer")
        self.assertEqual(code3, 0)
        self.assertIsNone(seen_efforts[-1])
        self.assertEqual(parent_model.get_reasoning_effort(), "low")

    def test_endpoint_aware_model_shallow_copy_for_effort_and_model_builder(self):
        """Invariant: EndpointAwareModel is shallow-copied for effort overrides without mutating parent."""
        http_parent = HttpChatModel(
            base_url="http://localhost:8000/v1",
            model="parent-model",
            reasoning_effort="low",
        )
        seen_child_models: list[tuple[str, str | None]] = []

        def fake_urlopen(req, timeout=90.0):
            payload = json.loads(req.data.decode("utf-8"))
            seen_child_models.append((payload["model"], payload.get("reasoning_effort")))
            resp = MagicMock()
            resp.__enter__.return_value = resp
            resp.read.return_value = json.dumps(
                {
                    "choices": [{"message": {"content": "http ok"}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5},
                }
            ).encode("utf-8")
            return resp

        subagents = {
            "fast_rev": Subagent(
                name="fast_rev",
                description="Fast reviewer",
                system_prompt="Review",
                scope="project",
                model="fast",
                effort="off",
            ),
            "worker": Subagent(
                name="worker",
                description="Worker",
                system_prompt="Work",
                scope="builtin",
            ),
        }

        built_calls: list[tuple[str, str | None]] = []

        def fake_builder(spec: str, eff: str | None):
            built_calls.append((spec, eff))
            return (
                True,
                "",
                HttpChatModel(
                    base_url="http://localhost:8000/v1",
                    model=f"resolved-{spec}",
                    reasoning_effort=eff,
                ),
            )

        tool = SubagentTool(
            env=self.env,
            parent_registry=self.registry,
            subagents=subagents,
            model_getter=lambda: http_parent,
            model_builder=fake_builder,
            effort_getter=http_parent.get_reasoning_effort,
        )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            # 1. Inherited model with explicit effort="high" copies parent without mutating parent
            code1, out1 = tool.execute(agent="worker", task="Inspect", effort="high")
            self.assertEqual(code1, 0)
            self.assertIn("http ok", out1)
            self.assertEqual(http_parent.get_reasoning_effort(), "low")
            self.assertEqual(seen_child_models[0], ("parent-model", "high"))
            self.assertEqual(http_parent.total_prompt_tokens, 20)
            self.assertEqual(http_parent.total_completion_tokens, 5)

            # 2. Explicit model="fast" + effort="off" calls model_builder with "off"
            code2, out2 = tool.execute(agent="fast_rev", task="Review")
            self.assertEqual(code2, 0)
            self.assertIn("http ok", out2)
            self.assertEqual(built_calls, [("fast", "off")])
            self.assertEqual(seen_child_models[1], ("resolved-fast", None))
            self.assertEqual(http_parent.total_prompt_tokens, 40)

    def test_optional_max_steps_and_negative_validations(self):
        """Invariant: max_steps enforces step budget and invalid inputs fail cleanly with exit_code=1."""
        looping_model = MockModel(
            [
                ("Step 1", [ToolCall(id="b1", name="bash", arguments={"command": "echo 1"})]),
                ("Step 2", [ToolCall(id="b2", name="bash", arguments={"command": "echo 2"})]),
            ]
        )
        subagents = discover_subagents(self.ws, None)
        self.assertIsNone(subagents["worker"].max_steps)

        tool = SubagentTool(
            env=self.env,
            parent_registry=self.registry,
            subagents=subagents,
            model_getter=lambda: looping_model,
        )

        # Negative validations
        for kwargs, expected_err in [
            ({"task": "   "}, "non-empty string"),
            ({"task": "ok", "agent": "nonexistent"}, "Unknown subagent 'nonexistent'"),
            ({"task": "ok", "max_steps": 0}, "positive integer"),
            ({"task": "ok", "max_steps": True}, "positive integer"),
            ({"task": "ok", "effort": "extreme"}, "Unsupported reasoning effort"),
            ({"task": "ok", "model": "fast"}, "no model builder configured"),
            ({"tasks": []}, "non-empty list"),
            ({"tasks": [{"agent": "worker"}]}, "non-empty string `task`"),
            ({"tasks": [{"task": "ok", "extra": 1}]}, "unsupported key"),
        ]:
            code, err = tool.execute(**kwargs)
            self.assertEqual(code, 1, f"expected error for {kwargs}, got {err}")
            self.assertIn(expected_err, err)

        # max_steps=1 bounds execution
        code, out = tool.execute(agent="worker", task="Keep looping", max_steps=1)
        self.assertEqual(code, 1)
        self.assertIn("max_steps_exceeded in 1 step(s)", out)

        # Omitted max_steps has no artificial cap (can run > 15 steps)
        twenty_step_model = MockModel(
            [
                (f"Step {i}", [ToolCall(id=f"b{i}", name="bash", arguments={"command": "echo ok"})])
                for i in range(18)
            ]
            + [("Finished after 19 steps", [])]
        )
        unbounded_tool = SubagentTool(
            env=self.env,
            parent_registry=self.registry,
            subagents=subagents,
            model_getter=lambda: twenty_step_model,
        )
        code_unbounded, out_unbounded = unbounded_tool.execute(agent="worker", task="Run 19 steps")
        self.assertEqual(code_unbounded, 0)
        self.assertIn("completed in 19 step(s)", out_unbounded)

    def test_parallel_tasks_batch_and_parallel_agent_loop_calls(self):
        """Invariant: both `tasks=[...]` and multiple `subagent` tool calls in one step run concurrently."""
        barrier = threading.Barrier(2, timeout=3.0)

        class ConcurrentModel:
            def generate(self, messages, tools):
                barrier.wait()
                return f"finished:{messages[1].content}", []

        active_model: list[object] = [ConcurrentModel()]
        tool = SubagentTool(
            env=self.env,
            parent_registry=self.registry,
            subagents=discover_subagents(self.ws, None),
            model_getter=lambda: active_model[0],  # type: ignore[return-value]
        )
        self.registry.register(tool)

        # 1. Parallel batch via tasks=[...]
        code, combined = tool.execute(
            tasks=[
                {"agent": "explore", "task": "task-A"},
                {"agent": "reviewer", "task": "task-B"},
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("[Subagent: explore]", combined)
        self.assertIn("finished:task-A", combined)
        self.assertIn("[Subagent: reviewer]", combined)
        self.assertIn("finished:task-B", combined)
        self.assertLess(combined.index("task-A"), combined.index("task-B"))

        # 2. Multiple subagent calls in a single AgentLoop step
        barrier.reset()

        class ParentAndChildModel:
            def __init__(self):
                self.parent_calls = 0

            def generate(self, messages, tools):
                # Parent system prompt includes Available Subagents; child prompts do not have subagent tool
                tool_names = {t["function"]["name"] for t in tools}
                if "subagent" in tool_names:
                    self.parent_calls += 1
                    if self.parent_calls == 1:
                        return "Fanning out", [
                            ToolCall(
                                id="s1",
                                name="subagent",
                                arguments={"agent": "explore", "task": "branch-1"},
                            ),
                            ToolCall(
                                id="s2",
                                name="subagent",
                                arguments={"agent": "reviewer", "task": "branch-2"},
                            ),
                        ]
                    return "Merged", []
                barrier.wait()
                return f"child:{messages[1].content}", []

        parent_model = ParentAndChildModel()
        active_model[0] = parent_model
        parent_mem = Memory(system_prompt="SYS", user_prompt="Run parallel")
        loop = AgentLoop(model=parent_model, memory=parent_mem, tools=self.registry)
        finished1, trace1 = loop.step(1)
        self.assertFalse(finished1)
        self.assertEqual(len(trace1.tool_results), 2)
        self.assertIn("child:branch-1", trace1.tool_results[0]["output"])
        self.assertIn("child:branch-2", trace1.tool_results[1]["output"])


class TestSubagentInteractiveAndTUI(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "app.py").write_text("x = 1\n", encoding="utf-8")
        self.subagents = {
            "explore": Subagent(
                name="explore",
                description="Explore codebase",
                system_prompt="Explore",
                scope="builtin",
                tools=("read_file", "bash"),
            ),
            "reviewer": Subagent(
                name="reviewer",
                description="Review code",
                system_prompt="Review",
                scope="project",
                tools=("read_file", "bash"),
                model="fast",
                effort="high",
                max_steps=10,
                path=".microharness/agents/reviewer.md",
            ),
        }

    def test_resolve_subagent_command_list_inspect_delegate_and_errors(self):
        """Invariant: /agent lists, inspects by name/index, delegates with task, and rejects invalid selectors."""
        listing = resolve_subagent_command("/agent", self.subagents)
        self.assertEqual(listing.action, "skip")
        self.assertFalse(listing.is_error)
        self.assertIn("explore", listing.message)
        self.assertIn("reviewer", listing.message)

        inspect_by_idx = resolve_subagent_command("/agent 2", self.subagents)
        self.assertEqual(inspect_by_idx.action, "skip")
        self.assertFalse(inspect_by_idx.is_error)
        self.assertIn("Subagent: reviewer [project]", inspect_by_idx.message)
        self.assertIn("Model: fast", inspect_by_idx.message)
        self.assertIn("Effort: high", inspect_by_idx.message)
        self.assertIn("Max Steps: 10", inspect_by_idx.message)

        delegate = resolve_subagent_command("/agent reviewer Check @app.py", self.subagents)
        self.assertEqual(delegate.action, "turn")
        self.assertFalse(delegate.is_error)
        self.assertIn('agent="reviewer"', delegate.prompt or "")
        self.assertIn("Check @app.py", delegate.prompt or "")

        bad_name = resolve_subagent_command("/agent missing do work", self.subagents)
        self.assertEqual(bad_name.action, "skip")
        self.assertTrue(bad_name.is_error)
        self.assertIn("not found", bad_name.message)

        bad_idx = resolve_subagent_command("/agent 99", self.subagents)
        self.assertEqual(bad_idx.action, "skip")
        self.assertTrue(bad_idx.is_error)
        self.assertIn("Invalid subagent number 99", bad_idx.message)

    def test_router_agent_command_and_help_integration(self):
        """Invariant: CommandRouter supports /agent, /agents, /subagent, expands @file tags, and lists subagents in /help."""
        memory = Memory(system_prompt="sys", user_prompt="")
        ui = RecordingUI()
        harness = MagicMock()
        ctx = CommandContext(
            harness=harness,
            memory=memory,
            ui=ui,
            workspace=self.tmp,
            subagents=self.subagents,
            turns=TurnState(),
        )
        router = CommandRouter(ctx)

        self.assertEqual(router.dispatch("/help"), SKIP)
        self.assertIn("/agent [name] [task]", ui.all_text)
        self.assertIn("Available subagents", ui.all_text)
        self.assertIn("reviewer", ui.all_text)

        self.assertEqual(router.dispatch("/agents"), SKIP)
        self.assertEqual(len(memory.messages), 1)

        self.assertEqual(router.dispatch("/subagent reviewer Review @app.py"), TURN)
        self.assertEqual(len(memory.messages), 2)
        self.assertIn('agent="reviewer"', memory.messages[-1].content)
        self.assertIn("[Attached File: @app.py]", memory.messages[-1].content)

        self.assertEqual(router.dispatch("/agent unknown task"), SKIP)
        self.assertIn("not found", ui.errors[0])
        self.assertEqual(len(memory.messages), 2)

        self.assertEqual(router.dispatch("/exit"), EXIT)

    def test_tui_subcommand_autocomplete_and_drawing(self):
        """Invariant: /agent opens 'Available Subagents' menu in TUIState and CursesTUI, supports Tab completion."""
        state = TUIState()
        state.set_subagents(self.subagents)

        state.input_buffer = "/ag"
        matches = state.get_matching_commands()
        self.assertEqual(matches[0][0], "/agent")

        tui = CursesTUI(state)
        state.selected_command_idx = 0
        tui.handle_key("\t", lambda _t: None)
        self.assertEqual(state.input_buffer, "/agent ")

        agent_matches = state.get_matching_commands()
        self.assertEqual(
            [cmd for cmd, _ in agent_matches],
            ["/agent explore", "/agent reviewer"],
        )
        self.assertIn("model:fast", agent_matches[1][1])
        self.assertIn("effort:high", agent_matches[1][1])

        state.input_buffer = "/agents rev"
        filtered = state.get_matching_commands()
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0][0], "/agents reviewer")

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 90)
        state.input_buffer = "/agent "
        tui.draw(mock_stdscr)
        drawn = " ".join(
            c.args[2]
            for c in mock_stdscr.addstr.call_args_list
            if len(c.args) >= 3 and isinstance(c.args[2], str)
        )
        self.assertIn("Available Subagents", drawn)
        self.assertIn("/agent reviewer", drawn)

    def test_tui_live_subagent_step_progress_and_esc_cancellation(self):
        """Invariant: run_tui discovers subagents, streams child tool actions live, and propagates Esc cancellation cleanly."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws = root / "ws"
            runs = root / "runs"
            ws.mkdir()
            runs.mkdir()
            _write(ws / "target.txt", "hello")
            _write(
                ws / ".microharness" / "agents" / "auditor.md",
                "---\ndescription: Custom auditor\ntools: read_file\n---\nAudit files.",
            )

            class TuiSubagentModel:
                def __init__(self) -> None:
                    self.probe = None
                    self.step = 0

                def set_cancel_probe(self, probe) -> None:
                    self.probe = probe

                def generate(self, messages, tools):
                    tool_names = {t["function"]["name"] for t in tools}
                    if "subagent" in tool_names:
                        self.step += 1
                        if self.step == 1:
                            return "Delegating", [
                                ToolCall(
                                    id="sub1",
                                    name="subagent",
                                    arguments={"agent": "auditor", "task": "Read target.txt"},
                                )
                            ]
                        return "Audit complete", []
                    # Child turn
                    if len(messages) == 2:
                        return "Reading file", [
                            ToolCall(id="r1", name="read_file", arguments={"path": "target.txt"})
                        ]
                    return "target.txt says hello", []

            harness = Harness(
                model=TuiSubagentModel(),
                workspace_dir=ws,
                runs_dir=runs,
                sandbox=False,
                verbose=False,
            )
            captured_states: list[TUIState] = []

            def mock_tui_run(tui_instance, on_user_submit, initial_instruction=None):
                captured_states.append(tui_instance.state)
                on_user_submit("/agent")
                on_user_submit("/agent auditor check target.txt")

            with patch("core.tui.CursesTUI.run", side_effect=mock_tui_run, autospec=True):
                harness.run_tui()

            state = captured_states[0]
            self.assertIn("auditor", state.subagents)
            texts = [line.text for line in state.lines]
            self.assertTrue(any("Custom auditor" in t for t in texts))
            self.assertTrue(any("↳ [auditor] ● Read target.txt" in t for t in texts))
            self.assertTrue(any("Audit complete" in t for t in texts))

            # Esc cancellation mid-subagent must abort cleanly and drop dangling tool_call
            class CancelChildModel:
                def __init__(self) -> None:
                    self.probe = None

                def set_cancel_probe(self, probe) -> None:
                    self.probe = probe

                def generate(self, messages, tools):
                    tool_names = {t["function"]["name"] for t in tools}
                    if "subagent" in tool_names:
                        return "Calling subagent", [
                            ToolCall(
                                id="sub_cancel",
                                name="subagent",
                                arguments={"agent": "explore", "task": "Search"},
                            )
                        ]
                    raise ModelCancelledError("Cancelled inside subagent")

            harness_cancel = Harness(
                model=CancelChildModel(),
                workspace_dir=ws,
                runs_dir=runs,
                sandbox=False,
                verbose=False,
            )
            cancel_states: list[TUIState] = []

            def mock_cancel_run(tui_instance, on_user_submit, initial_instruction=None):
                cancel_states.append(tui_instance.state)
                on_user_submit("Run explore")

            with patch("core.tui.CursesTUI.run", side_effect=mock_cancel_run, autospec=True):
                harness_cancel.run_tui()

            cancel_texts = [line.text for line in cancel_states[0].lines]
            self.assertTrue(any("cancelled by user (Esc)" in t for t in cancel_texts))
            self.assertTrue(any("Dropped 1 incomplete message(s)" in t for t in cancel_texts))

    def test_tracer_action_formatter_and_system_prompt_index(self):
        """Invariant: tracer formats single/parallel subagent calls and build_system_prompt includes subagents index."""
        self.assertEqual(
            _format_action_summary(
                "subagent",
                {"agent": "reviewer", "task": "Check auth.py\nExtra line"},
                color=False,
            ),
            "● Subagent [reviewer]: Check auth.py",
        )
        self.assertEqual(
            _format_action_summary(
                "subagent",
                {"tasks": [{"agent": "explore", "task": "a"}, {"agent": "reviewer", "task": "b"}]},
                color=False,
            ),
            "● Subagent parallel (2 tasks: explore, reviewer)",
        )
        self.assertEqual(
            format_subagent_desc(self.subagents["reviewer"]),
            "[project] Review code (model:fast, effort:high, tools:read_file,bash, max_steps:10)",
        )
        idx_text = render_subagents_index(self.subagents)
        self.assertIn("--- Available Subagents ---", idx_text)
        self.assertIn("[project] reviewer", idx_text)

    def test_harness_run_with_subagent_end_to_end_and_usage_rollup(self):
        """Invariant: Harness.run executes subagents end-to-end, injects subagent index into system prompt, and rolls up token usage."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws = root / "ws"
            runs = root / "runs"
            global_dir = root / "global"
            ws.mkdir()
            runs.mkdir()
            global_dir.mkdir()
            _write(ws / "data.txt", "magic=42\n")
            _write(
                global_dir / "agents" / "reader.md",
                "---\ndescription: Global reader\ntools: read_file\n---\nRead the requested file.",
            )

            captured_parent_prompts: list[str] = []

            class E2EModel:
                def __init__(self) -> None:
                    self.last_usage: dict[str, int] = {}

                def get_last_step_usage(self) -> dict[str, int]:
                    return dict(self.last_usage)

                def generate(self, messages, tools):
                    tool_names = {t["function"]["name"] for t in tools}
                    if "subagent" in tool_names:
                        captured_parent_prompts.append(messages[0].content)
                        self.last_usage = {"prompt_tokens": 100, "completion_tokens": 20}
                        if len(messages) == 2:
                            return "Delegating read", [
                                ToolCall(
                                    id="c1",
                                    name="subagent",
                                    arguments={"agent": "reader", "task": "Read data.txt"},
                                )
                            ]
                        return f"Final answer based on: {messages[-1].content}", []
                    self.last_usage = {"prompt_tokens": 50, "completion_tokens": 10}
                    if len(messages) == 2:
                        return "Reading", [
                            ToolCall(id="r1", name="read_file", arguments={"path": "data.txt"})
                        ]
                    return "data.txt contains magic=42", []

            harness = Harness(
                model=E2EModel(),
                workspace_dir=ws,
                global_dir=global_dir,
                runs_dir=runs,
                sandbox=False,
                verbose=False,
            )
            result = harness.run(Task(id="e2e_sub", instruction="Find magic value"))
            self.assertTrue(result.passed)
            self.assertEqual(result.finish_reason, "completed")
            self.assertIn("--- Available Subagents ---", captured_parent_prompts[0])
            self.assertIn("[global] reader", captured_parent_prompts[0])
            # Parent: 2 steps * (100 in, 20 out) + Child: 2 steps * (50 in, 10 out) = (300 in, 60 out)
            self.assertEqual(result.prompt_tokens, 300)
            self.assertEqual(result.completion_tokens, 60)
            self.assertIsNone(DEFAULT_SUBAGENT_MAX_STEPS)


if __name__ == "__main__":
    unittest.main()
