import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from core.config import ConfigLayers, HarnessConfig
from core.harness import Harness
from core.interactive import SKIP, CommandContext, CommandRouter, NullUI, TurnState
from core.memory import Memory
from core.model import MockModel
from core.presets import (
    discover_preset_commands,
    discover_preset_skills,
    discover_preset_subagents,
    discover_preset_tools,
    discover_presets,
    load_preset,
    resolve_preset,
)
from core.prompts import DEFAULT_SYSTEM_PROMPT
from core.types import Task, ToolCall


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestPresets(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.global_dir = self.root / "global"
        self.workspace = self.root / "workspace"
        self.global_dir.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_load_preset_parses_agent_md_and_config_json(self):
        preset_dir = self.global_dir / "presets" / "paper-researcher"
        _write(
            preset_dir / "agent.md",
            """---
name: Paper Researcher
description: Deep literature analysis and synthesis agent
model: openai/gpt-4o
effort: high
permission_mode: read-only
max_steps: 25
isolated: true
tools:
  - read
  - arxiv_search
  - subagent
---
You are a rigorous academic paper researcher.
Cite every claim with exact paper sections.
""",
        )
        _write(
            preset_dir / "config.json",
            """{
  // JSONC comments supported
  "model": "anthropic/claude-sonnet",
  "bash_timeout": 120,
  "max_context_tokens": 131072,
  "hooks": {
    "post_tool": [{"name": "audit", "command": "echo ok"}]
  }
}
""",
        )

        preset = load_preset(preset_dir, scope="global")
        self.assertEqual(preset.id, "paper-researcher")
        self.assertEqual(preset.name, "Paper Researcher")
        self.assertEqual(preset.description, "Deep literature analysis and synthesis agent")
        self.assertEqual(preset.scope, "global")
        self.assertEqual(
            preset.system_prompt,
            "You are a rigorous academic paper researcher.\nCite every claim with exact paper sections.",
        )
        # agent.md overrides config.json on shared keys
        self.assertEqual(preset.config["model"], "openai/gpt-4o")
        self.assertEqual(preset.config["reasoning_effort"], "high")
        self.assertEqual(preset.config["permission_mode"], "read-only")
        self.assertEqual(preset.config["max_steps"], 25)
        self.assertEqual(preset.config["bash_timeout"], 120.0)
        self.assertEqual(preset.config["max_context_tokens"], 131072)
        self.assertIn("post_tool", preset.config["hooks"])
        # Alias normalized ('read' -> 'read_file') and 'subagent' preserved
        self.assertEqual(preset.tools, ("read_file", "arxiv_search", "subagent"))
        self.assertTrue(preset.isolated)
        self.assertFalse(preset.include_builtin_subagents)
        self.assertFalse(preset.inherit_global)
        self.assertFalse(preset.inherit_project)
        self.assertFalse(preset.include_workspace_agents_md)

    def test_load_preset_negative_cases(self):
        with self.assertRaises(FileNotFoundError):
            load_preset(self.root / "missing-preset")

        empty_dir = self.root / "empty-dir"
        empty_dir.mkdir()
        with self.assertRaises(ValueError):
            load_preset(empty_dir)

        bad_tools_dir = self.root / "bad-tools"
        _write(bad_tools_dir / "config.json", json.dumps({"tools": 123}))
        with self.assertRaises(ValueError):
            load_preset(bad_tools_dir)

        bad_steps_dir = self.root / "bad-steps"
        _write(bad_steps_dir / "agent.md", "---\nmax_steps: -5\n---\nPrompt\n")
        with self.assertRaises(ValueError):
            load_preset(bad_steps_dir)

    def test_load_preset_yaml_disabled_hooks_null_limits_and_empty_tools(self):
        preset_dir = self.global_dir / "presets" / "no-tools"
        _write(
            preset_dir / "agent.md",
            """---
name: Pure Chat
subagent_max_steps: null
tools: []
disabled_hooks:
  - doom_loop_detection
  - python_syntax_diagnostics
---
Just answer from context.
""",
        )
        preset = load_preset(preset_dir, scope="global")
        self.assertEqual(preset.tools, ())
        self.assertIsNone(preset.config["subagent_max_steps"])
        self.assertEqual(
            preset.config["disabled_hooks"],
            ["doom_loop_detection", "python_syntax_diagnostics"],
        )

        harness = Harness(
            model=MockModel(script=[("Hello!", [])]),
            workspace_dir=self.workspace,
            global_dir=self.global_dir,
            runs_dir=self.root / "runs",
            preset=preset,
            verbose=False,
        )
        self.assertIsNone(harness.subagent_max_steps)
        self.assertIsNone(harness.hooks.get("doom_loop_detection"))
        result = harness.run(Task(id="pure_chat", instruction="Hi", max_steps=2))
        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        system_prompt = trace["steps"][0]["prompt_messages"][0]["content"]
        self.assertNotIn("You have access to tools", system_prompt)

    def test_discover_and_resolve_presets_with_shadowing(self):
        _write(
            self.global_dir / "presets" / "dev-1" / "agent.md",
            "---\nname: Developer Preset 1\ndescription: Global dev preset\n---\nGlobal dev prompt.\n",
        )
        _write(
            self.global_dir / "presets" / "paper-researcher" / "agent.md",
            "---\nname: Paper Researcher\ndescription: Reads papers\n---\nResearch prompt.\n",
        )
        # Project shadows dev-1
        _write(
            self.workspace / ".microharness" / "presets" / "dev-1" / "agent.md",
            "---\nname: Developer Preset 1\ndescription: Project dev preset\n---\nProject dev prompt.\n",
        )
        # Unrelated dir without agent.md / config.json is ignored during discovery
        (self.workspace / ".microharness" / "presets" / "scratch-notes").mkdir(parents=True)

        presets = discover_presets(workspace=self.workspace, global_dir=self.global_dir)
        self.assertEqual(sorted(presets.keys()), ["dev-1", "paper-researcher"])
        self.assertEqual(presets["dev-1"].scope, "project")
        self.assertEqual(presets["dev-1"].description, "Project dev preset")
        self.assertEqual(presets["paper-researcher"].scope, "global")

        # Resolve by slug, display name, and explicit path
        self.assertEqual(
            resolve_preset(
                "dev-1", workspace=self.workspace, global_dir=self.global_dir
            ).description,
            "Project dev preset",
        )
        self.assertEqual(
            resolve_preset(
                "Paper Researcher", workspace=self.workspace, global_dir=self.global_dir
            ).id,
            "paper-researcher",
        )
        by_path = resolve_preset(
            self.global_dir / "presets" / "paper-researcher",
            workspace=self.workspace,
            global_dir=self.global_dir,
        )
        self.assertEqual(by_path.scope, "global")

        with self.assertRaises(ValueError):
            resolve_preset("nonexistent", workspace=self.workspace, global_dir=self.global_dir)
        with self.assertRaises(ValueError):
            resolve_preset("   ", workspace=self.workspace, global_dir=self.global_dir)

    def test_config_layers_precedence_and_trust_boundary(self):
        _write(
            self.global_dir / "config.json",
            json.dumps({"model": "global-model", "bash_timeout": 100, "permission_mode": "auto"}),
        )
        _write(
            self.workspace / ".microharness" / "config.json",
            json.dumps({"permission_mode": "ask", "trust_project_tools": True}),
        )
        project_preset_dir = self.workspace / ".microharness" / "presets" / "untrusted"
        _write(
            project_preset_dir / "agent.md",
            "---\nmodel: preset-model\n---\nPrompt\n",
        )
        _write(
            project_preset_dir / "config.json",
            json.dumps(
                {
                    "bash_timeout": 45,
                    "trust_project_tools": True,
                    "hooks": {"post_tool": [{"name": "evil_hook", "command": "echo pwned"}]},
                }
            ),
        )
        _write(
            project_preset_dir / "tools" / "custom.py",
            """
class CustomTool:
    name = "custom_tool"
    description = "Custom preset tool"
    parameters = {"type": "object", "properties": {}, "required": []}
    def execute(self):
        return 0, "ran"
TOOLS = [CustomTool()]
""",
        )

        preset = resolve_preset("untrusted", workspace=self.workspace, global_dir=self.global_dir)
        layers = ConfigLayers.load(
            self.global_dir,
            self.workspace,
            preset_config=preset.config,
            preset_scope=preset.scope,
        )
        resolved = HarnessConfig().resolve(layers)
        # global < preset < project
        self.assertEqual(layers.merged["model"], "preset-model")
        self.assertEqual(resolved.bash_timeout, 45.0)
        self.assertEqual(resolved.permission_mode, "ask")

        # Untrusted project preset tools & command hooks are blocked without --trust-project-tools
        stderr_buf = io.StringIO()
        with contextlib.redirect_stderr(stderr_buf):
            blocked_tools = discover_preset_tools(preset, trust_project_tools=False)
            harness_untrusted = Harness(
                model=MockModel(script=[]),
                workspace_dir=self.workspace,
                global_dir=self.global_dir,
                preset=preset,
                verbose=False,
            )
        self.assertEqual(blocked_tools, [])
        self.assertFalse(harness_untrusted.trust_project_tools)
        self.assertIsNone(harness_untrusted.hooks.get("evil_hook"))
        self.assertIn("Skipping untrusted", stderr_buf.getvalue())

        # Trusted when explicit trust_project_tools=True
        trusted_tools = discover_preset_tools(preset, trust_project_tools=True)
        self.assertEqual([t.name for t in trusted_tools], ["custom_tool"])
        harness_trusted = Harness(
            model=MockModel(script=[]),
            workspace_dir=self.workspace,
            global_dir=self.global_dir,
            preset=preset,
            trust_project_tools=True,
            verbose=False,
        )
        self.assertIsNotNone(harness_trusted.hooks.get("evil_hook"))

    def test_harness_isolated_preset_bundle_execution(self):
        # Global and project distractions that isolated=True must exclude
        _write(self.global_dir / "AGENTS.md", "GLOBAL_RULE_DO_NOT_LEAK")
        _write(self.workspace / "AGENTS.md", "WORKSPACE_RULE_DO_NOT_LEAK")
        _write(
            self.global_dir / "skills" / "global-skill" / "SKILL.md",
            "---\ndescription: Global skill\n---\nGlobal skill body\n",
        )
        _write(
            self.workspace / ".microharness" / "skills" / "proj-skill" / "SKILL.md",
            "---\ndescription: Project skill\n---\nProject skill body\n",
        )
        _write(
            self.global_dir / "agents" / "global-helper.md",
            "---\ndescription: Global helper\n---\nGlobal helper prompt\n",
        )

        # Build isolated Paper Researcher preset in global_dir (so its tools are trusted)
        preset_dir = self.global_dir / "presets" / "paper-researcher"
        _write(
            preset_dir / "agent.md",
            """---
name: Paper Researcher
description: Literature review specialist
isolated: true
tools: [read_file, paper_lookup, subagent]
---
You are Paper Researcher. Always use paper_lookup before answering.
""",
        )
        _write(
            preset_dir / "skills" / "lit-review" / "SKILL.md",
            "---\ndescription: How to synthesize papers\n---\nRead abstract and methodology first.\n",
        )
        _write(
            preset_dir / "subagents" / "summarizer.md",
            "---\ndescription: Summarizes paper sections\ntools: [read_file]\n---\nSummarize concisely.\n",
        )
        _write(
            preset_dir / "commands" / "cite.md",
            "Format citations cleanly for $ARGUMENTS",
        )
        _write(
            preset_dir / "tools" / "paper_lookup.py",
            """
class PaperLookupTool:
    name = "paper_lookup"
    description = "Look up paper metadata"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    def execute(self, query: str):
        return 0, f"Found paper for: {query}"

TOOLS = [PaperLookupTool()]
""",
        )

        preset = resolve_preset(
            "paper-researcher", workspace=self.workspace, global_dir=self.global_dir
        )
        skill_path = discover_preset_skills(preset)["lit-review"].path
        self.assertEqual(sorted(discover_preset_subagents(preset).keys()), ["summarizer"])
        self.assertEqual(sorted(discover_preset_commands(preset).keys()), ["cite"])

        model = MockModel(
            script=[
                (
                    "Checking the lit-review skill and looking up the paper.",
                    [
                        ToolCall(id="c1", name="read_file", arguments={"path": skill_path}),
                        ToolCall(
                            id="c2",
                            name="paper_lookup",
                            arguments={"query": "attention is all you need"},
                        ),
                    ],
                ),
                (
                    "Done researching.",
                    [ToolCall(id="c3", name="finish", arguments={"summary": "Synthesized paper."})],
                ),
            ]
        )

        harness = Harness(
            model=model,
            workspace_dir=self.workspace,
            global_dir=self.global_dir,
            runs_dir=self.root / "runs",
            sandbox=True,
            preset="paper-researcher",
            require_finish_tool=True,
            verbose=False,
        )
        result = harness.run(
            Task(id="research_1", instruction="/cite Transformer paper", max_steps=5)
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.finish_reason, "completed")
        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        system_prompt = trace["steps"][0]["prompt_messages"][0]["content"]
        self.assertIn(
            "You are Paper Researcher. Always use paper_lookup before answering.", system_prompt
        )
        self.assertNotIn(DEFAULT_SYSTEM_PROMPT, system_prompt)
        self.assertNotIn("GLOBAL_RULE_DO_NOT_LEAK", system_prompt)
        self.assertNotIn("WORKSPACE_RULE_DO_NOT_LEAK", system_prompt)
        self.assertIn("lit-review", system_prompt)
        self.assertNotIn("global-skill", system_prompt)
        self.assertNotIn("proj-skill", system_prompt)
        self.assertIn("summarizer", system_prompt)
        self.assertNotIn("global-helper", system_prompt)
        self.assertNotIn("reviewer:", system_prompt)
        # Filtered tools: only read_file, paper_lookup, subagent, finish
        self.assertIn("paper_lookup", system_prompt)
        self.assertNotIn("bash(", system_prompt)
        self.assertNotIn("edit_file(", system_prompt)
        # Preset command /cite expanded into user instruction
        user_msg = trace["steps"][0]["prompt_messages"][1]["content"]
        self.assertEqual(user_msg, "Format citations cleanly for Transformer paper")
        # Both preset skill read_file and preset custom tool succeeded
        tool_outputs = [r["output"] for r in trace["steps"][0]["tool_results"]]
        self.assertTrue(any("Read abstract and methodology first." in o for o in tool_outputs))
        self.assertIn("Found paper for: attention is all you need", tool_outputs)

    def test_harness_layered_preset_merges_and_filters_subagent_tool(self):
        _write(self.global_dir / "AGENTS.md", "GLOBAL_STEERING")
        _write(
            self.global_dir / "skills" / "g-skill" / "SKILL.md",
            "---\ndescription: Global\n---\nG\n",
        )
        _write(
            self.workspace / ".microharness" / "skills" / "p-skill" / "SKILL.md",
            "---\ndescription: Project\n---\nP\n",
        )
        preset_dir = self.global_dir / "presets" / "dev-1"
        _write(
            preset_dir / "agent.md",
            """---
name: Developer Preset 1
tools: [read_file, grep, glob]
---
You are Developer Preset 1.
""",
        )
        _write(
            preset_dir / "skills" / "preset-skill" / "SKILL.md",
            "---\ndescription: Preset skill\n---\nPreset\n",
        )

        harness = Harness(
            model=MockModel(script=[("All done.", [])]),
            workspace_dir=self.workspace,
            global_dir=self.global_dir,
            runs_dir=self.root / "runs",
            preset="dev-1",
            verbose=False,
        )
        result = harness.run(Task(id="dev_task", instruction="Inspect codebase", max_steps=3))
        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        system_prompt = trace["steps"][0]["prompt_messages"][0]["content"]
        self.assertIn("You are Developer Preset 1.", system_prompt)
        self.assertIn("GLOBAL_STEERING", system_prompt)
        self.assertIn("g-skill", system_prompt)
        self.assertIn("preset-skill", system_prompt)
        self.assertIn("p-skill", system_prompt)
        # 'subagent' was excluded by preset tools allowlist, so subagents section is omitted
        self.assertNotIn("Available Subagents", system_prompt)
        self.assertNotIn("bash(", system_prompt)

    def test_cli_list_presets_and_context_resolution(self):
        from main import _build_arg_parser, _resolve_cli_context, main

        _write(
            self.global_dir / "presets" / "dev-2" / "agent.md",
            """---
name: Developer Preset 2
description: Fast coding preset
model: openai/gpt-4o-mini
reasoning_effort: low
permission_mode: ask
cache_ttl: 1h
---
Fast dev prompt.
""",
        )

        stdout_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf):
            main(
                [
                    "--list-presets",
                    "--global-dir",
                    str(self.global_dir),
                    "-d",
                    str(self.workspace),
                ]
            )
        listing = stdout_buf.getvalue()
        self.assertIn("dev-2 (Developer Preset 2)", listing)
        self.assertIn("Fast coding preset", listing)

        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--agent",
                "dev-2",
                "--global-dir",
                str(self.global_dir),
                "-d",
                str(self.workspace),
            ]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            ctx = _resolve_cli_context(args)
        self.assertIsNotNone(ctx.preset)
        assert ctx.preset is not None
        self.assertEqual(ctx.preset.id, "dev-2")
        self.assertEqual(ctx.reasoning_effort, "low")
        self.assertEqual(ctx.permission_mode, "ask")
        self.assertEqual(ctx.cache_ttl, "1h")
        self.assertEqual(ctx.config["model"], "openai/gpt-4o-mini")

        # Unknown preset exits non-zero
        bad_args = parser.parse_args(
            [
                "--preset",
                "does-not-exist",
                "--global-dir",
                str(self.global_dir),
                "-d",
                str(self.workspace),
            ]
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            _resolve_cli_context(bad_args)
        self.assertEqual(cm.exception.code, 1)

    def test_interactive_preset_and_status_commands(self):
        _write(
            self.global_dir / "presets" / "dev-1" / "agent.md",
            "---\nname: Developer Preset 1\ndescription: Standard dev\n---\nDev prompt.\n",
        )
        harness = Harness(
            model=MockModel(script=[]),
            workspace_dir=self.workspace,
            global_dir=self.global_dir,
            preset="dev-1",
            verbose=False,
        )

        class CaptureUI(NullUI):
            def __init__(self):
                self.lines: list[str] = []

            def info(self, text: str) -> None:
                self.lines.append(text)

            def status(self, text: str) -> None:
                self.lines.append(text)

        ui = CaptureUI()
        router = CommandRouter(
            CommandContext(
                harness=harness,
                memory=Memory(system_prompt="Dev prompt.", user_prompt=""),
                ui=ui,
                workspace=self.workspace,
                turns=TurnState(),
            )
        )
        self.assertEqual(router.dispatch("/preset"), SKIP)
        self.assertEqual(router.dispatch("/status"), SKIP)
        joined = "\n".join(ui.lines)
        self.assertIn("Active Preset: Developer Preset 1 (dev-1) [global]", joined)
        self.assertIn("dev-1 (Developer Preset 1)", joined)
        self.assertIn("🤖 Agent Preset: Developer Preset 1 (dev-1) [global]", joined)


if __name__ == "__main__":
    unittest.main()
