import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from core.environment import Environment
from core.evaluator import FileContentEvaluator
from core.harness import (
    Harness,
    _discover_commands,
    _expand_command,
    load_global_config,
    load_project_config,
)
from core.loop import AgentLoop
from core.memory import Memory
from core.model import MockModel
from core.tools import EditFileTool, ToolRegistry, WriteFileTool
from core.types import Message, Task, ToolCall


class TestHarnessIntegration(unittest.TestCase):
    def setUp(self):
        self.runs_dir = tempfile.mkdtemp()

    def test_full_successful_run_and_trace_integrity(self):
        script = [
            (
                "Writing file",
                [
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments={"path": "answer.txt", "content": "42"},
                    )
                ],
            ),
            ("Done", []),
        ]
        model = MockModel(script)
        evaluator = FileContentEvaluator(file_path="answer.txt", must_contain=["42"])

        harness = Harness(
            model=model,
            evaluator=evaluator,
            runs_dir=self.runs_dir,
            verbose=False,
        )

        task = Task(
            id="write_answer",
            instruction="Write 42 to answer.txt",
            max_steps=5,
        )

        result = harness.run(task)

        # Strict outcome invariants
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.finish_reason, "completed")
        self.assertEqual(result.total_steps, 2)

        # Deep trace integrity check
        json_trace = Path(result.trace_path)
        self.assertTrue(json_trace.exists())
        data = json.loads(json_trace.read_text(encoding="utf-8"))

        self.assertEqual(data["task"]["id"], "write_answer")
        self.assertEqual(data["finish_reason"], "completed")
        self.assertEqual(data["eval_result"]["passed"], True)
        self.assertEqual(len(data["steps"]), 2)

        # Step 1 must have the tool call and observation
        step1 = data["steps"][0]
        self.assertEqual(step1["step_index"], 1)
        self.assertEqual(step1["model_response"]["tool_calls"][0]["name"], "write_file")
        self.assertEqual(step1["tool_results"][0]["call_id"], "c1")
        self.assertEqual(step1["tool_results"][0]["exit_code"], 0)

        # Markdown trace check
        md_trace = json_trace.parent / "trace.md"
        self.assertTrue(md_trace.exists())
        md_content = md_trace.read_text(encoding="utf-8")
        self.assertIn("# Run Trace: Task write_answer", md_content)
        self.assertIn("write_file", md_content)

    def test_max_steps_strictly_halts_execution(self):
        # Model that would loop 10 times
        infinite_script = [
            (
                f"Looping step {i}",
                [ToolCall(id=f"c{i}", name="bash", arguments={"command": "echo loop"})],
            )
            for i in range(10)
        ]
        model = MockModel(infinite_script)

        # Hard guardrail at 3 steps
        harness = Harness(model=model, runs_dir=self.runs_dir, verbose=False)
        task = Task(id="infinite_task", instruction="Never stops", max_steps=3)

        result = harness.run(task)

        # Strict guardrail assertions
        self.assertEqual(result.finish_reason, "max_steps_exceeded")
        self.assertEqual(result.total_steps, 3)
        self.assertFalse(result.passed)

    def test_timeout_guardrail_strictly_halts_execution(self):
        # A model that causes time to elapse beyond timeout_seconds
        class SlowModel:
            def generate(self, messages, tools):
                time.sleep(0.15)
                return "Still working", [
                    ToolCall(id="s1", name="bash", arguments={"command": "echo tick"})
                ]

        harness = Harness(model=SlowModel(), runs_dir=self.runs_dir, verbose=False)
        # Timeout is 0.1s; step 1 takes 0.15s, so step 2 must trigger timeout
        task = Task(id="slow_task", instruction="Slow execution", timeout_seconds=0.1, max_steps=10)

        result = harness.run(task)

        # Strict: must be timeout, NOT completed or max_steps_exceeded
        self.assertEqual(result.finish_reason, "timeout")
        self.assertFalse(result.passed)

    def test_unhandled_model_crash_is_trapped_safely(self):
        class ExplodingModel:
            def generate(self, messages, tools):
                raise ConnectionResetError("Remote model socket terminated abruptly")

        harness = Harness(
            model=ExplodingModel(),
            runs_dir=self.runs_dir,
            verbose=False,
            max_step_retries=1,
            step_retry_delay=0.001,
        )
        task = Task(id="crash_task", instruction="Will crash", max_steps=5)

        # Invariant: Harness does not raise unhandled exception; records error cleanly
        result = harness.run(task)
        self.assertEqual(result.finish_reason, "error")
        self.assertFalse(result.passed)
        self.assertTrue(Path(result.trace_path).exists())

    def test_transient_step_error_retries_and_succeeds(self):
        class TransientFailModel:
            def __init__(self):
                self.calls = 0

            def generate(self, messages, tools):
                self.calls += 1
                if self.calls == 1:
                    raise ConnectionResetError("Transient network glitch")
                return "Successfully recovered after retry.", []

        model = TransientFailModel()
        harness = Harness(
            model=model,
            runs_dir=self.runs_dir,
            verbose=False,
            max_step_retries=2,
            step_retry_delay=0.001,
        )
        task = Task(id="retry_task", instruction="Fix network", max_steps=5)

        result = harness.run(task)
        self.assertEqual(result.finish_reason, "completed")
        self.assertEqual(model.calls, 2)

    def test_interactive_multi_turn_session(self):
        # Scripted model that handles 2 interactive turns:
        # Turn 1: writes step1.txt, then yields text
        # Turn 2: writes step2.txt, then yields text
        script = [
            (
                "Writing step 1",
                [
                    ToolCall(
                        id="t1",
                        name="write_file",
                        arguments={"path": "step1.txt", "content": "hello 1"},
                    )
                ],
            ),
            ("Finished turn 1.", []),
            (
                "Writing step 2",
                [
                    ToolCall(
                        id="t2",
                        name="write_file",
                        arguments={"path": "step2.txt", "content": "hello 2"},
                    )
                ],
            ),
            ("Finished turn 2.", []),
        ]
        model = MockModel(script)
        harness = Harness(model=model, runs_dir=self.runs_dir, verbose=False)

        # Mock user typing slash commands, follow-up, and then 'exit'
        user_inputs = iter(["/status", "/help", "/compact", "Now do step 2", "/status", "exit"])
        with patch("builtins.input", lambda *args: next(user_inputs)):
            harness.run_interactive(initial_instruction="Start step 1")

        # Invariant: Files from both turns exist in the same workspace!
        latest_run = max(Path(self.runs_dir).glob("interactive_session_*"))
        workspace = latest_run / "workspace"
        self.assertTrue((workspace / "step1.txt").exists())
        self.assertTrue((workspace / "step2.txt").exists())
        self.assertEqual((workspace / "step1.txt").read_text(), "hello 1")
        self.assertEqual((workspace / "step2.txt").read_text(), "hello 2")

        # Invariant: Trace captured all turns across the session
        trace_json = latest_run / "trace.json"
        self.assertTrue(trace_json.exists())
        trace_data = json.loads(trace_json.read_text(encoding="utf-8"))
        self.assertEqual(trace_data["finish_reason"], "session_ended")
        self.assertEqual(len(trace_data["steps"]), 4)

    def test_existing_project_directory_mode(self):
        # Simulate a developer's existing project folder
        project_dir = Path(tempfile.mkdtemp(prefix="my_existing_project_"))
        (project_dir / "app.py").write_text("def run():\n    return 'v1'\n")

        # Harness configured with workspace_dir pointing to the existing project
        script = [
            (
                "Inspecting app.py",
                [ToolCall(id="t1", name="read_file", arguments={"path": "app.py"})],
            ),
            (
                "Upgrading app.py",
                [
                    ToolCall(
                        id="t2",
                        name="write_file",
                        arguments={"path": "app.py", "content": "def run():\n    return 'v2'\n"},
                    )
                ],
            ),
            ("Done upgrading.", []),
        ]
        model = MockModel(script)
        harness = Harness(
            model=model, workspace_dir=project_dir, runs_dir=self.runs_dir, verbose=False
        )

        task = Task(id="upgrade_task", instruction="Upgrade app.py from v1 to v2")
        result = harness.run(task)

        self.assertTrue(result.passed)
        # Invariant: The actual project file was upgraded in-place!
        self.assertEqual((project_dir / "app.py").read_text(), "def run():\n    return 'v2'\n")

    def test_agents_md_and_skills_progressive_disclosure(self):
        project_dir = Path(tempfile.mkdtemp(prefix="project_with_skills_"))
        # 1. Project instructions
        (project_dir / "AGENTS.md").write_text(
            "Always format with ruff and follow standard conventions.\n"
        )

        # 2. Skill folder with SKILL.md
        skill_dir = project_dir / "skills" / "deploy-tool"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "# Deploy Workflow\nDetailed multi-step deployment commands and token rules.\n"
        )

        # Model captures the messages sent to it
        captured_prompts: list[list[Message]] = []

        class RecordingModel:
            def generate(self, messages, tools):
                captured_prompts.append(list(messages))
                # Agent uses read_file to inspect the skill on-demand
                if len(captured_prompts) == 1:
                    return "Need skill info", [
                        ToolCall(
                            id="s1",
                            name="read_file",
                            arguments={"path": "skills/deploy-tool/SKILL.md"},
                        )
                    ]
                return "Finished task using skill", []

        harness = Harness(
            model=RecordingModel(), workspace_dir=project_dir, runs_dir=self.runs_dir, verbose=False
        )
        task = Task(id="skill_task", instruction="Deploy the app")
        result = harness.run(task)

        self.assertTrue(result.passed)

        # Inspect the system prompt in the first step
        system_prompt = captured_prompts[0][0].content

        # Invariant 1: AGENTS.md instructions are injected
        self.assertIn("Always format with ruff", system_prompt)

        # Invariant 2: Progressive disclosure indexed the skill without dumping the entire file body
        self.assertIn("deploy-tool: Deploy Workflow", system_prompt)
        self.assertIn("read `skills/deploy-tool/SKILL.md`", system_prompt)
        # The full internal body of the skill was NOT dumped into the system prompt!
        self.assertNotIn("Detailed multi-step deployment commands and token rules", system_prompt)

        # Invariant 3: The agent was able to read the full skill on demand using read_file
        trace_data = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        step1_tool_output = trace_data["steps"][0]["tool_results"][0]["output"]
        self.assertIn("Detailed multi-step deployment commands", step1_tool_output)

    def test_global_and_project_cascading_and_shadowing(self):
        # 1. Setup mock global dir (~/.microharness)
        global_dir = Path(tempfile.mkdtemp(prefix="global_harness_")).resolve()
        (global_dir / "AGENTS.md").write_text(
            "GLOBAL RULE: Always write explicit type annotations.\n"
        )

        global_skill1 = global_dir / "skills" / "shared-tool"
        global_skill1.mkdir(parents=True)
        (global_skill1 / "SKILL.md").write_text("# Shared Tool\nGlobal shared tool instructions.\n")

        global_skill2 = global_dir / "skills" / "common-tool"
        global_skill2.mkdir(parents=True)
        (global_skill2 / "SKILL.md").write_text(
            "# Common Tool Global\nGlobal version that must be shadowed.\n"
        )

        # 2. Setup project dir
        project_dir = Path(tempfile.mkdtemp(prefix="project_dir_")).resolve()
        (project_dir / "AGENTS.md").write_text("PROJECT RULE: Target Python 3.11+ strictly.\n")

        project_skill = project_dir / "skills" / "common-tool"
        project_skill.mkdir(parents=True)
        (project_skill / "SKILL.md").write_text(
            "# Common Tool Project\nProject specialized override.\n"
        )

        captured_prompts: list[list[Message]] = []

        class MultiTurnModel:
            def generate(self, messages, tools):
                captured_prompts.append(list(messages))
                turn = len(captured_prompts)
                if turn == 1:
                    # Agent reads the global skill via absolute path
                    return "Read global skill", [
                        ToolCall(
                            id="g1",
                            name="read_file",
                            arguments={"path": str(global_skill1 / "SKILL.md")},
                        )
                    ]
                elif turn == 2:
                    # Agent reads the shadowed project skill via relative path
                    return "Read project skill", [
                        ToolCall(
                            id="g2",
                            name="read_file",
                            arguments={"path": "skills/common-tool/SKILL.md"},
                        )
                    ]
                elif turn == 3:
                    # Negative security test: Agent attempts to write into global dir (must fail)
                    return "Attempt write to global", [
                        ToolCall(
                            id="g3",
                            name="write_file",
                            arguments={"path": str(global_dir / "pwn.txt"), "content": "evil"},
                        )
                    ]
                return "All steps completed.", []

        harness = Harness(
            model=MultiTurnModel(),
            workspace_dir=project_dir,
            global_dir=global_dir,
            runs_dir=self.runs_dir,
            verbose=False,
        )

        task = Task(
            id="cascade_task",
            instruction="Execute tools following global and project rules",
            max_steps=5,
        )
        result = harness.run(task)

        self.assertTrue(result.passed)
        self.assertEqual(result.total_steps, 4)

        # Invariant 1: System prompt cascading order: Global baseline first, Project local second
        system_prompt = captured_prompts[0][0].content
        global_idx = system_prompt.find("Global User Instructions (~/.microharness/AGENTS.md)")
        project_idx = system_prompt.find("Project Instructions (AGENTS.md)")
        self.assertGreater(global_idx, -1)
        self.assertGreater(project_idx, -1)
        self.assertLess(global_idx, project_idx)  # Global comes before project
        self.assertIn("GLOBAL RULE: Always write explicit type annotations.", system_prompt)
        self.assertIn("PROJECT RULE: Target Python 3.11+ strictly.", system_prompt)

        # Invariant 2: Progressive disclosure skill index & shadowing
        self.assertIn("[global] shared-tool: Shared Tool", system_prompt)
        self.assertIn(f"read `{global_skill1 / 'SKILL.md'}`", system_prompt)
        # Project version shadows global version:
        self.assertIn("[project] common-tool: Common Tool Project", system_prompt)
        self.assertNotIn("Common Tool Global", system_prompt)
        self.assertIn("read `skills/common-tool/SKILL.md`", system_prompt)

        # Invariant 3: Global skill read succeeded
        trace_data = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        step1_out = trace_data["steps"][0]["tool_results"][0]["output"]
        self.assertIn("Global shared tool instructions", step1_out)

        # Invariant 4: Project skill read succeeded
        step2_out = trace_data["steps"][1]["tool_results"][0]["output"]
        self.assertIn("Project specialized override", step2_out)

        # Invariant 5: Negative security check - write to global dir was rejected with PermissionError
        step3_res = trace_data["steps"][2]["tool_results"][0]
        self.assertEqual(step3_res["exit_code"], 1)
        self.assertIn("Access denied", step3_res["output"])
        self.assertFalse((global_dir / "pwn.txt").exists())

    def test_edit_file_in_task_execution(self):
        project_dir = Path(tempfile.mkdtemp(prefix="edit_task_")).resolve()
        initial_code = (
            "def add(x, y):\n"
            "    return x + y\n\n"
            "def multiply(x, y):\n"
            "    # BUG: addition instead of multiplication\n"
            "    return x + y\n"
        )
        (project_dir / "math_lib.py").write_text(initial_code, encoding="utf-8")

        script = [
            (
                "Fixing multiplication bug using targeted edit",
                [
                    ToolCall(
                        id="e1",
                        name="edit_file",
                        arguments={
                            "path": "math_lib.py",
                            "old_str": "    return x + y\n",
                            "new_str": "    return x * y\n",
                        },
                    )
                ],
            ),
            (
                "First replacement had ambiguity, let's include surrounding context line",
                [
                    ToolCall(
                        id="e2",
                        name="edit_file",
                        arguments={
                            "path": "math_lib.py",
                            "old_str": "def multiply(x, y):\n    # BUG: addition instead of multiplication\n    return x + y\n",
                            "new_str": "def multiply(x, y):\n    return x * y\n",
                        },
                    )
                ],
            ),
            ("Done with fix.", []),
        ]
        model = MockModel(script)
        harness = Harness(
            model=model, workspace_dir=project_dir, runs_dir=self.runs_dir, verbose=False
        )

        task = Task(id="fix_math", instruction="Fix multiply in math_lib.py", max_steps=5)
        result = harness.run(task)

        self.assertTrue(result.passed)
        fixed_content = (project_dir / "math_lib.py").read_text(encoding="utf-8")
        self.assertIn("def multiply(x, y):\n    return x * y", fixed_content)
        self.assertIn("def add(x, y):\n    return x + y", fixed_content)

        # Check trace: step 1 failed due to multiple occurrences ("must be unique"), step 2 succeeded
        trace_data = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        step1_res = trace_data["steps"][0]["tool_results"][0]
        self.assertEqual(step1_res["exit_code"], 1)
        self.assertIn("must be unique", step1_res["output"])

        step2_res = trace_data["steps"][1]["tool_results"][0]
        self.assertEqual(step2_res["exit_code"], 0)
        self.assertIn("Successfully replaced", step2_res["output"])

    def test_git_patch_export_when_working_in_git_repo(self):
        project_dir = Path(tempfile.mkdtemp(prefix="git_repo_")).resolve()
        # Initialize real git repo
        subprocess.run("git init", shell=True, cwd=project_dir, check=True, capture_output=True)
        subprocess.run("git config user.name 'Test'", shell=True, cwd=project_dir, check=True)
        subprocess.run(
            "git config user.email 'test@test.com'", shell=True, cwd=project_dir, check=True
        )
        (project_dir / "server.py").write_text("PORT = 8000\nDEBUG = False\n", encoding="utf-8")
        subprocess.run(
            "git add . && git commit -m 'Initial commit'",
            shell=True,
            cwd=project_dir,
            check=True,
            capture_output=True,
        )

        # Agent modifies server.py
        script = [
            (
                "Enabling debug",
                [
                    ToolCall(
                        id="c1",
                        name="edit_file",
                        arguments={
                            "path": "server.py",
                            "old_str": "DEBUG = False",
                            "new_str": "DEBUG = True",
                        },
                    )
                ],
            ),
            ("Finished change", []),
        ]
        model = MockModel(script)
        harness = Harness(
            model=model, workspace_dir=project_dir, runs_dir=self.runs_dir, verbose=False
        )

        task = Task(id="enable_debug", instruction="Enable debug mode")
        result = harness.run(task)

        self.assertTrue(result.passed)
        # Invariant 1: patch_path is generated and points to patch.diff
        self.assertIsNotNone(result.patch_path)
        patch_file = Path(result.patch_path)
        self.assertTrue(patch_file.exists())
        patch_content = patch_file.read_text(encoding="utf-8")
        self.assertIn("-DEBUG = False", patch_content)
        self.assertIn("+DEBUG = True", patch_content)

        # Invariant 2: trace.json records patch_diff
        trace_data = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        self.assertIn("-DEBUG = False", trace_data["patch_diff"])

        # Invariant 3: trace.md renders git patch section
        md_content = (patch_file.parent / "trace.md").read_text(encoding="utf-8")
        self.assertIn("## Git Patch Diff", md_content)
        self.assertIn("+DEBUG = True", md_content)

    def test_custom_tools_dynamically_reflected_in_system_prompt(self):
        project_dir = Path(tempfile.mkdtemp(prefix="custom_tools_")).resolve()

        class CustomSearchTool:
            name = "custom_search"
            description = "Search our proprietary internal documentation."
            parameters = {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }

            def execute(self, **kwargs):
                return 0, "found docs"

        captured_prompts: list[list[Message]] = []

        class RecordingModel:
            def generate(self, messages, tools):
                captured_prompts.append(list(messages))
                return "All set", []

        harness = Harness(
            model=RecordingModel(),
            workspace_dir=project_dir,
            tools=[CustomSearchTool()],
            runs_dir=self.runs_dir,
            verbose=False,
        )

        task = Task(id="custom_task", instruction="Search docs")
        result = harness.run(task)

        self.assertTrue(result.passed)
        # Invariant: The custom tool is dynamically formatted into the system prompt
        system_prompt = captured_prompts[0][0].content
        self.assertIn(
            "custom_search(query: string): Search our proprietary internal documentation.",
            system_prompt,
        )

    def test_custom_commands_discovery_and_shadowing(self):
        global_dir = Path(tempfile.mkdtemp(prefix="global_cmds_")).resolve()
        project_dir = Path(tempfile.mkdtemp(prefix="project_cmds_")).resolve()

        # Global command
        (global_dir / "commands").mkdir(parents=True)
        (global_dir / "commands" / "explain.md").write_text(
            "# Explain codebase\nExplain the architecture of $ARGUMENTS", encoding="utf-8"
        )
        (global_dir / "commands" / "review.md").write_text(
            "# Global Review\nOld global review template", encoding="utf-8"
        )

        # Project command (shadows review.md)
        (project_dir / "commands").mkdir(parents=True)
        (project_dir / "commands" / "review.md").write_text(
            "# Project Review\nProject specialized review for $ARGUMENTS", encoding="utf-8"
        )
        (project_dir / "commands" / "test-all.md").write_text(
            "# Run All Tests\nExecute full test suite", encoding="utf-8"
        )

        cmds = _discover_commands(workspace=project_dir, global_dir=global_dir)

        # Invariant 1: explain is discovered from global scope
        self.assertIn("explain", cmds)
        self.assertEqual(cmds["explain"][0], "Explain codebase")
        self.assertEqual(cmds["explain"][2], "global")

        # Invariant 2: test-all is discovered from project scope
        self.assertIn("test-all", cmds)
        self.assertEqual(cmds["test-all"][0], "Run All Tests")
        self.assertEqual(cmds["test-all"][2], "project")

        # Invariant 3: project review shadows global review
        self.assertIn("review", cmds)
        self.assertEqual(cmds["review"][0], "Project Review")
        self.assertEqual(cmds["review"][2], "project")
        self.assertIn("Project specialized review", cmds["review"][1])
        self.assertNotIn("Old global review", cmds["review"][1])

    def test_expand_command_placeholders(self):
        # 1. $ARGUMENTS substitution
        t1 = "Review changes in $ARGUMENTS carefully."
        self.assertEqual(_expand_command(t1, "auth.py"), "Review changes in auth.py carefully.")

        # 2. {args} substitution
        t2 = "Run test suite on {args}."
        self.assertEqual(
            _expand_command(t2, "tests/test_harness.py"), "Run test suite on tests/test_harness.py."
        )

        # 3. Trailing append when no placeholder is present
        t3 = "Run all benchmarks."
        self.assertEqual(
            _expand_command(t3, "--quick"),
            "Run all benchmarks.\n\nAdditional Arguments / Target:\n--quick",
        )

        # 4. Empty arguments returns trimmed template
        self.assertEqual(_expand_command(t3, "   "), "Run all benchmarks.")

    def test_custom_command_execution_in_batch_mode(self):
        project_dir = Path(tempfile.mkdtemp(prefix="batch_cmd_")).resolve()
        (project_dir / "commands").mkdir(parents=True)
        (project_dir / "commands" / "lint.md").write_text(
            "# Lint code\nRun linter on $ARGUMENTS and fix issues.", encoding="utf-8"
        )

        captured_prompts: list[list[Message]] = []

        class RecordingModel:
            def generate(self, messages, tools):
                captured_prompts.append(list(messages))
                return "Linting complete.", []

        harness = Harness(
            model=RecordingModel(),
            workspace_dir=project_dir,
            runs_dir=self.runs_dir,
            verbose=False,
        )

        # Initial instruction invokes custom command /lint core/
        task = Task(id="run_lint_cmd", instruction="/lint core/")
        result = harness.run(task)

        self.assertTrue(result.passed)
        # Invariant: Prompt sent to model had /lint expanded with template and argument
        user_prompt = captured_prompts[0][1].content
        self.assertIn("Run linter on core/ and fix issues.", user_prompt)
        self.assertNotIn("/lint core/", user_prompt)

    def test_custom_command_in_interactive_mode(self):
        project_dir = Path(tempfile.mkdtemp(prefix="interactive_cmd_")).resolve()
        (project_dir / "commands").mkdir(parents=True)
        (project_dir / "commands" / "audit.md").write_text(
            "# Audit Security\nPerform security audit on target: $ARGUMENTS", encoding="utf-8"
        )

        captured_prompts: list[list[Message]] = []

        class RecordingModel:
            def generate(self, messages, tools):
                captured_prompts.append(list(messages))
                return "Audit report generated.", []

        harness = Harness(
            model=RecordingModel(),
            workspace_dir=project_dir,
            runs_dir=self.runs_dir,
            verbose=False,
        )

        # Mock interactive inputs: /help, custom command /audit api.py, then exit
        user_inputs = iter(["/help", "/audit api.py", "exit"])
        with patch("builtins.input", lambda *args: next(user_inputs)):
            harness.run_interactive(initial_instruction="Initial prompt")

        # Turn 1: Initial prompt
        # Turn 2: /audit api.py expanded
        self.assertEqual(len(captured_prompts), 2)
        turn2_user_prompt = captured_prompts[1][-1].content
        self.assertIn("Perform security audit on target: api.py", turn2_user_prompt)

    def test_load_global_config_scenarios(self):
        global_dir = Path(tempfile.mkdtemp(prefix="global_cfg_test_"))

        # 1. Missing config.json -> returns {}
        self.assertEqual(load_global_config(global_dir), {})

        # 2. None directory -> returns {}
        self.assertEqual(load_global_config(None), {})

        # 3. Valid config.json
        cfg_path = global_dir / "config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "endpoint": "https://api.openai.com/v1",
                    "model": "gpt-4o",
                    "max_context_chars": 24000,
                    "trigger_ratio": 0.85,
                    "sandbox": True,
                }
            ),
            encoding="utf-8",
        )
        loaded = load_global_config(global_dir)
        self.assertEqual(loaded["model"], "gpt-4o")
        self.assertEqual(loaded["endpoint"], "https://api.openai.com/v1")
        self.assertEqual(loaded["max_context_chars"], 24000)
        self.assertEqual(loaded["trigger_ratio"], 0.85)
        self.assertTrue(loaded["sandbox"])

        # 4. Corrupted config.json -> returns {} gracefully without crashing
        cfg_path.write_text("{malformed_json: true,", encoding="utf-8")
        self.assertEqual(load_global_config(global_dir), {})

    def test_harness_inherits_and_overrides_global_config(self):
        global_dir = Path(tempfile.mkdtemp(prefix="harness_cfg_test_"))
        cfg_path = global_dir / "config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "max_context_chars": 16000,
                    "trigger_ratio": 0.75,
                    "max_tool_chars": 1500,
                    "sandbox": True,
                }
            ),
            encoding="utf-8",
        )

        # Harness with defaults inheriting from config.json
        h1 = Harness(
            model=MockModel([]),
            global_dir=global_dir,
            runs_dir=self.runs_dir,
            verbose=False,
        )
        self.assertEqual(h1.max_context_chars, 16000)
        self.assertEqual(h1.trigger_ratio, 0.75)
        self.assertEqual(h1.max_tool_chars, 1500)
        self.assertTrue(h1.sandbox)

        # Harness with explicit overrides taking precedence over config.json
        h2 = Harness(
            model=MockModel([]),
            global_dir=global_dir,
            max_context_chars=32000,
            trigger_ratio=0.90,
            sandbox=False,
            runs_dir=self.runs_dir,
            verbose=False,
        )
        self.assertEqual(h2.max_context_chars, 32000)
        self.assertEqual(h2.trigger_ratio, 0.90)
        self.assertFalse(h2.sandbox)

    def test_agent_loop_records_modified_files_in_memory(self):
        ws_dir = Path(self.runs_dir) / "test_loop_ws"
        ws_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = Path(self.runs_dir) / "test_loop_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        env = Environment(
            workspace_dir=ws_dir,
            logs_dir=logs_dir,
            cleanup_on_teardown=False,
        )
        registry = ToolRegistry(env=env)
        registry.register(WriteFileTool(env))
        registry.register(EditFileTool(env))

        # Memory with small context to trigger compaction easily
        memory = Memory(
            system_prompt="SYS",
            user_prompt="TASK",
            max_context_chars=120,
            keep_recent_turns=1,
        )
        script = [
            (
                "Step 1: Write file",
                [
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments={"path": "code.py", "content": "x = 1\n"},
                    )
                ],
            ),
            (
                "Step 2: Negative failure case - edit non-existent text",
                [
                    ToolCall(
                        id="c2",
                        name="edit_file",
                        arguments={
                            "path": "unmodified.py",
                            "old_str": "non_existent",
                            "new_str": "val",
                        },
                    )
                ],
            ),
            (
                "Step 3: Edit file successfully",
                [
                    ToolCall(
                        id="c3",
                        name="edit_file",
                        arguments={
                            "path": "code.py",
                            "old_str": "x = 1",
                            "new_str": "x = 2",
                        },
                    )
                ],
            ),
            ("Step 4: Done", []),
        ]
        model = MockModel(script)
        loop = AgentLoop(model=model, memory=memory, tools=registry)

        # Step 1: write_file succeeds
        finished, trace1 = loop.step(1)
        self.assertFalse(finished)
        self.assertEqual(memory.modified_files, {"code.py"})

        # Step 2: edit_file fails (exit code 1) -> unmodified.py MUST NOT be recorded
        finished, trace2 = loop.step(2)
        self.assertFalse(finished)
        self.assertEqual(trace2.tool_results[0]["exit_code"], 1)
        self.assertEqual(memory.modified_files, {"code.py"})
        self.assertNotIn("unmodified.py", memory.modified_files)

        # Step 3: edit_file succeeds -> code.py recorded
        finished, trace3 = loop.step(3)
        self.assertFalse(finished)
        self.assertEqual(memory.modified_files, {"code.py"})

        # Step 4: final answer -> finishes
        finished, trace4 = loop.step(4)
        self.assertTrue(finished)

    def test_dot_microharness_full_project_namespace(self):
        """Invariant: .microharness/ provides a unified project namespace for config, agents, skills, and commands."""
        project_dir = Path(tempfile.mkdtemp(prefix="project_dot_mh_")).resolve()
        dot_mh = project_dir / ".microharness"
        dot_mh.mkdir(parents=True)

        # 1. Project config (.microharness/config.json)
        (dot_mh / "config.json").write_text(
            json.dumps({"max_context_chars": 12345, "trigger_ratio": 0.75, "sandbox": False}),
            encoding="utf-8",
        )
        loaded_cfg = load_project_config(project_dir)
        self.assertEqual(loaded_cfg["max_context_chars"], 12345)
        self.assertEqual(loaded_cfg["trigger_ratio"], 0.75)

        # 2. Project instructions (.microharness/AGENTS.md)
        (dot_mh / "AGENTS.md").write_text(
            "Project rule: Use strictly typed dataclasses only.\n", encoding="utf-8"
        )

        # 3. Project skill (.microharness/skills/lint/SKILL.md)
        skill_dir = dot_mh / "skills" / "lint-check"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "# Automated Lint Checker\nRun ruff and check all types.\n", encoding="utf-8"
        )

        # 4. Project command (.microharness/commands/test.md)
        cmd_dir = dot_mh / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "test.md").write_text(
            "# Run Tests Fast\npytest -k $ARGUMENTS\n", encoding="utf-8"
        )

        # Model captures the prompt and reads the skill via read_file
        captured_prompts: list[list[Message]] = []

        class RecordingModel:
            def generate(self, messages, tools):
                captured_prompts.append(list(messages))
                if len(captured_prompts) == 1:
                    return "Checking lint skill", [
                        ToolCall(
                            id="s1",
                            name="read_file",
                            arguments={"path": ".microharness/skills/lint-check/SKILL.md"},
                        )
                    ]
                return "All good", []

        harness = Harness(
            model=RecordingModel(),
            workspace_dir=project_dir,
            runs_dir=self.runs_dir,
            verbose=False,
        )

        # Verify harness merged config
        self.assertEqual(harness.max_context_chars, 12345)
        self.assertEqual(harness.trigger_ratio, 0.75)

        # Run task
        task = Task(id="test_dot_mh", instruction="Check code quality")
        result = harness.run(task)
        self.assertTrue(result.passed)

        # Invariants:
        # Prompt contains .microharness/AGENTS.md rule
        system_prompt = captured_prompts[0][0].content
        self.assertIn("Project rule: Use strictly typed dataclasses only.", system_prompt)

        # Prompt indexes .microharness/skills/lint-check/SKILL.md
        self.assertIn("lint-check: Automated Lint Checker", system_prompt)
        self.assertIn("read `.microharness/skills/lint-check/SKILL.md`", system_prompt)

        # Agent read the skill file without permission error
        trace_data = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        skill_content = trace_data["steps"][0]["tool_results"][0]["output"]
        self.assertIn("Run ruff and check all types.", skill_content)

        # Verify commands discovery includes .microharness/commands
        cmds = _discover_commands(workspace=project_dir)
        self.assertIn("test", cmds)
        self.assertEqual(cmds["test"][0], "Run Tests Fast")
        self.assertEqual(cmds["test"][2], "project")

    def test_runs_dir_precedence_and_global_default(self):
        """Invariant: runs_dir defaults to global_dir / 'runs' unless overridden by config or argument."""
        global_dir = Path(tempfile.mkdtemp(prefix="global_runs_")).resolve()
        project_dir = Path(tempfile.mkdtemp(prefix="project_runs_")).resolve()

        # 1. Default fallback: global_dir / "runs"
        h1 = Harness(
            model=MockModel([]), workspace_dir=project_dir, global_dir=global_dir, verbose=False
        )
        self.assertEqual(h1.runs_dir, (global_dir / "runs").resolve())

        # 2. Config override: .microharness/config.json specifying "runs_dir"
        custom_runs = Path(tempfile.mkdtemp(prefix="custom_runs_")).resolve()
        dot_mh = project_dir / ".microharness"
        dot_mh.mkdir(parents=True, exist_ok=True)
        (dot_mh / "config.json").write_text(
            json.dumps({"runs_dir": str(custom_runs)}), encoding="utf-8"
        )

        h2 = Harness(
            model=MockModel([]), workspace_dir=project_dir, global_dir=global_dir, verbose=False
        )
        self.assertEqual(h2.runs_dir, custom_runs)

        # 3. Explicit argument override wins over config
        explicit_runs = Path(tempfile.mkdtemp(prefix="explicit_runs_")).resolve()
        h3 = Harness(
            model=MockModel([]),
            workspace_dir=project_dir,
            global_dir=global_dir,
            runs_dir=explicit_runs,
            verbose=False,
        )
        self.assertEqual(h3.runs_dir, explicit_runs)

    def test_harness_cache_ttl_on_the_fly_and_config(self):
        model = MockModel([], cache_ttl="5m")
        harness = Harness(
            model=model,
            runs_dir=self.runs_dir,
            cache_ttl="5m",
            verbose=False,
        )
        self.assertEqual(harness.cache_ttl, "5m")
        self.assertEqual(model.cache_ttl, "5m")

        # Switch to 1h on the fly
        res = harness.set_cache_ttl("1h")
        self.assertEqual(res, "1h")
        self.assertEqual(harness.cache_ttl, "1h")
        self.assertEqual(model.cache_ttl, "1h")

        # Switch to off on the fly
        res_off = harness.set_cache_ttl("off")
        self.assertIsNone(res_off)
        self.assertIsNone(harness.cache_ttl)
        self.assertIsNone(model.cache_ttl)

        stats = harness.get_cache_stats()
        self.assertIsNone(stats.get("cache_ttl"))

        # Config precedence
        project_dir = Path(tempfile.mkdtemp(prefix="project_cache_"))
        dot_mh = project_dir / ".microharness"
        dot_mh.mkdir(parents=True, exist_ok=True)
        (dot_mh / "config.json").write_text(json.dumps({"cache_ttl": "1h"}), encoding="utf-8")

        m2 = MockModel([])
        h2 = Harness(model=m2, workspace_dir=project_dir, runs_dir=self.runs_dir, verbose=False)
        self.assertEqual(h2.cache_ttl, "1h")
        self.assertEqual(m2.cache_ttl, "1h")

    def test_harness_docker_sandboxing_integration(self):
        from core.environment import is_docker_available

        if not is_docker_available():
            self.skipTest("Docker is not available.")

        script = [
            (
                "Running in container",
                [
                    ToolCall(
                        id="c1",
                        name="bash",
                        arguments={"command": "uname -s > os_info.txt"},
                    )
                ],
            ),
            ("Done", []),
        ]
        model = MockModel(script)
        evaluator = FileContentEvaluator(file_path="os_info.txt", must_contain=["Linux"])

        harness = Harness(
            model=model,
            evaluator=evaluator,
            runs_dir=self.runs_dir,
            docker=True,
            docker_image="python:3.11",
            verbose=False,
        )

        task = Task(id="docker_task", instruction="Record OS name to os_info.txt", max_steps=5)
        result = harness.run(task)
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)

    def test_skills_frontmatter_description_summary(self):
        from core.harness import _discover_skills_index

        workspace = Path(tempfile.mkdtemp(prefix="ws_frontmatter_"))
        skill_dir = workspace / ".microharness" / "skills" / "my-linter"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-linter\ndescription: Comprehensive code style checks\n---\n# Linter\nDetails...",
            encoding="utf-8",
        )

        index = _discover_skills_index(workspace, global_dir=None)
        self.assertIn("Comprehensive code style checks", index)
        self.assertNotIn(": --- (read", index)

    def test_log_spill_readable_with_custom_runs_dir(self):
        custom_runs = Path(tempfile.mkdtemp(prefix="custom_runs_"))
        harness = Harness(
            model=MockModel([("Done", [])]),
            sandbox=True,
            runs_dir=custom_runs,
            verbose=False,
        )

        # Exercise the real harness environment setup wiring
        env, tool_registry, _, _, _, run_dir = harness._setup_run(
            task_id="spill_wiring_test",
            system_prompt="",
            initial_instruction="test",
            initial_files={},
            force_sandbox=True,
        )

        # Create a simulated spilled log file in the harness-configured logs_dir
        spill_file = env.logs_dir / "step_1_bash_call1.log"
        spill_file.write_text("AAAA_TRUNCATED_LOG_OUTPUT", encoding="utf-8")

        # 1. POSITIVE ASSERTION: read_file dispatch succeeds because logs_dir is in allowed_reads
        read_call = ToolCall(
            id="read_spill",
            name="read_file",
            arguments={"path": str(spill_file)},
        )
        res = tool_registry.dispatch(read_call, step_index=2)
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(res.output, "   1 | AAAA_TRUNCATED_LOG_OUTPUT")

        # 2. NEGATIVE ASSERTION: Attempting to read another file outside logs_dir in custom_runs fails
        outside_file = custom_runs / "unauthorized.txt"
        outside_file.write_text("SECRET_RUNS_DATA", encoding="utf-8")
        unauth_call = ToolCall(
            id="read_outside",
            name="read_file",
            arguments={"path": str(outside_file)},
        )
        unauth_res = tool_registry.dispatch(unauth_call, step_index=3)
        self.assertEqual(unauth_res.exit_code, 1)
        self.assertIn("Access denied", unauth_res.output)

    def test_global_config_json_not_readable_by_agent(self):
        global_d = Path(tempfile.mkdtemp(prefix="secret_global_"))
        (global_d / "config.json").write_text('{"api_key": "sk-secret-12345"}', encoding="utf-8")

        harness = Harness(
            model=MockModel([("Done", [])]),
            global_dir=global_d,
            sandbox=True,
            runs_dir=self.runs_dir,
            verbose=False,
        )
        env, _, _, _, _, _ = harness._setup_run(
            task_id="leak_test",
            system_prompt="",
            initial_instruction="test",
            initial_files={},
            force_sandbox=True,
        )

        with self.assertRaises(PermissionError):
            env.read_file(str(global_d / "config.json"))

    def test_step_trace_tool_call_id_preserved(self):
        script = [
            (
                "Calling tool",
                [ToolCall(id="call_abc_123", name="bash", arguments={"command": "echo hi"})],
            ),
            ("Done", []),
        ]
        model = MockModel(script)
        harness = Harness(model=model, runs_dir=self.runs_dir, verbose=False)
        result = harness.run(Task(id="trace_test", instruction="test", max_steps=2))
        self.assertEqual(result.finish_reason, "completed")

        # Read trace.json
        trace_data = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        step0_calls = trace_data["steps"][0]["model_response"]["tool_calls"]
        self.assertEqual(len(step0_calls), 1)
        self.assertEqual(step0_calls[0]["id"], "call_abc_123")
        self.assertEqual(step0_calls[0]["name"], "bash")

    def test_skills_frontmatter_block_scalar_summary(self):
        from core.harness import _extract_skill_summary

        # Test folded scalar (>)
        content_folded = (
            "---\n"
            "name: my-skill\n"
            "description: >\n"
            "  This is a multiline\n"
            "  description folded into\n"
            "  one line summary.\n"
            "---\n"
            "# My Skill\n"
        )
        summary_folded = _extract_skill_summary(content_folded, default="fallback")
        self.assertEqual(
            summary_folded,
            "This is a multiline description folded into one line summary.",
        )

        # Test literal scalar (|)
        content_literal = (
            "---\n"
            "name: my-skill-2\n"
            "description: |\n"
            "  First line of description.\n"
            "  Second line of description.\n"
            "---\n"
            "# My Skill 2\n"
        )
        summary_literal = _extract_skill_summary(content_literal, default="fallback")
        self.assertEqual(
            summary_literal,
            "First line of description. Second line of description.",
        )

    def test_skills_frontmatter_leading_blank_lines(self):
        from core.harness import _extract_skill_summary

        content_leading_blank = (
            "\n"
            "\n"
            "---\n"
            "name: test-skill\n"
            "description: Summary should be extracted despite leading blanks\n"
            "---\n"
            "# Heading\n"
        )
        summary = _extract_skill_summary(content_leading_blank, default="fallback")
        self.assertEqual(summary, "Summary should be extracted despite leading blanks")
        self.assertNotIn("description:", summary)
        self.assertNotIn("---", summary)

    def test_action_formatter_builtins_and_custom_registration(self):
        from core.tracer import (
            _format_action_summary,
            register_action_formatter,
            unregister_action_formatter,
        )

        # 1. Test built-in tool formatters with strict equality
        self.assertEqual(
            _format_action_summary("bash", {"command": "echo hi"}, color=False),
            "$ echo hi",
        )
        self.assertEqual(
            _format_action_summary(
                "read_file",
                {"path": "foo.py", "start_line": 5, "end_line": 15},
                color=False,
            ),
            "● Read foo.py (lines 5-15)",
        )
        self.assertEqual(
            _format_action_summary(
                "write_file",
                {"path": "bar.txt", "content": "line1\nline2\nline3"},
                color=False,
            ),
            "● Write bar.txt (3 lines)",
        )
        self.assertEqual(
            _format_action_summary("edit_file", {"path": "baz.py"}, color=False),
            "● Edit baz.py",
        )

        # 2. Test non-dict args with registered formatter
        self.assertEqual(
            _format_action_summary("bash", "echo string_cmd", color=False),
            "$ echo string_cmd",
        )

        # 3. Test generic fallback for unregistered tools
        self.assertEqual(
            _format_action_summary("unknown_tool", {"k": "v"}, color=False),
            "● unknown_tool(k='v')",
        )

        # 4. Test custom tool registration and ensure cleanup in finally block
        try:
            register_action_formatter(
                "custom_query",
                lambda args: f"Query({args.get('target', '') if isinstance(args, dict) else args})",
            )
            formatted = _format_action_summary(
                "custom_query",
                {"target": "users_table"},
                color=False,
            )
            self.assertEqual(formatted, "Query(users_table)")
        finally:
            unregister_action_formatter("custom_query")

        # Verify unregistration restored generic fallback
        self.assertEqual(
            _format_action_summary("custom_query", {"target": "users_table"}, color=False),
            "● custom_query(target='users_table')",
        )

        # 5. Test custom formatter fail-soft on malformed / non-dict args with warning
        try:
            register_action_formatter(
                "strict_dict_tool",
                lambda args: f"Strict({args['key']})",  # assumes dict, raises TypeError on string
            )
            import warnings

            with warnings.catch_warnings(record=True) as recorded_warnings:
                warnings.simplefilter("always")
                failsoft_out = _format_action_summary(
                    "strict_dict_tool",
                    "malformed_string_args",
                    color=False,
                )
                self.assertEqual(failsoft_out, "● strict_dict_tool(malformed_string_args)")
                self.assertEqual(len(recorded_warnings), 1)
                self.assertIn(
                    "Action formatter for tool 'strict_dict_tool' raised",
                    str(recorded_warnings[0].message),
                )
        finally:
            unregister_action_formatter("strict_dict_tool")

    def test_mock_model_reset_composition_factory(self):
        from main import create_benchmark_harness_factory

        # 1-entry script: executes write_file on step 0, exhausted on step 1
        script = [
            (
                "Write active flag",
                [
                    ToolCall(
                        id="w1",
                        name="write_file",
                        arguments={"path": "flag.txt", "content": "active"},
                    )
                ],
            )
        ]
        model = MockModel(script)
        factory = create_benchmark_harness_factory(
            model=model,
            global_dir=Path(tempfile.mkdtemp()),
            runs_dir=Path(self.runs_dir),
            verbose=False,
        )
        dummy_eval = FileContentEvaluator("flag.txt", "active")

        task1 = Task(id="t1", instruction="do 1", max_steps=2)
        h1 = factory(dummy_eval)
        res1 = h1.run(task1)
        self.assertEqual(res1.finish_reason, "completed")
        self.assertTrue((Path(res1.workspace_path) / "flag.txt").exists())
        self.assertEqual(
            (Path(res1.workspace_path) / "flag.txt").read_text(encoding="utf-8"),
            "active",
        )

        task2 = Task(id="t2", instruction="do 2", max_steps=2)
        h2 = factory(dummy_eval)
        res2 = h2.run(task2)
        self.assertEqual(res2.finish_reason, "completed")
        # If reset() was not called by create_benchmark_harness_factory, task 2 starts exhausted and fails this assertion!
        self.assertTrue((Path(res2.workspace_path) / "flag.txt").exists())
        self.assertEqual(
            (Path(res2.workspace_path) / "flag.txt").read_text(encoding="utf-8"),
            "active",
        )

    def test_run_dir_uniqueness_same_task_same_second(self):
        harness = Harness(model=MockModel([("Done", [])]), runs_dir=self.runs_dir, verbose=False)
        _, _, _, _, _, run_dir1 = harness._setup_run(
            task_id="same_second_task",
            system_prompt="",
            initial_instruction="test",
            initial_files={},
        )
        _, _, _, _, _, run_dir2 = harness._setup_run(
            task_id="same_second_task",
            system_prompt="",
            initial_instruction="test",
            initial_files={},
        )
        self.assertNotEqual(run_dir1, run_dir2)
        self.assertTrue(run_dir1.exists())
        self.assertTrue(run_dir2.exists())

    def test_harness_config_empty_string_fallback(self):
        """Invariant: Empty string values in config for runs_dir and docker_image fall back to defaults."""
        global_dir = Path(tempfile.mkdtemp(prefix="test_empty_cfg_")).resolve()
        config_path = global_dir / "config.json"
        config_path.write_text(
            json.dumps({"runs_dir": "", "docker_image": ""}),
            encoding="utf-8",
        )

        harness = Harness(
            model=MockModel([("Done", [])]),
            global_dir=global_dir,
            verbose=False,
        )
        self.assertEqual(harness.runs_dir, (global_dir / "runs").resolve())
        self.assertEqual(harness.docker_image, "python:3.11")

        # Explicit empty string argument should also fall back to default
        harness_explicit = Harness(
            model=MockModel([("Done", [])]),
            global_dir=global_dir,
            runs_dir="",
            docker_image="",
            verbose=False,
        )
        self.assertEqual(harness_explicit.runs_dir, (global_dir / "runs").resolve())
        self.assertEqual(harness_explicit.docker_image, "python:3.11")

    def test_interactive_dispatcher_slash_commands(self):
        """Invariant: Interactive dispatcher handles slash commands without adding them to memory or invoking model."""
        from core.memory import Memory

        harness = Harness(model=MockModel([("Done", [])]), verbose=False)
        memory = Memory(system_prompt="sys", user_prompt="")

        custom_commands = {"review": ("Review PR", "git diff HEAD~1", "project")}

        # 1. /status does not add to memory, returns skip
        action = harness._dispatch_interactive_input("/status", memory, custom_commands)
        self.assertEqual(action, "skip")
        self.assertEqual(len(memory.messages), 1)  # only system prompt

        # 2. /help returns skip
        action = harness._dispatch_interactive_input("/help", memory, custom_commands)
        self.assertEqual(action, "skip")

        # 3. /exit returns exit
        action = harness._dispatch_interactive_input("/exit", memory, custom_commands)
        self.assertEqual(action, "exit")

        # 4. /review expands custom command and returns turn
        action = harness._dispatch_interactive_input("/review", memory, custom_commands)
        self.assertEqual(action, "turn")
        self.assertEqual(len(memory.messages), 2)
        self.assertEqual(memory.messages[-1].content, "git diff HEAD~1")

        # 5. Regular input adds user message and returns turn
        action = harness._dispatch_interactive_input("Run the tests", memory, custom_commands)
        self.assertEqual(action, "turn")
        self.assertEqual(len(memory.messages), 3)
        self.assertEqual(memory.messages[-1].content, "Run the tests")

    def test_run_interactive_slash_command_initial_instruction(self):
        """Invariant: An initial instruction with a slash command (like /status) is dispatched, not fed as model turn."""
        from unittest.mock import patch

        harness = Harness(model=MockModel([("Done", [])]), verbose=False)

        # Calling run_interactive with initial_instruction="/status" and input="/exit"
        with patch("builtins.input", return_value="/exit"):
            harness.run_interactive(initial_instruction="/status")

    def test_cache_aware_model_protocol_compliance(self):
        """Invariant: Models implementing cache methods satisfy CacheAwareModel protocol."""
        from core.model import CacheAwareModel, HttpChatModel

        mock_model = MockModel([("Hi", [])])
        http_model = HttpChatModel(base_url="http://localhost:8000/v1", model="test")

        self.assertIsInstance(mock_model, CacheAwareModel)
        self.assertIsInstance(http_model, CacheAwareModel)

        # Minimal model with only generate does not satisfy CacheAwareModel
        class BareModel:
            def generate(self, messages, tools):
                return "bare", []

        bare_model = BareModel()
        self.assertNotIsInstance(bare_model, CacheAwareModel)

        # Harness with BareModel safely defaults cache methods without error
        harness = Harness(model=bare_model, verbose=False)
        self.assertEqual(harness.get_cache_stats(), {"cache_ttl": "5m"})

    def test_harness_rate_limit_finish_reason(self):
        """Invariant: When model raises RateLimitError, Harness finishes with finish_reason='rate_limited'."""
        from unittest.mock import MagicMock

        from core.model import RateLimitError

        mock_model = MagicMock()
        mock_model.generate.side_effect = RateLimitError("Rate limit reached", retry_after=30.0)

        harness = Harness(model=mock_model, runs_dir=self.runs_dir, verbose=False)
        task = Task(id="rate_limit_task", instruction="Do task")
        res = harness.run(task)

        self.assertEqual(res.finish_reason, "rate_limited")
        self.assertFalse(res.passed)

    def test_harness_reasoning_effort_control_and_slash_command(self):
        """Invariant: Harness allows dynamic reasoning effort configuration and /reasoning slash command."""
        mock_model = MockModel([("Hello", [])], reasoning_effort="high")
        harness = Harness(model=mock_model, reasoning_effort="high", verbose=False)
        self.assertEqual(harness.get_reasoning_effort(), "high")

        # Programmatic switch
        harness.set_reasoning_effort("low")
        self.assertEqual(harness.get_reasoning_effort(), "low")
        self.assertEqual(mock_model.get_reasoning_effort(), "low")

        # Interactive /reasoning slash command switch
        memory = Memory(system_prompt="sys", user_prompt="init")
        action = harness._dispatch_interactive_input("/reasoning medium", memory, custom_commands={})
        self.assertEqual(action, "skip")
        self.assertEqual(harness.get_reasoning_effort(), "medium")

        # Interactive /reasoning off
        action = harness._dispatch_interactive_input("/reasoning off", memory, custom_commands={})
        self.assertEqual(action, "skip")
        self.assertIsNone(harness.get_reasoning_effort())

    def test_harness_run_result_token_telemetry(self):
        """Invariant: RunResult captures prompt, completion, cached tokens and cache hit percentage."""
        mock_model = MockModel([("Done", [])])
        mock_model._update_usage({
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 750},
        })

        harness = Harness(model=mock_model, runs_dir=self.runs_dir, verbose=False)
        task = Task(id="token_task", instruction="Test tokens")
        result = harness.run(task)

        self.assertEqual(result.prompt_tokens, 1000)
        self.assertEqual(result.completion_tokens, 100)
        self.assertEqual(result.cached_tokens, 750)
        self.assertEqual(result.cache_hit_rate_pct, 75.0)


if __name__ == "__main__":
    unittest.main()
