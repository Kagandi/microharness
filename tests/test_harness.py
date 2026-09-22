import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from core.evaluator import FileContentEvaluator
from core.harness import Harness
from core.model import MockModel
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

        harness = Harness(model=ExplodingModel(), runs_dir=self.runs_dir, verbose=False)
        task = Task(id="crash_task", instruction="Will crash", max_steps=5)

        # Invariant: Harness does not raise unhandled exception; records error cleanly
        result = harness.run(task)
        self.assertEqual(result.finish_reason, "error")
        self.assertFalse(result.passed)
        self.assertTrue(Path(result.trace_path).exists())

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
        # 1. Setup mock global dir (~/.mini-harness)
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
        global_idx = system_prompt.find("Global User Instructions (~/.mini-harness/AGENTS.md)")
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


if __name__ == "__main__":
    unittest.main()
