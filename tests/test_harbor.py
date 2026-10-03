from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.harbor import (
    DEEPSWE_MINI_TASKS,
    HarborEvaluator,
    get_deepswe_mini_suite,
    load_harbor_task,
)
from core.environment import Environment


class TestHarborBenchmark(unittest.TestCase):
    def test_deepswe_mini_tasks_constant(self):
        self.assertEqual(len(DEEPSWE_MINI_TASKS), 16)
        self.assertEqual(len(set(DEEPSWE_MINI_TASKS)), 16)
        self.assertIn("aiomonitor-task-snapshots-diff", DEEPSWE_MINI_TASKS)
        self.assertIn("bandit-structured-nosec-directives", DEEPSWE_MINI_TASKS)
        self.assertIn("yaegi-go-embed-directives", DEEPSWE_MINI_TASKS)

    def test_load_harbor_task_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            toml_content = (
                'schema_version = "1.3"\n'
                "[metadata]\n"
                'task_id = "test-sample-task"\n'
                'base_commit_hash = "abc12345"\n'
                "[environment]\n"
                'docker_image = "python:3.11-slim"\n'
            )
            (td / "task.toml").write_text(toml_content)
            (td / "instruction.md").write_text("Fix the sample bug in auth.")

            task, evaluator = load_harbor_task(td, max_steps=12, timeout_seconds=180.0)

            self.assertEqual(task.id, "test-sample-task")
            self.assertEqual(task.instruction, "Fix the sample bug in auth.")
            self.assertEqual(task.docker_image, "python:3.11-slim")
            self.assertEqual(task.init_workdir, "/app")
            self.assertEqual(task.max_steps, 12)
            self.assertEqual(task.timeout_seconds, 180.0)

            self.assertIsInstance(evaluator, HarborEvaluator)
            self.assertEqual(evaluator.base_commit, "abc12345")
            self.assertEqual(evaluator.task_dir, td.resolve())

    def test_load_harbor_task_missing_toml_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            (td / "instruction.md").write_text("Do something")
            with self.assertRaises(FileNotFoundError):
                load_harbor_task(td)

    def test_load_harbor_task_missing_instruction_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            (td / "task.toml").write_text(
                '[metadata]\ntask_id = "t1"\n[environment]\ndocker_image = "img"\n'
            )
            with self.assertRaises(FileNotFoundError):
                load_harbor_task(td)

    def test_load_harbor_task_missing_docker_image_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            (td / "task.toml").write_text('[metadata]\ntask_id = "t1"\n[environment]\n')
            (td / "instruction.md").write_text("Do something")
            with self.assertRaises(ValueError) as ctx:
                load_harbor_task(td)
            self.assertIn("has no docker_image", str(ctx.exception))

    def test_harbor_evaluator_rejects_non_docker_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            env = Environment(workspace_dir=td)
            evaluator = HarborEvaluator(task_dir=td, base_commit="HEAD")
            res = evaluator.evaluate(env, steps=[])
            self.assertFalse(res.passed)
            self.assertEqual(res.score, 0.0)
            self.assertIn("active DockerEnvironment", res.feedback)

    def test_get_deepswe_mini_suite_filter(self):
        dataset_path = Path("datasets/deep-swe/tasks").resolve()
        if not dataset_path.is_dir():
            self.skipTest("datasets/deep-swe/tasks not cloned")

        suite = get_deepswe_mini_suite(dataset_dir=dataset_path, task_filter="aiomonitor")
        self.assertEqual(len(suite), 1)
        task, evaluator = suite[0]
        self.assertEqual(task.id, "aiomonitor-task-snapshots-diff")
        self.assertEqual(task.init_workdir, "/app")
        self.assertIn("aiomonitor", evaluator.task_dir.name)

    def test_harbor_evaluator_partial_score(self):
        """Invariant: When reward=0 but partial score is supplied, HarborEvaluator captures the partial score."""
        import json
        from unittest.mock import MagicMock

        from core.environment import DockerEnvironment

        mock_docker_env = MagicMock(spec=DockerEnvironment)
        mock_docker_env.is_running = True
        mock_docker_env.container_name = "test_container"
        mock_docker_env.execute_bash.side_effect = [
            (0, "patch applied"),
            (1, "1 test failed, 3 tests passed"),
            (
                0,
                json.dumps(
                    {
                        "reward": 0,
                        "partial": 0.75,
                        "f2p_passed": 3,
                        "f2p_total": 4,
                        "p2p_passed": 5,
                        "p2p_total": 5,
                    }
                ),
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = HarborEvaluator(task_dir=Path(tmpdir), base_commit="HEAD")
            res = evaluator.evaluate(mock_docker_env, steps=[])

            self.assertFalse(res.passed)
            self.assertEqual(res.score, 0.75)
            self.assertIn("Partial: 0.75", res.feedback)
            self.assertIn("F2P: 3/4", res.feedback)

    def test_harbor_evaluator_accepts_pluggable_container_env_not_docker_subclass(self):
        """Invariant: HarborEvaluator accepts any ContainerEnvironment protocol implementer without requiring DockerEnvironment inheritance."""
        import json

        from core.environment import ContainerEnvironment, DockerEnvironment

        class PluggableContainerEnv:
            container_name = "custom_podman_container"
            is_running = True

            def execute_bash(self, command: str, timeout: float = 30.0) -> tuple[int, str]:
                if "/logs/verifier/reward.json" in command:
                    return 0, json.dumps({"reward": 1, "f2p_passed": 2, "f2p_total": 2})
                return 0, "ok"

        custom_env = PluggableContainerEnv()
        self.assertTrue(isinstance(custom_env, ContainerEnvironment))
        self.assertFalse(isinstance(custom_env, DockerEnvironment))

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = HarborEvaluator(task_dir=Path(tmpdir), base_commit="HEAD")
            res = evaluator.evaluate(custom_env, steps=[])

            self.assertTrue(res.passed)
            self.assertEqual(res.score, 1.0)
            self.assertIn("PASSED ✅", res.feedback)


if __name__ == "__main__":
    unittest.main()
