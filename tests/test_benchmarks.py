import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.mini_swe import (
    MINI_SWE_TASKS,
    create_auth_middleware_task,
    create_circular_serializer_task,
    create_lru_ttl_cache_task,
    create_query_builder_task,
    create_rate_limiter_task,
    get_mini_swe_suite,
)
from benchmarks.runner import run_benchmark_suite
from core.environment import Environment
from core.harness import Harness
from core.model import MockModel
from core.types import ToolCall


class TestMiniSWEBenchmarks(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def _create_env(self, files: dict[str, str]) -> Environment:
        ws = Path(self.tmp_dir) / "ws"
        ws.mkdir(parents=True, exist_ok=True)
        logs = Path(self.tmp_dir) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        env = Environment(workspace_dir=ws, logs_dir=logs, cleanup_on_teardown=False)
        env.populate_files(files)
        return env

    def test_all_tasks_fail_initially_as_baseline(self):
        """Invariant: Every benchmark task must fail its baseline unit tests before fixes."""
        for item in MINI_SWE_TASKS:
            task, evaluator = item.task_factory()
            env = self._create_env(task.initial_files)
            res = evaluator.evaluate(env, [])
            self.assertFalse(
                res.passed,
                f"Task '{item.id}' unexpectedly passed before any fix was applied! Must fail baseline.",
            )
            self.assertEqual(res.score, 0.0)

    def test_pristine_evaluator_prevents_cheating(self):
        """Invariant: If an agent deletes or replaces the test file, evaluator restores pristine tests."""
        task, evaluator = create_rate_limiter_task()
        env = self._create_env(task.initial_files)

        # Cheating attempt: Agent overwrites test file with a fake passing test
        env.write_file(
            "tests/test_limiter.py",
            "import unittest\nclass Cheat(unittest.TestCase):\n    def test_cheat(self): pass\n",
        )
        res = evaluator.evaluate(env, [])
        # Evaluator restored pristine tests, so the buggy code still fails!
        self.assertFalse(res.passed)
        self.assertEqual(res.score, 0.0)

    def test_rate_limiter_reference_solution(self):
        task, evaluator = create_rate_limiter_task()
        env = self._create_env(task.initial_files)

        # Apply correct fix: continuous float token accumulation
        fixed_code = (
            "import time\n\n"
            "class TokenBucketLimiter:\n"
            "    def __init__(self, capacity: float, refill_rate: float):\n"
            "        self.capacity = float(capacity)\n"
            "        self.refill_rate = float(refill_rate)\n"
            "        self.tokens = float(capacity)\n"
            "        self.last_refill = time.time()\n\n"
            "    def allow_request(self, tokens: float = 1.0) -> bool:\n"
            "        now = time.time()\n"
            "        elapsed = now - self.last_refill\n"
            "        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)\n"
            "        self.last_refill = now\n"
            "        if self.tokens >= tokens:\n"
            "            self.tokens -= tokens\n"
            "            return True\n"
            "        return False\n"
        )
        env.write_file("src/limiter.py", fixed_code)
        res = evaluator.evaluate(env, [])
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 1.0)

    def test_circular_serializer_reference_solution(self):
        task, evaluator = create_circular_serializer_task()
        env = self._create_env(task.initial_files)

        fixed_code = (
            "from __future__ import annotations\n\n"
            "import dataclasses\n"
            "import datetime\n"
            "from typing import Any\n\n"
            "def serialize(obj: Any, _visited: set[int] | None = None) -> Any:\n"
            "    if _visited is None:\n"
            "        _visited = set()\n"
            "    if isinstance(obj, (int, float, str, bool)) or obj is None:\n"
            "        return obj\n"
            "    if isinstance(obj, (datetime.datetime, datetime.date)):\n"
            "        return obj.isoformat()\n"
            "    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):\n"
            "        obj = dataclasses.asdict(obj)\n"
            "    obj_id = id(obj)\n"
            "    if obj_id in _visited:\n"
            "        return {'$ref': '#'}\n"
            "    _visited.add(obj_id)\n"
            "    try:\n"
            "        if isinstance(obj, dict):\n"
            "            return {str(k): serialize(v, _visited.copy()) for k, v in obj.items()}\n"
            "        if isinstance(obj, (list, tuple, set)):\n"
            "            return [serialize(item, _visited.copy()) for item in obj]\n"
            "        if hasattr(obj, '__dict__'):\n"
            "            return {str(k): serialize(v, _visited.copy()) for k, v in obj.__dict__.items()}\n"
            "        return str(obj)\n"
            "    finally:\n"
            "        _visited.discard(obj_id)\n"
        )
        env.write_file("src/serializer.py", fixed_code)
        res = evaluator.evaluate(env, [])
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 1.0)

    def test_auth_middleware_reference_solution(self):
        task, evaluator = create_auth_middleware_task()
        env = self._create_env(task.initial_files)

        fixed_code = (
            "from __future__ import annotations\n\n"
            "class Request:\n"
            "    def __init__(self, headers: dict[str, str], path: str = '/'):\n"
            "        self.headers = headers\n"
            "        self.path = path\n\n"
            "    def get_header(self, name: str) -> str | None:\n"
            "        target = name.lower()\n"
            "        for k, v in self.headers.items():\n"
            "            if k.lower() == target:\n"
            "                return v\n"
            "        return None\n\n"
            "class AuthMiddleware:\n"
            "    def __init__(self, valid_tokens: set[str]):\n"
            "        self.valid_tokens = valid_tokens\n\n"
            "    def authenticate(self, request: Request) -> tuple[int, str]:\n"
            "        auth_header = request.get_header('Authorization')\n"
            "        if not auth_header:\n"
            "            return 401, 'Missing Authorization header'\n"
            "        parts = auth_header.split()\n"
            "        if len(parts) != 2 or parts[0] != 'Bearer':\n"
            "            return 401, 'Invalid Authorization header format'\n"
            "        token = parts[1]\n"
            "        if token in self.valid_tokens:\n"
            "            return 200, 'Authorized'\n"
            "        return 403, 'Forbidden'\n"
        )
        env.write_file("src/middleware.py", fixed_code)
        res = evaluator.evaluate(env, [])
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 1.0)

    def test_lru_ttl_cache_reference_solution(self):
        task, evaluator = create_lru_ttl_cache_task()
        env = self._create_env(task.initial_files)

        fixed_code = (
            "from __future__ import annotations\n\n"
            "import time\n"
            "from collections import OrderedDict\n"
            "from typing import Any\n\n"
            "class LRUTTLCache:\n"
            "    def __init__(self, capacity: int, default_ttl: float | None = None):\n"
            "        self.capacity = capacity\n"
            "        self.default_ttl = default_ttl\n"
            "        self.cache: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()\n\n"
            "    def get(self, key: str) -> Any | None:\n"
            "        if key not in self.cache:\n"
            "            return None\n"
            "        val, expiry = self.cache[key]\n"
            "        if expiry is not None and time.time() > expiry:\n"
            "            del self.cache[key]\n"
            "            return None\n"
            "        self.cache.move_to_end(key)\n"
            "        return val\n\n"
            "    def set(self, key: str, value: Any, ttl: float | None = None) -> None:\n"
            "        effective_ttl = ttl if ttl is not None else self.default_ttl\n"
            "        expiry = (time.time() + effective_ttl) if effective_ttl is not None else None\n"
            "        if key in self.cache:\n"
            "            self.cache[key] = (value, expiry)\n"
            "            self.cache.move_to_end(key)\n"
            "            return\n"
            "        if len(self.cache) >= self.capacity:\n"
            "            self.cache.popitem(last=False)\n"
            "        self.cache[key] = (value, expiry)\n"
        )
        env.write_file("src/cache.py", fixed_code)
        res = evaluator.evaluate(env, [])
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 1.0)

    def test_query_builder_reference_solution(self):
        task, evaluator = create_query_builder_task()
        env = self._create_env(task.initial_files)

        fixed_code = (
            "from __future__ import annotations\n\n"
            "from typing import Any\n\n"
            "class QueryBuilder:\n"
            "    def __init__(self, table: str):\n"
            "        self.table = table\n"
            "        self.conditions_and: list[str] = []\n"
            "        self.conditions_or: list[str] = []\n\n"
            "    def where(self, field: str, op: str, value: Any) -> QueryBuilder:\n"
            "        self.conditions_and.append(self._format_cond(field, op, value))\n"
            "        return self\n\n"
            "    def where_or(self, field: str, op: str, value: Any) -> QueryBuilder:\n"
            "        self.conditions_or.append(self._format_cond(field, op, value))\n"
            "        return self\n\n"
            "    def _format_cond(self, field: str, op: str, value: Any) -> str:\n"
            "        if isinstance(value, str):\n"
            '            escaped = value.replace("\'", "\'\'")\n'
            "            val_str = f\"'{escaped}'\"\n"
            "        elif value is None:\n"
            "            val_str = 'NULL'\n"
            "        else:\n"
            "            val_str = str(value)\n"
            '        return f"{field} {op} {val_str}"\n\n'
            "    def to_sql(self) -> str:\n"
            '        sql = f"SELECT * FROM {self.table}"\n'
            "        if not self.conditions_and and not self.conditions_or:\n"
            "            return sql\n"
            "        parts = []\n"
            "        if self.conditions_or:\n"
            "            or_clause = ' OR '.join(self.conditions_or)\n"
            "            if len(self.conditions_or) > 1 and self.conditions_and:\n"
            "                parts.append(f'({or_clause})')\n"
            "            else:\n"
            "                parts.append(or_clause)\n"
            "        if self.conditions_and:\n"
            "            parts.append(' AND '.join(self.conditions_and))\n"
            "        sql += ' WHERE ' + ' AND '.join(parts)\n"
            "        return sql\n"
        )
        env.write_file("src/query.py", fixed_code)
        res = evaluator.evaluate(env, [])
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 1.0)

    def test_run_benchmark_suite_with_mock_model(self):
        """End-to-end integration test of run_benchmark_suite runner."""
        fixed_limiter = (
            "import time\n\n"
            "class TokenBucketLimiter:\n"
            "    def __init__(self, capacity: float, refill_rate: float):\n"
            "        self.capacity = float(capacity)\n"
            "        self.refill_rate = float(refill_rate)\n"
            "        self.tokens = float(capacity)\n"
            "        self.last_refill = time.time()\n\n"
            "    def allow_request(self, tokens: float = 1.0) -> bool:\n"
            "        now = time.time()\n"
            "        elapsed = now - self.last_refill\n"
            "        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)\n"
            "        self.last_refill = now\n"
            "        if self.tokens >= tokens:\n"
            "            self.tokens -= tokens\n"
            "            return True\n"
            "        return False\n"
        )
        script = [
            (
                "Writing fix",
                [
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments={"path": "src/limiter.py", "content": fixed_limiter},
                    )
                ],
            ),
            ("Done", []),
        ]
        model = MockModel(script)
        task, evaluator = create_rate_limiter_task()

        def harness_factory(evaluator):
            return Harness(
                model=model,
                evaluator=evaluator,
                runs_dir=self.tmp_dir,
                verbose=False,
            )

        report = run_benchmark_suite(
            harness_factory=harness_factory,
            tasks=[(task, evaluator)],
            suite_name="Test-Mini-SWE",
            output_dir=self.tmp_dir,
            verbose=False,
        )

        self.assertEqual(report.total_tasks, 1)
        self.assertEqual(report.passed_tasks, 1)
        self.assertEqual(report.pass_rate, 100.0)
        self.assertEqual(report.mean_score, 1.0)

        # Verify exported report JSON
        report_file = Path(self.tmp_dir) / "test_mini_swe_report.json"
        self.assertTrue(report_file.is_file())
        data = json.loads(report_file.read_text(encoding="utf-8"))
        self.assertEqual(data["suite_name"], "Test-Mini-SWE")
        self.assertEqual(data["passed_tasks"], 1)

    def test_get_mini_swe_suite_filter(self):
        full_suite = get_mini_swe_suite()
        self.assertEqual(len(full_suite), 5)

        filtered = get_mini_swe_suite(task_filter="rate_limiter")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0][0].id, "mini_swe_rate_limiter")


if __name__ == "__main__":
    unittest.main()
