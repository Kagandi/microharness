from __future__ import annotations

from typing import Any, NamedTuple

from core.environment import Environment
from core.evaluator import Evaluator
from core.types import EvalResult, StepTrace, Task


class PristineTestEvaluator:
    """Evaluator that restores pristine test files before running verification to prevent cheating."""

    def __init__(
        self,
        command: str,
        pristine_files: dict[str, str],
        timeout: float = 30.0,
    ):
        self.command = command
        self.pristine_files = pristine_files
        self.timeout = timeout

    def evaluate(self, env: Environment, steps: list[StepTrace]) -> EvalResult:
        # Restore pristine test files so task cannot be bypassed by deleting or modifying tests
        for path, content in self.pristine_files.items():
            env.write_file(path, content)

        exit_code, output = env.execute_bash(self.command, timeout=self.timeout)
        passed = exit_code == 0
        status_msg = "passed" if passed else f"failed (exit {exit_code})"
        feedback = f"Verification `{self.command}` {status_msg}.\nOutput:\n{output.strip()}"
        return EvalResult(passed=passed, score=1.0 if passed else 0.0, feedback=feedback)


class MiniSWETaskDef(NamedTuple):
    id: str
    name: str
    description: str
    task_factory: Any


def create_rate_limiter_task() -> tuple[Task, Evaluator]:
    """Task 1: Token bucket rate limiter losing fractional refills under rapid polling."""
    limiter_code = (
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
        "        # BUG: Integer truncation and resetting last_refill on every call\n"
        "        # loses fractional replenishment under rapid polling!\n"
        "        added = int(elapsed * self.refill_rate)\n"
        "        self.tokens = min(self.capacity, self.tokens + added)\n"
        "        self.last_refill = now\n"
        "        if self.tokens >= tokens:\n"
        "            self.tokens -= tokens\n"
        "            return True\n"
        "        return False\n"
    )

    test_code = (
        "import time\n"
        "import unittest\n"
        "from src.limiter import TokenBucketLimiter\n\n"
        "class TestTokenBucketLimiter(unittest.TestCase):\n"
        "    def test_burst_within_capacity(self):\n"
        "        limiter = TokenBucketLimiter(capacity=3, refill_rate=1.0)\n"
        "        self.assertTrue(limiter.allow_request(1))\n"
        "        self.assertTrue(limiter.allow_request(1))\n"
        "        self.assertTrue(limiter.allow_request(1))\n"
        "        self.assertFalse(limiter.allow_request(1))\n\n"
        "    def test_rapid_polling_accumulates_fractional_refill(self):\n"
        "        # Empty bucket with refill_rate = 10 tokens/sec (0.1s per token)\n"
        "        limiter = TokenBucketLimiter(capacity=5, refill_rate=10.0)\n"
        "        self.assertTrue(limiter.allow_request(5))\n"
        "        self.assertFalse(limiter.allow_request(1))\n\n"
        "        # Rapid polling in intervals smaller than 0.1s\n"
        "        # In buggy implementation, elapsed*10 < 1 every time, so int(elapsed*10)=0\n"
        "        # and resetting last_refill=now repeatedly starves the bucket.\n"
        "        start = time.time()\n"
        "        while time.time() - start < 0.25:\n"
        "            limiter.allow_request(10)\n"
        "            time.sleep(0.02)\n\n"
        "        # After ~0.25s at 10 tokens/sec, bucket should have replenished at least 2 tokens\n"
        "        self.assertTrue(limiter.allow_request(1))\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )

    initial_files = {
        "src/limiter.py": limiter_code,
        "tests/test_limiter.py": test_code,
    }

    task = Task(
        id="mini_swe_rate_limiter",
        instruction=(
            "The token bucket rate limiter in `src/limiter.py` fails when polled rapidly. "
            "Run `python -m unittest discover -s tests -p test_limiter.py` to see the failure.\n\n"
            "Bug: `allow_request()` truncates fractional tokens with `int(elapsed * refill_rate)` "
            "and resets `last_refill = now`, which loses fractional replenishment and starves the bucket.\n\n"
            "Fix `src/limiter.py` so fractional tokens are accumulated continuously (as float) and "
            "all tests pass without modifying the test suite."
        ),
        initial_files=initial_files,
        max_steps=8,
        timeout_seconds=180.0,
    )

    evaluator = PristineTestEvaluator(
        command="python -m unittest discover -s tests -p test_limiter.py",
        pristine_files={"tests/test_limiter.py": test_code},
    )
    return task, evaluator


def create_circular_serializer_task() -> tuple[Task, Evaluator]:
    """Task 2: Serializer crashing on circular structures and missing datetime/dataclass support."""
    serializer_code = (
        "from __future__ import annotations\n\n"
        "import dataclasses\n"
        "import datetime\n"
        "from typing import Any\n\n"
        "def serialize(obj: Any) -> Any:\n"
        "    # BUG: No circular reference detection and missing datetime / dataclass support\n"
        "    if isinstance(obj, (int, float, str, bool)) or obj is None:\n"
        "        return obj\n"
        "    if isinstance(obj, dict):\n"
        "        return {str(k): serialize(v) for k, v in obj.items()}\n"
        "    if isinstance(obj, (list, tuple, set)):\n"
        "        return [serialize(item) for item in obj]\n"
        "    if hasattr(obj, '__dict__'):\n"
        "        return {str(k): serialize(v) for k, v in obj.__dict__.items()}\n"
        "    return str(obj)\n"
    )

    test_code = (
        "import dataclasses\n"
        "import datetime\n"
        "import unittest\n"
        "from src.serializer import serialize\n\n"
        "@dataclasses.dataclass\n"
        "class User:\n"
        "    id: int\n"
        "    name: str\n\n"
        "class TestSerializer(unittest.TestCase):\n"
        "    def test_primitives_and_nested(self):\n"
        "        data = {'a': 1, 'b': [2, 3, {'c': True}]}\n"
        "        self.assertEqual(serialize(data), data)\n\n"
        "    def test_datetime_and_dataclass(self):\n"
        "        dt = datetime.datetime(2026, 9, 25, 12, 0, 0)\n"
        "        user = User(id=1, name='Alice')\n"
        "        result = serialize({'user': user, 'created_at': dt})\n"
        "        self.assertEqual(result['created_at'], dt.isoformat())\n"
        "        self.assertEqual(result['user'], {'id': 1, 'name': 'Alice'})\n\n"
        "    def test_circular_reference_prevention(self):\n"
        "        # Circular dictionary reference\n"
        "        cyclic: dict = {'name': 'root'}\n"
        "        cyclic['self'] = cyclic\n"
        "        res = serialize(cyclic)\n"
        "        self.assertEqual(res['name'], 'root')\n"
        "        self.assertEqual(res['self'], {'$ref': '#'})\n\n"
        "        # Circular list reference\n"
        "        cycle_list: list = [1, 2]\n"
        "        cycle_list.append(cycle_list)\n"
        "        res_list = serialize(cycle_list)\n"
        "        self.assertEqual(res_list[0], 1)\n"
        "        self.assertEqual(res_list[1], 2)\n"
        "        self.assertEqual(res_list[2], {'$ref': '#'})\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )

    initial_files = {
        "src/serializer.py": serializer_code,
        "tests/test_serializer.py": test_code,
    }

    task = Task(
        id="mini_swe_circular_serializer",
        instruction=(
            "The serializer in `src/serializer.py` crashes on circular references and misses datetime/dataclass support. "
            "Run `python -m unittest discover -s tests -p test_serializer.py` to see the failures.\n\n"
            "Requirements:\n"
            "1. Prevent infinite recursion by tracking visited object IDs in the active recursion path. "
            'When a cycle is detected, return `{"$ref": "#"}`.\n'
            "2. Convert `datetime.datetime` and `datetime.date` objects into their `.isoformat()` string.\n"
            "3. Support dataclasses by converting them to dicts.\n"
            "4. Make all tests pass without modifying `tests/test_serializer.py`."
        ),
        initial_files=initial_files,
        max_steps=8,
        timeout_seconds=180.0,
    )

    evaluator = PristineTestEvaluator(
        command="python -m unittest discover -s tests -p test_serializer.py",
        pristine_files={"tests/test_serializer.py": test_code},
    )
    return task, evaluator


def create_auth_middleware_task() -> tuple[Task, Evaluator]:
    """Task 3: HTTP Auth middleware failing case-insensitive headers and whitespace."""
    middleware_code = (
        "from __future__ import annotations\n\n"
        "class Request:\n"
        "    def __init__(self, headers: dict[str, str], path: str = '/'):\n"
        "        self.headers = headers\n"
        "        self.path = path\n\n"
        "class AuthMiddleware:\n"
        "    def __init__(self, valid_tokens: set[str]):\n"
        "        self.valid_tokens = valid_tokens\n\n"
        "    def authenticate(self, request: Request) -> tuple[int, str]:\n"
        "        # BUG: Case-sensitive lookup fails when client sends 'authorization'\n"
        "        auth_header = request.headers.get('Authorization')\n"
        "        if not auth_header:\n"
        "            return 401, 'Missing Authorization header'\n\n"
        "        # BUG: Exact split on single space fails on multiple whitespace or malformed scheme\n"
        "        parts = auth_header.split(' ')\n"
        "        if len(parts) != 2 or parts[0] != 'Bearer':\n"
        "            return 401, 'Invalid Authorization header format'\n\n"
        "        token = parts[1]\n"
        "        if token in self.valid_tokens:\n"
        "            return 200, 'Authorized'\n"
        "        return 403, 'Forbidden'\n"
    )

    test_code = (
        "import unittest\n"
        "from src.middleware import AuthMiddleware, Request\n\n"
        "class TestAuthMiddleware(unittest.TestCase):\n"
        "    def setUp(self):\n"
        "        self.middleware = AuthMiddleware(valid_tokens={'secret-token-123'})\n\n"
        "    def test_standard_bearer_auth(self):\n"
        "        req = Request(headers={'Authorization': 'Bearer secret-token-123'})\n"
        "        code, msg = self.middleware.authenticate(req)\n"
        "        self.assertEqual(code, 200)\n"
        "        self.assertEqual(msg, 'Authorized')\n\n"
        "    def test_case_insensitive_header_keys(self):\n"
        "        # HTTP/2 sends headers lowercase\n"
        "        req1 = Request(headers={'authorization': 'Bearer secret-token-123'})\n"
        "        code1, msg1 = self.middleware.authenticate(req1)\n"
        "        self.assertEqual(code1, 200)\n\n"
        "        req2 = Request(headers={'AUTHORIZATION': 'Bearer secret-token-123'})\n"
        "        code2, _ = self.middleware.authenticate(req2)\n"
        "        self.assertEqual(code2, 200)\n\n"
        "    def test_whitespace_tolerance_and_malformed_headers(self):\n"
        "        # Extra spaces between Bearer and token\n"
        "        req1 = Request(headers={'Authorization': 'Bearer    secret-token-123'})\n"
        "        code1, msg1 = self.middleware.authenticate(req1)\n"
        "        self.assertEqual(code1, 200)\n\n"
        "        # Malformed scheme or missing token\n"
        "        req2 = Request(headers={'Authorization': 'Basic dXNlcjpwYXNz'})\n"
        "        code2, _ = self.middleware.authenticate(req2)\n"
        "        self.assertEqual(code2, 401)\n\n"
        "        req3 = Request(headers={'Authorization': 'Bearer'})\n"
        "        code3, _ = self.middleware.authenticate(req3)\n"
        "        self.assertEqual(code3, 401)\n\n"
        "        # Invalid token returns 403\n"
        "        req4 = Request(headers={'Authorization': 'Bearer wrong-token'})\n"
        "        code4, msg4 = self.middleware.authenticate(req4)\n"
        "        self.assertEqual(code4, 403)\n"
        "        self.assertEqual(msg4, 'Forbidden')\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )

    initial_files = {
        "src/middleware.py": middleware_code,
        "tests/test_auth.py": test_code,
    }

    task = Task(
        id="mini_swe_auth_middleware",
        instruction=(
            "The authentication middleware in `src/middleware.py` fails on lowercase HTTP/2 headers and whitespace variations. "
            "Run `python -m unittest discover -s tests -p test_auth.py` to see the failures.\n\n"
            "Requirements:\n"
            "1. Header lookup must be case-insensitive (e.g. `authorization`, `Authorization`, `AUTHORIZATION`).\n"
            "2. Token extraction must tolerate arbitrary whitespace between `Bearer` and the token.\n"
            "3. Malformed schemes (non-Bearer) or missing tokens must return 401 without crashing.\n"
            "4. Ensure all tests in `tests/test_auth.py` pass without modifying the test file."
        ),
        initial_files=initial_files,
        max_steps=8,
        timeout_seconds=180.0,
    )

    evaluator = PristineTestEvaluator(
        command="python -m unittest discover -s tests -p test_auth.py",
        pristine_files={"tests/test_auth.py": test_code},
    )
    return task, evaluator


def create_lru_ttl_cache_task() -> tuple[Task, Evaluator]:
    """Task 4: LRU Cache with TTL bug on updating existing keys."""
    cache_code = (
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
        "        expiry = (time.time() + effective_ttl) if effective_ttl is not None else None\n\n"
        "        if key in self.cache:\n"
        "            # BUG: Updates value, but forgets to move_to_end AND keeps the old expiry!\n"
        "            old_val, old_exp = self.cache[key]\n"
        "            self.cache[key] = (value, old_exp)\n"
        "            return\n\n"
        "        if len(self.cache) >= self.capacity:\n"
        "            self.cache.popitem(last=False)\n\n"
        "        self.cache[key] = (value, expiry)\n"
    )

    test_code = (
        "import time\n"
        "import unittest\n"
        "from src.cache import LRUTTLCache\n\n"
        "class TestLRUTTLCache(unittest.TestCase):\n"
        "    def test_basic_capacity_eviction(self):\n"
        "        cache = LRUTTLCache(capacity=2)\n"
        "        cache.set('a', 1)\n"
        "        cache.set('b', 2)\n"
        "        self.assertEqual(cache.get('a'), 1)  # marks 'a' as MRU\n"
        "        cache.set('c', 3)  # should evict 'b'\n"
        "        self.assertIsNone(cache.get('b'))\n"
        "        self.assertEqual(cache.get('a'), 1)\n"
        "        self.assertEqual(cache.get('c'), 3)\n\n"
        "    def test_updating_existing_key_refreshes_lru_order(self):\n"
        "        cache = LRUTTLCache(capacity=2)\n"
        "        cache.set('a', 1)\n"
        "        cache.set('b', 2)\n"
        "        # Update 'a' -> should make 'a' MRU, so 'b' is the next victim\n"
        "        cache.set('a', 10)\n"
        "        cache.set('c', 3)  # should evict 'b', NOT 'a'\n"
        "        self.assertEqual(cache.get('a'), 10)\n"
        "        self.assertIsNone(cache.get('b'))\n"
        "        self.assertEqual(cache.get('c'), 3)\n\n"
        "    def test_updating_existing_key_refreshes_ttl(self):\n"
        "        cache = LRUTTLCache(capacity=2, default_ttl=0.1)\n"
        "        cache.set('x', 100)\n"
        "        time.sleep(0.06)\n"
        "        # Update 'x' with fresh ttl of 0.2s\n"
        "        cache.set('x', 200, ttl=0.2)\n"
        "        time.sleep(0.06)  # 0.12s total: original ttl would have expired\n"
        "        self.assertEqual(cache.get('x'), 200)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )

    initial_files = {
        "src/cache.py": cache_code,
        "tests/test_cache.py": test_code,
    }

    task = Task(
        id="mini_swe_lru_ttl_cache",
        instruction=(
            "The `LRUTTLCache` in `src/cache.py` has an eviction and TTL bug when overwriting existing keys. "
            "Run `python -m unittest discover -s tests -p test_cache.py` to see the failures.\n\n"
            "Bug: In `set(key, value, ttl=None)`, when updating an existing key, the cache updates the value "
            "but forgets to refresh the LRU order (`move_to_end`) and preserves the old expiration timestamp.\n\n"
            "Fix `src/cache.py` so updating an existing key updates its value, refreshes its TTL expiry, "
            "and marks it as the most recently used (MRU) item. Ensure all tests in `tests/test_cache.py` pass."
        ),
        initial_files=initial_files,
        max_steps=8,
        timeout_seconds=180.0,
    )

    evaluator = PristineTestEvaluator(
        command="python -m unittest discover -s tests -p test_cache.py",
        pristine_files={"tests/test_cache.py": test_code},
    )
    return task, evaluator


def create_query_builder_task() -> tuple[Task, Evaluator]:
    """Task 5: SQL AST query builder with operator precedence and escaping bugs."""
    query_code = (
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
        "            # BUG: Unescaped single quotes\n"
        "            val_str = f\"'{value}'\"\n"
        "        elif value is None:\n"
        "            val_str = 'NULL'\n"
        "        else:\n"
        "            val_str = str(value)\n"
        '        return f"{field} {op} {val_str}"\n\n'
        "    def to_sql(self) -> str:\n"
        '        sql = f"SELECT * FROM {self.table}"\n'
        "        if not self.conditions_and and not self.conditions_or:\n"
        "            return sql\n\n"
        "        # BUG: Missing parenthesis grouping for OR conditions\n"
        "        parts = []\n"
        "        if self.conditions_or:\n"
        "            parts.append(' OR '.join(self.conditions_or))\n"
        "        if self.conditions_and:\n"
        "            parts.append(' AND '.join(self.conditions_and))\n"
        "        sql += ' WHERE ' + ' AND '.join(parts)\n"
        "        return sql\n"
    )

    test_code = (
        "import unittest\n"
        "from src.query import QueryBuilder\n\n"
        "class TestQueryBuilder(unittest.TestCase):\n"
        "    def test_simple_and_conditions(self):\n"
        "        qb = QueryBuilder('users').where('age', '>=', 18).where('active', '=', 1)\n"
        "        self.assertEqual(\n"
        "            qb.to_sql(),\n"
        "            'SELECT * FROM users WHERE age >= 18 AND active = 1',\n"
        "        )\n\n"
        "    def test_string_single_quote_escaping(self):\n"
        "        # String with single quote e.g. O'Connor should be escaped as 'O''Connor'\n"
        "        qb = QueryBuilder('authors').where('name', '=', \"O'Connor\")\n"
        "        self.assertEqual(\n"
        "            qb.to_sql(),\n"
        "            \"SELECT * FROM authors WHERE name = 'O''Connor'\",\n"
        "        )\n\n"
        "    def test_or_conditions_grouped_with_parentheses(self):\n"
        "        # When combining OR and AND conditions, OR branches must be wrapped in parentheses\n"
        "        qb = (\n"
        "            QueryBuilder('orders')\n"
        "            .where_or('status', '=', 'pending')\n"
        "            .where_or('status', '=', 'processing')\n"
        "            .where('amount', '>', 100)\n"
        "        )\n"
        "        expected = \"SELECT * FROM orders WHERE (status = 'pending' OR status = 'processing') AND amount > 100\"\n"
        "        self.assertEqual(qb.to_sql(), expected)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )

    initial_files = {
        "src/query.py": query_code,
        "tests/test_query.py": test_code,
    }

    task = Task(
        id="mini_swe_query_builder",
        instruction=(
            "The SQL query builder in `src/query.py` generates invalid SQL when escaping single quotes "
            "and causes precedence errors when combining `OR` and `AND` filters. "
            "Run `python -m unittest discover -s tests -p test_query.py` to see the failures.\n\n"
            "Requirements:\n"
            "1. Escape single quotes in string values by doubling them (e.g. `O'Connor` -> `'O''Connor'`).\n"
            "2. When `where_or()` conditions exist alongside `where()` conditions, wrap the grouped OR clauses "
            "in parentheses (e.g. `(status = 'pending' OR status = 'processing') AND amount > 100`).\n"
            "3. Ensure all tests in `tests/test_query.py` pass without modifying the test file."
        ),
        initial_files=initial_files,
        max_steps=8,
        timeout_seconds=180.0,
    )

    evaluator = PristineTestEvaluator(
        command="python -m unittest discover -s tests -p test_query.py",
        pristine_files={"tests/test_query.py": test_code},
    )
    return task, evaluator


MINI_SWE_TASKS: list[MiniSWETaskDef] = [
    MiniSWETaskDef(
        "rate_limiter",
        "Token Bucket Rate Limiter",
        "Fractional token accumulation under polling",
        create_rate_limiter_task,
    ),
    MiniSWETaskDef(
        "circular_serializer",
        "Circular JSON Serializer",
        "Cycle detection & ISO-8601 formatting",
        create_circular_serializer_task,
    ),
    MiniSWETaskDef(
        "auth_middleware",
        "HTTP Auth Middleware",
        "Case-insensitive headers & whitespace tolerance",
        create_auth_middleware_task,
    ),
    MiniSWETaskDef(
        "lru_ttl_cache",
        "LRU Cache with TTL",
        "Overwriting keys refreshes MRU order and TTL",
        create_lru_ttl_cache_task,
    ),
    MiniSWETaskDef(
        "query_builder",
        "SQL Query Builder AST",
        "Condition precedence grouping & quote escaping",
        create_query_builder_task,
    ),
]


def get_mini_swe_suite(task_filter: str | None = None) -> list[tuple[Task, Evaluator]]:
    """Returns list of (Task, Evaluator) instances for the Mini-SWE benchmark suite."""
    suite: list[tuple[Task, Evaluator]] = []
    for item in MINI_SWE_TASKS:
        if task_filter:
            tf = task_filter.lower().strip()
            allowed = {item.id.lower(), f"mini_swe_{item.id}".lower(), item.name.lower()}
            if tf not in allowed:
                continue
        task, evaluator = item.task_factory()
        suite.append((task, evaluator))
    return suite
