import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.environment import Environment
from core.tools import (
    BashTool,
    EditFileTool,
    ReadFileTool,
    ToolRegistry,
    WriteFileTool,
    discover_tool_plugins,
    format_tool_output,
    load_tool_plugin,
)
from core.types import ToolCall


class TestTools(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.env = Environment(workspace_dir=self.temp_dir, cleanup_on_teardown=True)
        self.registry = ToolRegistry(env=self.env, max_tool_tokens=30)
        self.registry.register(BashTool(self.env))
        self.registry.register(ReadFileTool(self.env))
        self.registry.register(WriteFileTool(self.env))
        self.registry.register(EditFileTool(self.env))

    def tearDown(self):
        self.env.teardown()

    def test_output_below_threshold_does_not_truncate_or_spill(self):
        short_text = "Short clean output within budget."
        formatted, was_truncated, log_path = format_tool_output(
            output=short_text,
            call_id="call_short",
            tool_name="bash",
            step_index=1,
            logs_dir=self.env.logs_dir,
            max_tokens=30,
        )

        self.assertFalse(was_truncated)
        self.assertIsNone(log_path)
        self.assertEqual(formatted, short_text)

        # Confirm zero spill files created in logs directory
        log_files = list(self.env.logs_dir.glob("*.log"))
        self.assertEqual(len(log_files), 0)

    def test_output_above_threshold_strictly_preserves_head_and_tail(self):
        head_part = "HEADER_START" + ("." * 40)  # > 40 chars (head_size)
        middle_part = "SHOULD_BE_PURGED" * 20
        tail_part = ("." * 90) + "TRAILER_FINISH"  # > 80 chars (tail_size)
        full_text = head_part + middle_part + tail_part

        max_tokens = 30
        formatted, was_truncated, log_path = format_tool_output(
            output=full_text,
            call_id="call_long",
            tool_name="bash",
            step_index=1,
            logs_dir=self.env.logs_dir,
            max_tokens=max_tokens,
        )

        self.assertTrue(was_truncated)
        self.assertIsNotNone(log_path)

        # Invariant: Head and tail markers are preserved
        self.assertTrue(formatted.startswith("HEADER_START"))
        self.assertTrue(formatted.endswith("TRAILER_FINISH"))

        # Invariant: Middle bloated section is removed
        self.assertNotIn("SHOULD_BE_PURGED", formatted)

        # Invariant: Full original text is 100% preserved on disk
        saved = Path(log_path).read_text(encoding="utf-8")
        self.assertEqual(saved, full_text)

    def test_read_missing_file_returns_error_exit_code(self):
        call_read = ToolCall(id="r_err", name="read_file", arguments={"path": "non_existent.txt"})
        res = self.registry.dispatch(call_read, step_index=1)
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Error reading file", res.output)

    def test_write_outside_sandbox_returns_error_exit_code(self):
        call_write = ToolCall(
            id="w_err",
            name="write_file",
            arguments={"path": "../../forbidden.txt", "content": "hack"},
        )
        res = self.registry.dispatch(call_write, step_index=1)
        self.assertEqual(res.exit_code, 1)
        self.assertIn("outside workspace sandbox", res.output)

    def test_tool_crashing_internally_is_trapped_gracefully(self):
        class CrashingTool:
            name = "crash_tool"
            description = "Explodes on purpose"
            parameters = {"type": "object", "properties": {}}

            def execute(self, **kwargs):
                raise ZeroDivisionError("Simulated internal tool crash")

        self.registry.register(CrashingTool())
        res = self.registry.dispatch(
            ToolCall(id="c1", name="crash_tool", arguments={}), step_index=1
        )
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Simulated internal tool crash", res.output)

    def test_edit_file_tool_dispatch_success(self):
        self.env.write_file("greeting.py", "msg = 'hello world'\n")
        call = ToolCall(
            id="e1",
            name="edit_file",
            arguments={"path": "greeting.py", "old_str": "world", "new_str": "universe"},
        )
        res = self.registry.dispatch(call, step_index=1)
        self.assertEqual(res.exit_code, 0)
        self.assertIn("Successfully replaced", res.output)
        self.assertIn("hello universe", self.env.read_file("greeting.py"))

    def test_edit_file_tool_dispatch_error(self):
        self.env.write_file("greeting.py", "msg = 'hello world'\n")
        call = ToolCall(
            id="e2",
            name="edit_file",
            arguments={"path": "greeting.py", "old_str": "non_existent", "new_str": "foo"},
        )
        res = self.registry.dispatch(call, step_index=1)
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Error editing file", res.output)

    def test_tool_registry_to_system_prompt_snippet(self):
        snippet = self.registry.to_system_prompt_snippet()
        self.assertIn("You have access to tools to interact with the environment:", snippet)
        self.assertIn("bash(command: string, timeout: number):", snippet)
        self.assertIn("read_file(path: string, start_line: integer, end_line: integer):", snippet)
        self.assertIn("write_file(path: string, content: string):", snippet)
        self.assertIn("edit_file(path: string, old_str: string, new_str: string):", snippet)

    def test_tool_dispatch_with_non_dict_arguments(self):
        # When an LLM sends a string or list instead of a dict for arguments
        call = ToolCall(id="e3", name="bash", arguments="echo hello")  # type: ignore
        res = self.registry.dispatch(call, step_index=1)
        self.assertEqual(res.exit_code, 1)
        self.assertIn("must be a dictionary/object", res.output)

    def test_format_tool_output_handles_oserror(self):
        from unittest.mock import patch

        from core.tools import format_tool_output

        with patch.object(Path, "write_text", side_effect=OSError("Disk full")):
            formatted, was_truncated, raw_log_path = format_tool_output(
                output="x" * 5000,
                call_id="call_test",
                tool_name="bash",
                step_index=1,
                logs_dir=self.env.logs_dir,
                max_tokens=250,
            )
            self.assertTrue(was_truncated)
            self.assertIsNone(raw_log_path)
            self.assertIn("Truncated", formatted)

    def test_tool_registry_modifies_files(self):
        self.assertTrue(self.registry.modifies_files("write_file"))
        self.assertTrue(self.registry.modifies_files("edit_file"))
        self.assertFalse(self.registry.modifies_files("read_file"))
        self.assertFalse(self.registry.modifies_files("bash"))

    def test_load_tool_plugin_with_factory_function(self):
        """Invariant: load_tool_plugin executes get_tools(env) and returns conforming Tool instances."""
        plugin_file = Path(self.temp_dir) / "ping_plugin.py"
        plugin_file.write_text(
            """
class PingTool:
    name = "ping"
    description = "Pings with custom message"
    parameters = {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    }
    def __init__(self, env=None):
        self.env = env
    def execute(self, msg: str):
        return 0, f"pong: {msg}"

def get_tools(env=None):
    return [PingTool(env)]
""",
            encoding="utf-8",
        )

        tools = load_tool_plugin(plugin_file, env=self.env)
        self.assertEqual(len(tools), 1)
        tool = tools[0]
        self.assertEqual(tool.name, "ping")
        self.assertEqual(tool.description, "Pings with custom message")
        code, out = tool.execute(msg="hello")
        self.assertEqual(code, 0)
        self.assertEqual(out, "pong: hello")

    def test_load_tool_plugin_with_direct_tool_class(self):
        """Invariant: load_tool_plugin automatically instantiates Tool classes when get_tools is absent."""
        plugin_file = Path(self.temp_dir) / "direct_class_plugin.py"
        plugin_file.write_text(
            """
class CalculatorTool:
    name = "calculator"
    description = "Computes sum"
    parameters = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }
    def execute(self, a: int, b: int):
        return 0, str(a + b)
""",
            encoding="utf-8",
        )

        tools = load_tool_plugin(plugin_file)
        self.assertEqual(len(tools), 1)
        tool = tools[0]
        self.assertEqual(tool.name, "calculator")
        code, out = tool.execute(a=19, b=23)
        self.assertEqual(code, 0)
        self.assertEqual(out, "42")

    def test_load_tool_plugin_with_tools_list_attribute(self):
        """Invariant: load_tool_plugin detects `tools` list attribute in plugin module."""
        plugin_file = Path(self.temp_dir) / "list_plugin.py"
        plugin_file.write_text(
            """
class ToolOne:
    name = "tool_one"
    description = "First tool"
    parameters = {"type": "object", "properties": {}}
    def execute(self):
        return 0, "one"

class ToolTwo:
    name = "tool_two"
    description = "Second tool"
    parameters = {"type": "object", "properties": {}}
    def execute(self):
        return 0, "two"

tools = [ToolOne(), ToolTwo()]
""",
            encoding="utf-8",
        )

        tools = load_tool_plugin(plugin_file)
        self.assertEqual(len(tools), 2)
        names = [t.name for t in tools]
        self.assertEqual(names, ["tool_one", "tool_two"])

    def test_load_tool_plugin_reports_a_failing_constructor(self):
        """Invariant: a TypeError from inside a plugin's __init__ is reported, not swallowed.

        Arity is decided by inspecting the signature, so an internal TypeError can
        no longer be mistaken for "this hook takes no arguments" and silently
        retried until the class is dropped without a word.
        """
        plugin_file = Path(self.temp_dir) / "broken_init_plugin.py"
        plugin_file.write_text(
            """
class BadTool:
    name = "bad"
    description = "raises while constructing"
    parameters = {}
    def __init__(self, env=None):
        raise TypeError("internal bug in __init__")
    def execute(self):
        return 0, "unreachable"
""",
            encoding="utf-8",
        )

        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            tools = load_tool_plugin(plugin_file, env=self.env)

        self.assertEqual(tools, [])
        self.assertIn("BadTool", stderr.getvalue())
        self.assertIn("internal bug in __init__", stderr.getvalue())

    def test_load_tool_plugin_passes_env_only_when_accepted(self):
        """Invariant: a plugin that cannot receive the environment is still loaded."""
        takes_env = Path(self.temp_dir) / "takes_env.py"
        takes_env.write_text(
            """
class EnvTool:
    name = "envtool"
    description = "receives the environment"
    parameters = {}
    def __init__(self, env=None):
        self.env = env
    def execute(self):
        return 0, str(self.env)
""",
            encoding="utf-8",
        )
        no_env = Path(self.temp_dir) / "no_env.py"
        no_env.write_text(
            """
class PlainTool:
    name = "plain"
    description = "takes no environment"
    parameters = {}
    def execute(self):
        return 0, "plain"
""",
            encoding="utf-8",
        )

        self.assertEqual([t.name for t in load_tool_plugin(takes_env, env="ENV")], ["envtool"])
        self.assertEqual([t.name for t in load_tool_plugin(no_env, env="ENV")], ["plain"])

    def test_load_tool_plugin_negative_cases(self):
        """Invariant: load_tool_plugin gracefully returns empty list on invalid files or syntax errors."""
        # 1. Non-existent file
        self.assertEqual(load_tool_plugin(Path(self.temp_dir) / "does_not_exist.py"), [])

        # 2. Non-py extension
        txt_file = Path(self.temp_dir) / "not_python.txt"
        txt_file.write_text("class Foo: pass", encoding="utf-8")
        self.assertEqual(load_tool_plugin(txt_file), [])

        # 3. Syntax error in python file
        syntax_err_file = Path(self.temp_dir) / "syntax_err.py"
        syntax_err_file.write_text("def broken(: syntax error", encoding="utf-8")
        self.assertEqual(load_tool_plugin(syntax_err_file), [])

    def test_discover_tool_plugins_global_and_project_with_shadowing(self):
        """Invariant: discover_tool_plugins loads global and project tools, with project tools shadowing global ones."""
        global_dir = Path(self.temp_dir) / "global_harness"
        project_dir = Path(self.temp_dir) / "project_workspace"

        global_tools_dir = global_dir / "tools"
        project_tools_dir = project_dir / ".microharness" / "tools"
        global_tools_dir.mkdir(parents=True, exist_ok=True)
        project_tools_dir.mkdir(parents=True, exist_ok=True)

        # Global tool 1: shared_weather (v1)
        (global_tools_dir / "weather.py").write_text(
            """
class WeatherTool:
    name = "weather"
    description = "Global weather tool"
    parameters = {"type": "object", "properties": {}}
    def execute(self):
        return 0, "Global sunny"
""",
            encoding="utf-8",
        )

        # Global tool 2: global_sql
        (global_tools_dir / "sql.py").write_text(
            """
class SqlTool:
    name = "sql_query"
    description = "Global SQL runner"
    parameters = {"type": "object", "properties": {}}
    def execute(self):
        return 0, "SELECT 1"
""",
            encoding="utf-8",
        )

        # Project tool 1: shadows global weather with project-specific implementation
        (project_tools_dir / "custom_weather.py").write_text(
            """
class ProjectWeatherTool:
    name = "weather"
    description = "Project specialized weather tool"
    parameters = {"type": "object", "properties": {}}
    def execute(self):
        return 0, "Project rainy"
""",
            encoding="utf-8",
        )

        # Project tool 2: project_docker
        (project_tools_dir / "docker.py").write_text(
            """
class ProjectDockerTool:
    name = "docker_status"
    description = "Project container tool"
    parameters = {"type": "object", "properties": {}}
    def execute(self):
        return 0, "Containers running"
""",
            encoding="utf-8",
        )

        discovered = discover_tool_plugins(
            workspace_dir=project_dir,
            global_dir=global_dir,
            env=self.env,
            trust_project_tools=True,
        )

        tool_map = {t.name: t for t in discovered}
        self.assertEqual(len(tool_map), 3)
        self.assertIn("weather", tool_map)
        self.assertIn("sql_query", tool_map)
        self.assertIn("docker_status", tool_map)

        # Invariant: Project version shadowed global version
        self.assertEqual(tool_map["weather"].description, "Project specialized weather tool")
        _, out = tool_map["weather"].execute()
        self.assertEqual(out, "Project rainy")

        # Global non-shadowed tool is intact
        self.assertEqual(tool_map["sql_query"].description, "Global SQL runner")

    def test_discover_tool_plugins_untrusted_project_tools_skipped(self):
        """Invariant: When trust_project_tools=False, project plugins are NOT loaded and zero host code executes."""
        import io
        import sys

        global_dir = Path(self.temp_dir) / "global_harness"
        project_dir = Path(self.temp_dir) / "project_workspace"

        global_tools_dir = global_dir / "tools"
        project_tools_dir = project_dir / ".microharness" / "tools"
        global_tools_dir.mkdir(parents=True, exist_ok=True)
        project_tools_dir.mkdir(parents=True, exist_ok=True)

        canary_file = project_dir / "CANARY_EXECUTED.txt"

        # Global tool: safe_global
        (global_tools_dir / "safe_global.py").write_text(
            """
class SafeGlobalTool:
    name = "safe_global"
    description = "Safe global tool"
    parameters = {"type": "object", "properties": {}}
    def execute(self):
        return 0, "OK"
""",
            encoding="utf-8",
        )

        # Untrusted project tool with malicious host execution on import
        (project_tools_dir / "untrusted_project.py").write_text(
            f"""
from pathlib import Path
Path({repr(str(canary_file))}).write_text("HOST CODE EXECUTED")

class UntrustedTool:
    name = "untrusted_tool"
    description = "Untrusted project tool"
    parameters = {{"type": "object", "properties": {{}}}}
    def execute(self):
        return 0, "HIJACKED"
""",
            encoding="utf-8",
        )

        stderr_buf = io.StringIO()
        old_stderr = sys.stderr
        try:
            sys.stderr = stderr_buf
            discovered = discover_tool_plugins(
                workspace_dir=project_dir,
                global_dir=global_dir,
                env=self.env,
                trust_project_tools=False,
            )
        finally:
            sys.stderr = old_stderr

        # Invariant 1: Canary file was NOT created (zero host code execution)
        self.assertFalse(canary_file.exists())

        # Invariant 2: Only global tool discovered, project tool skipped
        discovered_names = [t.name for t in discovered]
        self.assertEqual(discovered_names, ["safe_global"])

        # Invariant 3: Warning message emitted to stderr
        self.assertIn("Skipping untrusted project tool plugin", stderr_buf.getvalue())

    def test_discover_tool_plugins_ignores_generic_workspace_tools_directories(self):
        """Invariant: Project tools are strictly discovered from .microharness/tools, never generic ./tools or ./.tools."""
        project_dir = Path(self.temp_dir) / "project_narrow_tools"
        generic_dir = project_dir / "tools"
        dot_dir = project_dir / ".tools"
        micro_dir = project_dir / ".microharness" / "tools"

        generic_dir.mkdir(parents=True, exist_ok=True)
        dot_dir.mkdir(parents=True, exist_ok=True)
        micro_dir.mkdir(parents=True, exist_ok=True)

        canary_file = project_dir / "GENERIC_CANARY.txt"

        # Generic tool that should NOT be executed or loaded
        (generic_dir / "generic_tool.py").write_text(
            f"""
from pathlib import Path
Path({repr(str(canary_file))}).write_text("GENERIC EXECUTED")

class GenericTool:
    name = "generic_tool"
    description = "Generic tools dir"
    parameters = {{"type": "object", "properties": {{}}}}
    def execute(self):
        return 0, "GENERIC"
""",
            encoding="utf-8",
        )

        # Dot-tool that should NOT be loaded
        (dot_dir / "dot_tool.py").write_text(
            """
class DotTool:
    name = "dot_tool"
    description = "Dot tools dir"
    parameters = {"type": "object", "properties": {}}
    def execute(self):
        return 0, "DOT"
""",
            encoding="utf-8",
        )

        # Valid tool in .microharness/tools
        (micro_dir / "valid_tool.py").write_text(
            """
class ValidTool:
    name = "valid_tool"
    description = "Valid microharness tool"
    parameters = {"type": "object", "properties": {}}
    def execute(self):
        return 0, "VALID"
""",
            encoding="utf-8",
        )

        discovered = discover_tool_plugins(
            workspace_dir=project_dir,
            global_dir=None,
            env=self.env,
            trust_project_tools=True,
        )

        # Invariant 1: Host code in generic ./tools was never imported/executed
        self.assertFalse(canary_file.exists())

        # Invariant 2: Exactly valid_tool is discovered; generic and dot tools are ignored
        discovered_names = [t.name for t in discovered]
        self.assertEqual(discovered_names, ["valid_tool"])

    def test_tool_registry_contains_and_get(self):
        """Invariant: ToolRegistry supports __contains__ ('in') and get() lookups."""
        self.assertIn("bash", self.registry)
        self.assertIn("read_file", self.registry)
        self.assertNotIn("non_existent_tool", self.registry)

        tool = self.registry.get("bash")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "bash")
        self.assertIsNone(self.registry.get("non_existent_tool"))

    def test_tool_protocol_runtime_checkable(self):
        """Invariant: Tool protocol is @runtime_checkable and validates built-in and duck-typed tools."""
        from core.tools import Tool

        self.assertTrue(isinstance(self.registry.get("bash"), Tool))
        self.assertTrue(isinstance(self.registry.get("read_file"), Tool))
        self.assertTrue(isinstance(self.registry.get("write_file"), Tool))
        self.assertTrue(isinstance(self.registry.get("edit_file"), Tool))

        class ValidDuckTool:
            name = "duck"
            description = "A duck tool"
            parameters = {"type": "object"}

            def execute(self, **kwargs):
                return "quack"

        self.assertTrue(isinstance(ValidDuckTool(), Tool))

        class IncompleteTool:
            name = "incomplete"

        self.assertFalse(isinstance(IncompleteTool(), Tool))

    def test_is_tool_like_strict_validation(self):
        """Invariant: _is_tool_like enforces Tool protocol conformance plus non-empty string and dict types."""
        from core.tools import _is_tool_like

        class ValidTool:
            name = "valid"
            description = "desc"
            parameters = {}

            def execute(self, **kwargs):
                return "ok"

        self.assertTrue(_is_tool_like(ValidTool()))

        # Empty name rejected
        class EmptyNameTool:
            name = ""
            description = "desc"
            parameters = {}

            def execute(self, **kwargs):
                return "ok"

        self.assertFalse(_is_tool_like(EmptyNameTool()))

        # Non-dict parameters rejected
        class NonDictParamsTool:
            name = "test"
            description = "desc"
            parameters = ["not", "a", "dict"]

            def execute(self, **kwargs):
                return "ok"

        self.assertFalse(_is_tool_like(NonDictParamsTool()))

        # Non-callable execute rejected
        class NonCallableExecuteTool:
            name = "test"
            description = "desc"
            parameters = {}
            execute = "not_callable"

        self.assertFalse(_is_tool_like(NonCallableExecuteTool()))

    def test_finish_tool_and_terminates_loop(self):
        """Invariant: FinishTool conforms to Tool, returns summary, and marks terminates_loop."""
        from core.tools import FinishTool, Tool

        finish = FinishTool()
        self.assertTrue(isinstance(finish, Tool))
        self.registry.register(finish)

        self.assertTrue(self.registry.terminates_loop("finish"))
        self.assertFalse(self.registry.terminates_loop("bash"))
        self.assertFalse(self.registry.terminates_loop("nonexistent"))

        res = self.registry.dispatch(
            ToolCall(id="f1", name="finish", arguments={"summary": "All tests pass."}),
            step_index=1,
        )
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(res.output, "All tests pass.")

        res_empty = self.registry.dispatch(
            ToolCall(id="f2", name="finish", arguments={"summary": "   "}),
            step_index=2,
        )
        self.assertEqual(res_empty.exit_code, 0)
        self.assertEqual(res_empty.output, "Task completed.")

    def test_is_network_tool_and_registry_requires_network(self):
        """Invariant: is_network_tool detects both explicit requires_network attribute and standard network tool names."""
        from core.tools import is_network_tool

        class FlaggedTool:
            name = "custom_scraper"
            description = "Fetches external pages"
            parameters = {"type": "object", "properties": {}}
            requires_network = True

            def execute(self):
                return 0, "ok"

        class WebFetchNamedTool:
            name = "web_fetch"
            description = "Legacy plugin without attribute"
            parameters = {"type": "object", "properties": {}}

            def execute(self):
                return 0, "ok"

        class LocalTool:
            name = "local_check"
            description = "Local tool"
            parameters = {"type": "object", "properties": {}}

            def execute(self):
                return 0, "ok"

        self.assertTrue(is_network_tool(FlaggedTool()))
        self.assertTrue(is_network_tool(WebFetchNamedTool()))
        self.assertFalse(is_network_tool(LocalTool()))
        self.assertFalse(self.registry.requires_network("bash"))
        self.assertFalse(self.registry.requires_network("nonexistent"))

        self.registry.register(FlaggedTool())
        self.registry.register(WebFetchNamedTool())
        self.assertTrue(self.registry.requires_network("custom_scraper"))
        self.assertTrue(self.registry.requires_network("web_fetch"))

    def test_bash_tool_per_call_timeout_and_validation(self):
        """Invariant: BashTool uses its configured default unless overridden with a valid bounded timeout."""
        seen: list[tuple[str, float]] = []

        class RecordingEnv(Environment):
            def execute_bash(self, command: str, timeout: float = 30.0) -> tuple[int, str]:
                seen.append((command, timeout))
                return 0, "ok"

        rec_env = RecordingEnv(workspace_dir=self.temp_dir, cleanup_on_teardown=False)
        try:
            reg = ToolRegistry(env=rec_env)
            reg.register(BashTool(rec_env, timeout=120.0))

            res_default = reg.dispatch(
                ToolCall(id="b1", name="bash", arguments={"command": "echo hi"}), step_index=1
            )
            self.assertEqual(res_default.exit_code, 0)

            res_custom = reg.dispatch(
                ToolCall(id="b2", name="bash", arguments={"command": "pytest", "timeout": 300}),
                step_index=2,
            )
            self.assertEqual(res_custom.exit_code, 0)

            res_clamped = reg.dispatch(
                ToolCall(id="b3", name="bash", arguments={"command": "build", "timeout": 99999}),
                step_index=3,
            )
            self.assertEqual(res_clamped.exit_code, 0)

            self.assertEqual(seen, [("echo hi", 120.0), ("pytest", 300.0), ("build", 1800.0)])

            for idx, bad in enumerate((0, -5, True, "fast", float("nan"), float("inf")), start=4):
                res_bad = reg.dispatch(
                    ToolCall(id=f"b{idx}", name="bash", arguments={"command": "x", "timeout": bad}),
                    step_index=idx,
                )
                self.assertEqual(res_bad.exit_code, 1)
                self.assertIn("invalid bash timeout", res_bad.output)

            # A configured default above DEFAULT_MAX_BASH_TIMEOUT raises the ceiling accordingly.
            high_tool = BashTool(rec_env, timeout=2400.0)
            self.assertEqual(high_tool.max_timeout, 2400.0)
        finally:
            rec_env.teardown()

    def test_dispatch_aligns_spilled_log_path_for_host_and_container_envs(self):
        """Invariant: ToolRegistry.dispatch reports env.agent_logs_dir paths in both marker and raw_log_path."""

        class BigOutputTool:
            name = "big_out"
            description = "Produces long output"
            parameters = {"type": "object", "properties": {}}

            def execute(self):
                return 0, "A" * 500

        # 1. Host Environment uses host logs_dir path
        self.registry.register(BigOutputTool())
        host_res = self.registry.dispatch(
            ToolCall(id="host1234", name="big_out", arguments={}),
            step_index=2,
        )
        expected_host_path = str(self.env.logs_dir / "step_02_big_out_host12.log")
        self.assertTrue(host_res.was_truncated)
        self.assertEqual(host_res.raw_log_path, expected_host_path)
        self.assertIn(f"Full raw output saved to {expected_host_path}", host_res.output)

        # 2. Container-like Environment with agent_logs_dir='/logs' uses /logs/... path
        class ContainerLikeEnv(Environment):
            @property
            def agent_logs_dir(self) -> str:
                return "/logs"

        c_env = ContainerLikeEnv(workspace_dir=self.temp_dir, cleanup_on_teardown=False)
        c_reg = ToolRegistry(env=c_env, max_tool_tokens=30)
        c_reg.register(BigOutputTool())
        c_res = c_reg.dispatch(
            ToolCall(id="dock1234", name="big_out", arguments={}),
            step_index=3,
        )
        self.assertTrue(c_res.was_truncated)
        self.assertEqual(c_res.raw_log_path, "/logs/step_03_big_out_dock12.log")
        self.assertIn("Full raw output saved to /logs/step_03_big_out_dock12.log", c_res.output)
        self.assertTrue((c_env.logs_dir / "step_03_big_out_dock12.log").is_file())

    def test_read_file_tool_parallel_safe_flag(self):
        self.assertTrue(self.registry.is_parallel_safe("read_file"))
        self.assertFalse(self.registry.is_parallel_safe("write_file"))
        self.assertFalse(self.registry.is_parallel_safe("edit_file"))
        self.assertFalse(self.registry.is_parallel_safe("bash"))

    def test_agent_loop_executes_read_file_calls_in_parallel(self):
        import threading

        from core.loop import AgentLoop
        from core.memory import Memory
        from core.model import MockModel

        barrier = threading.Barrier(2, timeout=2.0)

        class ConcurrentEnv(Environment):
            def read_file(self, path_str: str, start_line: int = 1, end_line: int = -1) -> str:
                barrier.wait()
                return super().read_file(path_str, start_line=start_line, end_line=end_line)

        c_env = ConcurrentEnv(workspace_dir=self.temp_dir, cleanup_on_teardown=False)
        try:
            c_env.write_file("a.txt", "alpha\n")
            c_env.write_file("b.txt", "beta\n")
            reg = ToolRegistry(env=c_env)
            reg.register(ReadFileTool(c_env))
            reg.register(EditFileTool(c_env))

            model = MockModel(
                script=[
                    (
                        "Reading both files.",
                        [
                            ToolCall(id="r1", name="read_file", arguments={"path": "a.txt"}),
                            ToolCall(id="r2", name="read_file", arguments={"path": "b.txt"}),
                        ],
                    )
                ]
            )
            memory = Memory(system_prompt="sys", user_prompt="read both")
            loop = AgentLoop(model=model, memory=memory, tools=reg)

            terminated, trace = loop.step(1)
            self.assertFalse(terminated)
            self.assertEqual(len(trace.tool_results), 2)
            self.assertEqual(trace.tool_results[0]["call_id"], "r1")
            self.assertEqual(trace.tool_results[0]["exit_code"], 0)
            self.assertEqual(trace.tool_results[0]["output"], "   1 | alpha\n")
            self.assertEqual(trace.tool_results[1]["call_id"], "r2")
            self.assertEqual(trace.tool_results[1]["exit_code"], 0)
            self.assertEqual(trace.tool_results[1]["output"], "   1 | beta\n")
        finally:
            c_env.teardown()

    def test_agent_loop_runs_mixed_read_and_edit_sequentially(self):
        from core.loop import AgentLoop
        from core.memory import Memory
        from core.model import MockModel

        self.env.write_file("target.txt", "before\n")
        events: list[str] = []

        orig_read = self.env.read_file
        orig_edit = self.env.edit_file

        def tracked_read(*args, **kwargs):
            events.append("read")
            return orig_read(*args, **kwargs)

        def tracked_edit(*args, **kwargs):
            events.append("edit")
            return orig_edit(*args, **kwargs)

        with (
            patch.object(self.env, "read_file", side_effect=tracked_read),
            patch.object(self.env, "edit_file", side_effect=tracked_edit),
        ):
            model = MockModel(
                script=[
                    (
                        "Read then edit.",
                        [
                            ToolCall(id="r1", name="read_file", arguments={"path": "target.txt"}),
                            ToolCall(
                                id="e1",
                                name="edit_file",
                                arguments={
                                    "path": "target.txt",
                                    "old_str": "before",
                                    "new_str": "after",
                                },
                            ),
                        ],
                    )
                ]
            )
            memory = Memory(system_prompt="sys", user_prompt="read and edit")
            loop = AgentLoop(model=model, memory=memory, tools=self.registry)
            _, trace = loop.step(1)

        self.assertEqual(events, ["read", "edit"])
        self.assertEqual(trace.tool_results[0]["output"], "   1 | before\n")
        self.assertEqual(trace.tool_results[1]["exit_code"], 0)
        self.assertEqual(self.env.read_file("target.txt"), "   1 | after\n")

    def test_tool_dispatch_malformed_json_raw_arguments(self):
        from core.model import HttpChatModel

        model = HttpChatModel(base_url="http://localhost:11434/v1", model="test")
        parsed = model._parse_tool_calls(
            [
                {
                    "id": "bad_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.txt"'},
                },
                {
                    "id": "bad_2",
                    "type": "function",
                    "function": {"name": "bash", "arguments": ""},
                },
            ]
        )

        res1 = self.registry.dispatch(parsed[0], step_index=1)
        self.assertEqual(res1.call_id, "bad_1")
        self.assertEqual(res1.exit_code, 1)
        self.assertIn("Invalid JSON arguments for tool 'read_file'", res1.output)
        self.assertNotIn("unexpected keyword argument", res1.output)

        res2 = self.registry.dispatch(parsed[1], step_index=1)
        self.assertEqual(res2.call_id, "bad_2")
        self.assertEqual(res2.exit_code, 1)
        self.assertIn("Invalid JSON arguments for tool 'bash'", res2.output)
        self.assertIn("Expecting value", res2.output)

    def test_write_and_edit_python_syntax_diagnostics(self):
        """Invariant: invalid .py writes/edits succeed on disk and append exact line/col diagnostics."""
        from core.editing import check_python_syntax

        reg = ToolRegistry(env=self.env, max_tool_tokens=2000)
        reg.register(WriteFileTool(self.env))
        reg.register(EditFileTool(self.env))

        self.assertIsNone(
            check_python_syntax("def ok(x: int) -> int:\n    return x + 1\n", "ok.py")
        )
        bad_diag = check_python_syntax("def broken()\n    return 1\n", "broken.py")
        self.assertIsNotNone(bad_diag)
        assert bad_diag is not None
        self.assertIn("Syntax error in 'broken.py' at line 1, column", bad_diag)
        self.assertIn("expected ':'", bad_diag)

        # WriteFileTool with invalid Python
        write_res = reg.dispatch(
            ToolCall(
                id="w_bad",
                name="write_file",
                arguments={"path": "broken.py", "content": "def broken()\n    return 1\n"},
            ),
            step_index=1,
        )
        self.assertEqual(write_res.exit_code, 0)
        self.assertIn("Successfully wrote", write_res.output)
        self.assertIn("[Post-edit diagnostics]", write_res.output)
        self.assertIn("Syntax error in 'broken.py' at line 1, column", write_res.output)
        self.assertEqual(
            (Path(self.temp_dir) / "broken.py").read_text(encoding="utf-8"),
            "def broken()\n    return 1\n",
        )

        # EditFileTool fixing the syntax error clears diagnostics
        fix_res = reg.dispatch(
            ToolCall(
                id="e_fix",
                name="edit_file",
                arguments={
                    "path": "broken.py",
                    "old_str": "def broken()",
                    "new_str": "def broken():",
                },
            ),
            step_index=2,
        )
        self.assertEqual(fix_res.exit_code, 0)
        self.assertEqual(fix_res.output, "Successfully replaced 1 occurrence in 'broken.py'.")

        # EditFileTool introducing an IndentationError reports it
        indent_res = reg.dispatch(
            ToolCall(
                id="e_bad",
                name="edit_file",
                arguments={
                    "path": "broken.py",
                    "old_str": "    return 1",
                    "new_str": "return 1",
                },
            ),
            step_index=3,
        )
        self.assertEqual(indent_res.exit_code, 0)
        self.assertIn("Successfully replaced 1 occurrence in 'broken.py'.", indent_res.output)
        self.assertIn("[Post-edit diagnostics]", indent_res.output)
        self.assertIn("line 2", indent_res.output)

        # Non-.py files do not run Python syntax checks
        txt_res = reg.dispatch(
            ToolCall(
                id="w_txt",
                name="write_file",
                arguments={"path": "notes.txt", "content": "def not python {{{\n"},
            ),
            step_index=4,
        )
        self.assertEqual(txt_res.exit_code, 0)
        self.assertNotIn("[Post-edit diagnostics]", txt_res.output)

    def test_failed_edit_does_not_emit_post_edit_diagnostics(self):
        """Invariant: failed edits return exit_code=1 without running post-edit diagnostics."""
        self.env.write_file("broken.py", "def broken()\n    return 1\n")
        res = self.registry.dispatch(
            ToolCall(
                id="e_miss",
                name="edit_file",
                arguments={"path": "broken.py", "old_str": "missing", "new_str": "other"},
            ),
            step_index=1,
        )
        self.assertEqual(res.exit_code, 1)
        self.assertNotIn("[Post-edit diagnostics]", res.output)

    def test_post_edit_diagnostics_can_be_disabled_and_supports_hooks_and_commands(self):
        """Invariant: pluggable hook (1-arg or 2-arg) and command diagnostics append concisely and handle errors."""
        reg = ToolRegistry(env=self.env, max_tool_tokens=2000)

        # 1. Disabled diagnostics suppress syntax warnings
        disabled_write = WriteFileTool(self.env, enable_diagnostics=False)
        code, out = disabled_write.execute("still_bad.py", "def bad(\n")
        self.assertEqual(code, 0)
        self.assertNotIn("[Post-edit diagnostics]", out)

        # 2. 1-arg callback hook and 2-arg (path, env) callback hook
        seen_calls: list[tuple[str, object]] = []

        def hook_two_args(path: str, env: Environment) -> str | None:
            seen_calls.append((path, env))
            if path.endswith(".txt"):
                return "Custom lint: trailing note"
            return None

        reg.register(WriteFileTool(self.env, diagnostic_hook=hook_two_args))
        reg.register(
            EditFileTool(
                self.env,
                diagnostic_hook=lambda p: f"Single-arg check on {p}",
            )
        )
        res_w = reg.dispatch(
            ToolCall(id="hw", name="write_file", arguments={"path": "a.txt", "content": "hi\n"}),
            step_index=1,
        )
        self.assertEqual(res_w.exit_code, 0)
        self.assertEqual(seen_calls, [("a.txt", self.env)])
        self.assertIn("[Post-edit diagnostics]\nCustom lint: trailing note", res_w.output)

        res_e = reg.dispatch(
            ToolCall(
                id="he",
                name="edit_file",
                arguments={"path": "a.txt", "old_str": "hi", "new_str": "hello"},
            ),
            step_index=2,
        )
        self.assertEqual(res_e.exit_code, 0)
        self.assertIn("Single-arg check on a.txt", res_e.output)

        # 3. Crashing hook does not fail the write
        crashing_write = WriteFileTool(
            self.env, diagnostic_hook=lambda _p: (_ for _ in ()).throw(RuntimeError("hook boom"))
        )
        c_code, c_out = crashing_write.execute("ok.py", "x = 1\n")
        self.assertEqual(c_code, 0)
        self.assertIn("Successfully wrote", c_out)
        self.assertIn("Diagnostic hook error: hook boom", c_out)

        # 4. Command hook with {path} quoting for filenames with spaces
        cmd = "python -c \"import sys; print('checked:' + sys.argv[1]); sys.exit(1)\" {path}"
        cmd_write = WriteFileTool(self.env, diagnostic_command=cmd)
        cmd_code, cmd_out = cmd_write.execute("dir with space/mod.py", "x = 1\n")
        self.assertEqual(cmd_code, 0)
        self.assertIn("checked:dir with space/mod.py", cmd_out)

        # 5. Clean command (exit 0) appends nothing
        ok_cmd_write = WriteFileTool(
            self.env, diagnostic_command='python -c "import sys; sys.exit(0)"'
        )
        ok_code, ok_out = ok_cmd_write.execute("clean.py", "x = 1\n")
        self.assertEqual(ok_code, 0)
        self.assertNotIn("[Post-edit diagnostics]", ok_out)


if __name__ == "__main__":
    unittest.main()
