import tempfile
import unittest
from pathlib import Path

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
        self.registry = ToolRegistry(env=self.env, max_tool_chars=120)
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
            max_chars=120,
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

        max_chars = 120
        formatted, was_truncated, log_path = format_tool_output(
            output=full_text,
            call_id="call_long",
            tool_name="bash",
            step_index=1,
            logs_dir=self.env.logs_dir,
            max_chars=max_chars,
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
        self.assertIn("bash(command: string):", snippet)
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
                max_chars=1000,
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


if __name__ == "__main__":
    unittest.main()
