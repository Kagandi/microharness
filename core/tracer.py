from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

from core.session import (
    find_recent_sessions,
    load_session_messages,
    reconstruct_messages_from_steps,
)
from core.types import EvalResult, StepTrace, Task
from core.usage import summarize_run_usage

__all__ = [
    "ActionFormatter",
    "Tracer",
    "find_recent_sessions",
    "load_session_messages",
    "reconstruct_messages_from_steps",
    "register_action_formatter",
    "unregister_action_formatter",
]


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if _supports_color() else text


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _supports_color() else text


def _red(text: str) -> str:
    return f"\033[31m{text}\033[0m" if _supports_color() else text


def _format_bash_action(args: Any) -> str:
    if isinstance(args, dict):
        cmd = str(args.get("command", "")).strip()
    else:
        cmd = str(args).strip()
    return f"$ {cmd}"


def _format_read_file_action(args: Any) -> str:
    if not isinstance(args, dict):
        return f"● Read {args}"
    path = args.get("path", "")
    start = args.get("start_line")
    end = args.get("end_line")
    lines_info = f" (lines {start}-{end})" if start and end else ""
    return f"● Read {path}{lines_info}"


def _format_write_file_action(args: Any) -> str:
    if not isinstance(args, dict):
        return f"● Write {args}"
    path = args.get("path", "")
    content = args.get("content", "")
    lines = len(content.splitlines()) if isinstance(content, str) else 0
    return f"● Write {path} ({lines} lines)"


def _format_edit_file_action(args: Any) -> str:
    if not isinstance(args, dict):
        return f"● Edit {args}"
    path = args.get("path", "")
    return f"● Edit {path}"


ActionFormatter = Callable[[Any], str]
# The global registry of tool summaries, mutable via register/unregister.
_ACTION_FORMATTERS: dict[str, ActionFormatter] = {
    "bash": _format_bash_action,
    "read_file": _format_read_file_action,
    "write_file": _format_write_file_action,
    "edit_file": _format_edit_file_action,
}


def register_action_formatter(tool_name: str, formatter: ActionFormatter) -> None:
    """Registers a global custom action summary formatter for a tool."""
    _ACTION_FORMATTERS[tool_name] = formatter


def unregister_action_formatter(tool_name: str) -> None:
    """Unregisters a global action summary formatter for a tool."""
    _ACTION_FORMATTERS.pop(tool_name, None)


def _format_action_summary(
    name: str,
    args: Any,
    color: bool = True,
    formatters: dict[str, ActionFormatter] | None = None,
) -> str:
    """Formats a tool call concisely using registered formatters or a generic fallback."""
    active_formatters = formatters if formatters is not None else _ACTION_FORMATTERS
    formatter = active_formatters.get(name) or _ACTION_FORMATTERS.get(name)
    text: str | None = None
    if formatter:
        try:
            text = formatter(args)
        except Exception as exc:
            warnings.warn(
                f"Action formatter for tool '{name}' raised exception: {exc}. Falling back to default format.",
                stacklevel=2,
            )
            text = None

    if text is None:
        if isinstance(args, dict):
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            text = f"● {name}({args_str})"
        else:
            text = f"● {name}({args})"

    return _bold(text) if color else text


PREVIEW_MAX_CHARS = 90
ERROR_MAX_CHARS = 70
MAX_PREVIEW_LINES = 15


def _preview(text: str, limit: int = PREVIEW_MAX_CHARS) -> str:
    """Truncates text to a single line of at most ``limit`` characters."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    return f"{line[:limit]}..." if len(line) > limit else line


def _error_excerpt(result: dict[str, Any] | None, limit: int = ERROR_MAX_CHARS) -> str:
    """First line of a failed tool's output, bounded for display."""
    lines = str((result or {}).get("output", "")).strip().splitlines()
    return _preview(lines[0] if lines else "Failed", limit)


def _format_call(
    call: dict[str, Any] | None,
    formatters: dict[str, ActionFormatter] | None = None,
    color: bool = True,
) -> str:
    """Formats a tool call, tolerating a missing call (an unpaired result)."""
    call = call or {}
    return _format_action_summary(
        call.get("name", ""), call.get("arguments", {}), color=color, formatters=formatters
    )


def _token_suffix(usage: dict[str, Any], bracketed: bool = False) -> str:
    """Trailing token counts for a step header, empty when nothing was reported.

    The console divider parenthesizes the cache figures while the Markdown report
    brackets them, so ``bracketed`` selects the delimiter.
    """
    if not usage:
        return ""
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    if prompt <= 0 and completion <= 0:
        return ""
    cached = int(usage.get("cached_tokens") or 0)
    if cached <= 0:
        return f" · {prompt:,} in, {completion:,} out"

    rate = float(usage.get("cache_hit_rate_pct") or 0.0)
    open_delim, close_delim = ("[", "]") if bracketed else ("(", ")")
    return (
        f" · {prompt:,} in {open_delim}{cached:,} cached - {rate:.1f}%{close_delim}, "
        f"{completion:,} out"
    )


def _pair_calls_with_results(
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    """Pairs tool calls with corresponding results, first by call_id, then by index."""
    results_by_id: dict[str, dict[str, Any]] = {
        tr["call_id"]: tr for tr in tool_results if tr.get("call_id")
    }
    remaining_results = list(tool_results)
    paired_results: list[dict[str, Any] | None] = [None] * len(tool_calls)

    # 1. Match by call_id
    for i, tc in enumerate(tool_calls):
        call_id = tc.get("id")
        if call_id and call_id in results_by_id:
            tr = results_by_id.pop(call_id)
            remaining_results.remove(tr)
            paired_results[i] = tr

    # 2. Match remaining unmatched calls with remaining results in order
    for i in range(len(tool_calls)):
        if paired_results[i] is None and remaining_results:
            paired_results[i] = remaining_results.pop(0)

    pairs: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = [
        (tc, paired_results[i]) for i, tc in enumerate(tool_calls)
    ]

    # 3. Any leftover results without a corresponding tool call
    for tr in remaining_results:
        pairs.append((None, tr))

    return pairs


class Tracer:
    """Collects telemetry during the agent run and exports structured traces."""

    def __init__(
        self,
        task: Task,
        output_dir: Path,
        workspace: Path | str | None = None,
        verbose: bool = True,
        compact: bool = False,
        formatters: dict[str, ActionFormatter] | None = None,
    ):
        self.task = task
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workspace = str(Path(workspace).resolve()) if workspace else None
        self.jsonl_path = self.output_dir / "trajectory.jsonl"
        self.verbose = verbose
        self.compact = compact
        self.formatters: dict[str, ActionFormatter] = {
            **_ACTION_FORMATTERS,
            **(formatters or {}),
        }
        self.steps: list[StepTrace] = []
        self.eval_result: EvalResult | None = None
        self.finish_reason: str = "running"
        self.patch_diff: str | None = None
        self.telemetry: dict[str, Any] = {}

    def log(self, text: str) -> None:
        if self.verbose:
            print(text)

    def _append_jsonl(self, step: StepTrace) -> None:
        try:
            record = {
                "step_index": step.step_index,
                "duration_s": step.duration_s,
                "compaction_applied": step.compaction_applied,
                "usage": step.usage,
                "prompt_messages": step.prompt_messages,
                "model_response": step.model_response,
                "tool_results": step.tool_results,
            }
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
                f.flush()
        except Exception:
            try:
                # Fallback: preserve step_index so crash recovery never has index holes
                fallback = {
                    "step_index": step.step_index,
                    "duration_s": getattr(step, "duration_s", 0.0),
                    "compaction_applied": getattr(step, "compaction_applied", False),
                    "usage": {},
                    "prompt_messages": [],
                    "model_response": {
                        "content": str(step.model_response.get("content", ""))
                        if isinstance(getattr(step, "model_response", None), dict)
                        else ""
                    },
                    "tool_results": [],
                    "serialization_error": True,
                }
                with self.jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(fallback, default=str) + "\n")
                    f.flush()
            except Exception:
                pass

    def record_step(self, step: StepTrace) -> None:
        """Persists a completed step, then renders it when verbose output is on."""
        self.steps.append(step)
        self._append_jsonl(step)

        if not self.verbose:
            return
        if self.compact:
            self._render_compact_step(step)
        else:
            self._render_verbose_step(step)

    def _render_compact_step(self, step: StepTrace) -> None:
        """One clean line per action, so an interactive session stays readable."""
        text = step.model_response.get("content") or ""
        tool_calls = step.model_response.get("tool_calls") or []

        if not tool_calls:
            if text.strip():
                print(f"\n{text.strip()}\n")
            return

        if text.strip():
            print(_dim(f"● Thinking: {_preview(text)}"))
        for call, result in _pair_calls_with_results(tool_calls, step.tool_results):
            summary = _format_call(call, self.formatters)
            exit_code = (result or {}).get("exit_code", 0)
            if exit_code == 0:
                print(f"{summary} " + _dim(f"({step.duration_s:.1f}s)"))
            else:
                print(f"{summary} " + _red(f"❌ (exit {exit_code}: {_preview(_error_excerpt(result))})"))
        if step.compaction_applied:
            print(_dim("🧹 [Context compacted to conserve budget]"))

    def _render_verbose_step(self, step: StepTrace) -> None:
        """Full step detail: divider, token counts, model text, and raw tool output."""
        text = step.model_response.get("content") or ""
        tool_calls = step.model_response.get("tool_calls") or []

        divider = _dim(f"─── Step {step.step_index} · {step.duration_s:.2f}s{_token_suffix(step.usage)} " + "─" * 25)
        print(f"\n{divider}")

        if text:
            print(f"{text.strip()}\n")
        if step.compaction_applied:
            print(_dim("🧹 [Memory Compaction]: Pruned older observations to preserve context.\n"))

        for call, result in _pair_calls_with_results(tool_calls, step.tool_results):
            if call:
                print(_format_call(call, self.formatters))
            if result:
                self._print_tool_output(result)

    def _print_tool_output(self, tr: dict[str, Any]) -> None:
        exit_code = tr.get("exit_code", 0)
        if exit_code != 0:
            print(_red(f"  ❌ (exit {exit_code})"))

        output = tr.get("output", "").strip()
        if not output:
            return

        lines = output.splitlines()
        for line in lines[:MAX_PREVIEW_LINES]:
            print(f"  {line}")
        if len(lines) > MAX_PREVIEW_LINES:
            print(_dim(f"  ... ({len(lines) - MAX_PREVIEW_LINES} more lines)"))

    def _compute_telemetry(self) -> dict[str, Any]:
        """Derives run totals from step usage, reusing the shared token normalizer."""
        usage = summarize_run_usage((s.usage for s in self.steps))
        return {
            "total_prompt_tokens": usage.prompt_tokens,
            "total_completion_tokens": usage.completion_tokens,
            "total_cached_tokens": usage.cached_tokens,
            "cache_hit_rate_pct": usage.cache_hit_rate_pct,
        }

    def finalize(
        self,
        finish_reason: str,
        eval_result: EvalResult | None = None,
        patch_diff: str | None = None,
        telemetry: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        self.finish_reason = finish_reason
        self.eval_result = eval_result
        self.patch_diff = patch_diff
        self.telemetry = telemetry if isinstance(telemetry, dict) else self._compute_telemetry()

        json_path = self.output_dir / "trace.json"
        md_path = self.output_dir / "trace.md"

        self._export_json(json_path)
        self._export_markdown(md_path)

        return json_path, md_path

    def _reconstruct_final_messages(self) -> list[dict[str, Any]]:
        return reconstruct_messages_from_steps(
            [
                {
                    "step_index": s.step_index,
                    "prompt_messages": s.prompt_messages,
                    "model_response": s.model_response,
                    "tool_results": s.tool_results,
                }
                for s in self.steps
            ]
        )

    def _export_json(self, path: Path) -> None:
        data = {
            "task": {
                "id": self.task.id,
                "instruction": self.task.instruction,
                "max_steps": self.task.max_steps,
            },
            "workspace": self.workspace,
            "finish_reason": self.finish_reason,
            "total_steps": len(self.steps),
            "telemetry": self.telemetry,
            "patch_diff": self.patch_diff,
            "eval_result": {
                "passed": self.eval_result.passed if self.eval_result else None,
                "score": self.eval_result.score if self.eval_result else None,
                "feedback": self.eval_result.feedback if self.eval_result else None,
            },
            "messages": self._reconstruct_final_messages(),
            "steps": [
                {
                    "step_index": s.step_index,
                    "duration_s": s.duration_s,
                    "compaction_applied": s.compaction_applied,
                    "usage": s.usage,
                    "prompt_messages": s.prompt_messages,
                    "model_response": s.model_response,
                    "tool_results": s.tool_results,
                }
                for s in self.steps
            ],
        }
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def _export_markdown(self, path: Path) -> None:
        """Writes the human-readable trace report."""
        sections = [
            self._markdown_header(),
            self._markdown_task(),
            self._markdown_steps(),
            self._markdown_patch(),
        ]
        path.write_text("\n".join(section for section in sections if section), encoding="utf-8")

    def _markdown_header(self) -> str:
        """Run summary: status, step count, tokens, and evaluation verdict."""
        lines = [
            f"# Run Trace: Task {self.task.id}",
            f"- **Status**: `{self.finish_reason}`",
            f"- **Total Steps**: {len(self.steps)}",
        ]
        prompt = int(self.telemetry.get("total_prompt_tokens") or 0)
        completion = int(self.telemetry.get("total_completion_tokens") or 0)
        cached = int(self.telemetry.get("total_cached_tokens") or 0)
        if self.telemetry and (prompt or completion):
            cached_tag = (
                f" ({cached:,} cached - {self.telemetry.get('cache_hit_rate_pct', 0.0)}% hit rate)"
                if cached
                else ""
            )
            lines.append(f"- **Tokens**: {prompt:,} in{cached_tag}, {completion:,} out")

        if self.eval_result:
            verdict = "PASSED" if self.eval_result.passed else "FAILED"
            lines.append(f"- **Evaluation**: `{verdict}` (Score: {self.eval_result.score})")
            lines.append(f"- **Feedback**: {self.eval_result.feedback}")
        return "\n".join(lines)

    def _markdown_task(self) -> str:
        """The instruction the run was given."""
        return f"## Task Instruction\n```\n{self.task.instruction}\n```\n"

    def _markdown_steps(self) -> str:
        """One section per step: header, compaction note, actions, and outputs."""
        sections = ["## Execution Steps"]
        sections += [self._markdown_step(step) for step in self.steps]
        return "\n".join(section for section in sections if section)

    def _markdown_step(self, step: StepTrace) -> str:
        header = (
            f"### Step {step.step_index} "
            f"({step.duration_s:.2f}s{_token_suffix(step.usage, bracketed=True)})"
        )
        parts = [header]
        if step.compaction_applied:
            parts.append("> ℹ️ *Context Compaction was applied during this step.*")
        if content := step.model_response.get("content"):
            parts.append(content.strip())

        tool_calls = step.model_response.get("tool_calls") or []
        for call, result in _pair_calls_with_results(tool_calls, step.tool_results):
            if call:
                parts.append(self._markdown_action(call))
            if result:
                parts.append(self._markdown_output(result))
        return "\n\n".join(parts)

    def _markdown_action(self, call: dict[str, Any]) -> str:
        """Renders a tool call, giving bash its own fenced block and others a summary line."""
        name = call.get("name", "")
        args = call.get("arguments", {})
        if name == "bash":
            command = args.get("command", "").strip() if isinstance(args, dict) else str(args)
            return f"**Command** (`bash`):\n```bash\n{command}\n```"
        if name in _ACTION_FORMATTERS:
            return f"**Action** (`{name}`): `{_format_action_summary(name, args, color=False)}`"
        return f"**Action**: `{name}`\n```json\n{json.dumps(args, indent=2, default=str)}\n```"

    def _markdown_output(self, result: dict[str, Any]) -> str:
        truncated = " (Truncated)" if result.get("was_truncated") else ""
        exit_code = result.get("exit_code", 0)
        output = str(result.get("output", "") or "").strip()
        return f"**Output** (exit={exit_code}){truncated}:\n```\n{output}\n```"

    def _markdown_patch(self) -> str:
        """The unified diff of everything the run changed, when there is one."""
        if not self.patch_diff:
            return ""
        return f"## Git Patch Diff\n```diff\n{self.patch_diff}\n```\n"
