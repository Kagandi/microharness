from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

from core.types import EvalResult, StepTrace, Task


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


ActionFormatter = Callable[[Any], str]
_ACTION_FORMATTERS: dict[str, ActionFormatter] = {}


def register_action_formatter(tool_name: str, formatter: ActionFormatter) -> None:
    """Registers a custom action summary formatter for a tool."""
    _ACTION_FORMATTERS[tool_name] = formatter


def unregister_action_formatter(tool_name: str) -> None:
    """Unregisters an action summary formatter for a tool."""
    _ACTION_FORMATTERS.pop(tool_name, None)


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


register_action_formatter("bash", _format_bash_action)
register_action_formatter("read_file", _format_read_file_action)
register_action_formatter("write_file", _format_write_file_action)
register_action_formatter("edit_file", _format_edit_file_action)


def _format_action_summary(name: str, args: Any, color: bool = True) -> str:
    """Formats a tool call concisely using registered formatters or a generic fallback."""
    formatter = _ACTION_FORMATTERS.get(name)
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
        verbose: bool = True,
        compact: bool = False,
    ):
        self.task = task
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.compact = compact
        self.steps: list[StepTrace] = []
        self.eval_result: EvalResult | None = None
        self.finish_reason: str = "running"
        self.patch_diff: str | None = None
        self.telemetry: dict[str, Any] = {}

    def log(self, text: str) -> None:
        if self.verbose:
            print(text)

    def record_step(self, step: StepTrace) -> None:
        self.steps.append(step)

        if not self.verbose:
            return

        model_text = step.model_response.get("content") or ""
        tool_calls = step.model_response.get("tool_calls") or []

        if self.compact:
            # Modern compact UX: collapse intermediate step details into clean 1-liners
            if tool_calls:
                # 1. If model emitted intermediate reasoning/thought before calling tools, show 1 dimmed line
                if model_text.strip():
                    first_line = model_text.strip().splitlines()[0]
                    preview = (first_line[:90] + "...") if len(first_line) > 90 else first_line
                    print(_dim(f"● Thinking: {preview}"))

                # 2. For each tool call, show a single clean action line
                for tc, tr in _pair_calls_with_results(tool_calls, step.tool_results):
                    name = tc.get("name", "") if tc else ""
                    args = tc.get("arguments", {}) if tc else {}
                    action_summary = _format_action_summary(name, args, color=True)
                    exit_code = tr.get("exit_code", 0) if tr else 0
                    if exit_code == 0:
                        print(f"{action_summary} " + _dim(f"({step.duration_s:.1f}s)"))
                    else:
                        out_lines = tr.get("output", "").strip().splitlines() if tr else []
                        err_snip = out_lines[0] if out_lines else "Failed"
                        err_snip = (err_snip[:70] + "...") if len(err_snip) > 70 else err_snip
                        print(f"{action_summary} " + _red(f"❌ (exit {exit_code}: {err_snip})"))

                if step.compaction_applied:
                    print(_dim("🧹 [Context compacted to conserve budget]"))
            else:
                # Final response (no tool calls): print full response text cleanly
                if model_text.strip():
                    print(f"\n{model_text.strip()}\n")
            return

        token_str = ""
        if step.usage:
            p_tok = int(step.usage.get("prompt_tokens") or 0)
            c_tok = int(step.usage.get("completion_tokens") or 0)
            cached_tok = int(step.usage.get("cached_tokens") or 0)
            hit_pct = float(step.usage.get("cache_hit_rate_pct") or 0.0)
            if p_tok > 0 or c_tok > 0:
                if cached_tok > 0:
                    token_str = f" · {p_tok:,} in ({cached_tok:,} cached - {hit_pct:.1f}%), {c_tok:,} out"
                else:
                    token_str = f" · {p_tok:,} in, {c_tok:,} out"

        divider = _dim(f"─── Step {step.step_index} · {step.duration_s:.2f}s{token_str} " + "─" * 25)
        print(f"\n{divider}")

        if model_text:
            print(f"{model_text.strip()}\n")

        if step.compaction_applied:
            print(_dim("🧹 [Memory Compaction]: Pruned older observations to preserve context.\n"))

        for tc, tr in _pair_calls_with_results(tool_calls, step.tool_results):
            if tc:
                name = tc.get("name", "")
                args = tc.get("arguments", {})
                action_text = _format_action_summary(name, args, color=True)
                print(action_text)

            if tr:
                self._print_tool_output(tr)

    def _print_tool_output(self, tr: dict[str, Any]) -> None:
        exit_code = tr.get("exit_code", 0)
        if exit_code != 0:
            print(_red(f"  ❌ (exit {exit_code})"))

        output = tr.get("output", "").strip()
        if not output:
            return

        lines = output.splitlines()
        max_preview_lines = 15
        for line in lines[:max_preview_lines]:
            print(f"  {line}")
        if len(lines) > max_preview_lines:
            remaining = len(lines) - max_preview_lines
            print(_dim(f"  ... ({remaining} more lines)"))

    def _compute_telemetry(self) -> dict[str, Any]:
        p_tok = sum(
            int(s.usage.get("prompt_tokens", 0) or 0)
            for s in self.steps
            if isinstance(s.usage, dict) and isinstance(s.usage.get("prompt_tokens"), (int, float))
        )
        c_tok = sum(
            int(s.usage.get("completion_tokens", 0) or 0)
            for s in self.steps
            if isinstance(s.usage, dict)
            and isinstance(s.usage.get("completion_tokens"), (int, float))
        )
        cached_tok = sum(
            int(s.usage.get("cached_tokens", 0) or 0)
            for s in self.steps
            if isinstance(s.usage, dict) and isinstance(s.usage.get("cached_tokens"), (int, float))
        )
        hit_pct = round((cached_tok / p_tok * 100), 1) if p_tok > 0 else 0.0
        return {
            "total_prompt_tokens": p_tok,
            "total_completion_tokens": c_tok,
            "total_cached_tokens": cached_tok,
            "cache_hit_rate_pct": hit_pct,
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

    def _export_json(self, path: Path) -> None:
        data = {
            "task": {
                "id": self.task.id,
                "instruction": self.task.instruction,
                "max_steps": self.task.max_steps,
            },
            "finish_reason": self.finish_reason,
            "total_steps": len(self.steps),
            "telemetry": self.telemetry,
            "patch_diff": self.patch_diff,
            "eval_result": {
                "passed": self.eval_result.passed if self.eval_result else None,
                "score": self.eval_result.score if self.eval_result else None,
                "feedback": self.eval_result.feedback if self.eval_result else None,
            },
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
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _export_markdown(self, path: Path) -> None:
        lines = [
            f"# Run Trace: Task {self.task.id}",
            f"- **Status**: `{self.finish_reason}`",
            f"- **Total Steps**: {len(self.steps)}",
        ]
        if self.telemetry:
            p_tok = self.telemetry.get("total_prompt_tokens", 0)
            c_tok = self.telemetry.get("total_completion_tokens", 0)
            cached_tok = self.telemetry.get("total_cached_tokens", 0)
            hit_pct = self.telemetry.get("cache_hit_rate_pct", 0.0)
            if p_tok or c_tok:
                cached_tag = (
                    f" ({cached_tok:,} cached - {hit_pct:.1f}% hit rate)" if cached_tok else ""
                )
                lines.append(f"- **Tokens**: {p_tok:,} in{cached_tag}, {c_tok:,} out")

        if self.eval_result:
            status_tag = "PASSED" if self.eval_result.passed else "FAILED"
            lines.extend(
                [
                    f"- **Evaluation**: `{status_tag}` (Score: {self.eval_result.score})",
                    f"- **Feedback**: {self.eval_result.feedback}",
                ]
            )

        lines.extend(
            [
                "",
                "## Task Instruction",
                f"```\n{self.task.instruction}\n```",
                "",
                "## Execution Steps",
            ]
        )

        for s in self.steps:
            step_header = f"### Step {s.step_index} ({s.duration_s:.2f}s"
            if s.usage:
                p = s.usage.get("prompt_tokens", 0)
                c = s.usage.get("completion_tokens", 0)
                ca = s.usage.get("cached_tokens", 0)
                h = s.usage.get("cache_hit_rate_pct", 0.0)
                if p or c:
                    ca_str = f" [{ca:,} cached - {h:.1f}%]" if ca else ""
                    step_header += f" · {p:,} in{ca_str}, {c:,} out"
            step_header += ")"
            lines.extend(
                [
                    step_header,
                    "",
                ]
            )
            if s.compaction_applied:
                lines.append("> ℹ️ *Context Compaction was applied during this step.*")

            content = s.model_response.get("content")
            if content:
                lines.extend([content.strip(), ""])

            tool_calls = s.model_response.get("tool_calls") or []
            for tc, tr in _pair_calls_with_results(tool_calls, s.tool_results):
                if tc:
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})
                    if name == "bash":
                        cmd = (
                            args.get("command", "").strip() if isinstance(args, dict) else str(args)
                        )
                        lines.extend(["**Command** (`bash`):", f"```bash\n{cmd}\n```"])
                    elif name in _ACTION_FORMATTERS:
                        action_summary = _format_action_summary(name, args, color=False)
                        lines.append(f"**Action** (`{name}`): `{action_summary}`")
                    else:
                        lines.extend(
                            [
                                f"**Action**: `{name}`",
                                f"```json\n{json.dumps(args, indent=2)}\n```",
                            ]
                        )

                if tr:
                    trunc = " (Truncated)" if tr.get("was_truncated") else ""
                    exit_code = tr.get("exit_code", 0)
                    lines.extend(
                        [
                            f"**Output** (exit={exit_code}){trunc}:",
                            f"```\n{tr.get('output', '').strip()}\n```",
                            "",
                        ]
                    )

        if self.patch_diff:
            lines.extend(
                [
                    "## Git Patch Diff",
                    f"```diff\n{self.patch_diff}\n```",
                    "",
                ]
            )

        path.write_text("\n".join(lines), encoding="utf-8")
