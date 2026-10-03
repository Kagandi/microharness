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
_DEFAULT_ACTION_FORMATTERS: dict[str, ActionFormatter] = {
    "bash": _format_bash_action,
    "read_file": _format_read_file_action,
    "write_file": _format_write_file_action,
    "edit_file": _format_edit_file_action,
}
_ACTION_FORMATTERS: dict[str, ActionFormatter] = dict(_DEFAULT_ACTION_FORMATTERS)


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
                f.write(json.dumps(record) + "\n")
                f.flush()
        except Exception:
            pass

    def record_step(self, step: StepTrace) -> None:
        self.steps.append(step)
        self._append_jsonl(step)

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
                    action_summary = _format_action_summary(
                        name, args, color=True, formatters=self.formatters
                    )
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
                action_text = _format_action_summary(
                    name, args, color=True, formatters=self.formatters
                )
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


def reconstruct_messages_from_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconstructs full conversation message history from a list of step trace dictionaries."""
    if not steps:
        return []
    last_step = steps[-1]
    msgs: list[dict[str, Any]] = [dict(m) for m in (last_step.get("prompt_messages") or [])]
    resp = last_step.get("model_response") or {}
    content = resp.get("content") or ""
    tool_calls = resp.get("tool_calls") or []
    if content or tool_calls:
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if tool_calls:
            formatted_calls: list[dict[str, Any]] = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    cid = tc.get("id", "")
                    fn = tc.get("function") or {}
                    name = fn.get("name") or tc.get("name", "")
                    args = fn.get("arguments") or tc.get("arguments") or {}
                    if isinstance(args, str):
                        raw_args = args
                    else:
                        raw_args = json.dumps(args) if isinstance(args, dict) else str(args)
                    call_dict = {
                        "id": cid,
                        "type": "function",
                        "function": {"name": name, "arguments": raw_args},
                    }
                    if tc.get("extra_content"):
                        call_dict["extra_content"] = tc.get("extra_content")
                    formatted_calls.append(call_dict)
            if formatted_calls:
                assistant_msg["tool_calls"] = formatted_calls
        msgs.append(assistant_msg)

    for tr in last_step.get("tool_results") or []:
        if isinstance(tr, dict) and "call_id" in tr:
            msgs.append(
                {
                    "role": "tool",
                    "content": tr.get("output", ""),
                    "tool_call_id": tr.get("call_id"),
                    "raw_log_path": tr.get("raw_log_path"),
                }
            )
    return msgs


def load_session_messages(run_dir: Path | str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Loads reconstructed messages and metadata from a session directory."""
    dir_path = Path(run_dir)
    trace_file = dir_path / "trace.json"
    jsonl_file = dir_path / "trajectory.jsonl"
    metadata: dict[str, Any] = {
        "run_id": dir_path.name,
        "run_dir": str(dir_path),
        "total_steps": 0,
        "workspace": None,
        "finish_reason": "unknown",
    }

    # 1. Try trace.json
    if trace_file.is_file():
        try:
            data = json.loads(trace_file.read_text(encoding="utf-8"))
            metadata["total_steps"] = int(data.get("total_steps") or 0)
            metadata["workspace"] = data.get("workspace")
            metadata["finish_reason"] = data.get("finish_reason", "unknown")
            metadata["task"] = data.get("task", {})
            if "messages" in data and isinstance(data["messages"], list) and data["messages"]:
                return data["messages"], metadata
            steps = data.get("steps") or []
            if steps:
                msgs = reconstruct_messages_from_steps(steps)
                return msgs, metadata
        except Exception:
            pass

    # 2. Try trajectory.jsonl (if trace.json was absent or corrupted after sudden crash)
    if jsonl_file.is_file():
        try:
            steps = []
            for line in jsonl_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    steps.append(json.loads(line))
            if steps:
                metadata["total_steps"] = len(steps)
                msgs = reconstruct_messages_from_steps(steps)
                return msgs, metadata
        except Exception:
            pass

    return [], metadata


def find_recent_sessions(
    runs_dir: Path | str,
    workspace: Path | str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Discovers previous sessions from runs_dir, sorted by recency and workspace affinity."""
    base = Path(runs_dir)
    if not base.is_dir():
        return []

    target_ws = str(Path(workspace).resolve()) if workspace else None
    sessions: list[dict[str, Any]] = []

    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        trace_json = entry / "trace.json"
        traj_jsonl = entry / "trajectory.jsonl"
        if not (trace_json.is_file() or traj_jsonl.is_file()):
            continue

        try:
            mtime = entry.stat().st_mtime
        except Exception:
            mtime = 0.0

        total_steps = 0
        preview = ""
        sess_ws = None
        finish_reason = ""
        task_id = entry.name

        if trace_json.is_file():
            try:
                data = json.loads(trace_json.read_text(encoding="utf-8"))
                total_steps = int(data.get("total_steps") or 0)
                sess_ws = data.get("workspace")
                finish_reason = data.get("finish_reason", "")
                task_info = data.get("task") or {}
                task_id = task_info.get("id") or entry.name
                preview = (task_info.get("instruction") or "").strip()
                if not preview and data.get("messages"):
                    for m in data["messages"]:
                        if m.get("role") == "user" and m.get("content"):
                            preview = m["content"].strip()
                            break
                elif not preview and data.get("steps"):
                    last_step = data["steps"][-1]
                    for m in (last_step.get("prompt_messages") or []):
                        if m.get("role") == "user" and m.get("content"):
                            preview = m["content"].strip()
                            break
            except Exception:
                pass

        if not preview and traj_jsonl.is_file():
            try:
                lines = [line for line in traj_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
                total_steps = max(total_steps, len(lines))
                if lines:
                    last_step = json.loads(lines[-1])
                    for m in (last_step.get("prompt_messages") or []):
                        if m.get("role") == "user" and m.get("content"):
                            preview = m["content"].strip()
                            break
            except Exception:
                pass

        if len(preview) > 60:
            preview = preview[:57] + "..."

        is_match = bool(target_ws is not None and sess_ws is not None and target_ws == sess_ws)

        sessions.append(
            {
                "run_id": entry.name,
                "run_dir": str(entry),
                "task_id": task_id,
                "total_steps": total_steps,
                "preview": preview or "(no user prompt)",
                "workspace": sess_ws,
                "mtime": mtime,
                "finish_reason": finish_reason,
                "matches_workspace": is_match,
            }
        )

    # Sort: matching workspace first, then newest mtime
    sessions.sort(key=lambda s: (1 if s["matches_workspace"] else 0, s["mtime"]), reverse=True)
    return sessions[:limit]
