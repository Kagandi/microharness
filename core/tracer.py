from __future__ import annotations

import json
from pathlib import Path

from core.types import EvalResult, StepTrace, Task


class Tracer:
    """Collects telemetry during the agent run and exports structured traces."""

    def __init__(self, task: Task, output_dir: Path, verbose: bool = True):
        self.task = task
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.steps: list[StepTrace] = []
        self.eval_result: EvalResult | None = None
        self.finish_reason: str = "running"

    def log(self, text: str) -> None:
        if self.verbose:
            print(text)

    def record_step(self, step: StepTrace) -> None:
        self.steps.append(step)

        if not self.verbose:
            return

        print(f"\n{'=' * 25} Step {step.step_index} ({step.duration_s:.2f}s) {'=' * 25}")
        model_text = step.model_response.get("content")
        tool_calls = step.model_response.get("tool_calls") or []

        if model_text:
            print(f"🧠 Thought / Message:\n{model_text.strip()}")

        if step.compaction_applied:
            print("🧹 [Memory Compaction]: Pruned older observations to preserve context.")

        for tc in tool_calls:
            args_repr = json.dumps(tc.get("arguments", {}))
            print(f"⚡ Action: {tc.get('name')}({args_repr})")

        for tr in step.tool_results:
            exit_info = f"exit={tr.get('exit_code')}"
            trunc_info = " [TRUNCATED]" if tr.get("was_truncated") else ""
            print(f"👁️ Observation ({exit_info}{trunc_info}):")
            output = tr.get("output", "").strip()
            # Indent observation for readable terminal output
            for line in output.splitlines()[:15]:
                print(f"   {line}")
            if output.count("\n") > 15:
                print(f"   ... ({output.count(chr(10)) - 15} more lines in observation)")

    def finalize(
        self, finish_reason: str, eval_result: EvalResult | None = None
    ) -> tuple[Path, Path]:
        self.finish_reason = finish_reason
        self.eval_result = eval_result

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
            lines.extend(
                [
                    f"### Step {s.step_index} ({s.duration_s:.2f}s)",
                    "",
                ]
            )
            if s.compaction_applied:
                lines.append("> ℹ️ *Context Compaction was applied during this step.*")

            content = s.model_response.get("content")
            if content:
                lines.extend(["**Agent Response / Reasoning**:", f"```\n{content}\n```", ""])

            tool_calls = s.model_response.get("tool_calls") or []
            for tc in tool_calls:
                lines.extend(
                    [
                        f"**Action**: `{tc.get('name')}`",
                        f"```json\n{json.dumps(tc.get('arguments', {}), indent=2)}\n```",
                    ]
                )

            for tr in s.tool_results:
                trunc = " (Truncated)" if tr.get("was_truncated") else ""
                lines.extend(
                    [
                        f"**Observation** (Exit code: {tr.get('exit_code')}){trunc}:",
                        f"```\n{tr.get('output', '').strip()}\n```",
                    ]
                )
                if tr.get("raw_log_path"):
                    lines.append(f"*Full log saved at: `{tr['raw_log_path']}`*")
                lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
