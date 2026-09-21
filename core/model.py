from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol

from core.types import Message, ToolCall


class Model(Protocol):
    """Protocol for pluggable LLM callers."""

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> tuple[str, list[ToolCall]]:
        """Given conversation history and tool schemas, returns (response_text, tool_calls)."""
        ...


class OpenAICompatibleModel:
    """Direct API caller for any OpenAI-compatible /v1/chat/completions endpoint.

    Works with OpenAI, DeepSeek, Ollama, Groq, vLLM, etc., using Python's standard library.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout = timeout

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> tuple[str, list[ToolCall]]:
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        formatted_messages = []
        for msg in messages:
            item: dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
            if msg.tool_call_id:
                item["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                            if isinstance(tc.arguments, dict)
                            else str(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            formatted_messages.append(item)

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": formatted_messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                res_body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"API call failed with HTTP {exc.code}: {err_body}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Network error calling model endpoint: {exc}") from exc

        choice = res_body["choices"][0]["message"]
        content = choice.get("content") or ""
        tool_calls: list[ToolCall] = []

        if "tool_calls" in choice and choice["tool_calls"]:
            for raw_tc in choice["tool_calls"]:
                func_data = raw_tc["function"]
                args_raw = func_data.get("arguments", "{}")
                parsed_args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                tool_calls.append(
                    ToolCall(
                        id=raw_tc["id"],
                        name=func_data["name"],
                        arguments=parsed_args,
                    )
                )

        return content, tool_calls


class MockModel:
    """A deterministic scripted model for offline testing and educational demonstrations."""

    def __init__(self, script: list[tuple[str, list[ToolCall]]]):
        """Accepts a sequence of (content, tool_calls) tuples to yield sequentially."""
        self.script = list(script)
        self.step_idx = 0

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> tuple[str, list[ToolCall]]:
        if self.step_idx < len(self.script):
            result = self.script[self.step_idx]
            self.step_idx += 1
            return result
        # Default stop condition if script runs out: respond without tools
        return "I have completed the task and verified all results.", []
