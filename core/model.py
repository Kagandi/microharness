from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol

from core.types import Message, ToolCall


class Model(Protocol):
    """Protocol for pluggable LLM callers.

    Any class or adapter with a `generate` method can be plugged directly
    into the Harness without modifying any harness code.
    """

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> tuple[str, list[ToolCall]]:
        ...


class HttpChatModel:
    """A generic HTTP model client for any OpenAI-compatible completions endpoint.

    Works identically with:
    - OpenRouter (https://openrouter.ai/api/v1)
    - Ollama (http://localhost:11434/v1)
    - vLLM (http://localhost:8000/v1)
    - LM Studio (http://localhost:1234/v1)
    - Any standard completions gateway
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 90.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.extra_headers = headers or {}
        self.timeout = timeout

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> tuple[str, list[ToolCall]]:
        endpoint = f"{self.base_url}/chat/completions"
        req_headers = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        if self.api_key:
            req_headers["Authorization"] = f"Bearer {self.api_key}"

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
            "model": self.model,
            "messages": formatted_messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=req_headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                res_body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Model API returned HTTP {exc.code} from {endpoint}:\n{err_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Could not connect to model endpoint at {endpoint}.\n"
                f"Is the server running and reachable? (Detail: {exc.reason})"
            ) from exc

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
    """Deterministic model for offline testing and educational walkthroughs."""

    def __init__(self, script: list[tuple[str, list[ToolCall]]]):
        self.script = list(script)
        self.step_idx = 0

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> tuple[str, list[ToolCall]]:
        if self.step_idx < len(self.script):
            result = self.script[self.step_idx]
            self.step_idx += 1
            return result
        return "I have completed the task and verified all results.", []
