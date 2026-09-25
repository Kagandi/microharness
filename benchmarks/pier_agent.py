from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

# Gracefully inherit from BaseAgent if Harbor/Pier is present in the environment
try:
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except ImportError:
    BaseAgent = object  # type: ignore
    BaseEnvironment = Any  # type: ignore
    AgentContext = Any  # type: ignore

from core.harness import Harness
from core.model import HttpChatModel
from core.types import Task


class MicroharnessAgent(BaseAgent):  # type: ignore
    """Adapter for running microharness inside the Datacurve Pier evaluation framework."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        if BaseAgent is not object:
            super().__init__(logs_dir=logs_dir, model_name=model_name, logger=logger, **kwargs)
        self.logs_dir = Path(logs_dir)
        self.model_name = model_name or os.getenv("MODEL_NAME") or "stealth/space-bunny-alpha"
        self.endpoint = os.getenv("MODEL_ENDPOINT", "https://openrouter.ai/api/v1")
        self.api_key = os.getenv("MODEL_API_KEY") or os.getenv("OPENROUTER_API_KEY")

    def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext = None,
    ) -> None:
        """Executes the microharness agent on the task instruction."""
        model = HttpChatModel(
            base_url=self.endpoint,
            model=self.model_name,
            api_key=self.api_key,
            cache_ttl="5m",
        )
        # Agent operates directly in the container's workspace (/app)
        task = Task(
            id="pier_trial",
            instruction=instruction,
            max_steps=35,
            timeout_seconds=900.0,
        )
        harness = Harness(
            model=model,
            workspace_dir=Path("/app"),
            runs_dir=self.logs_dir,
            sandbox=False,
            docker=False,  # Pier already manages container isolation
            cache_ttl="5m",
            verbose=True,
        )
        harness.run(task)
