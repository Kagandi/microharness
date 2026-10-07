"""microharness CLI entry point."""

from core.cli import *  # noqa: F403
from core.cli import (  # noqa: F401
    BENCHMARK_RUNNERS,
    CliContext,
    _build_arg_parser,
    _dispatch_benchmark,
    _resolve_model,
    create_benchmark_harness_factory,
    main,
)

if __name__ == "__main__":
    main()
