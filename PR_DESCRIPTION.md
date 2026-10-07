# feat: make TUI default, add local providers, non-blocking execution, and token limits

## Summary of Changes

This PR modernizes the interactive developer experience and configuration architecture across `microharness`:

1. **TUI as Default Interactive Interface**:
   - `Harness.run_interactive()` now launches the full-screen Curses TUI by default.
   - Removed legacy `ConsoleUI`; interactive mode is now 100% full-screen TUI.
   - Headless execution is strictly handled via `--batch` (or `Harness.run(task)` in code). If an interactive run is attempted without a TTY, the CLI provides helpful guidance to use `--batch`.
   - Extracted clean CLI parsing and dispatch into [`core/cli.py`](file:///Users/dimakagan/Projects/mini-harness/core/cli.py).

2. **Non-Blocking Curses Execution & Thread-Safe Redraw**:
   - Agent turns now run asynchronously on a background worker thread (`worker_thread`), with non-blocking key polling (`stdscr.timeout(50)`).
   - Thread-safe drawing synchronization using `_draw_lock` prevents curses race conditions and display corruption.
   - Status spinner and UI remain responsive while model/tool execution takes place.

3. **Safe Escape Key Handling**:
   - Pressing <kbd>Esc</kbd> during an active turn signals execution cancellation (`_cancel_requested = True`).
   - Pressing <kbd>Esc</kbd> when idle dismisses menus, clears autocomplete queries, or clears input buffer.
   - **Never exits the session on an empty prompt** when pressing <kbd>Esc</kbd> (exit is via `/exit` or <kbd>Ctrl+C</kbd> / <kbd>Ctrl+D</kbd>).

4. **Interactive Reasoning / Thinking Effort Control**:
   - Added interactive `/reasoning` popup menu in TUI with choices: `off`, `low`, `medium`, `high`, `xhigh`, `max`.
   - Dynamic update of active reasoning effort in TUI header and model serialization.
   - Support for `xhigh` and `max` levels across model payloads (OpenAI and OpenRouter specs).

5. **First-Class Local Provider Presets & Scoped Endpoints**:
   - Added built-in presets for local inference engines in [`core/providers.json`](file:///Users/dimakagan/Projects/mini-harness/core/providers.json): `lmstudio` (`http://localhost:1234/v1`), `vllm` (`http://localhost:8000/v1`), `llamacpp` (`http://localhost:8080/v1`), `ollama` (`http://localhost:11434/v1`), and generic `local`.
   - Added provider aliases (`lm-studio`, `llama-cpp`, `llama.cpp`).
   - Refreshed frontier defaults to Claude 3.7 Sonnet, Gemini 2.5 Flash, and GPT-4o.
   - Fixed endpoint scoping: models defined under `models: { ... }` in config retain their provider-specific endpoints without being overwritten by top-level defaults.

6. **Unified Model Naming (`default_model` / `defaultModel`)**:
   - Supports `default_model` and `defaultModel` keys with precedence over legacy `model` in configuration.
   - Clean documentation and schema examples throughout README.

7. **Token-First Context and Tool Limits**:
   - Replaced char-centric limits with token controls: `max_context_tokens` (default 64,000 tokens ≈ 256k chars) and `max_tool_tokens` (default 8,000 tokens ≈ 32,000 chars).
   - Backward-compatible cascade handles both `*_tokens` and `*_chars` transparently.
   - Head (10k chars) + Tail (22k chars) preservation for tool outputs with disk log spillover.

8. **Fixed Live Model Switching**:
   - `/model <name>` now dynamically synchronizes active model across `Harness`, `TuiTurnRunner`, and `TUIState` (updating title bar and provider display immediately).

9. **Offline Interactive Guidance**:
   - Interactive sessions without configured credentials or offline status fail fast with clear setup instructions instead of silently falling back to a dummy `MockModel`.

## Verification & Testing

- **Zero Runtime Dependencies**: Standard library only (`pyproject.toml` dependencies remain empty).
- **Unit Test Suite**: 556 passing tests (`uv run pytest`):
  - `tests/test_config.py` (token limits, cascade priority, reasoning choices)
  - `tests/test_harness.py` (TUI defaults, headless NullUI, session handling)
  - `tests/test_interactive.py` (model switching runner synchronization, reasoning status update)
  - `tests/test_providers.py` (local provider presets, scoped endpoints)
  - `tests/test_tui.py` (Esc key safety, reasoning subcommand menus, non-blocking threading)
- **Lint & Style**: Passed `uv run ruff check` with 0 warnings.