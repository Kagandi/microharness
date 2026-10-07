# feat: make TUI default, add local providers, non-blocking execution, and token limits

## Summary of Changes

This PR modernizes the interactive developer experience, provider registry, and context limits across `microharness`:

1. **TUI as Default Interactive Interface**:
   - `Harness.run_interactive()` launches the full-screen Curses TUI by default.
   - Removed legacy `ConsoleUI`; interactive mode is now 100% full-screen TUI.
   - Headless execution is strictly handled via `--batch` (or `Harness.run(task)` in code). If an interactive run is attempted without a TTY, the CLI provides helpful guidance to use `--batch`.
   - Single entrypoint in `main.py` without introducing extra CLI modules into `core/`.

2. **Non-Blocking Curses Execution & Thread-Safe Redraw**:
   - Agent turns run asynchronously on a background worker thread (`worker_thread`), with non-blocking key polling (`stdscr.timeout(50)`).
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
   - Preserved `LAST_RESORT_MODEL = "cohere/north-mini-code:free"` and configured `anthropic/claude-5.5-sonnet` / `claude-5.5-sonnet` frontier models.
   - Fixed endpoint scoping: models defined under `models: { ... }` in config retain their provider-specific endpoints without being overwritten by top-level defaults.

6. **Unified Model Naming (`default_model` / `defaultModel`)**:
   - Supports `default_model` and `defaultModel` keys with precedence over legacy `model` in configuration.
   - Clean documentation and schema examples throughout README.

7. **Token-Only Context and Tool Limits**:
   - Fully replaced `max_context_chars` and `max_tool_chars` with `max_context_tokens` (default 64,000 tokens ≈ 256k chars) and `max_tool_tokens` (default 8,000 tokens ≈ 32k chars).
   - `HarnessConfig`, `ResolvedConfig`, `Harness`, `Memory`, and `ToolRegistry` expose only token limits.
   - Legacy character settings from old config files are automatically converted to tokens during resolution.
   - Head (1/3 ≈ 10.6k chars) + Tail (2/3 ≈ 21.3k chars) preservation for tool outputs (at default 8,000 tokens) with disk log spillover.

8. **Robust TUI Thread Safety & Cancellation UX**:
   - `TUIState` protects transcript mutations and line slices under `threading.RLock()`.
   - Curses rendering calls are kept strictly on the main thread via `request_redraw()`.
   - Submitting Enter while a turn is running gives immediate feedback (`⏳ Turn is running (press Esc or Ctrl+C to cancel)`).
   - `Harness.run_tui(task_id="interactive_session")` standardizes run-directory naming, with `allow_bare_words=True` enabled for interactive shell commands.

9. **Configuration Migration & Backward Compatibility**:
   - `get_default_global_dir()` gracefully falls back to `~/.mini-harness` if `~/.microharness` does not exist.
   - Legacy `max_tool_chars` and `max_context_chars` in configuration files are automatically converted to tokens (`chars // 4`).
   - Re-exported `find_recent_sessions`, `load_session_messages`, and `reconstruct_messages_from_steps` in `core.tracer` for backward compatibility.

10. **Portable Reasoning Effort**:
    - Maps `xhigh` and `max` to `"high"` on standard OpenAI endpoints to prevent API validation 400 errors, while preserving full values on OpenRouter endpoints.

11. **Offline Interactive Guidance**:
    - Interactive TUI launch verifies endpoint and credentials for non-local providers and fails fast with setup instructions, while headless batch and benchmark runs continue using `MockModel` when offline.

## Verification & Testing

- **Zero Runtime Dependencies**: Standard library only (`pyproject.toml` dependencies remain empty).
- **Correct Repository URLs**: Pointing to `https://github.com/Kagandi/microharness`.
- **Unit Test Suite**: 556 passing tests (`uv run pytest`):
  - `tests/test_config.py` (token limits, legacy char migration, global dir fallback, cascade priority, reasoning choices)
  - `tests/test_harness.py` (TUI defaults, headless NullUI, session handling)
  - `tests/test_interactive.py` (model switching runner synchronization, reasoning status update)
  - `tests/test_memory.py` (token limits, high-water mark compaction)
  - `tests/test_model.py` (reasoning effort mapping for OpenAI vs OpenRouter)
  - `tests/test_providers.py` (local provider presets, scoped endpoints)
  - `tests/test_tools.py` (token limits and spillover)
  - `tests/test_tui.py` (Esc key safety, reasoning subcommand menus, non-blocking threading)
- **Lint & Style**: Passed `uv run ruff check` with 0 warnings.