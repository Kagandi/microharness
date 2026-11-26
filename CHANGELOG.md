# Changelog

All notable changes to `microharness` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Central CSV experiment tracking:** Added `benchmarks/experiments.py` (`CsvExperimentTracker`, `ExperimentTracker`, `backfill_runs`, `collect_agent_metadata`, `collect_subagents_summary`), persisting run-level (`~/.microharness/experiments.csv`) and per-task (`~/.microharness/experiments_tasks.csv`) agent versions, git commit/dirty state, configurations, subagent presets, and scores/token telemetry automatically across benchmark runs and `--resume`.
- **Experiments CLI:** Added `microharness experiments backfill` and `microharness experiments list` to import existing saved reports and inspect runs directly from the terminal.

## [0.1.21] - 2026-11-26

### Added
- **Atomic multi-hunk `edit_file`:** Added `apply_text_edits` (`core/editing.py`) and updated `EditFileTool` / `Environment.edit_file` / `ReadOnlyEnvironment` / `GuardedEnvironment` / `core/tracer.py` to support sequential `edits: [{old_str, new_str}, ...]` batches with atomic all-or-nothing rollback.
- **Advertised `edit_file(path, edits)` schema:** Simplified the LLM tool schema to require `edits` directly (preserving internal `old_str`/`new_str` fallback compatibility in `EditFileTool.execute`).
- **Brace expansion in `glob`:** Added `{a,b}` alternative support to `_compile_glob_pattern` in `core/tools.py` (fixing failures on patterns such as `**/*.{ts,tsx,json,md}`).
- **Project changelog:** Added `CHANGELOG.md` documenting releases `0.1.0` through `0.1.21`.

## [0.1.20] - 2026-11-24

### Added
- **Benchmark task exclusions & `arm64` guard (#38):** Added `!term` / `-term` exclusion filters to `benchmarks/runner.py`, `benchmarks/harbor.py`, and `benchmarks/mini_swe.py`, and automatically skip `narwhals-rolling-window-suite` on `arm64` hosts where prebuilt wheels fail.

## [0.1.19] - 2026-11-24

### Added
- **Dedicated read-only inspection tools (#37):** Added `grep`, `glob`, and `git_read` (`parallel_safe=True`), switched `explore` and `reviewer` subagent presets to `[read_file, grep, glob, git_read]`, and surfaced explicit `read-only` access labels in `render_subagents_index`.

### Fixed
- **Shell chaining validation:** Fixed `&&` / `||` operator parsing in `core/environment.py`.

## [0.1.18] - 2026-11-24

### Fixed
- **Container skills & capped subagents (#36):** Skip host repository skills inside container sandboxes and ensure subagents that hit `max_steps` produce a structured wrap-up summary.

## [0.1.17] - 2026-11-23

### Fixed
- **Benchmark ergonomics & metadata (#35):** Added multi-task filter support, deferred live context-window probing until needed, tightened task filter matching, and fixed macOS curses test compatibility.

## [0.1.16] - 2026-11-23

### Fixed
- **Sandbox project discovery (#34):** Discovered project `.microharness/` config, subagent presets, and project skills inside sandbox workspaces without duplicate discovery.

## [0.1.15] - 2026-11-22

### Fixed
- **Subagent context & spill logs (#33):** Resolved per-subagent model context windows, bumped default `bash_timeout` to `300s`, and prevented log spill file collisions.

## [0.1.14] - 2026-11-22

### Added
- **UTC trace timestamps (#32):** Added ISO-8601 UTC timestamps to trajectories and run traces.

## [0.1.13] - 2026-11-22

### Fixed
- **TUI interaction polish (#31):** Required leading `/` for slash commands, added mouse/keyboard scrolling, unified `Enter`/`Tab` input behavior, and added visual turn separators.

## [0.1.12] - 2026-11-21

### Changed
- **Declarative defaults (#30):** Centralized operational defaults in `core/defaults.json` and `core/providers.json`, and raised the default fallback context window to `256k`.

## [0.1.11] - 2026-11-21

### Added
- **Dynamic context-window discovery (#29):** Auto-detected model context windows from provider `/models` endpoints.

## [0.1.10] - 2026-11-21

### Fixed
- **Docker path resolution & fail-fast HTTP errors (#28):** Resolved container relative paths cleanly and failed fast on non-retryable `4xx` HTTP responses instead of burning retry budgets.

## [0.1.9] - 2026-11-21

### Fixed
- **Permissions & review cleanup (#27):** Added `/permissions ro` alias, deduplicated config/environment helpers, and removed dead hook code.

## [0.1.7] - 2026-11-21

### Added
- **Lifecycle hooks & loop guardrails (#25):** Added `HookRegistry` (`pre_turn`, `post_turn`, `pre_step`, `post_step`, `pre_tool`, `post_tool`), built-in `python_syntax_diagnostics` via `ast.parse`, and `doom_loop_detection`.
- **Checkpoints & permission guardrails (#26):** Added pre-turn git/file checkpoints with `/undo`, `GuardedEnvironment` (`auto`/`ask`/`read-only`/`ro`), and Docker resource limits.

### Fixed
- **Hooks & Python 3.12+ proxy compatibility:** Fired `post_step` on final steps, passed turn outcome to `post_turn`, and preserved `ContainerEnvironment` protocol compliance in `GuardedEnvironment`.

## [0.1.5] - 2026-11-21

### Fixed
- **Preset defaults:** Tightened read-only preset defaults for subagent roles.

## [0.1.4] - 2026-11-21

### Added
- **KV-cache friendly memory compaction (#24):** Added immutable prefix preservation, `prompt_tokens`-calibrated budgeting, structured compaction summaries, and read-only Docker `/logs` inspection.
- **Resilient editing & parallel reads (#21):** Extracted `core/editing.py` with whitespace/dedent fallback and `difflib` closest-match hints, enabled `parallel_safe` on `read_file`, and improved JSON parse error feedback.
- **Subagent isolation hardening (#22):** Hardened `ReadOnlyEnvironment` isolation, prevented concurrent write-worker races, and added configurable `max_steps` per preset.

## [0.1.3] - 2026-11-20

### Added
- **Subagent orchestration (#14):** Added `SubagentTool` (`parallel_safe=True`), `.microharness/agents/` Markdown presets (`explore`, `worker`, `reviewer`), concurrent `tasks` batching, and per-call `bash` timeout up to `1800s`.

### Fixed
- **Provider routing & reasoning replay:** Prevented non-local global endpoints from hijacking local presets, and preserved assistant `reasoning` / `reasoning_details` / `reasoning_content` across tool turns.

## [0.1.2] - 2026-11-17

### Added
- **`finish` tool & sandbox hardening (#12):** Added explicit `FinishTool`, Docker `--network none` isolation, bash timeout hardening, and preserved trailing newlines in `get_git_patch`.
- **Static type checking & CI (#8, #13):** Added GitHub Actions CI (`ruff`, `ty`, `pytest` across Python 3.11–3.13), CodeQL security scanning, and `.pre-commit-config.yaml`.

## [0.1.1] - 2026-11-15

### Added
- **Global auth store & Vercel AI Gateway (#9, #10, #11):** Added global credential storage, CLI login, `/login` command, Vercel AI Gateway provider support, and CodeQL secret-taint hardening.

## [0.1.0] - 2026-11-08

### Added
- **Initial PyPI release (#1–#7):** Modular stdlib-only core (`Model`, `Tool`, `Evaluator` protocols), curses TUI with caret editing and `/help` command router, local provider presets, Amazon Bedrock OpenAI-compatible shim with SigV4/SSO auth, cascading `AGENTS.md` and skills discovery, Mini-SWE & DeepSWE-mini benchmark runners, and PyPI OIDC Trusted Publishing workflow (`.github/workflows/publish.yml`).
