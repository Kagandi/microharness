# Agent Config Presets

A **preset** is a self-contained directory that packages a runnable agent persona—its system prompt, model/runtime defaults, subagents, custom tools, progressive-disclosure skills, and slash commands—into one portable bundle.

## Directory Layout

Place presets under `~/.microharness/presets/<name>/` (global, trusted) or `.microharness/presets/<name>/` (project-local, shadows global presets sharing the same ID), or load any folder directly by path.

```text
~/.microharness/presets/
└── paper-researcher/
    ├── agent.md          # YAML frontmatter + base system prompt (or AGENT.md)
    ├── config.json       # Optional JSONC overrides (HarnessConfig, hooks, models)
    ├── subagents/        # Preset-scoped subagents (*.md or <name>/AGENT.md; agents/ also supported)
    ├── tools/            # Preset-scoped Python tool plugins (*.py)
    ├── skills/           # Preset-scoped skills (<name>/SKILL.md or *.md)
    └── commands/         # Optional preset-scoped slash commands (*.md)
```

---

## Usage

```bash
# List discovered presets (global + project)
uv run microharness --list-presets        # alias: --list-agents

# Run with a named preset (matched by directory ID or frontmatter display name)
uv run microharness --preset paper-researcher "Summarize the attached paper"
uv run microharness --agent "Paper Researcher" --batch "Summarize the attached paper"

# Or load a preset from any filesystem path
uv run microharness --preset ./path/to/my-preset
```

Inside an interactive session or TUI, `/preset` (or `/presets`) displays the active preset and lists all available presets, and `/status` includes the active preset banner.

In Python:

```python
from microharness import Harness

harness = Harness(
    model=my_model,
    workspace_dir=".",
    preset="paper-researcher",
)
```

---

## `agent.md` Frontmatter Reference

`agent.md` (or `AGENT.md`) uses optional YAML frontmatter followed by the Markdown body, which replaces `DEFAULT_SYSTEM_PROMPT` as the root agent's base system prompt.

| Field | Type | Purpose |
| :--- | :--- | :--- |
| `name` | string | Human-readable display name (defaults to directory name) |
| `description` | string | One-line summary shown in `--list-presets` and `/preset` |
| `model` | string | Model alias from `config.json` or `provider/model` spec (overridden by `--model`) |
| `provider`, `endpoint`, `api_key` | string | Optional provider/endpoint overrides |
| `reasoning_effort` (`effort`, `thinking`, `reasoning`) | string | `low`, `medium`, `high`, `xhigh`, `max`, numeric token budget, or `off` |
| `cache_ttl` | string | `5m`, `1h`, or `off` |
| `permission_mode` | string | `auto`, `ask`, or `read-only` |
| `tools` | list / CSV | Tool allow-list (`read`/`write`/`edit` aliases supported; `[]` disables all tools; `*`/`all` or omitted allows all) |
| `disable_default_tools` | bool | Skip registering built-in coding tools (`bash`, `read_file`, etc.) |
| `auto_discover_tools`, `allow_network_tools` | bool | Control plugin auto-discovery and network tool filtering |
| `sandbox`, `docker`, `docker_image`, `docker_network` | bool / string | Execution environment defaults |
| `max_steps`, `max_context_tokens`, `max_tool_tokens`, `bash_timeout`, `subagent_max_steps` | int / float / `null` | Budget and timeout defaults |
| `disabled_hooks` | list / CSV | Hook names to disable (supports inline or YAML `- item` lists) |
| `isolated` | bool | Shorthand setting all four `inherit_*` / `include_*` flags below to `false` |
| `include_builtin_subagents` | bool | Include built-in `explore`, `reviewer`, and `worker` subagents (default `true`, or `false` when `isolated: true`) |
| `inherit_global` | bool | Inherit global `skills/`, `tools/`, `agents/`, `commands/`, `AGENTS.md`, and `hooks` |
| `inherit_project` | bool | Inherit project `.microharness/` config, `skills/`, `tools/`, `agents/`, and `commands/` |
| `include_workspace_agents_md` | bool | Append workspace `AGENTS.md` to the system prompt |

---

## Heterogeneous Multi-Model Combos

Multi-agent presets are most effective when different roles use different models or reasoning efforts—for example, pairing a strong reasoning model for the main loop with a fast, low-cost model for read-only exploration or a second model family for independent code review:

```yaml
# .microharness/presets/main-explore/agent.md
---
name: Main + Explore
description: DeepSeek main agent delegating broad discovery to Gemini Flash Lite
model: deepseek
reasoning_effort: high
include_builtin_subagents: false
tools: [bash, read_file, grep, glob, git_read, write_file, edit_file, subagent]
---
Delegate open-ended multi-file discovery to `explore` subagents; edit and test directly in the main loop.
```

```yaml
# .microharness/presets/main-explore/subagents/explore.md
---
name: explore
description: Fast read-only codebase exploration specialist
model: google/gemini-3.5-flash-lite
effort: low
tools: [read_file, grep, glob, git_read]
read_only: true
---
Search and read relevant files in parallel and report exact file paths, symbols, and line numbers.
```

Notes from benchmark runs:
- Setting `include_builtin_subagents: false` keeps `Available Subagents` scoped strictly to the subagents defined in the preset's `subagents/` folder.
- Avoid overly tight `max_steps` caps on subagents working in large repositories; leaving `max_steps` unset inherits `subagent_max_steps` (`null` by default).

---

## Precedence & Security Boundary

- **Config precedence (lowest to highest):** Built-in defaults → Global `~/.microharness/config.json` → Preset (`config.json`, then `agent.md` frontmatter) → Project `.microharness/config.json` (unless `inherit_project: false` / `isolated: true`) → Explicit CLI flags / `Harness(...)` kwargs.
- **Resource precedence:** Global → Preset → Project (later layers shadow earlier ones with the same ID).
- **Trust boundary (`tools/` & command hooks):** Presets discovered in `<global_dir>/presets/` are trusted automatically. Presets loaded from a repository (`.microharness/presets/`) or an explicit path (`--preset ./dir`) have their `tools/*.py` plugins and command hooks skipped unless `--trust-project-tools` (or `"trust_project_tools": true` in `~/.microharness/config.json`) is set.
