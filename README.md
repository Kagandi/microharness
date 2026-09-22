# mini-harness

An educational, minimalist, and fully pluggable AI agent harness in Python.

Built with **zero external runtime dependencies** (standard library only) to teach the core architectural boundaries of agentic systems. Compatible with modern packaging tools like [`uv`](https://github.com/astral-sh/uv).

---

## 💡 The Core Mental Model: Harness vs. Loop

| Component | Responsibility | Location in `mini-harness` |
| :--- | :--- | :--- |
| **Agent Loop** *(The Engine)* | Pure Thought $\to$ Action $\to$ Observation cycle. Agnostic to benchmarks, sandboxes, and models. | [`core/loop.py`](file:///Users/dimakagan/Projects/mini-harness/core/loop.py) (~40 LOC) |
| **Harness** *(The Chassis)* | The outer environment: workspace directory, sandboxing, budgets, memory compaction, telemetry/tracing, and grading. | [`core/harness.py`](file:///Users/dimakagan/Projects/mini-harness/core/harness.py) |

---

## 🧭 Configuration, Steering, Skills & Commands

`mini-harness` supports cascading configuration at two distinct scopes:
1. **Global User Baseline** (`~/.mini-harness/` or `--global-dir`)
2. **Project-Local Scope** (`./` in current workspace root)

### 1. Global Configuration File (`~/.mini-harness/config.json`)
Configure your baseline model, endpoint, API key, and limits once on your machine so you can run `mini-harness` from any repository without copying `.env` files:

```json
{
  "endpoint": "https://openrouter.ai/api/v1",
  "model": "anthropic/claude-3.5-sonnet",
  "api_key": "sk-or-v1-...",
  "max_context_chars": 32000,
  "trigger_ratio": 0.80,
  "max_tool_chars": 2500,
  "sandbox": false
}
```

#### Precedence Hierarchy:
1. **CLI Flags** (`--model`, `--endpoint`, `--api-key`, `--sandbox`, `-d`) — *Highest priority*
2. **Environment Variables / Local `.env`** (`MODEL_ENDPOINT`, `MODEL_NAME`, `MODEL_API_KEY`)
3. **Global Config** (`~/.mini-harness/config.json`) — *Machine baseline*
4. **Hardcoded Fallbacks** (`MockModel`, default prompts) — *Lowest priority*

### 2. Cascading Instructions (`AGENTS.md`)
You can provide personal agent preferences globally, while projects can specialize or override them:
* **Global Baseline**: `~/.mini-harness/AGENTS.md` (e.g., your personal coding habits, formatting preferences).
* **Project Conventions**: `./AGENTS.md` in the repo root (e.g., repo-specific test commands, architecture rules).

The harness loads both in cascading order (Global baseline first, Project local second) into the system prompt.

### 3. Progressive Skills Disclosure (`skills/` & Shadowing) — Agent-Facing
Define reusable task workflows for the **agent** in `skills/<skill_name>/SKILL.md` (or `.skills/`):
* **Global Skills** (`~/.mini-harness/skills/`): Available across all projects on your machine.
* **Project Skills** (`./skills/`): Project-specific tools and workflows.
* **Skill Shadowing**: If a project defines a skill with the same name as a global skill, the project version cleanly overrides it.
* **Progressive Disclosure**: Full skill files are never dumped into the prompt. Instead, the harness injects a 1-line index with scope badges (`[global]` or `[project]`).
* **Lazy Loading & Security**: The agent reads full instructions on demand via `read_file`. Global skills are authorized for reading, while all file writes remain strictly sandboxed to the project directory.

### 4. Custom Slash Commands (`commands/*.md`) — User-Facing
Define reusable prompt templates and macros for the **user** in `commands/<name>.md` (or `.commands/`, globally in `~/.mini-harness/commands/`):
```markdown
# Review code changes
Review the git diff or specified files: $ARGUMENTS.
Check for:
1. Correctness and edge-case logic bugs
2. Unit test coverage
3. Adherence to conventions in AGENTS.md
```
* **Skills vs. Commands**: Skills teach the **agent** how to perform complex workflows; Commands give the **user** prompt shortcuts.
* **Invocation**: Type `/review src/auth.py` in interactive chat or pass `mini-harness "/review src/auth.py"`.
* **Argument Substitution**: `$ARGUMENTS` or `{args}` is substituted with your trailing arguments.
* **Dynamic Help & Shadowing**: `/help` lists all discovered custom commands with their descriptions and scopes (`[project]` or `[global]`). Project commands shadow global ones.

---

## 🛠️ Working on Existing Projects

By default, the agent runs directly in your **current working directory** (like Claude Code or Aider). It sees all your existing files, respects git, and resumes right where you left off:

```bash
# Operates directly on the current directory:
uv run mini-harness "Inspect the files in this directory and explain the architecture"

# Or point to any other project directory with -d:
uv run mini-harness -d /path/to/my-project "Fix the bug in auth.py"
```

If you ever want an isolated disposable scratchpad (e.g. for testing without touching real files), pass `--sandbox`:
```bash
uv run mini-harness --sandbox "Experiment with writing a new scraper"
```

If you prefer a one-and-done batch run (e.g. for CI), pass `--batch`:
```bash
uv run mini-harness "Run pytest and fix any failures" --batch
```

---

## 💬 Continuous Interactive Chat & Meta-Commands

When you chat with the agent, the harness maintains persistent memory and renders a clean **context budget status line** before every turn:

```text
─── 📊 Context: 1,420 / 8,000 chars (17.8%) | Messages: 6 ───
👤 You: 
```

### Slash Commands (No Model Call)
You can inspect or control the harness runtime directly:

* **`/status`**: Displays detailed memory metrics (system, user, assistant, and tool counts + budget percentage).
* **`/compact`**: Manually triggers memory compaction to prune older tool observations and reclaim context space.
* **`/new`**: Starts a fresh conversation context while keeping all your project files intact.
* **`/help`**: Lists available commands.
* **`/exit`**: Finalizes and exports the session trace.

---

## ⚡ Quickstart with `uv`

### 1. Install & Sync Environment
```bash
uv sync
```

### 2. Run the Test Suite
```bash
uv run pytest
```
*(All 58 unit and integration tests run in ~0.9s)*

### 3. Run Predefined Benchmark Demos
```bash
# Calculator SWE bug-fix benchmark with automated grading:
uv run mini-harness --demo calculator

# Lightweight greeting benchmark:
uv run mini-harness --demo greeting
```

---

## 🔑 True Pluggability (Zero Code Modifications)

Because components follow minimal Python `Protocols`, you can plug in any Model, Tool, or Evaluator from the outside without touching a single line of harness code:

```python
from core.harness import Harness

# In-place coding agent on your repo with global and project steering:
harness = Harness(
    model=my_custom_model,  # Or HttpChatModel(endpoint, model, key)
    workspace_dir=".",  # Works directly on your project
    global_dir="~/.mini-harness",  # Global config, instructions & skills
    tools=[bash_tool, my_tool],  # Custom tools (dynamically documented in system prompt)
)
harness.run_interactive("Help me refactor the database layer")
```

---

## 🔑 Key Educational Paradigms Demonstrated

1. **In-Place Project Operation vs. Sandboxing (`core/environment.py`)**
   - By default, operates directly on existing project files.
   - Optional `--sandbox` creates an isolated scratchpad in `.runs/`.
   - Prevents path traversal outside project boundaries while permitting read access to authorized global directories (`allowed_read_dirs`).
   - Executes bash with timeouts and combined stdout/stderr.

2. **Cascading Configuration & Steering (`core/harness.py`)**
   - Loads global defaults from `~/.mini-harness/config.json`.
   - Injects global user instructions (`~/.mini-harness/AGENTS.md`) and project instructions (`./AGENTS.md`) in cascading order.
   - Discovers `skills/` and `commands/` globally and locally; project definitions cleanly shadow global ones.
   - Indexes skills lazily with scope badges to conserve context tokens.

3. **Targeted File Editing vs. Full Rewrites (`core/tools.py`)**
   - Built-in `edit_file(path, old_str, new_str)` enforces exact, unique string replacement.
   - Prevents token wastage and LLM hallucination bugs common when rewriting large files with `write_file`.

4. **Dynamic Tool Introspection (`core/tools.py`)**
   - `ToolRegistry.to_system_prompt_snippet()` automatically reflects registered tools in the system prompt. Custom tools require zero prompt tweaking.

5. **Large Log Handling & Spillover (`core/tools.py`)**
   - Massive command outputs are head + tail truncated.
   - The full, raw output is spilled to `.runs/<run_id>/logs/` so it never pollutes the user's project root.

6. **Proactive 80% Context Compaction & Memory Optimization (`core/memory.py`)**
   - Proactively prunes older tool observations at an 80% high-water mark (`trigger_ratio = 0.80`), keeping 20% reserved for output tokens and model reasoning.
   - **Enriched Tombstones with Log Pointers**: Collapsed tool outputs embed the exact spilled log file path (`Full log: <path>. Use read_file to inspect if needed.`), giving the agent a deterministic bridge to inspect full outputs.
   - **Historical Reasoning Pruning**: Automatically prunes transient `<think>...</think>` and `<thinking>...</thinking>` reasoning streams from older assistant turns to reclaim thousands of tokens for reasoning models (DeepSeek-R1, Claude 3.7 Thinking, o1/o3).
   - **Session Files Modified Ledger**: Injects and maintains an active session ledger (`[Session Files Modified: file1.py, file2.py]`) across compactions at the safe prefix.
   - Preserves system prompt, initial task, and recent working turns.

7. **Git Patch Export (`patch.diff`) & Tracing (`core/tracer.py`)**
   - Automatic git diff extraction: generates `patch.diff` for coding tasks in git repositories.
   - Real-time terminal streaming of thoughts, actions, observations, truncations, and compactions.
   - Persists machine-readable [`trace.json`](file:///Users/dimakagan/Projects/mini-harness/.runs) and human-readable [`trace.md`](file:///Users/dimakagan/Projects/mini-harness/.runs) for every run.

8. **Automated Evaluation (`core/evaluator.py`)**
   - Evaluators test whether the agent actually succeeded (e.g. running test commands or asserting file contents).

---

## 📁 Project Structure

```
mini-harness/
├── pyproject.toml      # Modern PEP 621 packaging for uv & hatchling
├── uv.lock             # Deterministic lockfile generated by uv
├── SPEC.md             # Detailed specification & design rationale
├── README.md           # Quickstart and overview
├── AGENTS.md           # Project conventions for agents
├── skills/             # On-demand skills (progressive disclosure)
│   └── run-tests/
│       └── SKILL.md
├── commands/           # Custom slash commands (user prompt shortcuts)
│   └── review.md
├── .env.example        # Environment variable template
├── main.py             # Runnable demonstration & CLI entrypoint
├── core/
│   ├── types.py        # Clean dataclasses (Message, ToolCall, Task, Trajectory, RunResult)
│   ├── prompts.py      # System prompt templates
│   ├── environment.py  # Workspace & bash runner (supports in-place and sandbox)
│   ├── tools.py        # Pluggable tool protocol & built-ins (bash, read_file, write_file)
│   ├── memory.py       # Context manager & 80% high-water mark compaction
│   ├── model.py        # Model protocol, generic HttpChatModel & MockModel
│   ├── loop.py         # Pure ReAct loop (~40 lines)
│   ├── evaluator.py    # Automated graders (CommandTestEvaluator, FileContentEvaluator)
│   ├── tracer.py       # Telemetry collector & trace exporter
│   └── harness.py      # Harness orchestrator with global config & commands support
└── tests/
    ├── test_environment.py # Path sandboxing & bash timeout tests
    ├── test_tools.py       # Tools, truncation & disk spillover tests
    ├── test_memory.py      # Conversation history, stats & 80% compaction tests
    ├── test_model.py       # Endpoint normalization, serialization & mock model tests
    ├── test_evaluator.py   # Grader assertions tests
    └── test_harness.py     # End-to-end lifecycle, global config, commands & guardrails tests
```
