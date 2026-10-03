# microharness

A minimalist, zero-dependency, and fully pluggable AI agent microframework in Python.

Built with **zero external runtime dependencies** (standard library only) to teach the core architectural boundaries of agentic systems. Compatible with modern packaging tools like [`uv`](https://github.com/astral-sh/uv).

---

## 💡 The Core Mental Model: Harness vs. Loop

| Component | Responsibility | Location in `microharness` |
| :--- | :--- | :--- |
| **Agent Loop** *(The Engine)* | Pure Thought $\to$ Action $\to$ Observation cycle. Agnostic to benchmarks, sandboxes, and models. | [`core/loop.py`](file:///Users/dimakagan/Projects/mini-harness/core/loop.py) (~40 LOC) |
| **Harness** *(The Chassis)* | The outer environment: workspace directory, sandboxing, budgets, memory compaction, telemetry/tracing, and grading. | [`core/harness.py`](file:///Users/dimakagan/Projects/mini-harness/core/harness.py) |

---

## 🧭 Configuration, Steering, Skills, Commands & Tool Plugins

`microharness` supports cascading configuration at two distinct scopes:
1. **Global User Baseline** (`~/.microharness/` or `--global-dir`)
2. **Project-Local Scope** (`.microharness/` in current workspace root)

### 1. Configuration Files (`config.json`)
Configure your baseline model, endpoint, API key, and limits globally or override them per project:
* **Global Baseline**: `~/.microharness/config.json`
* **Project Override**: `.microharness/config.json`

```json
{
  "endpoint": "https://openrouter.ai/api/v1",
  "model": "anthropic/claude-3.5-sonnet",
  "api_key": "sk-or-v1-...",
  "max_context_chars": 32000,
  "trigger_ratio": 0.80,
  "max_tool_chars": 2000,
  "sandbox": false,
  "models": {
    "fast": { "model": "gemini-2.5-flash", "provider": "google", "description": "Fast flash model" },
    "smart": { "model": "anthropic/claude-3.5-sonnet", "provider": "openrouter", "description": "Frontier reasoning model" }
  }
}
```

#### Precedence Hierarchy:
1. **CLI Flags** (`--model`, `--endpoint`, `--api-key`, `--sandbox`, `-d`) — *Highest priority*
2. **Environment Variables / Local `.env`** (`MODEL_ENDPOINT`, `MODEL_NAME`, `MODEL_API_KEY`)
3. **Project Config** (`.microharness/config.json`) — *Project overrides*
4. **Global Config** (`~/.microharness/config.json`) — *Machine baseline*
5. **Hardcoded Fallbacks** (`MockModel`, default prompts) — *Lowest priority*

### 2. Cascading Instructions (`AGENTS.md`)
You can provide personal agent preferences globally, while projects can specialize or override them:
* **Global Baseline**: `~/.microharness/AGENTS.md` (e.g., your personal coding habits, formatting preferences).
* **Project Conventions**: `.microharness/AGENTS.md` or `./AGENTS.md` in the repo root (e.g., repo-specific test commands, architecture rules).

The harness loads both in cascading order (Global baseline first, Project local second) into the system prompt.

### 3. Progressive Skills Disclosure (`.microharness/skills/` & Shadowing) — Agent-Facing
Define reusable task workflows for the **agent** in `.microharness/skills/<skill_name>/SKILL.md`:
* **Global Skills** (`~/.microharness/skills/`): Available across all projects on your machine.
* **Project Skills** (`.microharness/skills/`): Project-specific tools and workflows (namespaced cleanly without polluting the project root).
* **Skill Shadowing**: If a project defines a skill with the same name as a global skill, the project version cleanly overrides it.
* **Progressive Disclosure**: Full skill files are never dumped into the prompt. Instead, the harness injects a 1-line index with scope badges (`[global]` or `[project]`).
* **Lazy Loading & Security**: The agent reads full instructions on demand via `read_file`. Global skills are authorized for reading, while all file writes remain strictly sandboxed to the project directory.

### 4. Custom Slash Commands (`.microharness/commands/*.md`) — User-Facing
Define reusable prompt templates and macros for the **user** in `.microharness/commands/<name>.md`:
```markdown
# Review code changes
Review the git diff or specified files: $ARGUMENTS.
Check for:
1. Correctness and edge-case logic bugs
2. Unit test coverage
3. Adherence to conventions in AGENTS.md
```
* **Skills vs. Commands**: Skills teach the **agent** how to perform complex workflows; Commands give the **user** prompt shortcuts.
* **Invocation**: Type `/review src/auth.py` in interactive chat or pass `microharness "/review src/auth.py"`.
* **Argument Substitution**: `$ARGUMENTS` or `{args}` is substituted with your trailing arguments.
* **Dynamic Help & Shadowing**: `/help` lists all discovered custom commands with their descriptions and scopes (`[project]` or `[global]`). Project commands shadow global ones.
 
### 5. Custom Tool Plugins (`.microharness/tools/*.py`) — Zero-Commit Extensibility
Add new capabilities to the agent dynamically without modifying harness core code or making git commits:
* **Global Tools** (`~/.microharness/tools/*.py`): Available across all projects on your machine (always trusted and loaded automatically).
* **Project Tools** (`.microharness/tools/*.py`): Scoped to the current repository. Requires explicit opt-in via `--trust-project-tools` (or `"trust_project_tools": true` in `~/.microharness/config.json`) to prevent untrusted code execution from cloned repositories.
* **Tool Shadowing & Overrides**: A project tool with the same `name` overrides the global version. If a custom plugin shares a name with a built-in tool (e.g. `bash`), it will override it and emit a warning to ensure full visibility.

#### Authoring Conventions
A tool plugin file can follow any of these patterns:

##### Option A: Standalone Tool Class (Simplest)
Any class conforming to the `Tool` protocol (`name`, `description`, `parameters`, `execute`):
```python
# ~/.microharness/tools/web_fetch.py or .microharness/tools/calculator.py
class CalculatorTool:
    name = "calculate"
    description = "Evaluates basic math expressions safely"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "The math expression (e.g. '24 * 7')"}
        },
        "required": ["expression"],
    }

    def execute(self, expression: str) -> tuple[int, str]:
        try:
            return 0, str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as exc:
            return 1, f"Error: {exc}"
```

##### Option B: Environment-Aware Factory (`get_tools`)
Use `get_tools(env)` if your tool needs access to the harness execution environment (e.g. `env.workspace_dir`):
```python
# .microharness/tools/db_tool.py
class DatabaseTool:
    name = "query_db"
    description = "Queries a project SQLite database"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(self, env=None):
        self.workspace = env.workspace_dir if env else "."

    def execute(self, query: str) -> tuple[int, str]:
        # Query local database in self.workspace
        return 0, "Query results..."

def get_tools(env=None):
    return [DatabaseTool(env)]
```

##### Option C: Module-Level `tools` List
Export an explicit list of pre-instantiated tool instances:
```python
# .microharness/tools/helpers.py
tools = [ToolOne(), ToolTwo()]
```

---

## 🛠️ Working on Existing Projects

By default, the agent runs directly in your **current working directory** (like Claude Code or Aider). It sees all your existing files, respects git, and resumes right where you left off:

```bash
# Operates directly on the current directory:
uv run microharness "Inspect the files in this directory and explain the architecture"

# Or point to any other project directory with -d:
uv run microharness -d /path/to/my-project "Fix the bug in auth.py"
```

If you ever want an isolated disposable scratchpad (e.g. for testing without touching real files), pass `--sandbox`:
```bash
uv run microharness --sandbox "Experiment with writing a new scraper"
```

### 🐳 Docker Sandboxing (via OrbStack or Docker Desktop)
For 100% containerized process, operating system, and filesystem isolation (e.g. running untrusted code, heavy test suites, or SWE benchmarks), pass `--docker`:
```bash
# Run educational demo inside an isolated Linux Docker container
uv run microharness --demo calculator --docker

# Run the Mini-SWE benchmark suite inside Docker
uv run microharness --benchmark mini-swe --docker

# Specify a custom Docker image (default: python:3.11)
uv run microharness --docker --docker-image python:3.11 "Run pytest and fix bugs"
```
* **Instant Hybrid I/O**: The workspace directory is mounted at `/workspace`. File reading, writing, and editing occur on the host filesystem with zero container latency, while all bash commands execute inside the isolated Docker Linux container via `docker exec`.
* **Zero Runtime Dependencies**: Operates entirely via Python's standard library `subprocess` without third-party Docker SDKs. Works out of the box with **OrbStack** or standard Docker Desktop.
* **Automatic Teardown**: Containers are automatically stopped and removed upon task completion with zero orphaned background containers.

If you prefer a one-and-done batch run (e.g. for CI), pass `--batch`:
```bash
uv run microharness "Run pytest and fix any failures" --batch
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

* **`/model [name|number]`**: Inspect available models or switch active model dynamically on the fly (e.g. `/model fast` or `/model 2`). In TUI mode, Tab autocompletes configured model names. A name is matched against the `models` aliases in `config.json`, their underlying model ids, and the provider registry (`openai` or `openai/gpt-4o` both work). An unrecognized name is **rejected with the list of valid models** rather than silently constructing a client for a model that does not exist. The `--model` startup flag applies the same recognition rules but only warns on stderr, so it remains usable for endpoints whose provider is not in the registry.
* **`/resume [id|number]`**: List previous sessions in `~/.microharness/runs/` (with workspace affinity) or resume conversation history from an earlier session in-place. Discovery reads only the newest sessions first and falls back to a full scan when your repository has no recent session, so an older session for the same repo stays reachable.
* **`/cache [5m|1h|off]`**: Inspect or switch prompt caching duration on the fly (`5m` standard ephemeral, `1h` extended beta, or `off`).
* **`/status`**: Displays detailed memory metrics (system, user, assistant, and tool counts + budget percentage) and prompt cache hit rates.
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
*(All 250+ unit and integration tests run in ~7s. Correctness guarantees are enforced by mutation testing — see SPEC.md §4.)*

### 3. Interactive and TUI Sessions
```bash
# Launch interactive mode (or full-screen TUI with --tui):
uv run microharness
uv run microharness --tui

# Resume previous interactive session:
uv run microharness --resume

# Resume a specific session by ID:
uv run microharness --resume run-2026-10-03-xxxx
```

### 4. Run Mini-SWE for Rapid Sanity Testing
Run the lightweight 5-task sanity test suite before launching heavy benchmarks:
```bash
# Run the entire Mini-SWE sanity benchmark suite:
uv run microharness --benchmark mini-swe

# Filter to a specific challenge (e.g. rate_limiter or query_builder):
uv run microharness --benchmark mini-swe --task rate_limiter

# Built-in demo tasks:
uv run microharness --demo calculator
uv run microharness --demo greeting
```

---

## 🏆 Mini-SWE Benchmark Suite (Sanity & Smoke Testing)

`microharness` includes **Mini-SWE**, a fast, lightweight, pure-Python benchmark designed primarily for **rapid sanity testing, local iteration, and CI smoke testing**:

* **Sanity Testing Focus**: Validates the agent's tool-calling loop, multi-turn file editing, memory compaction, and test assertion grading in seconds—without burning model quota or waiting on multi-hour containerized benchmarks.
* **Pure Python & Zero Docker**: Operates entirely with Python's standard library. No multi-gigabyte container downloads, instant startup.
* **Anti-Cheating Evaluator (`PristineTestEvaluator`)**: Evaluators restore pristine test files prior to running verification. An agent cannot bypass tests by deleting or rewriting assertions in the test suite.
* **5 Realistic Engineering Challenges**:
  1. `rate_limiter`: Token Bucket rate limiter losing fractional refills under rapid polling.
  2. `circular_serializer`: Circular JSON serializer cycle detection & ISO-8601 formatting.
  3. `auth_middleware`: HTTP auth middleware case-insensitive headers & whitespace tolerance.
  4. `lru_ttl_cache`: LRU cache with TTL updating existing keys & refreshing MRU order.
  5. `query_builder`: SQL query builder AST precedence grouping & single-quote escaping.
* **Automated Scorecard & JSON Telemetry**: Generates a formatted summary table in the terminal and exports comprehensive JSON results to `~/.microharness/runs/benchmark_report.json`.

---

## 🌐 DeepSWE-mini Benchmark Suite (Harbor Format & Docker)

`microharness` natively supports **DeepSWE-mini** (curated from [Datacurve-AI DeepSWE](https://github.com/datacurve-ai/deep-swe) and [`LocalLLaMA/deepswe-mini`](https://huggingface.co/datasets/LocalLLaMA/deepswe-mini)):

* **16 Curated Real-World Repositories**: Across Python, Go, and TypeScript (e.g. `aiomonitor`, `bandit`, `ipython`, `narwhals`, `textual`, `prometheus`, `valibot`, `dasel`).
* **Harbor Standard Compliance**: Automatically parses `task.toml` metadata (via Python 3.11's standard library `tomllib`), pulls task ECR container images, mounts behavioral verifiers, and grades via `tests/grader.py`.
* **Zero Dependency Container Sandboxing**: Spawns isolated Docker containers via **OrbStack** or Docker Desktop, syncs repo code at NVMe speed, and automatically captures `model.patch` diffs.

```bash
# Clone tasks once (only ~3.5MB repository metadata):
git clone --depth 1 https://github.com/datacurve-ai/deep-swe datasets/deep-swe

# Run the entire 16-task DeepSWE-mini benchmark suite in Docker:
uv run main.py --benchmark deepswe-mini --docker

# Run a specific task with custom step & timeout budgets:
uv run main.py --benchmark deepswe-mini --task aiomonitor --max-steps 500 --timeout 7200 --docker

# Run with official Datacurve Pier framework using microharness adapter:
uvx --from datacurve-pier pier run -p datasets/deep-swe/tasks/aiomonitor-task-snapshots-diff \
  --agent-import-path benchmarks.pier_agent:MicroharnessAgent --env docker
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
    global_dir="~/.microharness",  # Global config, instructions & skills
    tools=[bash_tool, my_tool],  # Custom tools (dynamically documented in system prompt)
)
harness.run_interactive("Help me refactor the database layer")
```

---

## 🔑 Key Educational Paradigms Demonstrated

1. **In-Place Project Operation vs. Sandboxing (`core/environment.py`)**
   - By default, operates directly on existing project files with zero root directory pollution.
   - Optional `--sandbox` creates an isolated scratchpad in `~/.microharness/runs/` (or `--runs-dir`).
   - Prevents path traversal outside project boundaries while permitting read access to authorized global directories (`allowed_read_dirs`).
   - Executes bash with timeouts and combined stdout/stderr.

2. **Cascading Configuration & Steering (`core/harness.py`)**
   - Loads global defaults from `~/.microharness/config.json`.
   - Injects global user instructions (`~/.microharness/AGENTS.md`) and project instructions (`./AGENTS.md`) in cascading order.
   - Discovers `skills/` and `commands/` globally and locally; project definitions cleanly shadow global ones.
   - Indexes skills lazily with scope badges to conserve context tokens.

3. **Targeted File Editing vs. Full Rewrites (`core/tools.py`)**
   - Built-in `edit_file(path, old_str, new_str)` enforces exact, unique string replacement.
   - Prevents token wastage and LLM hallucination bugs common when rewriting large files with `write_file`.

4. **Dynamic Tool Introspection (`core/tools.py`)**
   - `ToolRegistry.to_system_prompt_snippet()` automatically reflects registered tools in the system prompt. Custom tools require zero prompt tweaking.

5. **Large Log Handling & Spillover (`core/tools.py`)**
   - Massive command outputs are head + tail truncated.
   - The full, raw output is spilled to `~/.microharness/runs/<run_id>/logs/` so it never pollutes the user's project root.

6. **Proactive 80% Context Compaction & Memory Optimization (`core/memory.py`)**
   - Proactively prunes older tool observations at an 80% high-water mark (`trigger_ratio = 0.80`), keeping 20% reserved for output tokens and model reasoning.
   - **Enriched Tombstones with Log Pointers**: Collapsed tool outputs embed the exact spilled log file path (`Full log: <path>. Use read_file to inspect if needed.`), giving the agent a deterministic bridge to inspect full outputs.
   - **Historical Reasoning Pruning**: Automatically prunes transient `<think>...</think>` and `<thinking>...</thinking>` reasoning streams from older assistant turns to reclaim thousands of tokens for reasoning models (DeepSeek-R1, Claude 3.7 Thinking, o1/o3).
   - **Session Files Modified Ledger**: Injects and maintains an active session ledger (`[Session Files Modified: file1.py, file2.py]`) across compactions at the safe prefix.
   - Preserves system prompt, initial task, and recent working turns.

7. **Git Patch Export (`patch.diff`) & Modern CLI Tracing (`core/tracer.py`)**
   - Automatic git diff extraction: generates `patch.diff` for coding tasks in git repositories.
   - **Modern Developer CLI Output**: Eliminates academic ReAct tags (`Thought:` / `Action:` / `Observation:`) in favor of natural assistant speech, `$ <command>` execution, `● Read/Write/Edit <path>`, real-time duration headers, and clean output gutters.
   - Persists machine-readable `trace.json` and human-readable `trace.md` in `~/.microharness/runs/<run_id>/` for every run.

8. **Automated Evaluation (`core/evaluator.py`)**
   - Evaluators test whether the agent actually succeeded (e.g. running test commands, asserting file contents, or Docker-based SWE-bench graders).

---

## 📁 Project Structure

```
microharness/
├── pyproject.toml      # Modern PEP 621 packaging for uv & hatchling
├── uv.lock             # Deterministic lockfile generated by uv
├── SPEC.md             # Detailed specification & design rationale
├── README.md           # Quickstart and overview
├── AGENTS.md           # Project conventions for agents
├── .microharness/      # Clean project-level configuration namespace
│   ├── skills/         # Progressive disclosure skills
│   │   └── run-tests/
│   │       └── SKILL.md
│   └── commands/       # Custom slash commands (user prompt shortcuts)
│       └── review.md
├── .env.example        # Environment variable template
├── main.py             # Runnable demonstration & CLI entrypoint
├── benchmarks/         # SWE benchmark suites & runners
│   ├── mini_swe.py     # 5 real-world pure-Python SWE tasks & PristineTestEvaluator
│   ├── harbor.py       # Harbor / DeepSWE-mini benchmark loader & Docker verifier adapter
│   ├── pier_agent.py   # Datacurve Pier framework agent adapter (MicroharnessAgent)
│   └── runner.py       # Benchmark orchestrator, scorecard table & incremental JSON report export
├── core/
│   ├── types.py        # Clean dataclasses (Message, ToolCall, Task, Trajectory, RunResult)
│   ├── prompts.py      # System prompt templates
│   ├── environment.py  # Workspace & bash runner (supports in-place and sandbox)
│   ├── tools.py        # Pluggable tool protocol & built-ins (bash, read_file, write_file)
│   ├── memory.py       # Context manager & 80% high-water mark compaction
│   ├── model.py        # Model protocol, generic HttpChatModel & MockModel
│   ├── loop.py         # Native tool-calling loop (~40 lines)
│   ├── evaluator.py    # Automated graders (CommandTestEvaluator, FileContentEvaluator)
│   ├── tracer.py       # Telemetry collector, modern CLI renderer & trace exporter
│   └── harness.py      # Harness orchestrator with global config & commands support
└── tests/
    ├── test_benchmarks.py  # Mini-SWE baseline failure, anti-cheating & reference tests
    ├── test_harbor.py      # Harbor task parsing, Docker image validation & evaluator tests
    ├── test_environment.py # Path sandboxing & bash timeout tests
    ├── test_tools.py       # Tools, truncation & disk spillover tests
    ├── test_memory.py      # Conversation history, stats & 80% compaction tests
    ├── test_model.py       # Endpoint normalization, serialization & mock model tests
    ├── test_evaluator.py   # Grader assertions tests
    └── test_harness.py     # End-to-end lifecycle, global config, commands & guardrails tests
```
