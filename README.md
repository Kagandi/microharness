# mini-harness

An educational, minimalist, and fully pluggable AI agent harness in Python.

Built with **zero external dependencies** (standard library only) to teach the core architectural boundaries of agentic systems.

---

## 💡 The Core Mental Model: Harness vs. Loop

| Component | Responsibility | Examples in `mini-harness` |
| :--- | :--- | :--- |
| **Agent Loop** *(The Engine)* | Pure Thought $\to$ Action $\to$ Observation cycle. Agnostic to benchmarks, sandboxes, and evaluations. | [`core/loop.py`](file:///Users/dimakagan/Projects/mini-harness/core/loop.py) (~40 LOC) |
| **Harness** *(The Chassis)* | The outer environment: sandboxing, guardrails/budgets, memory compaction, telemetry/tracing, and post-run grading. | [`core/harness.py`](file:///Users/dimakagan/Projects/mini-harness/core/harness.py) |

---

## 🔑 Key Educational Paradigms Demonstrated

1. **Workspace Sandboxing (`core/environment.py`)**
   - Each task runs in an isolated directory (`.runs/<task_id>_<timestamp>/workspace/`).
   - Prevents path traversal outside the sandbox.
   - Executes bash with timeouts and combined stdout/stderr.

2. **Large Log Handling & Spillover (`core/tools.py`)**
   - When a command generates massive output, it is **not** dumped directly into the LLM context.
   - Head + tail truncation preserves the initial setup and final stack trace/errors.
   - The full, raw output is spilled to `.logs/step_N.log` inside the workspace so the model can inspect it using tools (e.g. `grep` or `read_file`).

3. **Context & Memory Compaction (`core/memory.py`)**
   - Tracks the ongoing conversation history.
   - When context size exceeds a threshold, older tool observations in middle turns are compacted into concise markers (`[Tool output collapsed...]`) while preserving the system prompt, initial task, and recent turns.

4. **Structured Tracing & Debugging (`core/tracer.py`)**
   - Real-time terminal streaming of thoughts, actions, observations, truncations, and compactions.
   - Persists machine-readable [`trace.json`](file:///Users/dimakagan/Projects/mini-harness/.runs) and human-readable [`trace.md`](file:///Users/dimakagan/Projects/mini-harness/.runs) for every run.

5. **Automated Evaluation (`core/evaluator.py`)**
   - Evaluators test whether the agent actually succeeded (e.g. running test commands or asserting file contents).

6. **Pluggable Model Architecture (`core/model.py`)**
   - `MockModel`: Deterministic, offline replay model for testing without API keys or costs.
   - `OpenAICompatibleModel`: Direct API client built with Python's standard library `urllib.request` (works with OpenAI, DeepSeek, Ollama, Groq, vLLM, etc.).

---

## 📁 Project Structure

```
mini-harness/
├── SPEC.md             # Detailed specification & design rationale
├── README.md           # Quickstart and overview
├── main.py             # Runnable demonstration
└── core/
    ├── types.py        # Clean dataclasses (Message, ToolCall, Task, Trajectory, RunResult)
    ├── prompts.py      # System prompt templates
    ├── environment.py  # Isolated workspace & bash runner
    ├── tools.py        # Pluggable tool protocol & built-ins (bash, read_file, write_file)
    ├── memory.py       # Context manager & compaction policies
    ├── model.py        # Model protocol (Direct API + MockModel)
    ├── loop.py         # Pure ReAct loop (~40 lines)
    ├── evaluator.py    # Automated graders (CommandTestEvaluator, FileContentEvaluator)
    ├── tracer.py       # Telemetry collector & trace exporter
    └── harness.py      # Harness orchestrator
```

---

## 🚀 Quickstart

### 1. Run the Offline Educational Demo (No API Key needed)
```bash
python3 main.py
```
This executes a full software engineering bug-fix scenario with the built-in deterministic `MockModel`:
- Runs tests and captures the initial test failure.
- Triggers a verbose log to demonstrate log truncation and disk spillover.
- Inspects code, applies the bug fix, and triggers memory compaction.
- Verifies test passage and grades the final workspace state.

### 2. Run with Live LLM (OpenAI / DeepSeek / Ollama)
```bash
export OPENAI_API_KEY="sk-..."
# Optional custom base URL (e.g. Ollama, Groq, DeepSeek):
# export OPENAI_BASE_URL="http://localhost:11434/v1"

python3 main.py --live --model gpt-4o
```

---

## 🔍 Inspecting Traces
Every run produces artifacts under `.runs/<task_id>_<timestamp>/`:
- `trace.md`: Markdown summary of thoughts, actions, tool results, and evaluation.
- `trace.json`: Full JSON record of prompt states, latencies, and tool calls.
- `workspace/`: The real file directory manipulated by the agent.
- `workspace/.logs/`: Full raw outputs of truncated commands.
