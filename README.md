# mini-harness

An educational, minimalist, and fully pluggable AI agent harness in Python.

Built with **zero external dependencies** (standard library only) to teach the core architectural boundaries of agentic systems.

---

## 💡 The Core Mental Model: Harness vs. Loop

| Component | Responsibility | Location in `mini-harness` |
| :--- | :--- | :--- |
| **Agent Loop** *(The Engine)* | Pure Thought $\to$ Action $\to$ Observation cycle. Agnostic to benchmarks, sandboxes, and models. | [`core/loop.py`](file:///Users/dimakagan/Projects/mini-harness/core/loop.py) (~40 LOC) |
| **Harness** *(The Chassis)* | The outer environment: sandboxing, guardrails/budgets, memory compaction, telemetry/tracing, and post-run grading. | [`core/harness.py`](file:///Users/dimakagan/Projects/mini-harness/core/harness.py) |

---

## 🔑 True Pluggability (Zero Code Modifications)

Because components follow minimal Python `Protocols`, you can plug in any Model, Tool, or Evaluator from the outside without touching a single line of harness code:

```python
from core.harness import Harness

# Plug in ANY object or adapter that implements `generate(messages, tools)`
harness = Harness(
    model=my_custom_model,      # Or HttpChatModel(endpoint, model, key)
    tools=[bash_tool, my_tool], # Custom tools
    evaluator=my_evaluator,     # Custom evaluator
)
harness.run(task)
```

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
   - Tracks conversation history.
   - When context size exceeds a threshold, older tool observations in middle turns are compacted into concise markers (`[Tool output collapsed...]`) while preserving the system prompt, initial task, and recent turns.

4. **Structured Tracing & Debugging (`core/tracer.py`)**
   - Real-time terminal streaming of thoughts, actions, observations, truncations, and compactions.
   - Persists machine-readable [`trace.json`](file:///Users/dimakagan/Projects/mini-harness/.runs) and human-readable [`trace.md`](file:///Users/dimakagan/Projects/mini-harness/.runs) for every run.

5. **Automated Evaluation (`core/evaluator.py`)**
   - Evaluators test whether the agent actually succeeded (e.g. running test commands or asserting file contents).

---

## 📁 Project Structure

```
mini-harness/
├── SPEC.md             # Detailed specification & design rationale
├── README.md           # Quickstart and overview
├── .env.example        # Environment variable template
├── main.py             # Runnable demonstration
├── core/
│   ├── types.py        # Clean dataclasses (Message, ToolCall, Task, Trajectory, RunResult)
│   ├── prompts.py      # System prompt templates
│   ├── environment.py  # Isolated workspace & bash runner
│   ├── tools.py        # Pluggable tool protocol & built-ins (bash, read_file, write_file)
│   ├── memory.py       # Context manager & compaction policies
│   ├── model.py        # Model protocol, generic HttpChatModel & MockModel
│   ├── loop.py         # Pure ReAct loop (~40 lines)
│   ├── evaluator.py    # Automated graders (CommandTestEvaluator, FileContentEvaluator)
│   ├── tracer.py       # Telemetry collector & trace exporter
│   └── harness.py      # Harness orchestrator
└── tests/
    ├── test_environment.py # Path sandboxing & bash timeout tests
    ├── test_tools.py       # Tools, truncation & disk spillover tests
    ├── test_memory.py      # Conversation history & compaction tests
    ├── test_model.py       # Endpoint normalization & mock model tests
    ├── test_evaluator.py   # Grader assertions tests
    └── test_harness.py     # End-to-end lifecycle & guardrails tests
```

---

## 🧪 Running the Test Suite

Run the automated test suite with Python's standard library (no installs required):

```bash
python3 -m unittest discover -s tests -v
```

All 16 unit and integration tests run in ~1.3s.

---

## 🚀 Quickstart

### 1. Offline Educational Demo (No API Key / Endpoint needed)
```bash
python3 main.py
```

### 2. Connect to Any Live Endpoint

#### OpenRouter:
```bash
python3 main.py \
  --endpoint https://openrouter.ai/api/v1 \
  --model cohere/north-mini-code:free \
  --api-key sk-or-v1-...
```

#### Local Endpoint (Ollama / vLLM / LM Studio):
```bash
# Ollama default:
python3 main.py \
  --endpoint http://localhost:11434/v1 \
  --model qwen2.5-coder
```

Or copy `.env.example` to `.env` and simply run:
```bash
python3 main.py
```

---

## 🔍 Inspecting Traces
Every run produces artifacts under `.runs/<task_id>_<timestamp>/`:
- `trace.md`: Markdown summary of thoughts, actions, tool results, and evaluation.
- `trace.json`: Full JSON record of prompt states, latencies, and tool calls.
- `workspace/`: The real file directory manipulated by the agent.
- `workspace/.logs/`: Full raw outputs of truncated commands.
