## .github/copilot-instructions.md — quick guide for AI contributors

Purpose
- Help an AI coding assistant be immediately productive in this Proof-of-Concept "intent-routed agent" repo (see `todo.md`).

Big picture (what the code should do)
- This repo is a POC: an agent that maps user intent → picks tools → runs a small workflow graph → aggregates results → makes a simple inference → applies a feedback loop → returns an answer plus a short trace.
- Core intent classes to expect (from `todo.md`): `metrics lookup`, `knowledge lookup`, `calc/compare`.

Architecture and data flow (concise)
- User query → Intent classifier → Orchestrator / graph engine (2–4 nodes) → Tool nodes (HTTP REST, vector-DB/knowledge, utility) → Aggregator → Inference & feedback → Final answer + run trace.
- Instrument outputs at node boundaries: record selected intent, tools called, node outputs, and confidence score for the inference step.

What to look for in the repo (first places to inspect)
- `todo.md` — authoritative project description and acceptance criteria.
- Look for files or folders named `agent`, `orchestrator`, `tools`, `mocks`, `fixtures`, or `service` for implementation hints.
- If you find Python files, prefer existing virtualenv/requirements files; otherwise, follow common Python layout (src/, tests/, requirements.txt).

Project-specific conventions and rules
- The trace returned with answers must include: `selected_intent`, `tools_called` (list), `why` (short reasoning), and `confidence` (0.0–1.0). Follow this exact shape when adding examples/tests.
- External dependencies should be safely stubbed in CI/local dev with small JSON fixtures or a tiny mock HTTP server (FastAPI suggested in `todo.md`). If a `mocks/` or `fixtures/` folder exists, reuse it.
- Orchestration nodes should be small, single-responsibility functions (e.g., `understand`, `fetch`, `aggregate`, `answer`) so components are testable and combinable.

Examples from this project to mirror
- Intent types: `metrics lookup`, `knowledge lookup`, `calc/compare` (use these labels in tests and classifiers).
- Tool types required: one HTTP/REST tool, one knowledge/vector-search tool, and one utility tool (calculator/SQL/filesystem).
- Workflow composition example (recommended shape): `understand -> fetch -> aggregate -> answer`.

Testing & verification (how an AI should add tests)
- Add small unit tests that call node functions directly and assert the run-trace shape and the final answer format.
- For integration-style tests, use JSON fixtures and/or a small mock HTTP endpoint to avoid network calls.

When editing or creating files
- Keep changes small and focused. Each PR should include: (a) code, (b) a short README or comment describing the node’s responsibility, and (c) at least one unit test showing expected inputs → outputs and run-trace.
- If you add a new tool implementation, include a lightweight mock fixture under `fixtures/` and a test that uses it.

What not to do
- Don't assume a production-grade infra (no databases, cloud services, or long-running schedulers unless present). If external services are required, prefer mocks.
- Don't change the required run-trace fields or intent labels without updating `todo.md` and adding a migration note.

If you need to run or build the project
- If you see Python code: create a venv and install deps from `requirements.txt` or `pyproject.toml` when present. Typical commands:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

- Run tests with `pytest` if tests exist.

Notes and priorities for an AI agent
- Prioritize delivering a minimal working orchestration: intent classifier + 2 tools + an aggregator + trace output.
- Prefer deterministic, testable behavior over clever heuristics.

Files created/edited by AI
- When you add code, create/update a top-level `README.md` or an `ARCHITECTURE.md` describing the implemented node graph and where fixtures live.

If anything in this guide is unclear or you find conflicting files, ask for clarification and show the minimal diff you propose.

---

See `todo.md` for the full POC acceptance criteria.