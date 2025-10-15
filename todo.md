POC: Build a small “intent-routed agent” that takes a user’s natural-language query, chooses the right tools, runs a workflow, aggregates results, makes an inference, applies a simple feedback loop, and returns a final answer with a short trace of what it did.

What to implement

Intent → tool selection
Classify the user’s query (e.g., “metrics lookup”, “knowledge lookup”, “calc/compare”).
Use LangChain or LangGraph to route the request to the right nodes/tools.
Tools (at least three)
One HTTP/REST tool (mock or real; e.g., a metrics API or any public REST).
One knowledge tool (vector search over a small doc set you provide).
One utility tool (calculator, simple SQL over a local table, or a filesystem/doc loader).
You may stub external services with small JSON fixtures or a tiny FastAPI mock.
Workflow orchestration (nodes/graph)
Compose 2–4 nodes (e.g., “understand → fetch → aggregate → answer”).
Aggregate outputs from multiple tools when needed.
Inference + feedback loop
Do a simple inference step (e.g., “is latency above threshold?”, “which source is most reliable?”).
Add a feedback loop: if confidence is low or data is missing, either re-query a different tool or ask the user one clarifying question, then proceed.
Return an answer with a short run trace
Final answer + a brief trace: selected intent, tools called, and why.
Submission requirements:

A brief write-up explaining the architecture, design choices, and how the feedback loop is handled.
The complete, runnable code for the agent.