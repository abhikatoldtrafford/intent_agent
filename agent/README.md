

# Agent Documentation

## Overview

An **intent-routed agent** built with LangGraph that:
- Classifies user queries by intent (metrics, knowledge, calculation)
- Selects and executes appropriate tools
- Aggregates results from multiple sources
- Performs inference and analysis
- Implements confidence-based feedback loop
- Returns answers with complete execution trace

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AGENT SYSTEM                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐ │
│  │   Intent    │───▶│Tool Selection│───▶│Tool Execution  │ │
│  │Classification│    │              │    │                │ │
│  └─────────────┘    └──────────────┘    └────────────────┘ │
│                                                ▲             │
│                                                │             │
│  ┌─────────────┐    ┌──────────────┐    ┌─────┴──────────┐ │
│  │  Response   │◀───│   Feedback   │◀───│  Aggregation   │ │
│  │  Formatting │    │    Check     │    │  & Inference   │ │
│  └─────────────┘    └──────────────┘    └────────────────┘ │
│                           │ retry                            │
│                           └──────────────────────────────────┘
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        TOOLS LAYER                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Metrics    │  │ Knowledge  │  │  SQL     │  │Calculator│ │
│  │ API (HTTP) │  │ RAG (FAISS)│  │ Database │  │  Tool   │ │
│  └────────────┘  └────────────┘  └──────────┘  └─────────┘ │
│       ▼                ▼              ▼              ▼       │
└───────┼────────────────┼──────────────┼──────────────┼──────┘
        │                │              │              │
        ▼                ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                      SERVICES LAYER                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │  FastAPI   │  │  RAG       │  │   DB Service         │  │
│  │  Service   │  │  Service   │  │   (SQLite)           │  │
│  │ (REST API) │  │  (Python)  │  │   (Python)           │  │
│  └────────────┘  └────────────┘  └──────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Tools (`agent/tools.py`)

Four tools available to the agent:

#### Tool 1: Metrics API Tool
```python
query_metrics_api(metric_type: str, service: str = None, period: str = "1h")
```
- **Purpose**: Get real-time metrics via HTTP
- **Access**: REST API calls to `http://127.0.0.1:8001`
- **Metrics**: latency, throughput, errors, health, services
- **Use case**: Current/recent metrics lookup

#### Tool 2: Knowledge RAG Tool
```python
search_knowledge_base(query: str, top_k: int = 3, search_mode: str = "hybrid")
```
- **Purpose**: Search documentation using semantic search
- **Access**: Direct Python calls to RAGService
- **Search modes**: hybrid (vector + BM25), vector, bm25
- **Use case**: Documentation, how-to guides, best practices

#### Tool 3: SQL Database Tool
```python
query_sql_database(question: str)
```
- **Purpose**: Query historical metrics using natural language
- **Access**: Direct Python calls to DatabaseService
- **Features**: NL-to-SQL conversion, 840 rows of time-series data
- **Use case**: Historical analysis, trends, comparisons

#### Tool 4: Calculator Tool
```python
calculate(expression: str)
```
- **Purpose**: Perform mathematical calculations
- **Access**: Safe evaluation of math expressions
- **Operations**: +, -, *, /, **, %, comparisons, functions
- **Use case**: Calculations, comparisons, percentages

### 2. State (`agent/state.py`)

**AgentState TypedDict** - Complete workflow state passed through all nodes:

```python
class AgentState(TypedDict):
    # Input
    query: str
    session_id: str

    # Intent Classification
    intent: str
    intent_confidence: float

    # Tool Selection & Execution
    tools_to_use: List[str]
    tool_outputs: Dict[str, Any]
    tools_executed: List[str]

    # Aggregation & Inference
    aggregated_data: Dict[str, Any]
    inference_result: Dict[str, Any]
    findings: List[str]
    recommendations: List[str]

    # Feedback Loop
    confidence: float
    feedback_needed: bool
    retry_count: int

    # Response
    final_answer: str

    # Trace & Metadata
    trace: List[Dict[str, Any]]
    total_duration_ms: float
    node_durations: Dict[str, float]
```

**Intent Types**:
- `metrics_lookup` - Query about service metrics
- `knowledge_lookup` - Documentation/how-to question
- `calculation` - Mathematical computation
- `mixed` - Requires multiple tool types
- `clarification` - Need more info from user
- `unknown` - Cannot determine intent

**Confidence Thresholds**:
- High: ≥ 0.8 (proceed with confidence)
- Medium: 0.6-0.8 (proceed with some uncertainty)
- Low: < 0.6 (may trigger retry)

### 3. Nodes (`agent/nodes.py`)

Seven workflow nodes:

#### Node 1: `classify_intent`
- Uses OpenAI GPT-4o-mini to classify query intent
- Fallback to keyword-based classification
- Updates: `intent`, `intent_confidence`, `confidence`

#### Node 2: `select_tools`
- Maps intent to appropriate tools
- Considers query keywords for refinement
- Updates: `tools_to_use`, `tool_selection_reasoning`

#### Node 3: `execute_tools`
- Executes all selected tools
- Handles parameter extraction from query
- Error handling for tool failures
- Updates: `tool_outputs`, `tool_errors`, `tools_executed`

#### Node 4: `aggregate_results`
- Combines outputs from multiple tools
- Assesses data quality and completeness
- Updates: `aggregated_data`, `data_quality`

#### Node 5: `perform_inference`
- Analyzes aggregated data
- Threshold checks (latency > 100ms, error_rate > 5%)
- Trend analysis for time-series data
- Updates: `inference_result`, `findings`, `recommendations`

#### Node 6: `check_feedback`
- Evaluates confidence score
- Decides: retry, clarify, or proceed
- Max retries: 2
- Updates: `feedback_needed`, `retry_reason`, `retry_count`

#### Node 7: `format_response`
- Creates final formatted answer
- Includes findings, recommendations, trace
- Markdown formatting
- Updates: `final_answer`, `end_time`, `total_duration_ms`

### 4. Graph (`agent/graph.py`)

**LangGraph StateGraph** with conditional edges:

```
START → classify_intent → select_tools → execute_tools →
aggregate_results → perform_inference → check_feedback
                                            │
                    ┌───────────────────────┴───────────────────┐
                    │                                           │
              feedback_needed?                          feedback_needed?
                  (True)                                    (False)
                    │                                           │
                    ▼                                           ▼
              select_tools                               format_response
               (retry)                                         │
                                                               ▼
                                                              END
```

**Conditional Edge**: `should_retry`
- If `feedback_needed=True` → retry (back to select_tools)
- If `feedback_needed=False` → respond (to format_response)
- Max retries enforced to prevent infinite loops

## Usage

### Basic Usage

```python
from agent import run_agent

# Run a query
result = run_agent("What is the latency for api-gateway?")

# Print the answer
print(result["final_answer"])
```

### Detailed Usage

```python
from agent import run_agent

# Run with verbose trace
result = run_agent(
    query="Show me error rates for auth-service",
    session_id="my_session",
    verbose=True  # Prints execution trace
)

# Access result components
print(f"Intent: {result['intent']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Tools used: {result['tools_executed']}")
print(f"Processing time: {result['total_duration_ms']:.0f}ms")

# Get findings
for finding in result['findings']:
    print(f"- {finding}")

# Get recommendations
for rec in result['recommendations']:
    print(f"→ {rec}")

# Examine trace
for event in result['trace']:
    print(f"[{event['node']}] {event['event_type']}")
```

### Streaming Usage

```python
from agent import stream_agent

# Stream intermediate states
for state in stream_agent("What is the latency for api-gateway?"):
    current_node = state.get('current_node')
    confidence = state.get('confidence', 0)
    print(f"Node: {current_node}, Confidence: {confidence:.2f}")
```

### Async Usage

```python
from agent import arun_agent
import asyncio

async def main():
    result = await arun_agent("What is the latency for api-gateway?")
    print(result["final_answer"])

asyncio.run(main())
```

## Query Examples

### Metrics Lookup
```python
run_agent("What is the current latency for api-gateway?")
run_agent("Show me error rates for auth-service")
run_agent("Is the business-logic service healthy?")
run_agent("List all available services")
```

### Knowledge Lookup
```python
run_agent("How do I configure API rate limiting?")
run_agent("What are the best practices for deployment?")
run_agent("Explain the monitoring architecture")
```

### Calculations
```python
run_agent("Calculate the average of 150, 200, and 250")
run_agent("If latency is 95ms and threshold is 100ms, is it within limits?")
run_agent("What is (500 - 450) / 450 as a percentage?")
```

### Historical Queries
```python
run_agent("What was the average CPU usage for api-gateway over the past week?")
run_agent("Compare memory usage between api-gateway and auth-service")
run_agent("Show me services with error counts above 100")
```

### Mixed Queries
```python
run_agent("What is the latency for api-gateway and how does it compare to best practices?")
run_agent("Show me error rates and explain how to reduce them")
```

## Installation

### 1. Install Dependencies

```bash
# Install agent dependencies
pip install -r requirements-agent.txt

# Install service dependencies
pip install -r requirements-rag.txt
pip install -r requirements-api.txt
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
export METRICS_API_URL="http://127.0.0.1:8001"  # Optional, defaults to this
```

### 3. Initialize Services

```bash
# Initialize RAG service
python demo_rag.py

# Create database
python tests/validate_db_service.py

# Start API server (in separate terminal)
python start_api_server.py
```

## Testing

### Test Tools
```bash
python tests/test_tools.py
```

Tests all 4 tools individually.

### Test Agent Workflow
```bash
python tests/test_agent.py
```

Tests complete agent workflow with various query types.

### Run Demo
```bash
python demo_agent.py
```

Interactive demonstration with example queries.

## Inference & Feedback Loop

### Inference Rules

**Threshold Checks**:
- Latency P95 > 100ms → Flag as high
- Error rate > 5% → Flag as critical
- Service status != "healthy" → Flag for investigation

**Comparisons**:
- Compare metrics between services
- Identify best/worst performers

**Trend Analysis**:
- Analyze time-series data for trends
- Detect anomalies

### Feedback Loop Logic

```python
if confidence < 0.6:
    if tool_failures:
        retry_with_different_tools()
    elif incomplete_data:
        retry_with_additional_tools()
    elif unclear_intent:
        ask_clarifying_question()
else:
    proceed_to_response()
```

**Confidence Updates**:
- Tool success → confidence × 1.1
- Tool failure → confidence × 0.8
- Missing data → confidence × 0.9
- Multiple sources agree → confidence × 1.2

**Max Retries**: 2 (prevents infinite loops)

## Observability

### Tracing

Every agent execution includes:
- Complete trace of all events
- Node-by-node execution log
- Tool calls and responses
- Confidence updates with reasoning
- Timing information

```python
result = run_agent("query", verbose=True)

for event in result['trace']:
    print(f"[{event['timestamp']}] {event['node']} - {event['event_type']}")
    print(f"  Data: {event['data']}")
```

### LangSmith Integration (Ready)

The agent is ready for LangSmith tracing:

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "intent-agent"

# Now all runs are traced in LangSmith
result = run_agent("query")
```

### OpenTelemetry Integration (Ready)

The state includes trace_id and span_id fields for OpenTelemetry:

```python
from agent.state import create_initial_state

state = create_initial_state("query")
print(f"Trace ID: {state['trace_id']}")
print(f"Span ID: {state['span_id']}")
```

## File Structure

```
agent/
├── __init__.py          # Package exports
├── tools.py             # 4 tool definitions
├── state.py             # AgentState schema
├── nodes.py             # 7 node implementations
├── graph.py             # LangGraph workflow
└── README.md            # This file

tests/
├── test_tools.py        # Tool tests
└── test_agent.py        # Agent workflow tests

demo_agent.py            # Interactive demonstration
requirements-agent.txt   # Agent dependencies
```

## Performance

Typical execution times:

- Intent classification: 200-500ms
- Tool selection: 10-20ms
- Tool execution: 50-300ms (varies by tool)
- Aggregation: 5-10ms
- Inference: 10-50ms
- Feedback check: 5-10ms
- Response formatting: 10-20ms

**Total**: 300-1000ms for typical queries

## Extending the Agent

### Adding a New Tool

1. Define tool in `agent/tools.py`:
```python
@tool
def my_new_tool(param: str) -> Dict[str, Any]:
    """My new tool description."""
    # Implementation
    return {"result": "data"}
```

2. Add to tool registry:
```python
ALL_TOOLS = [
    query_metrics_api,
    search_knowledge_base,
    query_sql_database,
    calculate,
    my_new_tool  # Add here
]
```

3. Update tool selection logic in `agent/nodes.py`:
```python
def select_tools(state: AgentState) -> AgentState:
    # Add logic to select your new tool
    if "my_condition" in query:
        tools_to_use.append("my_new_tool")
```

### Adding a New Node

1. Implement node in `agent/nodes.py`:
```python
def my_new_node(state: AgentState) -> AgentState:
    # Node implementation
    add_trace_event(state, "my_new_node", "node_start", {})
    # ... processing ...
    add_trace_event(state, "my_new_node", "node_end", {})
    return state
```

2. Add to graph in `agent/graph.py`:
```python
workflow.add_node("my_new_node", my_new_node)
workflow.add_edge("previous_node", "my_new_node")
workflow.add_edge("my_new_node", "next_node")
```

### Customizing Inference

Edit `perform_inference` in `agent/nodes.py`:

```python
def perform_inference(state: AgentState) -> AgentState:
    # Add custom inference logic
    if custom_condition:
        findings.append("Custom finding")
        recommendations.append("Custom recommendation")
    return state
```

## Troubleshooting

### "Cannot connect to metrics API"
```bash
# Start the API server
python start_api_server.py
```

### "RAG service not initialized"
```bash
# Initialize RAG
python demo_rag.py
```

### "Database not found"
```bash
# Create database
python tests/validate_db_service.py
```

### "OpenAI API key not set"
```bash
export OPENAI_API_KEY="your-key"
```

### Low confidence scores
- Check that all services are running
- Verify data quality in services
- Review intent classification accuracy
- Check tool selection logic

### Infinite retry loop
- Max retries is capped at 2
- If you see more, check `should_retry` logic
- Verify feedback_needed flag is set correctly

## Best Practices

1. **Always set OPENAI_API_KEY** before running
2. **Start all services** (API, RAG, DB) before testing
3. **Use verbose=True** for debugging
4. **Check confidence scores** to assess answer quality
5. **Review trace** to understand agent decisions
6. **Monitor tool failures** in tool_errors field
7. **Test with various query types** to ensure coverage

## Summary

The intent-routed agent provides:

✅ **Intent Classification** - Automatic query categorization
✅ **Multi-Tool Orchestration** - 4 specialized tools
✅ **Result Aggregation** - Unified data from multiple sources
✅ **Inference Engine** - Threshold checks, comparisons, analysis
✅ **Feedback Loop** - Confidence-based retry logic
✅ **Complete Tracing** - Full execution visibility
✅ **LangGraph Workflow** - Robust state management
✅ **Extensible Design** - Easy to add tools/nodes

**Ready for production with LangSmith and OpenTelemetry integration!**
