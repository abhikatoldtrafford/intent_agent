# Intent-Routed Agent Architecture

## Overview

This document explains the architecture, design choices, and implementation details of the Intent-Routed Agent POC. The system demonstrates an intelligent agent that classifies user intents, selects appropriate tools, executes workflows, and implements a confidence-based feedback loop for improved accuracy.

**Last Updated:** 2025-10-15
**Version:** 1.0.0

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Design Choices](#design-choices)
4. [Feedback Loop Implementation](#feedback-loop-implementation)
5. [Data Flow](#data-flow)
6. [Service Architecture](#service-architecture)
7. [Observability & Tracing](#observability--tracing)
8. [Performance Considerations](#performance-considerations)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   CLI Tool   │  │ Streamlit UI │  │  Programmatic API   │  │
│  │  (main.py)   │  │ (8-tab app)  │  │ (from agent import) │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LangGraph Agent Workflow                    │
│  ┌────────────┐   ┌─────────────┐   ┌──────────────┐          │
│  │  Classify  │──▶│Select Tools │──▶│Execute Tools │          │
│  │   Intent   │   │             │   │              │          │
│  └────────────┘   └─────────────┘   └──────┬───────┘          │
│                                              │                   │
│  ┌────────────┐   ┌─────────────┐   ┌──────▼───────┐          │
│  │   Format   │◀──│   Check     │◀──│  Aggregate   │          │
│  │  Response  │   │  Feedback   │   │   Results    │          │
│  └────────────┘   └──────┬──────┘   └──────────────┘          │
│         ▲                 │ retry?                               │
│         │                 │                                      │
│         │          ┌──────▼───────┐                             │
│         └──────────│   Perform    │                             │
│          proceed   │  Inference   │                             │
│                    └──────────────┘                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Tools Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Metrics API  │  │ Knowledge    │  │   SQL Database      │  │
│  │ (REST HTTP)  │  │ RAG (Direct) │  │   (Direct Call)     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────────────┘  │
│         │                  │                  │                  │
│  ┌──────▼──────────────────▼──────────────────▼───────────────┐ │
│  │               Calculator (Utility)                         │ │
│  └────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Services Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   FastAPI    │  │  RAG Service │  │   Database Service  │  │
│  │   REST API   │  │ (FAISS+BM25) │  │   (SQLite)          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Data Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Generated   │  │  Embeddings  │  │   Metrics Database  │  │
│  │   Metrics    │  │  + FAISS     │  │   (840 rows)        │  │
│  │   (Mock)     │  │  Index       │  │                     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. LangGraph Workflow

**Technology:** LangGraph StateGraph
**File:** `agent/graph.py`
**Purpose:** Orchestrates the agent's decision-making process through 7 connected nodes.

#### Node Sequence

```
START → classify_intent → select_tools → execute_tools →
        aggregate_results → perform_inference → check_feedback →
        [if retry: back to select_tools] →
        [if proceed: format_response] → END
```

#### State Management

The workflow maintains complete state in `AgentState` TypedDict (323 lines in `agent/state.py`):

```python
class AgentState(TypedDict):
    # Input
    query: str
    session_id: str

    # Intent Classification
    intent: str                          # metrics_lookup, knowledge_lookup, etc.
    intent_confidence: float

    # Tool Selection & Execution
    tools_to_use: List[str]
    tool_outputs: Dict[str, Any]
    tool_errors: Dict[str, str]
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
    retry_reason: Optional[str]

    # Response & Observability
    final_answer: str
    trace: List[Dict[str, Any]]
    node_durations: Dict[str, float]
    total_duration_ms: float
```

### 2. Intent Classification Node

**File:** `agent/nodes.py` (lines 46-124)
**LLM:** OpenAI GPT-4o-mini
**Fallback:** Keyword-based classification

#### Supported Intents

| Intent | Description | Example Query |
|--------|-------------|---------------|
| `metrics_lookup` | Real-time metrics queries | "What is the latency for api-gateway?" |
| `knowledge_lookup` | Documentation searches | "How do I configure rate limiting?" |
| `calculation` | Mathematical operations | "Calculate (150 + 200) / 2" |
| `mixed` | Multi-tool queries | "Show latency and how to improve it" |
| `clarification` | Ambiguous queries | "Tell me about the system" |
| `unknown` | Unclassifiable queries | Random text |

#### Classification Process

```python
def classify_intent(state: AgentState) -> AgentState:
    query = state["query"]

    # Step 1: Try LLM classification
    prompt = f"""Classify this query into one of:
                 metrics_lookup, knowledge_lookup, calculation, mixed,
                 clarification, unknown

                 Query: {query}

                 Return JSON: {{"intent": "...", "confidence": 0.X, "reasoning": "..."}}"""

    response = llm.invoke(prompt)
    intent_data = parse_json(response)

    # Step 2: Fallback to keyword matching if LLM fails
    if not intent_data:
        intent_data = keyword_classify(query)

    # Step 3: Update state with trace
    state["intent"] = intent_data["intent"]
    state["intent_confidence"] = intent_data["confidence"]
    state["confidence"] = intent_data["confidence"]
    add_trace_event(state, "classify_intent", "completed", intent_data)

    return state
```

**Design Choice:** LLM-first with keyword fallback ensures robustness. The LLM handles complex queries while keywords provide a safety net.

### 3. Tool Selection Node

**File:** `agent/nodes.py` (lines 127-189)
**Strategy:** Rule-based intent-to-tool mapping

#### Intent-to-Tool Mapping

```python
INTENT_TOOL_MAP = {
    INTENT_METRICS_LOOKUP: [TOOL_METRICS_API],
    INTENT_KNOWLEDGE_LOOKUP: [TOOL_KNOWLEDGE_RAG],
    INTENT_CALCULATION: [TOOL_CALCULATOR],
    INTENT_MIXED: [TOOL_METRICS_API, TOOL_KNOWLEDGE_RAG],  # Multi-tool
}
```

#### Additional Logic

- **Historical data detection:** If query mentions "history", "average over time", "past week" → Add `TOOL_SQL_DATABASE`
- **Comparison detection:** If query mentions "compare", "versus" → Add `TOOL_CALCULATOR`
- **Mixed intent handling:** Automatically selects multiple tools

**Design Choice:** Rule-based tool selection is fast, predictable, and easy to debug. More complex strategies (e.g., LLM-based tool selection) can be added later.

### 4. Tool Execution Node

**File:** `agent/nodes.py` (lines 192-344)
**Strategy:** Parallel-safe execution with error isolation

#### Execution Pattern

```python
def execute_tools(state: AgentState) -> AgentState:
    tools_to_use = state["tools_to_use"]
    tool_outputs = {}
    tool_errors = {}

    for tool_name in tools_to_use:
        try:
            # Extract parameters from query
            params = extract_params(state["query"], tool_name)

            # Execute tool
            tool = get_tool_by_name(tool_name)
            result = tool.invoke(params)

            # Handle errors in result
            if "error" in result:
                tool_errors[tool_name] = result["error"]
                update_confidence(state, -0.2)  # Penalize confidence
            else:
                tool_outputs[tool_name] = result
                update_confidence(state, +0.1)  # Boost confidence

        except Exception as e:
            tool_errors[tool_name] = str(e)
            update_confidence(state, -0.2)

    state["tool_outputs"] = tool_outputs
    state["tool_errors"] = tool_errors

    return state
```

**Design Choices:**
- **Error isolation:** Each tool's error is captured separately, doesn't crash entire workflow
- **Confidence updates:** Success boosts confidence, failure reduces it
- **Parameter extraction:** Uses regex patterns to extract service names, periods, etc. from natural language

### 5. Aggregation Node

**File:** `agent/nodes.py` (lines 347-434)
**Purpose:** Combines data from multiple tools and assesses quality

#### Data Quality Metrics

```python
def assess_data_quality(state: AgentState) -> Dict[str, float]:
    tool_outputs = state["tool_outputs"]
    tools_to_use = state["tools_to_use"]

    # Completeness: What % of tools succeeded?
    completeness = len(tool_outputs) / max(len(tools_to_use), 1)

    # Consistency: Do results make sense together?
    consistency = 1.0  # Default
    if len(tool_outputs) >= 2:
        # Check for conflicting data
        if has_conflicts(tool_outputs):
            consistency = 0.7

    return {
        "completeness": completeness,
        "consistency": consistency,
        "overall": (completeness + consistency) / 2
    }
```

#### Aggregated Data Structure

```python
{
    "metrics": {
        "latency": {"p50": 25, "p95": 69, "p99": 156},
        "throughput": 1500,
        "error_rate": 0.02
    },
    "knowledge": [
        {"filename": "troubleshooting.md", "score": 0.89, "content": "..."},
        {"filename": "api_guide.md", "score": 0.76, "content": "..."}
    ],
    "calculations": [
        {"expression": "(150+200)/2", "result": 175}
    ],
    "sql_data": {
        "rows": [...],
        "columns": ["service", "cpu_usage", "memory_usage"]
    }
}
```

**Design Choice:** Structured aggregation by data type makes inference easier. Quality metrics inform confidence scoring.

### 6. Inference Node

**File:** `agent/nodes.py` (lines 437-587)
**Purpose:** Apply business logic and generate insights

#### Inference Rules

**Threshold Checks:**
```python
# Latency threshold
if metrics.get("p95") > 100:  # 100ms threshold
    findings.append(f"⚠️ High latency detected: P95 = {p95}ms (threshold: 100ms)")
    recommendations.append("Consider implementing caching or optimizing database queries")

# Error rate threshold
if error_rate > 0.05:  # 5% threshold
    findings.append(f"⚠️ High error rate: {error_rate*100:.1f}% (threshold: 5%)")
    recommendations.append("Investigate error logs and implement better error handling")
```

**Service Comparisons:**
```python
if "compare" in query.lower():
    services = extract_services(sql_data)
    for metric in ["cpu_usage", "memory_usage"]:
        values = [s[metric] for s in services]
        best = min(values)
        worst = max(values)
        findings.append(f"📊 {metric}: Best={best:.1f}%, Worst={worst:.1f}%")
```

**Trend Analysis:**
```python
if len(time_series) >= 3:
    trend = calculate_trend(time_series)
    if trend > 0.1:  # 10% increase
        findings.append(f"📈 Increasing trend detected: +{trend*100:.1f}%")
    elif trend < -0.1:
        findings.append(f"📉 Decreasing trend detected: {trend*100:.1f}%")
```

**Design Choice:** Rule-based inference is deterministic and explainable. Each finding is traceable to specific data.

### 7. Feedback Loop Node

**File:** `agent/nodes.py` (lines 590-685)
**Purpose:** Decide whether to retry or proceed based on confidence

This is covered in detail in the [Feedback Loop Implementation](#feedback-loop-implementation) section below.

### 8. Response Formatting Node

**File:** `agent/nodes.py` (lines 688-769)
**Purpose:** Create human-readable markdown response

#### Response Template

```markdown
# Answer

{natural_language_answer}

## 🔍 Findings
- Finding 1
- Finding 2
- Finding 3

## 💡 Recommendations
- Recommendation 1
- Recommendation 2

## 📊 Data Summary
- Tools used: metrics_api, knowledge_rag
- Confidence: 0.87
- Processing time: 1,234ms

---
*Generated by Intent-Routed Agent*
```

---

## Design Choices

### 1. Local-First Architecture

**Choice:** Only external dependency is OpenAI API. Everything else runs locally.

**Rationale:**
- **Privacy:** No data leaves the local environment (except LLM calls)
- **Performance:** Direct Python calls are faster than HTTP for local services
- **Simplicity:** No infrastructure dependencies, easy to run
- **Debugging:** Full control over all components

**Trade-offs:**
- Harder to scale horizontally (would need to externalize services)
- RAG and database are single-threaded (but fast enough for POC)

### 2. Hybrid Service Access Pattern

**Choice:** Mix of HTTP and direct calls

| Service | Access Method | Rationale |
|---------|---------------|-----------|
| Metrics API | HTTP (REST) | Simulates external service, demonstrates HTTP integration |
| RAG Service | Direct Python call | Maximize performance, no serialization overhead |
| Database | Direct Python call | Maximize performance, avoid network stack |
| Calculator | Direct Python call | Simple utility, no reason for HTTP |

**Rationale:**
- **Demonstrates both patterns:** Shows agent can work with external APIs and local services
- **Optimizes performance:** Avoids HTTP overhead for high-frequency operations (RAG, DB)
- **Realistic:** Production systems often mix local and remote services

### 3. LangGraph Over Custom Workflow

**Choice:** Use LangGraph StateGraph instead of custom orchestration

**Rationale:**
- **Built-in features:** State management, streaming, checkpointing
- **Observability:** LangSmith integration for tracing
- **Conditional edges:** Easy to implement retry logic
- **Type safety:** TypedDict state schema catches errors early
- **Community:** Well-maintained, good documentation

**Trade-offs:**
- Learning curve (LangGraph-specific concepts)
- Less control over execution (but rarely needed)

### 4. Intent Classification Strategy

**Choice:** LLM-first with keyword fallback

**Rationale:**
- **Flexibility:** LLM handles complex, ambiguous queries
- **Robustness:** Keywords provide safety net if LLM fails
- **Cost-effective:** GPT-4o-mini is cheap (~$0.15 per 1M tokens)

**Alternative considered:**
- Pure keyword-based: Too rigid, misses nuanced queries
- Pure LLM-based: Fails catastrophically if API is down

### 5. Confidence Scoring Approach

**Choice:** Additive/multiplicative updates throughout workflow

**Base Confidence Sources:**
```python
# Starting point: Intent classification confidence (from LLM)
confidence = intent_confidence  # 0.0 - 1.0

# Tool execution adjustments
for tool in tools_executed:
    if tool_succeeded:
        confidence = min(confidence * 1.1, 1.0)  # +10% boost, capped at 1.0
    else:
        confidence *= 0.8  # -20% penalty

# Data quality adjustment
confidence *= data_quality["overall"]  # 0.0 - 1.0

# Multi-source agreement boost
if multiple_tools_agree:
    confidence = min(confidence * 1.2, 1.0)  # +20% boost
```

**Rationale:**
- **Transparent:** Each adjustment is logged in trace
- **Bounded:** Confidence always stays in [0.0, 1.0]
- **Intuitive:** Higher confidence when more tools succeed and data is consistent

**Alternative considered:**
- Machine learning model: Too complex for POC, needs training data

### 6. Error Handling Philosophy

**Choice:** Graceful degradation with detailed error tracking

**Implementation:**
```python
# Tool errors don't crash the workflow
try:
    result = tool.invoke(params)
except Exception as e:
    tool_errors[tool_name] = str(e)
    # Continue with other tools

# Agent still generates answer even with errors
if has_partial_data:
    answer = generate_answer_from_partial_data()
    answer += "\n\n⚠️ Note: Some data sources failed. Answer may be incomplete."
```

**Rationale:**
- **User experience:** Better to give partial answer than crash
- **Debugging:** Errors are logged and visible in trace
- **Resilience:** System works even if some components fail

### 7. Streaming Support

**Choice:** Support both sync and streaming execution

```python
# Synchronous (default)
result = run_agent("query")  # Returns complete result

# Streaming (for real-time UI updates)
for state in stream_agent("query"):
    node = list(state.keys())[0]
    print(f"Completed: {node}")
```

**Rationale:**
- **Flexibility:** Different use cases need different patterns
- **User feedback:** Streaming shows progress in long-running queries
- **LangGraph feature:** Built-in, easy to expose

---

## Feedback Loop Implementation

### Overview

The feedback loop is the system's self-correction mechanism. It evaluates confidence after inference and decides whether to retry with different tools or proceed with response generation.

### Confidence Thresholds

```python
CONFIDENCE_HIGH = 0.8    # Proceed with confidence
CONFIDENCE_MEDIUM = 0.6  # Proceed with note
CONFIDENCE_LOW = 0.6     # May trigger retry
MAX_RETRIES = 2          # Prevent infinite loops
```

### Decision Logic

**File:** `agent/nodes.py` (lines 590-685)

```python
def check_feedback(state: AgentState) -> AgentState:
    confidence = state.get("confidence", 0.0)
    retry_count = state.get("retry_count", 0)
    tool_errors = state.get("tool_errors", {})
    data_quality = state.get("data_quality", {})
    intent = state.get("intent")

    # Decision tree
    feedback_needed = False
    retry_reason = None

    # Case 1: Max retries reached → Always proceed
    if retry_count >= MAX_RETRIES:
        feedback_needed = False
        add_trace_event(state, "check_feedback", "max_retries_reached",
                       {"retry_count": retry_count})

    # Case 2: High confidence → Proceed
    elif confidence >= CONFIDENCE_HIGH:
        feedback_needed = False
        add_trace_event(state, "check_feedback", "high_confidence",
                       {"confidence": confidence})

    # Case 3: Medium confidence → Proceed with note
    elif confidence >= CONFIDENCE_MEDIUM:
        feedback_needed = False
        add_trace_event(state, "check_feedback", "medium_confidence",
                       {"confidence": confidence})

    # Case 4: Low confidence → Analyze why
    else:
        # Sub-case 4a: Tool failures → Retry with different tools
        if tool_errors:
            feedback_needed = True
            retry_reason = "tool_failures"
            add_trace_event(state, "check_feedback", "retry_tool_failures",
                           {"failed_tools": list(tool_errors.keys())})

        # Sub-case 4b: Incomplete data → Retry with additional tools
        elif data_quality.get("completeness", 1.0) < 0.7:
            feedback_needed = True
            retry_reason = "incomplete_data"
            add_trace_event(state, "check_feedback", "retry_incomplete_data",
                           {"completeness": data_quality["completeness"]})

        # Sub-case 4c: Unclear intent → Ask for clarification
        elif intent == INTENT_UNKNOWN or intent == INTENT_CLARIFICATION:
            feedback_needed = True
            retry_reason = "unclear_intent"
            state["clarification_question"] = generate_clarification(state)
            add_trace_event(state, "check_feedback", "needs_clarification",
                           {"intent": intent})

        # Sub-case 4d: Low confidence but unclear why → Proceed anyway
        else:
            feedback_needed = False
            add_trace_event(state, "check_feedback", "proceeding_despite_low_confidence",
                           {"confidence": confidence})

    # Update state
    state["feedback_needed"] = feedback_needed
    state["retry_reason"] = retry_reason

    return state
```

### Conditional Edge

**File:** `agent/graph.py` (lines 97-104)

```python
def should_retry(state: AgentState) -> Literal["retry", "respond"]:
    """Route based on feedback_needed flag."""
    if state.get("feedback_needed", False):
        # Increment retry count
        state["retry_count"] = state.get("retry_count", 0) + 1
        return "retry"  # Go back to select_tools
    else:
        return "respond"  # Go to format_response
```

### Retry Strategy

When retry is triggered, the workflow goes back to `select_tools` node:

**Tool Selection on Retry:**
```python
def select_tools(state: AgentState) -> AgentState:
    retry_reason = state.get("retry_reason")
    retry_count = state.get("retry_count", 0)

    # First attempt: Use standard intent-to-tool mapping
    if retry_count == 0:
        tools = INTENT_TOOL_MAP.get(state["intent"], [])

    # Retry attempts: Adjust tool selection based on reason
    else:
        tools = []

        # If tools failed, try alternatives
        if retry_reason == "tool_failures":
            failed_tools = state.get("tool_errors", {}).keys()
            tools = get_alternative_tools(state["intent"], exclude=failed_tools)

        # If data incomplete, add more tools
        elif retry_reason == "incomplete_data":
            tools = state.get("tools_to_use", [])
            tools.extend(get_supplementary_tools(state["intent"]))

        # If intent unclear, try broader search
        elif retry_reason == "unclear_intent":
            tools = [TOOL_KNOWLEDGE_RAG, TOOL_METRICS_API]  # Cast wide net

    state["tools_to_use"] = tools
    return state
```

### Example Feedback Loop Execution

**Query:** "Show me metrics for api-gateway"

**Attempt 1:**
```
1. classify_intent: metrics_lookup (confidence: 0.85)
2. select_tools: [metrics_api]
3. execute_tools: metrics_api fails (API down)
   → confidence: 0.85 * 0.8 = 0.68
4. aggregate_results: data_quality = {completeness: 0.0, consistency: 1.0}
   → confidence: 0.68 * 0.5 = 0.34
5. perform_inference: No data to analyze
6. check_feedback: confidence < 0.6 AND tool_errors → retry!
   → retry_reason: tool_failures
```

**Attempt 2 (Retry):**
```
7. select_tools: Try alternative → [sql_database] (historical data fallback)
8. execute_tools: sql_database succeeds
   → confidence: 0.34 * 1.1 = 0.37
9. aggregate_results: data_quality = {completeness: 1.0, consistency: 1.0}
   → confidence: 0.37 * 1.0 = 0.37
10. perform_inference: Generate findings from SQL data
11. check_feedback: confidence < 0.6 but retry_count >= 1 → proceed
    (Better to give answer than retry forever)
12. format_response: Generate answer with disclaimer about API failure
```

**Final Answer:**
```markdown
Based on historical data from the database, api-gateway had an average
latency of 45ms over the past 24 hours.

⚠️ Note: Real-time metrics API is currently unavailable.
This answer is based on historical data only.

Confidence: 0.37 (Low - data source limited)
```

### Feedback Loop Benefits

1. **Resilience:** System recovers from tool failures automatically
2. **Completeness:** Seeks additional data when initial results are insufficient
3. **User experience:** Better to retry than give empty answer
4. **Transparency:** Retry reasons logged in trace for debugging

### Feedback Loop Limitations

1. **Max retries:** Prevents infinite loops but may give up too early
2. **No user interaction:** Can't ask user for clarification in current implementation
3. **Retry strategy:** Simple rule-based, could be more intelligent with ML

---

## Data Flow

### Complete Query Execution Flow

```
User Query: "What is the latency for api-gateway and how can I improve it?"
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. classify_intent                                          │
│    LLM: "This is a mixed intent - metrics + knowledge"     │
│    Result: intent=mixed, confidence=0.85                   │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. select_tools                                             │
│    Intent=mixed → [metrics_api, knowledge_rag]             │
│    Reasoning: "Need current metrics + improvement advice"   │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. execute_tools                                            │
│    • metrics_api("latency", "api-gateway", "1h")           │
│      → {p50: 25ms, p95: 69ms, p99: 156ms}                 │
│      confidence: 0.85 * 1.1 = 0.94                         │
│                                                             │
│    • knowledge_rag("improve latency")                      │
│      → [troubleshooting.md: "Add caching...", score: 0.89] │
│      confidence: 0.94 * 1.1 = 1.0 (capped)                │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. aggregate_results                                        │
│    Combine:                                                 │
│      metrics: {latency: {p50: 25, p95: 69, p99: 156}}     │
│      knowledge: [{file: troubleshooting.md, ...}]          │
│    Data quality: {completeness: 1.0, consistency: 1.0}     │
│    confidence: 1.0 * 1.0 = 1.0                             │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. perform_inference                                        │
│    Check thresholds:                                        │
│      ✓ p95 (69ms) < 100ms → Healthy                       │
│    Extract recommendations from knowledge:                  │
│      • "Implement Redis caching for frequently accessed..."│
│      • "Consider CDN for static assets..."                 │
│    Findings: ["Latency is within acceptable range"]        │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. check_feedback                                           │
│    confidence (1.0) >= HIGH (0.8) → Proceed!               │
│    feedback_needed: false                                   │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. format_response                                          │
│    Generate markdown answer combining:                      │
│      • Current metrics summary                              │
│      • Findings (latency status)                           │
│      • Recommendations (from knowledge base)                │
│      • Metadata (confidence, duration, tools used)         │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
                  Final Answer
```

### State Evolution Example

**Initial State:**
```python
{
    "query": "What is the latency for api-gateway?",
    "session_id": "abc123",
    "confidence": 0.5,  # Default
    "retry_count": 0,
    "trace": []
}
```

**After classify_intent:**
```python
{
    "query": "What is the latency for api-gateway?",
    "session_id": "abc123",
    "intent": "metrics_lookup",
    "intent_confidence": 0.85,
    "confidence": 0.85,
    "retry_count": 0,
    "trace": [
        {
            "node": "classify_intent",
            "event_type": "completed",
            "timestamp": "2025-10-15T10:30:45.123Z",
            "data": {"intent": "metrics_lookup", "confidence": 0.85}
        }
    ]
}
```

**After execute_tools (success):**
```python
{
    # ... previous fields ...
    "tools_to_use": ["metrics_api"],
    "tools_executed": ["metrics_api"],
    "tool_outputs": {
        "metrics_api": {
            "service": "api-gateway",
            "metrics": {"p50": 25, "p95": 69, "p99": 156}
        }
    },
    "tool_errors": {},
    "confidence": 0.94,  # Boosted from 0.85
    "trace": [
        # ... previous events ...
        {
            "node": "execute_tools",
            "event_type": "tool_success",
            "timestamp": "2025-10-15T10:30:46.456Z",
            "data": {"tool": "metrics_api", "duration_ms": 45}
        }
    ]
}
```

**Final State:**
```python
{
    "query": "What is the latency for api-gateway?",
    "session_id": "abc123",
    "intent": "metrics_lookup",
    "intent_confidence": 0.85,
    "tools_to_use": ["metrics_api"],
    "tools_executed": ["metrics_api"],
    "tool_outputs": {...},
    "tool_errors": {},
    "aggregated_data": {...},
    "data_quality": {"completeness": 1.0, "consistency": 1.0},
    "inference_result": {...},
    "findings": ["✓ Latency is within acceptable range"],
    "recommendations": [],
    "confidence": 1.0,
    "feedback_needed": false,
    "retry_count": 0,
    "final_answer": "The api-gateway has a P95 latency of 69ms...",
    "start_time": "2025-10-15T10:30:45.000Z",
    "end_time": "2025-10-15T10:30:47.234Z",
    "total_duration_ms": 2234,
    "node_durations": {
        "classify_intent": 523,
        "select_tools": 12,
        "execute_tools": 456,
        "aggregate_results": 89,
        "perform_inference": 234,
        "check_feedback": 45,
        "format_response": 123
    },
    "trace": [
        # 15-20 trace events with full execution history
    ]
}
```

---

## Service Architecture

### Local Services Design

All services run locally with no external dependencies except OpenAI API.

#### 1. Metrics REST API Service

**File:** `services/api_service.py` (550 lines)
**Technology:** FastAPI + Uvicorn
**Port:** 8001

**Endpoints:**
```python
GET  /                              # Welcome page
GET  /health                         # Health check
GET  /services                       # List services
GET  /metrics/latency?service=X&period=Y
GET  /metrics/throughput?service=X&period=Y
GET  /metrics/errors?service=X&period=Y
POST /metrics/query                  # Batch query
```

**Data Generation:**
- Realistic patterns: Business hours variation, weekend dips
- Occasional incidents: Random latency spikes
- Multiple services: api-gateway, auth-service, business-logic, data-processor, payment-service

**Performance:**
- Response time: <50ms
- No database: Generates data on-the-fly with deterministic randomness

#### 2. RAG Service

**File:** `services/rag_service.py` (838 lines)
**Technology:** OpenAI Embeddings + FAISS + BM25

**Components:**
```
┌─────────────────────────────────────────┐
│         RAG Service                     │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  Document Loader                  │ │
│  │  - Loads 5 markdown files         │ │
│  │  - Total: 3,795 lines             │ │
│  └──────────┬────────────────────────┘ │
│             ↓                           │
│  ┌───────────────────────────────────┐ │
│  │  Semantic Chunker                 │ │
│  │  - LangChain SemanticChunker      │ │
│  │  - ~150 chunks with metadata      │ │
│  └──────────┬────────────────────────┘ │
│             ↓                           │
│  ┌───────────────────────────────────┐ │
│  │  Embedding Model                  │ │
│  │  - text-embedding-3-small         │ │
│  │  - 1536 dimensions                │ │
│  └──────────┬────────────────────────┘ │
│             ↓                           │
│  ┌───────────────┬───────────────────┐ │
│  │  FAISS Index  │  BM25 Index       │ │
│  │  (Vector)     │  (Keyword)        │ │
│  └───────────────┴───────────────────┘ │
│             ↓                           │
│  ┌───────────────────────────────────┐ │
│  │  Hybrid Search (0.7 vec + 0.3 bm25)│
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

**Features:**
- **Caching:** Persistent FAISS index in `data/embeddings/`
- **Fast initialization:** ~200ms from cache
- **Rich metadata:** Filename, chunk_id, prev/next IDs, span positions
- **Multiple search modes:** vector, bm25, hybrid

#### 3. Database Service

**File:** `services/db_service.py` (629 lines)
**Technology:** SQLite
**Size:** 192KB, 840 rows

**Schema:**
```sql
CREATE TABLE service_metrics (
    id INTEGER PRIMARY KEY,
    service_name TEXT,
    timestamp TEXT,
    cpu_usage REAL,
    memory_usage REAL,
    request_count INTEGER,
    error_count INTEGER,
    avg_latency REAL,
    status TEXT,
    region TEXT,
    instance_id TEXT
);

CREATE INDEX idx_service_name ON service_metrics(service_name);
CREATE INDEX idx_timestamp ON service_metrics(timestamp);
```

**Data:**
- 7 days of hourly metrics (168 hours)
- 5 services × 168 hours = 840 rows
- Realistic patterns: Business hours, incidents, status changes

**Features:**
- **Safe SQL execution:** Blocks DROP, DELETE, UPDATE, INSERT
- **Natural language to SQL:** Uses OpenAI GPT-4o-mini
- **Predefined queries:** 5 common query methods
- **Fast:** <10ms query performance

---

## Observability & Tracing

### Built-in Tracing

Every agent execution includes a complete trace with 15-20 events:

```python
{
    "node": "classify_intent",
    "event_type": "completed",
    "timestamp": "2025-10-15T10:30:45.123Z",
    "duration_ms": 523,
    "data": {
        "intent": "metrics_lookup",
        "confidence": 0.85,
        "reasoning": "Query mentions 'latency' and specific service"
    }
}
```

### Trace Event Types

| Node | Event Types | Key Data |
|------|-------------|----------|
| classify_intent | started, completed, failed | intent, confidence, reasoning |
| select_tools | completed | tools_selected, reasoning |
| execute_tools | tool_started, tool_success, tool_error | tool_name, duration, result/error |
| aggregate_results | completed | data_quality, aggregated_counts |
| perform_inference | completed | findings_count, recommendations_count |
| check_feedback | high_confidence, low_confidence, retry_triggered | confidence, retry_reason |
| format_response | completed | answer_length, format |

### LangSmith Integration

**Setup:**
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls__...
export LANGCHAIN_PROJECT=intent-agent-poc
```

**Features:**
- Automatic tracing of all LangChain/LangGraph operations
- Tool inputs/outputs captured
- LLM calls logged with prompts and responses
- State transitions visible
- Playground for testing

### OpenTelemetry Support

State includes trace_id and span_id fields for correlation:

```python
state["trace_id"] = uuid.uuid4().hex
state["span_id"] = uuid.uuid4().hex
state["parent_span_id"] = None  # For nested spans
```

Can be integrated with any OpenTelemetry collector.

---

## Performance Considerations

### Latency Breakdown

Typical query execution time: **1,500-2,500ms**

| Phase | Duration | % of Total |
|-------|----------|------------|
| Intent classification (LLM) | 400-700ms | 25-35% |
| Tool selection | 5-10ms | <1% |
| Tool execution | 500-1000ms | 30-50% |
| - Metrics API (HTTP) | 30-50ms | 2-3% |
| - RAG search | 15-30ms | 1-2% |
| - SQL query | 5-10ms | <1% |
| - Calculator | <1ms | <1% |
| Aggregation | 50-100ms | 3-5% |
| Inference | 100-200ms | 5-10% |
| Feedback check | 20-50ms | 1-2% |
| Response formatting | 50-150ms | 3-8% |

**Optimization Opportunities:**
1. **Cache LLM calls:** Same query → same intent (could save 400-700ms)
2. **Parallel tool execution:** Currently sequential, could parallelize (save 50%)
3. **Streaming:** Start response generation before all tools complete

### Memory Usage

Typical memory footprint: **150-250MB**

| Component | Memory | Notes |
|-----------|--------|-------|
| FAISS index | 50-80MB | ~150 vectors × 1536 dimensions |
| BM25 index | 10-20MB | Keyword index |
| SQLite connection | 5-10MB | Small database |
| LangGraph state | 1-5MB | Depends on tool outputs |
| FastAPI server | 50-100MB | If running |
| Python runtime | 30-50MB | Base overhead |

**Optimization Opportunities:**
1. **Lazy loading:** Load FAISS index only when needed (save 50-80MB)
2. **Index compression:** Use FAISS IVF index (save 30-40MB, slower search)
3. **State pruning:** Remove old trace events from state

### Scalability Considerations

**Current limitations:**
- Single-threaded RAG search (but fast enough: 15-30ms)
- No caching of LLM responses
- No distributed execution
- SQLite (single-writer)

**If scaling needed:**
- Replace SQLite with PostgreSQL
- Add Redis for caching (LLM responses, RAG results)
- Externalize services (FastAPI, separate RAG service)
- Use LangGraph checkpointing for fault tolerance
- Horizontal scaling: Multiple agent instances with shared services

---

## Summary

This architecture demonstrates a production-grade agentic system with:

✅ **Intelligent routing:** LLM-based intent classification
✅ **Multi-tool orchestration:** Parallel-safe execution with error isolation
✅ **Self-correction:** Confidence-based feedback loop with retry logic
✅ **Observability:** Complete tracing, LangSmith integration
✅ **Performance:** Sub-3-second responses for complex queries
✅ **Resilience:** Graceful degradation, detailed error tracking
✅ **Local-first:** Privacy-preserving, easy to run
✅ **Extensible:** Easy to add new tools, intents, inference rules

The feedback loop is the key innovation, enabling the agent to recover from failures, seek additional data when needed, and provide transparency about confidence levels.

---

**For more details, see:**
- `agent/README.md` - Detailed agent documentation
- Service-specific READMEs in `services/` directory
