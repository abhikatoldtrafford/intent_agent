# Intent-Routed Agent POC

> A local agent that classifies natural language queries, routes to appropriate tools, performs inference with feedback loops, and returns complete execution traces.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/framework-LangGraph-green.svg)](https://github.com/langchain-ai/langgraph)
[![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)]()

---

## 🎥 Video Demonstration

📹 **[Watch the Full Demo on YouTube](https://youtu.be/rCsbiqlGTbA)**

See the agent in action: intent classification, tool selection, workflow execution, and complete tracing.

---

## 🎯 Overview

This project demonstrates a sophisticated AI agent that:
- **Classifies user intent** using OpenAI GPT-4.1-mini (latest, fastest model - 50% faster than GPT-4o)
- **Routes to specialized tools** (REST API, RAG, SQL Database, Calculator)
- **Executes workflows** with 7-node LangGraph state machine
- **Performs intelligent inference** (threshold checks, comparisons, recommendations)
- **Implements feedback loops** with confidence-based retry logic
- **Returns complete traces** of all decisions and tool calls

**Key Constraint**: Only external service is OpenAI API. Everything else runs locally.

---

## 🎯 Orchestration & Feedback Loops

### Why This Matters for Production Systems

In production AI systems, **orchestration** and **feedback loops** are critical for:
- **Reliability**: Automatic fallback when primary data sources fail
- **Self-healing**: Intelligent retry with different strategies
- **Explainability**: Complete visibility into WHY decisions were made
- **Cost optimization**: Avoiding wasted API calls for off-topic queries

This agent showcases **enterprise-grade orchestration patterns** that clients value:

### 🔄 Intelligent Orchestration

**What it does**: Makes smart decisions about which tools to use and when to switch strategies.

**Real-world examples**:

1. **Intelligent Fallback Routing** (lines 806-834 in `agent/nodes.py`)
   - API returns empty → automatically tries SQL database
   - SQL returns empty → automatically tries API
   - Avoids wasted retries by suggesting the RIGHT alternative tool

   ```python
   # Example: User asks for current metrics, but API is down
   # Agent automatically falls back to SQL database historical data
   Query: "What's the latency for api-gateway?"
   → Tries REST API (empty/error)
   → Orchestration decision: Switch to SQL database
   → Returns historical data with disclaimer
   ```

2. **Context-Aware Tool Selection** (lines 424-468 in `agent/nodes.py`)
   - "current latency" → REST API
   - "average latency over past week" → SQL database
   - "latency and how to improve" → REST API + Knowledge RAG
   - CPU/Memory queries → ONLY SQL database (not in API)

   ```python
   # Smart routing prevents tool failures
   Query: "Show me CPU usage for api-gateway"
   → Orchestration: CPU only in database, skip API call
   → Uses SQL database directly (saves API call)
   ```

3. **Off-Topic Query Detection** (lines 544-550, 1547-1578 in `agent/nodes.py`)
   - Detects queries like "What's the weather?" or "Hello"
   - **Skips all tools** (no wasted API calls)
   - Returns friendly guidance message

   ```python
   # Before fix: wasted RAG API call on "Hi"
   # After fix: skips all tools, returns guidance
   Query: "Tell me a joke"
   → Orchestration: Unknown intent, no tools needed
   → Returns: "I'm a monitoring agent. I can help with..."
   ```

### 🔁 Confidence-Based Feedback Loops

**What it does**: Evaluates results quality and decides whether to retry with different tools.

**Real-world examples**:

1. **Multi-Level Confidence Scoring** (lines 322-482 in `agent/nodes.py`)
   - **HIGH (≥0.8)**: Proceed immediately
   - **MEDIUM (0.6-0.8)**: Proceed with caveat note
   - **LOW (<0.6)**: Trigger feedback loop (up to 2 retries)

   ```python
   # Confidence factors:
   # - Intent classification confidence: 0.95
   # - Tool success: +10% per successful tool
   # - Tool failure: -20% per failed tool
   # - Empty results: -50% completeness penalty
   # - All sources empty: Forces confidence to 0.4 (triggers retry)
   ```

2. **Adaptive Retry Strategy** (lines 390-426 in `agent/nodes.py`)
   - **Retry 1**: Try fallback tools suggested by aggregation layer
   - **Retry 2**: Try different approach or ask for clarification
   - **After 2 retries**: Proceed with LLM general knowledge + disclaimer

   ```python
   # Example feedback loop in action:
   Iteration 0: Query "Show latency for unknown-service"
   → Tools: [REST API]
   → API returns: {"error": "Service not found"}
   → Confidence: 0.4 (low)
   → Feedback: Retry needed

   Iteration 1: Retry with fallback
   → Tools: [SQL database] (suggested by orchestration)
   → Database: No matching records
   → Confidence: 0.4 (low)
   → Feedback: Max retries reached

   Iteration 2: Final attempt
   → Answer from LLM general knowledge
   → Response: "Service 'unknown-service' not found. Available services: ..."
   ```

3. **Orchestration Decision Logging** (lines 555-567, 487-499 in `agent/nodes.py`)
   - Every tool selection logged with reasoning
   - Every retry logged with reason and suggested alternatives
   - Complete visibility into agent's decision-making process

   ```python
   # Orchestration log example:
   state["orchestration_log"] = [
     {
       "stage": "tool_selection",
       "intent": "metrics_lookup",
       "decision": "Selected 2 tool(s): query_metrics_api, query_sql_database",
       "reasoning": "General metrics query - using both REST API (current) and SQL database (trends)",
       "retry_iteration": 0,
       "timestamp": "2025-10-17T..."
     }
   ]

   state["feedback_iterations"] = [
     {
       "iteration": 1,
       "reason": "empty_results_fallback",
       "confidence_at_retry": 0.45,
       "fallback_tools": ["query_sql_database"],
       "timestamp": "2025-10-17T..."
     }
   ]
   ```

### 📊 Observability Dashboard

All orchestration decisions and feedback iterations are visible in:
- **Streamlit Agent Testing Tab**: Real-time decision timeline
- **State Trace**: Complete `orchestration_log` and `feedback_iterations` arrays
- **Node Durations**: Performance breakdown for each decision point

**Try it**:
```bash
streamlit run streamlit_app.py
# Navigate to Agent Testing → Try a complex query
# View orchestration decisions in real-time
```

**Challenge Queries** (to see orchestration in action):
```python
# Triggers fallback routing
"What is the latency for payment-service in the last 6 hours?"

# Triggers multi-tool coordination
"Show me CPU usage and explain how to optimize it"

# Triggers off-topic detection
"What's the weather like today?"

# Triggers retry loop (if service doesn't exist)
"Show metrics for nonexistent-service"
```

---

## ✨ Features

### Core Capabilities
- ✅ **6 Intent Types**: metrics_lookup, knowledge_lookup, calculation, mixed, clarification, unknown
- ✅ **4 Specialized Tools**:
  - 🌐 REST API - Real-time metrics from FastAPI service
  - 📚 Knowledge RAG - Semantic search over documentation (FAISS + BM25)
  - 🗄️ SQL Database - Natural language to SQL queries
  - 🧮 Calculator - Safe mathematical computations
- ✅ **7-Node Workflow**: classify → select → execute → aggregate → infer → feedback → format
- ✅ **Inference Engine**: Threshold checks, service comparisons, trend analysis
- ✅ **Feedback Loop**: Confidence-based retry (max 2 attempts, thresholds: 0.8 high, 0.6 medium)
- ✅ **Complete Tracing**: Every decision, tool call, and state transition logged
- ✅ **Trace Caching**: 24-hour persistent cache for traces across sessions
- ✅ **API Testing Tab**: Interactive Swagger-like interface with prefilled parameters
- ✅ **Enhanced Test Visibility**: Detailed execution traces in all test outputs

### Technical Highlights
- **LangGraph** workflow orchestration with conditional edges
- **OpenAI GPT-4.1-mini** for intent classification and NL-to-SQL
- **FAISS + BM25** hybrid search for document retrieval
- **FastAPI** REST service with auto-generated OpenAPI docs
- **SQLite** database with 840 rows of time-series metrics
- **LangSmith** integration with real-time API fetching and trace caching
- **Trace Display Utilities** for comprehensive test execution visualization
- **Auto-Population** of traces from LangSmith API on Streamlit startup

---

## 📊 Architecture

```
┌─────────────┐
│    User     │
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│         LangGraph Agent Workflow        │
│                                         │
│  classify_intent → select_tools         │
│       ↓                                 │
│  execute_tools → aggregate_results      │
│       ↓                                 │
│  perform_inference → check_feedback     │
│       ↓                ↓                │
│  format_response (or retry)             │
└────────┬────────────────────────────────┘
         │
         ├──→ REST API Tool (HTTP)
         │    └─→ FastAPI Service (localhost:8001)
         │
         ├──→ Knowledge RAG Tool (Direct)
         │    └─→ FAISS + BM25 Hybrid Search
         │
         ├──→ SQL Database Tool (Direct)
         │    └─→ SQLite + NL-to-SQL (OpenAI)
         │
         └──→ Calculator Tool (Direct)
              └─→ Safe eval with restricted namespace
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key 
- Langsmith API key 

### Installation

```bash
# 1. Navigate to project directory
cd agent_poc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set OpenAI API key (automatically loaded from .env)
echo "OPENAI_API_KEY=sk-your-key-here" > .env
# Note: The .env file is automatically loaded and takes precedence over system environment variables
```

### Single Command Startup

```bash
# This automatically:
# - Checks prerequisites
# - Initializes RAG embeddings (first run only, ~30 seconds)
# - Starts API server on http://127.0.0.1:8001

python start_api_server.py
```

### Usage Options

#### 1. Interactive CLI
```bash
# In a separate terminal
python main.py
```

Example session:
```
> What is the latency for api-gateway?
> How do I reduce error rates?
> Calculate (150 + 200) / 2
> Compare CPU usage between api-gateway and auth-service
> help
> quit
```

#### 2. Programmatic API
```python
from agent import run_agent

# Run a query
result = run_agent("What is the latency for api-gateway?")

# Access the answer
print(result["final_answer"])

# Access metadata
print(f"Intent: {result['intent']}")               # metrics_lookup
print(f"Confidence: {result['confidence']:.2f}")    # 1.00
print(f"Tools: {result['tools_executed']}")         # ['query_metrics_api']
print(f"Time: {result['total_duration_ms']:.0f}ms") # 1504ms

# View complete trace
for event in result['trace']:
    print(f"[{event['node']}] {event['event_type']}")
```

#### 3. Streamlit Web UI
```bash
# Launch comprehensive web dashboard
streamlit run streamlit_app.py
```

**Features** (10 tabs):
- 🏠 **Home** - System status and metrics overview with trace auto-population summary
- 📚 **Documentation** - Complete usage guide and API reference
- 🌐 **API Testing** - Interactive Swagger-like interface with prefilled parameters (zero-configuration)
- 🤖 **Agent Testing** - Interactive query interface with execution history
- 🔍 **RAG Service** - Search knowledge base and browse documents
- 💾 **SQL Database** - Query database with natural language or SQL
- 🧪 **Tests** - Run and view test results with detailed execution traces
- 🎮 **Demos** - Execute demo scripts with live output
- 🔀 **Workflow** - Visualize agent routing and execution flow
- 📡 **Observability** - LangSmith trace fetching, cache management, and tracing dashboard

Access at: http://localhost:8501 (auto-opens in browser)

#### 4. Demo Scripts
```bash
# Interactive demo with multiple query types
python demo/demo_agent.py

# RAG service initialization demo
python demo/demo_rag.py
```

---

## 📂 Project Structure

```
agent_poc/
├── agent/                      # Core agent implementation (1,948 lines)
│   ├── __init__.py             # Package exports
│   ├── graph.py                # LangGraph workflow definition
│   ├── nodes.py                # 7 node implementations (enhanced intent classification)
│   ├── state.py                # AgentState schema with 40+ fields
│   ├── tools.py                # 4 tool definitions
│   └── README.md               # Agent documentation
│
├── services/                   # Local services (2,017 lines)
│   ├── api_service.py          # FastAPI REST endpoints (6 routes)
│   ├── db_service.py           # SQLite + NL-to-SQL
│   ├── rag_service.py          # FAISS + BM25 hybrid search
│   ├── README_API.md           # API documentation
│   ├── README_DB.md            # Database documentation
│   └── README_RAG.md           # RAG documentation
│
├── utils/                      # Utility modules (NEW - 799 lines)
│   ├── trace_cache.py          # 24-hour trace caching system (387 lines)
│   └── trace_display.py        # Test trace visualization utilities (412 lines)
│
├── data/                       # Data files
│   ├── docs/                   # 5 markdown documents (5,435 lines)
│   │   ├── architecture.md     # Complete system architecture (1,640 lines)
│   │   ├── api_guide.md
│   │   ├── troubleshooting.md
│   │   ├── deployment.md
│   │   └── monitoring.md
│   ├── metrics.db              # SQLite database (840 rows, 192KB)
│   ├── embeddings/             # FAISS cache (auto-generated)
│   └── trace_cache/            # Trace cache directory (24-hour lifetime)
│
├── test/                       # Test suite (13 test files, 41 tests)
│   ├── test_agent.py           # Agent workflow tests
│   ├── test_rag_service.py     # RAG tests
│   ├── test_tools.py           # Tool tests
│   ├── test_individual_tools.py    # Individual tool validation (18 checks)
│   ├── test_end_to_end.py      # End-to-end workflow tests (8 scenarios)
│   ├── test_feedback_loop.py   # Feedback loop tests (8 checks, enhanced tracing)
│   ├── test_orchestration.py   # Orchestration decision tests (enhanced tracing)
│   ├── test_trace_cache.py     # Trace caching system tests (6 tests)
│   ├── test_api_endpoints.py   # API endpoint validation (5 endpoints)
│   ├── verify_documentation.py # Documentation verification (9 checks)
│   ├── validate_api_service.py # API validation
│   └── validate_db_service.py  # Database validation
│
├── demo/                       # Demo scripts
│   ├── demo_agent.py           # Interactive agent demonstration
│   └── demo_rag.py             # RAG initialization demo
│
├── main.py                     # Interactive CLI interface
├── streamlit_app.py            # Streamlit web UI (10 tabs, 4,400+ lines)
├── start_api_server.py         # System startup (single entry point)
├── requirements.txt            # All dependencies
├── .env                        # Environment configuration (user-created)
├── .env.example                # Environment template
├── README.md                   # This file

```


---

## 🛠️ Components

### 1. LangGraph Workflow (7 Nodes)

| Node | Function | Output |
|------|----------|--------|
| **classify_intent** | Classify query using OpenAI GPT-4.1-mini | intent, confidence |
| **select_tools** | Map intent to appropriate tools | tools_to_use |
| **execute_tools** | Run selected tools (parallel/sequential) | tool_outputs, tools_executed |
| **aggregate_results** | Combine outputs, assess data quality | aggregated_data, data_quality |
| **perform_inference** | Threshold checks, comparisons, recommendations | findings, recommendations |
| **check_feedback** | Evaluate confidence, decide retry/respond | feedback_needed, retry_reason |
| **format_response** | Create final markdown-formatted answer | final_answer, trace |

**Conditional Edge**: After `check_feedback`, routes to either `select_tools` (retry) or `format_response` (respond).

### 2. Tools

#### Tool 1: REST API (query_metrics_api)
- **Access**: HTTP calls to http://127.0.0.1:8001
- **Endpoints**: /metrics/latency, /metrics/throughput, /metrics/errors, /health, /services
- **Response Time**: <50ms
- **Features**: Real-time metrics, auto-generated OpenAPI docs

#### Tool 2: Knowledge RAG (search_knowledge_base)
- **Technology**: OpenAI Embeddings + FAISS + BM25
- **Documents**: 5 markdown files, 101 chunks
- **Search Modes**: hybrid (default), vector, bm25
- **Performance**: ~15ms per query (cached)

#### Tool 3: SQL Database (query_sql_database)
- **Technology**: SQLite + OpenAI NL-to-SQL
- **Data**: 840 rows, 5 services, 7 days of hourly metrics
- **Features**: Safe execution (blocks DROP/DELETE/UPDATE/INSERT)
- **Performance**: <10ms per query

#### Tool 4: Calculator (calculate)
- **Operations**: +, -, *, /, **, %, <, >, <=, >=, ==, !=
- **Functions**: abs, min, max, round, sum, len
- **Security**: Safe eval, blocks import/exec/eval

### 3. Services

#### FastAPI REST Service
**Endpoint**: http://127.0.0.1:8001

**Routes**:
- `GET /metrics/latency` - Latency metrics (p50, p95, p99)
- `GET /metrics/throughput` - Request throughput data
- `GET /metrics/errors` - Error rates with 4xx/5xx breakdown
- `GET /health` - Service health check
- `GET /services` - List all services
- `GET /docs` - Auto-generated API documentation

**Data Pattern**: Realistic patterns with business hours variation and occasional incidents

#### RAG Service
**Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
**Index**: FAISS IndexFlatIP (cosine similarity)
**Keyword**: BM25 for keyword matching
**Hybrid**: Weighted fusion (0.7 vector + 0.3 BM25)

**Metadata per chunk**: filename, chunk_id, prev/next IDs, span positions, section headings, timestamps

#### Database Service
**Schema**: service_name, timestamp, cpu_usage, memory_usage, request_count, error_count, avg_latency, status, region, instance_id

**Services**: api-gateway, auth-service, business-logic, data-processor, payment-service

**Distribution**: 95% healthy, 3% degraded, 2% unhealthy

---

## 📖 Usage Examples

### Example 1: Metrics Lookup
```python
result = run_agent("What is the current latency for api-gateway?")

# Output:
# Intent: metrics_lookup
# Confidence: 1.00
# Tools: ['query_metrics_api']
# Time: 1504ms
# Answer: "P95 latency is 64.63ms (healthy)"
```

### Example 2: Knowledge Search
```python
result = run_agent("How do I troubleshoot high latency issues?")

# Output:
# Intent: knowledge_lookup
# Confidence: 0.99
# Tools: ['search_knowledge_base']
# Time: 2531ms
# Answer: "Found 3 relevant documentation sections:
#          1. troubleshooting.md (score: 1.00)..."
```

### Example 3: Calculation
```python
result = run_agent("Calculate the average of 150, 200, and 250")

# Output:
# Intent: calculation
# Confidence: 1.00
# Tools: ['calculate']
# Time: 1683ms
# Answer: "Result: 200"
```

### Example 4: Mixed Query (Multiple Tools)
```python
result = run_agent("What is the latency for api-gateway and how can I improve it?")

# Output:
# Intent: mixed
# Confidence: 1.00
# Tools: ['query_metrics_api', 'search_knowledge_base', 'query_sql_database']
# Time: 3710ms
# Answer: "Current P95 latency is 64.98ms (healthy).
#          Based on best practices: [recommendations]..."
```

### Example 5: Historical Data (SQL)
```python
result = run_agent("What was the average CPU usage for api-gateway over the past week?")

# Output:
# Intent: metrics_lookup
# Confidence: 1.00
# Tools: ['query_metrics_api', 'query_sql_database']
# Time: 3498ms
# Answer: "Historical average CPU usage: 45.2%"
```

---

## 🧪 Testing

### Test Coverage
- ✅ **41+ individual checks** across 13 test files
- ✅ **100% success rate** in all tests
- ✅ All intent types, workflows, feedback loops, caching, and API endpoints verified
- ✅ **Enhanced trace visibility** - all tests show detailed execution traces

### Run Tests

```bash
# Individual tool tests (18 checks)
python test/test_individual_tools.py

# End-to-end workflow tests (8 scenarios)
python test/test_end_to_end.py

# Feedback loop tests (8 checks, with detailed trace output)
python test/test_feedback_loop.py

# Orchestration decision tests (with detailed trace output)
python test/test_orchestration.py

# Trace caching system tests (6 tests)
python test/test_trace_cache.py

# API endpoint validation (5 endpoints with prefilled parameters)
python test/test_api_endpoints.py

# Documentation verification (9 checks)
python test/verify_documentation.py

# Agent workflow tests
python test/test_agent.py

# RAG service tests
python test/test_rag_service.py

# Tool tests
python test/test_tools.py

# API service tests (requires server running)
python test/validate_api_service.py

# Database tests
python test/validate_db_service.py
```

**Trace Display Control**:
```bash
# Show detailed traces (default)
SHOW_TRACES=true python test/test_feedback_loop.py

# Hide traces
SHOW_TRACES=false python test/test_feedback_loop.py
```

---

## 🔍 Observability

### Built-in Tracing

Every agent execution includes complete trace:

```python
result = run_agent("query", verbose=True)  # Prints trace to console

# Access trace programmatically
for event in result['trace']:
    timestamp = event['timestamp']
    node = event['node']
    event_type = event['event_type']
    data = event['data']

# Node durations
for node, duration_ms in result['node_durations'].items():
    print(f"{node}: {duration_ms:.0f}ms")

# Total duration
print(f"Total: {result['total_duration_ms']:.0f}ms")
```

### Trace Caching System (NEW)

Persistent 24-hour cache for all execution traces:

```python
from utils.trace_cache import (
    get_agent_executions,      # Load cached agent executions
    cache_agent_execution,     # Cache a new execution
    get_langsmith_traces,      # Get LangSmith traces (cached or API)
    auto_populate_traces,      # Auto-load all traces
    get_cache_status,          # Check cache status
    refresh_langsmith_cache    # Force refresh from API
)

# Auto-populated on Streamlit startup
summary = auto_populate_traces()
print(f"Loaded {summary['total']} traces from cache")

# Check cache status
status = get_cache_status()
for name, info in status.items():
    print(f"{name}: {'valid' if info['valid'] else 'expired'}")
```

**Features**:
- 24-hour automatic expiration
- Multiple sources: LangSmith API, agent executions, demos, tests
- Auto-population on Streamlit app startup
- Cache status reporting and manual refresh

### LangSmith Integration

Enable advanced tracing with LangSmith:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls__your-key
export LANGCHAIN_PROJECT=intent-agent-poc
```

All agent runs will automatically appear in LangSmith dashboard with:
- Complete state transitions
- Tool inputs/outputs
- Error tracking
- Performance metrics

**Real-time API Fetching**:
- Streamlit Observability tab fetches latest traces from LangSmith API
- Caches traces locally for 24 hours
- View traces directly in the dashboard without leaving the app
- Direct links to LangSmith platform for detailed inspection

### API Testing Tab (NEW)

Interactive Swagger-like interface in Streamlit:

**Features**:
- Zero-configuration testing with prefilled parameters
- 5 REST API endpoints (latency, throughput, errors, health, services)
- Live URL preview before execution
- Visual metrics cards for responses
- Copy-paste cURL and Python examples
- Real-time response time measurement

**Access**: `streamlit run streamlit_app.py` → Navigate to "API Testing" tab

### OpenTelemetry Support

The agent includes `trace_id`, `span_id`, and `parent_span_id` fields for distributed tracing integration.

---

## ⚙️ Configuration

### Environment Variables

**Automatic Loading**: The `.env` file is automatically loaded by all entry points and takes precedence over system environment variables. No manual `export` commands needed!

See `.env.example` for all options:

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional - LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your-key
LANGCHAIN_PROJECT=intent-agent-poc

# Optional - Service Configuration
METRICS_API_URL=http://127.0.0.1:8001
DB_PATH=data/metrics.db
DOCS_PATH=data/docs
EMBEDDINGS_PATH=data/embeddings

# Optional - Agent Configuration
MAX_RETRIES=2
CONFIDENCE_HIGH=0.8
CONFIDENCE_MEDIUM=0.6

# Optional - Test Configuration
SHOW_TRACES=true  # Show detailed execution traces in test output
```

**Note**: All Python entry points (`main.py`, `streamlit_app.py`, agent modules, services) use `load_dotenv(override=True)` to ensure the `.env` file always takes precedence.

### Trace Caching Configuration

Trace caching is automatically enabled with default settings:
- **Cache Lifetime**: 24 hours (configurable in `utils/trace_cache.py`)
- **Cache Location**: `data/trace_cache/` (auto-created)
- **Auto-Population**: Enabled on Streamlit startup
- **Sources**: LangSmith API, agent executions, demos, tests

### Confidence Thresholds

- **HIGH (≥0.8)**: Proceed with confidence, no retry
- **MEDIUM (0.6-0.8)**: Proceed with some uncertainty
- **LOW (<0.6)**: May trigger retry if retry_count < MAX_RETRIES

### Feedback Loop

- **Max Retries**: 2 (prevents infinite loops)
- **Retry Triggers**: Tool failures, incomplete data (<70%), unclear intent
- **Retry Logic**: Routes back to `select_tools` with different tool selection

---

## 📚 Documentation

### Core Documentation
- **[README.md](README.md)** - This file (comprehensive guide with quick start)
- **[data/docs/architecture.md](data/docs/architecture.md)** - Complete system architecture (1,640 lines)
  - Module-by-module breakdown with line numbers
  - Every file documented with purpose, functionality, and code snippets
  - Data flow diagrams and end-to-end examples
  - Observability, testing, and deployment guides
- **[agent/README.md](agent/README.md)** - Agent architecture and workflow details

### Service Documentation
- **[services/README_API.md](services/README_API.md)** - API endpoints and usage
- **[services/README_DB.md](services/README_DB.md)** - Database schema and queries
- **[services/README_RAG.md](services/README_RAG.md)** - RAG service and search

### Utility Documentation
- **[utils/trace_cache.py](utils/trace_cache.py)** - Trace caching system (inline documentation)
- **[utils/trace_display.py](utils/trace_display.py)** - Test trace visualization (inline documentation)

---

## 🔧 Troubleshooting

### RAG Embeddings Not Found
```bash
# Initialize manually
python demo/demo_rag.py
```

### API Server Not Running
```bash
# Start server
python start_api_server.py
```

### Database Not Found
```bash
# Create database
python test/validate_db_service.py
```

### OpenAI API Errors
```bash
# Check your API key
echo $OPENAI_API_KEY

# Verify it's valid
python -c "from openai import OpenAI; print(OpenAI().models.list())"
```

### Import Errors
```bash
# Make sure you're in the project root
cd agent_poc

# Reinstall dependencies
pip install -r requirements.txt
```
---

## 🎯 Requirements Compliance

| Requirement | Status | Implementation |
|------------|--------|----------------|
| Intent → tool selection | ✅ COMPLETE | OpenAI GPT-4.1-mini + keyword fallback |
| 3+ tools | ✅ COMPLETE | 4 tools (REST API, RAG, SQL, Calculator) |
| HTTP/REST tool | ✅ COMPLETE | FastAPI metrics service |
| Knowledge tool | ✅ COMPLETE | RAG with FAISS + BM25 hybrid search |
| Utility tool | ✅ COMPLETE | Calculator + SQL database |
| Workflow orchestration | ✅ COMPLETE | 7 nodes with LangGraph |
| Aggregation | ✅ COMPLETE | aggregate_results node |
| Inference | ✅ COMPLETE | Threshold checks, comparisons, trends |
| Feedback loop | ✅ COMPLETE | Confidence-based retry (max 2) |
| Return answer + trace | ✅ COMPLETE | Complete trace with all events |



### Recent Improvements (Implemented)
1. ✅ **Trace Caching**: 24-hour persistent cache with auto-population
2. ✅ **API Testing Interface**: Swagger-like UI with prefilled parameters
3. ✅ **Enhanced Test Visibility**: Detailed execution traces in all test outputs
4. ✅ **LangSmith API Integration**: Real-time trace fetching and caching
5. ✅ **Intent Classification Guardrails**: Improved classification with reasonable defaults
6. ✅ **Comprehensive Architecture Docs**: 1,640-line architecture documentation

### Future Enhancements
1. **Logging**: Replace print statements with structured logging framework (e.g., loguru)
2. **Query Caching**: Cache agent responses for repeated/similar queries
3. **Streaming**: Add streaming responses for better UX with SSE or WebSockets
4. **Sessions**: Multi-turn conversation support with context management
5. **Rate Limiting**: Add OpenAI API rate limiting and retry logic
6. **Docker**: Containerization for easy deployment with docker-compose
7. **CI/CD**: Automated testing pipeline with GitHub Actions
8. **Benchmarks**: Performance profiling and optimization
9. **Additional Tools**: Web search, code execution, file system access
10. **MCP Server**: Model Context Protocol server for fallback tools

