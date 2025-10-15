# Intent-Routed Agent POC

> A local agent that classifies natural language queries, routes to appropriate tools, performs inference with feedback loops, and returns complete execution traces.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/framework-LangGraph-green.svg)](https://github.com/langchain-ai/langgraph)
[![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)]()

---

## 🎥 Video Demonstration

📹 **[Watch the Full Demo on YouTube]**<!-- https://youtu.be/vD4zDR47uAs -->

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

### Technical Highlights
- **LangGraph** workflow orchestration with conditional edges
- **OpenAI GPT-4.1-mini** for intent classification and NL-to-SQL 
- **FAISS + BM25** hybrid search for document retrieval
- **FastAPI** REST service with auto-generated OpenAPI docs
- **SQLite** database with 840 rows of time-series metrics
- **LangSmith** integration ready for advanced tracing

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

**Features**:
- 🏠 **Dashboard** - System status and metrics overview
- 📚 **Documentation** - Complete usage guide and API reference
- 🤖 **Agent Testing** - Interactive query interface with history
- 🔍 **RAG Explorer** - Search knowledge base and browse documents
- 💾 **SQL Viewer** - Query database with natural language or SQL
- 🧪 **Test Runner** - Run and view test results
- 🎮 **Demo Runner** - Execute demo scripts
- 🔀 **Workflow Viz** - Visualize agent routing and execution flow

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
│   ├── nodes.py                # 7 node implementations
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
├── data/                       # Data files
│   ├── docs/                   # 5 markdown documents (3,795 lines)
│   │   ├── architecture.md
│   │   ├── api_guide.md
│   │   ├── troubleshooting.md
│   │   ├── deployment.md
│   │   └── monitoring.md
│   ├── metrics.db              # SQLite database (840 rows, 192KB)
│   └── embeddings/             # FAISS cache (auto-generated)
│
├── test/                       # Test suite (10 test files)
│   ├── test_agent.py           # Agent workflow tests
│   ├── test_rag_service.py     # RAG tests
│   ├── test_tools.py           # Tool tests
│   ├── test_individual_tools.py    # Individual tool validation
│   ├── test_end_to_end.py      # End-to-end workflow tests
│   ├── test_feedback_loop.py   # Feedback loop tests
│   ├── verify_documentation.py # Documentation verification
│   ├── validate_api_service.py # API validation
│   └── validate_db_service.py  # Database validation
│
├── demo/                       # Demo scripts
│   ├── demo_agent.py           # Interactive agent demonstration
│   └── demo_rag.py             # RAG initialization demo
│
├── main.py                     # Interactive CLI interface
├── streamlit_app.py            # Streamlit web UI (comprehensive dashboard)
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
- ✅ **43 individual checks** across 4 test suites
- ✅ **100% success rate** in all tests
- ✅ All intent types, workflows, feedback loops verified

### Run Tests

```bash
# Individual tool tests (18 checks)
python test/test_individual_tools.py

# End-to-end workflow tests (8 scenarios)
python test/test_end_to_end.py

# Feedback loop tests (8 checks)
python test/test_feedback_loop.py

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
```

**Note**: All Python entry points (`main.py`, `streamlit_app.py`, agent modules, services) use `load_dotenv(override=True)` to ensure the `.env` file always takes precedence.

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
- **[README.md](README.md)** - This file (comprehensive guide)
- **[agent/README.md](agent/README.md)** - Agent architecture and workflow

### Service Documentation
- **[services/README_API.md](services/README_API.md)** - API endpoints and usage
- **[services/README_DB.md](services/README_DB.md)** - Database schema and queries
- **[services/README_RAG.md](services/README_RAG.md)** - RAG service and search

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



### Potential Improvements
1. **Logging**: Replace print statements with structured logging framework
2. **Caching**: Implement query caching for repeated requests
3. **Streaming**: Add streaming responses for better UX
4. **Sessions**: Multi-turn conversation support
5. **Rate Limiting**: Add OpenAI API rate limiting
6. **Docker**: Containerization for easy deployment
7. **CI/CD**: Automated testing pipeline
8. **Benchmarks**: Performance profiling and optimization
9. **Additional Tools**: Web search, code execution, etc.

