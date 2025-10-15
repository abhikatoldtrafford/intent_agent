"""
Agent Package

Intent-routed agent with LangGraph workflow, multiple tools,
and feedback loop for confidence-based retry logic.

Main Components:
- Tools: REST API, Knowledge RAG, SQL Database, Calculator
- State: AgentState TypedDict with full workflow state
- Nodes: 7 workflow nodes (intent, tools, aggregation, inference, feedback, response)
- Graph: LangGraph workflow with conditional retry logic

Usage:
    from agent import run_agent

    result = run_agent("What is the latency for api-gateway?")
    print(result["final_answer"])
"""

# Load environment variables from .env file (override system env)
from dotenv import load_dotenv
load_dotenv(override=True)

# Configure observability (LangSmith, OpenTelemetry, logging)
import os
import sys
from pathlib import Path

# Add utils to path if not already there
utils_path = Path(__file__).parent.parent / "utils"
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path.parent))

# Configure observability based on environment variables
try:
    from utils.observability import configure_observability

    # Enable LangSmith by default (set to "true" in .env or disabled with "false")
    enable_langsmith = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true"

    # Enable OpenTelemetry if ENABLE_OPENTELEMETRY is set or OTEL_EXPORTER is configured
    enable_otel = os.getenv("ENABLE_OPENTELEMETRY", "true").lower() == "true"

    # Configure all observability features
    observability_status = configure_observability(
        enable_logging=True,
        enable_langsmith=enable_langsmith,
        enable_opentelemetry=enable_otel,
        langsmith_project=os.getenv("LANGCHAIN_PROJECT", "intent-agent-poc"),
        otel_service_name=os.getenv("OTEL_SERVICE_NAME", "intent-agent"),
        otel_exporter=os.getenv("OTEL_EXPORTER", "console")
    )

    # Store status for later access
    OBSERVABILITY_STATUS = observability_status

except Exception as e:
    # Fallback if observability setup fails
    print(f"Warning: Observability configuration failed: {e}")
    OBSERVABILITY_STATUS = {"logging": False, "langsmith": False, "opentelemetry": False}

# Tools
from agent.tools import (
    query_metrics_api,
    search_knowledge_base,
    query_sql_database,
    calculate,
    ALL_TOOLS,
    TOOL_DESCRIPTIONS,
    get_tool_by_name,
    list_available_tools
)

# State
from agent.state import (
    AgentState,
    create_initial_state,
    add_trace_event,
    update_confidence,
    # Intent types
    INTENT_METRICS_LOOKUP,
    INTENT_KNOWLEDGE_LOOKUP,
    INTENT_CALCULATION,
    INTENT_MIXED,
    INTENT_CLARIFICATION,
    INTENT_UNKNOWN,
    # Tool names
    TOOL_METRICS_API,
    TOOL_KNOWLEDGE_RAG,
    TOOL_SQL_DATABASE,
    TOOL_CALCULATOR,
    # Inference types
    INFERENCE_THRESHOLD,
    INFERENCE_COMPARISON,
    INFERENCE_TREND,
    INFERENCE_RECOMMENDATION,
    INFERENCE_AGGREGATION,
    INFERENCE_NONE,
    # Confidence thresholds
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW,
    MAX_RETRIES
)

# Nodes
from agent.nodes import (
    classify_intent,
    select_tools,
    execute_tools,
    aggregate_results,
    perform_inference,
    check_feedback,
    format_response
)

# Graph
from agent.graph import (
    agent_graph,
    create_agent_graph,
    run_agent,
    stream_agent,
    arun_agent,
    get_graph_visualization
)


__all__ = [
    # Main functions
    "run_agent",
    "stream_agent",
    "arun_agent",
    "create_agent_graph",
    "get_graph_visualization",

    # Tools
    "query_metrics_api",
    "search_knowledge_base",
    "query_sql_database",
    "calculate",
    "ALL_TOOLS",
    "TOOL_DESCRIPTIONS",
    "get_tool_by_name",
    "list_available_tools",

    # State
    "AgentState",
    "create_initial_state",
    "add_trace_event",
    "update_confidence",

    # Constants - Intents
    "INTENT_METRICS_LOOKUP",
    "INTENT_KNOWLEDGE_LOOKUP",
    "INTENT_CALCULATION",
    "INTENT_MIXED",
    "INTENT_CLARIFICATION",
    "INTENT_UNKNOWN",

    # Constants - Tools
    "TOOL_METRICS_API",
    "TOOL_KNOWLEDGE_RAG",
    "TOOL_SQL_DATABASE",
    "TOOL_CALCULATOR",

    # Constants - Inference
    "INFERENCE_THRESHOLD",
    "INFERENCE_COMPARISON",
    "INFERENCE_TREND",
    "INFERENCE_RECOMMENDATION",
    "INFERENCE_AGGREGATION",
    "INFERENCE_NONE",

    # Constants - Confidence
    "CONFIDENCE_HIGH",
    "CONFIDENCE_MEDIUM",
    "CONFIDENCE_LOW",
    "MAX_RETRIES",

    # Nodes
    "classify_intent",
    "select_tools",
    "execute_tools",
    "aggregate_results",
    "perform_inference",
    "check_feedback",
    "format_response",

    # Graph
    "agent_graph",

    # Observability
    "OBSERVABILITY_STATUS",
]


__version__ = "1.0.0"
