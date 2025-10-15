"""
Agent State Schema

Defines the state structure for the LangGraph agent workflow.
The state is passed between nodes and updated as the agent progresses.
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime


class AgentState(TypedDict, total=False):
    """
    Complete state for the agent workflow.

    This state is passed through all nodes in the graph and accumulates
    information as the agent processes the user's query.

    Attributes:
        # Input
        query: Original user query/question
        session_id: Unique session identifier for tracking

        # Intent Classification
        intent: Classified intent (e.g., "metrics_lookup", "knowledge_lookup", "calculation")
        intent_confidence: Confidence score for intent classification (0-1)

        # Tool Selection
        tools_to_use: List of tool names selected for this query
        tool_selection_reasoning: Why these tools were chosen

        # Tool Execution
        tool_outputs: Dictionary mapping tool names to their outputs
        tool_errors: Dictionary mapping tool names to any errors encountered
        tools_executed: List of tools that were successfully executed

        # Aggregation
        aggregated_data: Combined and structured data from all tools
        data_quality: Assessment of data quality (completeness, consistency)

        # Inference
        inference_result: Results of inference/analysis
        inference_type: Type of inference performed (threshold, comparison, trend, etc.)
        findings: Key findings from the analysis
        recommendations: Actionable recommendations based on findings

        # Feedback Loop
        confidence: Overall confidence in the answer (0-1)
        feedback_needed: Whether additional iteration is needed
        retry_count: Number of retry attempts made
        retry_reason: Reason for retry (if applicable)
        clarification_question: Question to ask user (if needed)

        # Response
        final_answer: The final formatted answer to the user
        answer_format: Format of the answer (text, markdown, structured)

        # Trace/Metadata
        trace: List of trace events showing what the agent did
        start_time: Timestamp when query processing started
        end_time: Timestamp when query processing completed
        total_duration_ms: Total processing time in milliseconds
        node_durations: Duration of each node execution

        # OpenTelemetry / LangSmith Integration
        trace_id: Unique trace ID for observability
        span_id: Current span ID
        parent_span_id: Parent span ID (if nested)
    """

    # Input
    query: str
    session_id: str

    # Intent Classification
    intent: str
    intent_confidence: float

    # Tool Selection
    tools_to_use: List[str]
    tool_selection_reasoning: str

    # Tool Execution
    tool_outputs: Dict[str, Any]
    tool_errors: Dict[str, str]
    tools_executed: List[str]

    # Aggregation
    aggregated_data: Dict[str, Any]
    data_quality: Dict[str, Any]

    # Inference
    inference_result: Dict[str, Any]
    inference_type: str
    findings: List[str]
    recommendations: List[str]

    # Feedback Loop
    confidence: float
    feedback_needed: bool
    retry_count: int
    retry_reason: Optional[str]
    clarification_question: Optional[str]

    # Response
    final_answer: str
    answer_format: str

    # Trace/Metadata
    trace: List[Dict[str, Any]]
    start_time: str
    end_time: str
    total_duration_ms: float
    node_durations: Dict[str, float]

    # Observability
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]


# Intent types
INTENT_METRICS_LOOKUP = "metrics_lookup"          # Query real-time or historical metrics
INTENT_KNOWLEDGE_LOOKUP = "knowledge_lookup"      # Search documentation
INTENT_CALCULATION = "calculation"                # Perform calculations/comparisons
INTENT_MIXED = "mixed"                           # Requires multiple tool types
INTENT_CLARIFICATION = "clarification"           # Need more info from user
INTENT_UNKNOWN = "unknown"                       # Cannot determine intent

VALID_INTENTS = [
    INTENT_METRICS_LOOKUP,
    INTENT_KNOWLEDGE_LOOKUP,
    INTENT_CALCULATION,
    INTENT_MIXED,
    INTENT_CLARIFICATION,
    INTENT_UNKNOWN
]

# Tool names (matching tools.py)
TOOL_METRICS_API = "query_metrics_api"
TOOL_KNOWLEDGE_RAG = "search_knowledge_base"
TOOL_SQL_DATABASE = "query_sql_database"
TOOL_CALCULATOR = "calculate"

AVAILABLE_TOOLS = [
    TOOL_METRICS_API,
    TOOL_KNOWLEDGE_RAG,
    TOOL_SQL_DATABASE,
    TOOL_CALCULATOR
]

# Inference types
INFERENCE_THRESHOLD = "threshold_check"          # Check if metric exceeds threshold
INFERENCE_COMPARISON = "comparison"              # Compare metrics between services
INFERENCE_TREND = "trend_analysis"               # Analyze trends over time
INFERENCE_RECOMMENDATION = "recommendation"      # Provide recommendations
INFERENCE_AGGREGATION = "aggregation"            # Aggregate data from multiple sources
INFERENCE_NONE = "none"                         # No inference needed

# Confidence thresholds
CONFIDENCE_HIGH = 0.8
CONFIDENCE_MEDIUM = 0.6
CONFIDENCE_LOW = 0.4

# Max retries to prevent infinite loops
MAX_RETRIES = 2


def create_initial_state(query: str, session_id: Optional[str] = None) -> AgentState:
    """
    Create initial state for a new query.

    Args:
        query: User's question/query
        session_id: Optional session identifier

    Returns:
        AgentState with initialized values
    """
    if session_id is None:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return AgentState(
        # Input
        query=query,
        session_id=session_id,

        # Intent
        intent="",
        intent_confidence=0.0,

        # Tools
        tools_to_use=[],
        tool_selection_reasoning="",
        tool_outputs={},
        tool_errors={},
        tools_executed=[],

        # Aggregation
        aggregated_data={},
        data_quality={},

        # Inference
        inference_result={},
        inference_type="",
        findings=[],
        recommendations=[],

        # Feedback
        confidence=0.0,
        feedback_needed=False,
        retry_count=0,
        retry_reason=None,
        clarification_question=None,

        # Response
        final_answer="",
        answer_format="text",

        # Trace
        trace=[],
        start_time=datetime.now().isoformat(),
        end_time="",
        total_duration_ms=0.0,
        node_durations={},

        # Observability
        trace_id=f"trace_{session_id}",
        span_id="",
        parent_span_id=None
    )


def add_trace_event(
    state: AgentState,
    node_name: str,
    event_type: str,
    data: Dict[str, Any]
) -> None:
    """
    Add a trace event to the state.

    Args:
        state: Current agent state
        node_name: Name of the node adding the event
        event_type: Type of event (e.g., "decision", "tool_call", "error")
        data: Event data
    """
    if "trace" not in state:
        state["trace"] = []

    state["trace"].append({
        "timestamp": datetime.now().isoformat(),
        "node": node_name,
        "event_type": event_type,
        "data": data
    })


def update_confidence(
    state: AgentState,
    factor: float,
    reason: str
) -> None:
    """
    Update confidence score with reasoning.

    Args:
        state: Current agent state
        factor: Multiplicative factor to apply to confidence
        reason: Reason for confidence change
    """
    current = state.get("confidence", 1.0)
    new_confidence = max(0.0, min(1.0, current * factor))

    state["confidence"] = new_confidence

    add_trace_event(
        state,
        "confidence_update",
        "confidence_change",
        {
            "previous": current,
            "new": new_confidence,
            "factor": factor,
            "reason": reason
        }
    )
