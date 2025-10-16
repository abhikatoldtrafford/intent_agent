"""
LangGraph Nodes

Implements all nodes for the agent workflow:
1. classify_intent - Classifies user query intent
2. select_tools - Selects appropriate tools
3. execute_tools - Executes selected tools
4. aggregate_results - Combines tool outputs
5. perform_inference - Makes decisions and analyzes data
6. check_feedback - Evaluates confidence and decides on retry
7. format_response - Creates final answer with trace
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List
from openai import OpenAI

from agent.state import (
    AgentState,
    add_trace_event,
    update_confidence,
    INTENT_METRICS_LOOKUP,
    INTENT_KNOWLEDGE_LOOKUP,
    INTENT_CALCULATION,
    INTENT_MIXED,
    INTENT_CLARIFICATION,
    INTENT_UNKNOWN,
    TOOL_METRICS_API,
    TOOL_KNOWLEDGE_RAG,
    TOOL_SQL_DATABASE,
    TOOL_CALCULATOR,
    INFERENCE_THRESHOLD,
    INFERENCE_COMPARISON,
    INFERENCE_TREND,
    INFERENCE_RECOMMENDATION,
    INFERENCE_AGGREGATION,
    INFERENCE_NONE,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    MAX_RETRIES
)
from agent.tools import (
    query_metrics_api,
    search_knowledge_base,
    query_sql_database,
    calculate
)


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# NODE 1: Intent Classification
# ============================================================================

def classify_intent(state: AgentState) -> AgentState:
    """
    Classify the user's query to determine intent.

    Uses OpenAI to analyze the query and categorize it into one of:
    - metrics_lookup: Query about service metrics (latency, errors, etc.)
    - knowledge_lookup: Question about documentation/how-to
    - calculation: Needs mathematical computation
    - mixed: Requires multiple tool types
    - clarification: Need more information from user
    - unknown: Cannot determine intent

    Updates:
        - state["intent"]
        - state["intent_confidence"]
        - state["confidence"]
        - state["trace"]
    """
    start_time = time.time()
    query = state["query"]

    # CRITICAL: Extract CURRENT QUERY from context-aware prompts
    # The Streamlit UI sends: "CONVERSATION CONTEXT...\n\nCURRENT QUERY: {user_input}"
    # We need to extract just the current query for intent classification
    current_query = query
    conversation_context = None

    if "CURRENT QUERY:" in query:
        parts = query.split("CURRENT QUERY:")
        if len(parts) == 2:
            conversation_context = parts[0].strip()
            current_query = parts[1].strip()

            # Store context for later use in answer generation
            state["conversation_context"] = conversation_context

            add_trace_event(
                state,
                "classify_intent",
                "context_extracted",
                {
                    "full_query_length": len(query),
                    "extracted_query": current_query,
                    "has_context": True
                }
            )

    add_trace_event(state, "classify_intent", "node_start", {"query": current_query})

    try:
        # Use OpenAI to classify intent with enhanced prompting
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert intent classifier for a metrics monitoring agent with access to multiple data sources.

YOUR TASK: Analyze user queries and classify them into the correct intent category based on what data sources are needed.

═══════════════════════════════════════════════════════════════════════
DATA SOURCES AVAILABLE (Critical for Intent Classification):
═══════════════════════════════════════════════════════════════════════

1. METRICS REST API (Real-time, Last 1-7 days)
   └─ Services: api-gateway, auth-service, business-logic, data-processor, payment-service
   └─ Data Available:
      • Current latency metrics (p50, p95, p99) in milliseconds
      • Real-time throughput (requests per second, total requests)
      • Live error rates (4xx, 5xx breakdown, total errors)
      • Service health status (healthy/degraded/unhealthy)
      • Service instance information
   └─ Best For: "Current", "now", "latest", "real-time" queries

2. SQL DATABASE (Historical, Last 7 days hourly data)
   └─ 840 rows: 5 services × 168 hours = 7 days of hourly metrics
   └─ Data Available:
      • Historical CPU usage (percentage over time)
      • Historical memory usage (percentage over time)
      • Request counts per hour
      • Error counts per hour
      • Average latency per hour
      • Service status history (healthy/degraded/unhealthy)
      • Regional deployment data (us-east-1, us-west-2, eu-west-1)
      • Instance-level metrics
   └─ Best For: "Average", "trend", "over time", "last week", "compare over period", "historical"

3. KNOWLEDGE BASE (Documentation, Best Practices)
   └─ 5 markdown documents: architecture.md, api_guide.md, troubleshooting.md, deployment.md, monitoring.md
   └─ Content Available:
      • How-to guides (configuration, setup, deployment)
      • Troubleshooting procedures (debugging, fixing issues)
      • Best practices (optimization, performance tuning)
      • Architecture explanations (system design, components)
      • API documentation (endpoints, parameters)
   └─ Best For: "How to", "how do I", "explain", "best practices", "guide", "documentation"

4. CALCULATOR (Mathematical Operations)
   └─ Operations: +, -, *, /, **, %, <, >, <=, >=, ==, !=
   └─ Functions: abs(), min(), max(), round(), sum(), len()
   └─ Best For: "Calculate", "compute", "average of numbers", "percentage", "compare numbers"

═══════════════════════════════════════════════════════════════════════
INTENT CLASSIFICATION RULES:
═══════════════════════════════════════════════════════════════════════

1. metrics_lookup
   → Query asks for ACTUAL METRIC VALUES (numbers, statistics, status)
   → Can be satisfied by REST API or SQL DATABASE
   → Keywords: "what is", "show me", "current", "latency", "errors", "throughput", "CPU", "memory", "status", "performance"
   → IMPORTANT: "Performance" queries are metrics_lookup (asking for latency/CPU/memory/throughput data)
   → Examples: "What's the latency?", "Show error rate", "Is service healthy?", "Tell me about performance"

2. knowledge_lookup
   → Query asks for PROCEDURAL KNOWLEDGE or EXPLANATIONS
   → Can be satisfied by KNOWLEDGE BASE only
   → Keywords: "how to", "how do I", "explain", "why", "what causes", "best practices", "guide", "configure"
   → Examples: "How do I reduce latency?", "Explain monitoring setup", "Best practices for deployment"

3. calculation
   → Query asks to PERFORM MATH on provided numbers OR metric-derived numbers
   → Requires CALCULATOR (and maybe metrics if calculating from metric values)
   → Keywords: "calculate", "compute", "average of", "sum of", "percentage", "ratio"
   → Examples: "Calculate average of 10, 20, 30", "What's 50% of 1000 requests?"

4. mixed
   → Query needs MULTIPLE DATA SOURCES to answer completely
   → Common patterns:
      • Metrics + Knowledge: "What's the error rate AND how do I fix it?"
      • Metrics + Calculation: "Show latency for 3 services and calculate average"
      • Historical + Real-time: "Compare current CPU to last week's average"
   → Keywords: "and", "also", "compare... and...", "both", "as well as"

5. clarification
   → Query is too VAGUE or AMBIGUOUS to determine WHAT data is needed
   → Missing critical information about WHAT to retrieve (not WHO/WHICH)
   → IMPORTANT: Requests mentioning specific data types are NOT vague:
      • "All services" or "all metrics" = metrics_lookup (comprehensive query)
      • "Performance" = metrics_lookup (asking for latency/CPU/memory/throughput)
      • "Status" or "health" = metrics_lookup (asking for service status)
      • Time modifications ("last 30 days", "past week") with context = metrics_lookup
   → IMPORTANT: Context-aware follow-up queries (with CONVERSATION CONTEXT):
      • If query contains "CONVERSATION CONTEXT", extract the CURRENT QUERY
      • Interpret follow-ups using the context (e.g., "last 30 days" + "latency stats" = metrics_lookup)
      • Time ranges alone ("30 days", "last week") imply expanding previous metrics request
   → Clarification ONLY needed when: Cannot determine if asking for metrics, knowledge, or calculation
   → Examples of TRULY VAGUE queries: "Tell me about the system", "What's going on?", "Show me stuff", "Help me"
   → Examples of NOT vague: "Show me metrics", "Tell me about performance", "Check service health", "last 30 days" (with context)

6. unknown
   → Query is COMPLETELY UNRELATED to monitoring/metrics OR is a casual greeting/chitchat
   → Cannot be answered with available data sources
   → Examples: "What's the weather?", "Tell me a joke", "Hi", "Hello", "How are you?", Random gibberish
   → NOTE: Greetings should be classified as unknown with HIGH confidence (0.9+) to trigger friendly response

═══════════════════════════════════════════════════════════════════════
COMPLEX TRAINING EXAMPLES (Study these decision patterns):
═══════════════════════════════════════════════════════════════════════

Example 1 - Historical Trend Analysis with Comparison:
Query: "Compare the average CPU usage between api-gateway and auth-service over the last 72 hours"
Response:
{
    "intent": "metrics_lookup",
    "confidence": 0.95,
    "reasoning": "Requires historical data comparison from SQL database. Multiple services, time range specified, aggregate function (average) - pure metrics query."
}
Why not mixed? The calculation (average) is part of SQL query, not separate calculator tool.

Example 2 - Multi-step Diagnostic:
Query: "The payment-service has high latency, show me the current metrics and explain what might be causing it"
Response:
{
    "intent": "mixed",
    "confidence": 0.98,
    "reasoning": "Two-part query: (1) Show current metrics from REST API, (2) Explain causes from knowledge base troubleshooting docs."
}
Why mixed? Explicitly asks for both metric data AND explanatory knowledge.

Example 3 - Best Practice with Context:
Query: "Our error rate keeps spiking during peak hours, what are the recommended strategies to handle this?"
Response:
{
    "intent": "knowledge_lookup",
    "confidence": 0.92,
    "reasoning": "Question asks for recommended strategies (knowledge), not current error rate values. Context about 'spiking' is informational, not a data request."
}
Why not mixed? User doesn't ask to SEE the error rate, they want solutions from best practices docs.

Example 4 - Cross-service Performance Analysis:
Query: "Which service had the most 5xx errors in the last 24 hours and what was the error rate percentage?"
Response:
{
    "intent": "metrics_lookup",
    "confidence": 0.97,
    "reasoning": "Requires SQL database query with aggregation (SUM, MAX) and calculation (percentage). Both are satisfied by SQL query capabilities."
}
Why not mixed/calculation? SQL can do the aggregation and percentage calc in one query - single data source.

Example 5 - Threshold-based Alert Investigation:
Query: "Show me all services where memory usage exceeded 85 percent in the past week and calculate how many times it happened for each"
Response:
{
    "intent": "metrics_lookup",
    "confidence": 0.94,
    "reasoning": "Historical threshold query with COUNT aggregation. SQL database can filter (WHERE > 85) and count occurrences (GROUP BY, COUNT) in single query."
}
Why not mixed? Filtering and counting are SQL operations, not separate calculator needs.

═══════════════════════════════════════════════════════════════════════
CLASSIFICATION DECISION TREE:
═══════════════════════════════════════════════════════════════════════

1. Does query ask for HOW TO do something or EXPLAIN something?
   → YES: knowledge_lookup (unless also asking for current metrics = mixed)
   → NO: Continue to step 2

2. Does query involve ONLY arithmetic on explicit numbers (not from metrics)?
   → YES: calculation
   → NO: Continue to step 3

3. Does query ask for MULTIPLE types of information from DIFFERENT categories?
   → Metrics + Knowledge: mixed
   → Metrics + Standalone Math: mixed
   → Current + Historical comparison: metrics_lookup (same category)
   → NO: Continue to step 4

4. Does query ask for metric values, status, or performance data?
   → YES: metrics_lookup
   → NO: Continue to step 5

5. Is query vague or missing critical context?
   → YES: clarification
   → NO: unknown

═══════════════════════════════════════════════════════════════════════
OUTPUT REQUIREMENTS:
═══════════════════════════════════════════════════════════════════════

CRITICAL RULES:
1. ONLY respond with valid JSON - no markdown, no explanations outside JSON
2. Intent MUST be exactly one of: metrics_lookup, knowledge_lookup, calculation, mixed, clarification, unknown
3. Confidence MUST be between 0.0 and 1.0
4. Reasoning MUST explain which data source(s) are needed and why
5. Use the decision tree above for edge cases

Response Format:
{
    "intent": "metrics_lookup",
    "confidence": 0.95,
    "reasoning": "Brief explanation referencing data sources and decision logic"
}

YOUR RESPONSE MUST BE VALID JSON ONLY."""
                },
                {
                    "role": "user",
                    "content": f"Classify this query: {current_query}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        result = json.loads(response.choices[0].message.content)

        intent = result.get("intent", INTENT_UNKNOWN)
        intent_confidence = result.get("confidence", 0.5)
        reasoning = result.get("reasoning", "")

        state["intent"] = intent
        state["intent_confidence"] = intent_confidence
        state["confidence"] = intent_confidence
        state["current_query"] = current_query  # Store extracted query for tool execution

        add_trace_event(
            state,
            "classify_intent",
            "classification",
            {
                "intent": intent,
                "confidence": intent_confidence,
                "reasoning": reasoning
            }
        )

    except Exception as e:
        # Fallback: simple keyword-based classification
        query_lower = current_query.lower().strip()

        if any(kw in query_lower for kw in ["latency", "error", "throughput", "health", "metric", "performance"]):
            intent = INTENT_METRICS_LOOKUP
            intent_confidence = 0.6
        elif any(kw in query_lower for kw in ["how to", "how do", "what is", "explain", "documentation", "guide"]):
            intent = INTENT_KNOWLEDGE_LOOKUP
            intent_confidence = 0.6
        elif any(kw in query_lower for kw in ["calculate", "compute", "average", "compare", "percentage"]):
            intent = INTENT_CALCULATION
            intent_confidence = 0.6
        else:
            intent = INTENT_UNKNOWN
            intent_confidence = 0.3

        state["intent"] = intent
        state["intent_confidence"] = intent_confidence
        state["confidence"] = intent_confidence
        state["current_query"] = current_query  # Store extracted query for tool execution

        add_trace_event(
            state,
            "classify_intent",
            "fallback_classification",
            {
                "error": str(e),
                "intent": intent,
                "confidence": intent_confidence
            }
        )

    duration_ms = (time.time() - start_time) * 1000
    if "node_durations" not in state:
        state["node_durations"] = {}
    state["node_durations"]["classify_intent"] = duration_ms

    add_trace_event(state, "classify_intent", "node_end", {"duration_ms": duration_ms})

    return state


# ============================================================================
# NODE 2: Tool Selection
# ============================================================================

def select_tools(state: AgentState) -> AgentState:
    """
    Select appropriate tools based on classified intent.

    Maps intent to one or more tools:
    - metrics_lookup → query_metrics_api, query_sql_database
    - knowledge_lookup → search_knowledge_base
    - calculation → calculate
    - mixed → multiple tools

    Updates:
        - state["tools_to_use"]
        - state["tool_selection_reasoning"]
        - state["trace"]
    """
    start_time = time.time()
    intent = state["intent"]
    # Use extracted current_query if available (from context-aware prompts)
    query = state.get("current_query", state["query"])

    add_trace_event(state, "select_tools", "node_start", {"intent": intent})

    tools_to_use = []
    reasoning = ""
    query_lower = query.lower()

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED TOOL SELECTION LOGIC (Aligned with Intent Classification)
    # ═══════════════════════════════════════════════════════════════════

    if intent == INTENT_METRICS_LOOKUP:
        # Analyze query to determine optimal data source(s)

        # Keywords for real-time data (REST API)
        realtime_keywords = ["current", "now", "real-time", "live", "latest", "right now", "at this moment"]
        # Keywords for historical data (SQL Database)
        historical_keywords = ["history", "historical", "past", "trend", "over time", "last week", "last month",
                              "average over", "72 hours", "24 hours", "7 days", "compare over", "past week"]
        # Keywords for database-specific features
        db_specific_keywords = ["cpu", "memory", "average cpu", "average memory", "exceeded", "threshold",
                               "how many times", "count", "aggregate", "group by"]

        has_realtime = any(kw in query_lower for kw in realtime_keywords)
        has_historical = any(kw in query_lower for kw in historical_keywords)
        has_db_specific = any(kw in query_lower for kw in db_specific_keywords)

        if has_db_specific or (has_historical and not has_realtime):
            # Pure historical/aggregation query - SQL database only
            tools_to_use = [TOOL_SQL_DATABASE]
            reasoning = (f"Query requires historical data analysis from SQL database "
                        f"({'CPU/memory metrics' if has_db_specific else 'time-series analysis'})")

        elif has_realtime and not has_historical:
            # Pure real-time query - REST API only
            tools_to_use = [TOOL_METRICS_API]
            reasoning = "Query asks for current/real-time metrics - using REST API for latest data"

        elif has_historical and has_realtime:
            # Comparison between current and historical - both sources
            tools_to_use = [TOOL_METRICS_API, TOOL_SQL_DATABASE]
            reasoning = "Query compares current vs historical metrics - using both REST API and SQL database"

        else:
            # Ambiguous query - need to determine best source
            # CPU/Memory are ONLY in database, not in REST API
            if has_db_specific:
                # Has CPU/memory keywords but no time context - use database
                tools_to_use = [TOOL_SQL_DATABASE]
                reasoning = "Query mentions CPU/memory metrics - using SQL database (only available source)"
            else:
                # General metrics query without clear time context
                # Use both sources for comprehensive answer (current + historical context)
                tools_to_use = [TOOL_METRICS_API, TOOL_SQL_DATABASE]
                reasoning = "General metrics query - using both REST API (current) and SQL database (trends) for comprehensive answer"

    elif intent == INTENT_KNOWLEDGE_LOOKUP:
        # Pure documentation/how-to query
        tools_to_use = [TOOL_KNOWLEDGE_RAG]
        reasoning = "Procedural knowledge question - using RAG to search documentation (architecture.md, api_guide.md, troubleshooting.md, deployment.md, monitoring.md)"

    elif intent == INTENT_CALCULATION:
        # Check if calculation needs metric data first
        metric_keywords = ["latency", "error", "throughput", "cpu", "memory", "requests", "metric"]
        needs_metrics = any(kw in query_lower for kw in metric_keywords)

        if needs_metrics:
            # Need to fetch metrics before calculating
            if any(kw in query_lower for kw in ["average over", "last week", "historical"]):
                tools_to_use = [TOOL_SQL_DATABASE, TOOL_CALCULATOR]
                reasoning = "Calculation on historical metrics - fetching from SQL database then calculating"
            else:
                tools_to_use = [TOOL_METRICS_API, TOOL_CALCULATOR]
                reasoning = "Calculation on current metrics - fetching from REST API then calculating"
        else:
            # Pure arithmetic on provided numbers
            tools_to_use = [TOOL_CALCULATOR]
            reasoning = "Direct arithmetic calculation - using calculator only (no data fetching needed)"

    elif intent == INTENT_MIXED:
        # Complex query needing multiple data source types
        # Analyze which combinations are needed

        needs_metrics = any(kw in query_lower for kw in ["latency", "error", "throughput", "metric", "performance", "cpu", "memory"])
        needs_knowledge = any(kw in query_lower for kw in ["how", "why", "explain", "configure", "best practice", "guide", "troubleshoot", "fix", "improve", "reduce"])
        needs_calculation = any(kw in query_lower for kw in ["calculate", "average", "compare", "percentage", "ratio"])

        tools_to_use = []
        reasoning_parts = []

        if needs_metrics:
            # Decide between API or DB based on time context
            if any(kw in query_lower for kw in ["current", "now", "latest"]):
                tools_to_use.append(TOOL_METRICS_API)
                reasoning_parts.append("REST API for current metrics")
            elif any(kw in query_lower for kw in ["historical", "past", "over time", "last week"]):
                tools_to_use.append(TOOL_SQL_DATABASE)
                reasoning_parts.append("SQL database for historical metrics")
            else:
                tools_to_use.append(TOOL_METRICS_API)
                reasoning_parts.append("REST API for metrics")

        if needs_knowledge:
            tools_to_use.append(TOOL_KNOWLEDGE_RAG)
            reasoning_parts.append("RAG for documentation/best practices")

        if needs_calculation:
            tools_to_use.append(TOOL_CALCULATOR)
            reasoning_parts.append("Calculator for computations")

        # Ensure we have at least metrics + knowledge (common mixed pattern)
        if not tools_to_use:
            tools_to_use = [TOOL_METRICS_API, TOOL_KNOWLEDGE_RAG]
            reasoning_parts = ["REST API for metrics", "RAG for knowledge"]

        reasoning = f"Mixed query requires: {'; '.join(reasoning_parts)}"

    elif intent == INTENT_CLARIFICATION:
        # Query is too vague - no tools selected yet
        tools_to_use = []
        reasoning = "Query requires clarification - no tools selected until user provides more context"
        state["feedback_needed"] = True
        state["clarification_question"] = (
            "Could you please clarify what you're looking for?\n\n"
            "Are you asking about:\n"
            "1. **Current service metrics** (latency, errors, throughput, health)?\n"
            "2. **Historical trends** (CPU usage over time, error patterns, etc.)?\n"
            "3. **Documentation** (how to configure, troubleshoot, or optimize)?\n"
            "4. **Calculations** (comparing numbers, computing averages, etc.)?"
        )

    else:  # INTENT_UNKNOWN
        # Try knowledge base as fallback - might find something relevant
        tools_to_use = [TOOL_KNOWLEDGE_RAG]
        reasoning = "Intent unclear - attempting knowledge base search as fallback (may help interpret query)"

    state["tools_to_use"] = tools_to_use
    state["tool_selection_reasoning"] = reasoning

    add_trace_event(
        state,
        "select_tools",
        "tools_selected",
        {
            "tools": tools_to_use,
            "reasoning": reasoning
        }
    )

    duration_ms = (time.time() - start_time) * 1000
    state["node_durations"]["select_tools"] = duration_ms

    add_trace_event(state, "select_tools", "node_end", {"duration_ms": duration_ms})

    return state


# ============================================================================
# NODE 3: Tool Execution
# ============================================================================

def execute_tools(state: AgentState) -> AgentState:
    """
    Execute all selected tools and collect outputs.

    For each tool in tools_to_use:
    1. Parse parameters from query
    2. Execute tool
    3. Store output or error
    4. Update confidence based on success

    Updates:
        - state["tool_outputs"]
        - state["tool_errors"]
        - state["tools_executed"]
        - state["confidence"]
        - state["trace"]
    """
    start_time = time.time()
    tools_to_use = state["tools_to_use"]
    # Use extracted current_query if available (from context-aware prompts)
    query = state.get("current_query", state["query"])

    add_trace_event(state, "execute_tools", "node_start", {"tools_count": len(tools_to_use)})

    tool_outputs = {}
    tool_errors = {}
    tools_executed = []

    for tool_name in tools_to_use:
        tool_start = time.time()

        try:
            add_trace_event(
                state,
                "execute_tools",
                "tool_call_start",
                {"tool": tool_name}
            )

            # Execute tool based on name
            if tool_name == TOOL_METRICS_API:
                output = _execute_metrics_api_tool(query, state)
            elif tool_name == TOOL_KNOWLEDGE_RAG:
                output = _execute_knowledge_tool(query, state)
            elif tool_name == TOOL_SQL_DATABASE:
                output = _execute_sql_tool(query, state)
            elif tool_name == TOOL_CALCULATOR:
                output = _execute_calculator_tool(query, state)
            else:
                output = {"error": f"Unknown tool: {tool_name}"}

            # Store output
            tool_outputs[tool_name] = output

            # Check for errors
            if isinstance(output, dict) and "error" in output:
                tool_errors[tool_name] = output["error"]
                update_confidence(state, 0.8, f"Tool {tool_name} returned error")
            else:
                tools_executed.append(tool_name)
                update_confidence(state, 1.1, f"Tool {tool_name} executed successfully")

            tool_duration = (time.time() - tool_start) * 1000

            add_trace_event(
                state,
                "execute_tools",
                "tool_call_end",
                {
                    "tool": tool_name,
                    "duration_ms": tool_duration,
                    "success": tool_name in tools_executed,
                    "output_size": len(str(output))
                }
            )

        except Exception as e:
            tool_errors[tool_name] = str(e)
            tool_outputs[tool_name] = {"error": str(e)}
            update_confidence(state, 0.7, f"Tool {tool_name} failed with exception")

            add_trace_event(
                state,
                "execute_tools",
                "tool_call_error",
                {
                    "tool": tool_name,
                    "error": str(e)
                }
            )

    state["tool_outputs"] = tool_outputs
    state["tool_errors"] = tool_errors
    state["tools_executed"] = tools_executed

    # Overall success rate
    if tools_to_use:
        success_rate = len(tools_executed) / len(tools_to_use)
        if success_rate < 0.5:
            update_confidence(state, 0.6, "More than half of tools failed")

    duration_ms = (time.time() - start_time) * 1000
    state["node_durations"]["execute_tools"] = duration_ms

    add_trace_event(
        state,
        "execute_tools",
        "node_end",
        {
            "duration_ms": duration_ms,
            "tools_executed": len(tools_executed),
            "tools_failed": len(tool_errors)
        }
    )

    return state


def _execute_metrics_api_tool(query: str, state: AgentState) -> Dict[str, Any]:
    """Execute metrics API tool with parameter extraction."""
    query_lower = query.lower()

    # Determine metric type
    if "latency" in query_lower:
        metric_type = "latency"
    elif "throughput" in query_lower or "requests" in query_lower or "rps" in query_lower:
        metric_type = "throughput"
    elif "error" in query_lower:
        metric_type = "errors"
    elif "health" in query_lower:
        metric_type = "health"
    elif "services" in query_lower or "list" in query_lower:
        metric_type = "services"
    else:
        metric_type = "latency"  # Default

    # Extract service name
    services = ["api-gateway", "auth-service", "business-logic", "data-processor", "payment-service"]
    service = None
    for svc in services:
        if svc in query_lower:
            service = svc
            break

    if service is None and metric_type not in ["services"]:
        service = "api-gateway"  # Default

    # Execute tool
    return query_metrics_api.invoke({"metric_type": metric_type, "service": service})


def _execute_knowledge_tool(query: str, state: AgentState) -> Dict[str, Any]:
    """Execute knowledge RAG tool."""
    return search_knowledge_base.invoke({"query": query, "top_k": 3})


def _execute_sql_tool(query: str, state: AgentState) -> Dict[str, Any]:
    """Execute SQL database tool."""
    return query_sql_database.invoke({"question": query})


def _execute_calculator_tool(query: str, state: AgentState) -> Dict[str, Any]:
    """Execute calculator tool with expression extraction."""
    # Try to extract mathematical expression from query
    query_lower = query.lower()

    # Look for explicit expressions
    import re
    math_pattern = r'[\d\+\-\*/\(\)\.\s><=]+'
    matches = re.findall(math_pattern, query)

    if matches:
        expression = max(matches, key=len).strip()
        return calculate.invoke({"expression": expression})
    else:
        return {"error": "Could not extract mathematical expression from query"}


# ============================================================================
# NODE 4: Aggregate Results
# ============================================================================

def aggregate_results(state: AgentState) -> AgentState:
    """
    Combine and structure outputs from multiple tools.

    Creates a unified view of all data:
    - Merges metrics from different sources
    - Combines knowledge base results
    - Structures calculation results

    Updates:
        - state["aggregated_data"]
        - state["data_quality"]
        - state["trace"]
    """
    start_time = time.time()
    tool_outputs = state["tool_outputs"]
    tools_executed = state["tools_executed"]

    add_trace_event(state, "aggregate_results", "node_start", {})

    aggregated = {
        "metrics": {},
        "knowledge": [],
        "calculations": {},
        "errors": []
    }

    data_quality = {
        "completeness": 1.0,
        "consistency": 1.0,
        "issues": []
    }

    # ═══════════════════════════════════════════════════════════════════════
    # INTELLIGENT FALLBACK ROUTING
    # ═══════════════════════════════════════════════════════════════════════

    # Track which sources returned empty/error results for intelligent fallback
    empty_sources = []
    fallback_tools_needed = []

    # Aggregate metrics API data
    if TOOL_METRICS_API in tools_executed:
        api_output = tool_outputs.get(TOOL_METRICS_API, {})
        has_error = "error" in api_output
        is_empty = (not has_error and
                   (not api_output or
                    (isinstance(api_output, dict) and len(api_output.get("data", api_output)) == 0)))

        if has_error or is_empty:
            data_quality["completeness"] *= 0.5
            if has_error:
                data_quality["issues"].append(f"API returned error: {api_output.get('error', 'Unknown')}")
            else:
                data_quality["issues"].append("API returned empty results")
            empty_sources.append(TOOL_METRICS_API)

            # FALLBACK: If API failed/empty and we haven't tried DB yet, suggest DB
            if TOOL_SQL_DATABASE not in tools_executed:
                fallback_tools_needed.append(TOOL_SQL_DATABASE)
                add_trace_event(state, "aggregate_results", "fallback_suggestion",
                               {"failed_tool": TOOL_METRICS_API, "suggested": TOOL_SQL_DATABASE})
        else:
            aggregated["metrics"]["api"] = api_output

    # Aggregate SQL database data
    if TOOL_SQL_DATABASE in tools_executed:
        db_output = tool_outputs.get(TOOL_SQL_DATABASE, {})
        has_error = "error" in db_output
        is_empty = (not has_error and
                   (isinstance(db_output, dict) and
                    (db_output.get("row_count", 0) == 0 or len(db_output.get("data", [])) == 0)))

        if has_error or is_empty:
            data_quality["completeness"] *= 0.5
            if has_error:
                data_quality["issues"].append(f"Database query error: {db_output.get('error', 'Unknown')}")
            else:
                data_quality["issues"].append("Database returned no matching records")
            empty_sources.append(TOOL_SQL_DATABASE)

            # FALLBACK: If DB failed/empty and we haven't tried API yet, suggest API
            if TOOL_METRICS_API not in tools_executed:
                fallback_tools_needed.append(TOOL_METRICS_API)
                add_trace_event(state, "aggregate_results", "fallback_suggestion",
                               {"failed_tool": TOOL_SQL_DATABASE, "suggested": TOOL_METRICS_API})
        else:
            aggregated["metrics"]["database"] = db_output

    # Aggregate knowledge base results
    if TOOL_KNOWLEDGE_RAG in tools_executed:
        rag_output = tool_outputs.get(TOOL_KNOWLEDGE_RAG, {})
        has_error = "error" in rag_output
        is_empty = (not has_error and
                   (not rag_output.get("results") or len(rag_output.get("results", [])) == 0))

        if has_error or is_empty:
            data_quality["completeness"] *= 0.5
            if has_error:
                data_quality["issues"].append(f"Knowledge base error: {rag_output.get('error', 'Unknown')}")
            else:
                data_quality["issues"].append("No relevant documentation found")
            empty_sources.append(TOOL_KNOWLEDGE_RAG)

            # Note: Knowledge base has no fallback - if docs don't exist, we'll use LLM knowledge
            add_trace_event(state, "aggregate_results", "knowledge_empty",
                           {"reason": "No matching documents - will use LLM general knowledge"})
        else:
            aggregated["knowledge"] = rag_output["results"]

    # Aggregate calculations
    if TOOL_CALCULATOR in tools_executed:
        calc_output = tool_outputs.get(TOOL_CALCULATOR, {})
        if "result" in calc_output:
            aggregated["calculations"] = calc_output
        else:
            data_quality["completeness"] *= 0.9
            data_quality["issues"].append("Calculation failed")

    # Collect all errors
    aggregated["errors"] = list(state.get("tool_errors", {}).values())

    # Check for data consistency if multiple sources
    if TOOL_METRICS_API in tools_executed and TOOL_SQL_DATABASE in tools_executed:
        # Both sources available - could check for consistency
        # For now, just note that we have redundant data
        data_quality["consistency"] = 1.0

    state["aggregated_data"] = aggregated
    state["data_quality"] = data_quality

    # INTELLIGENT FALLBACK: Store suggested tools for retry mechanism
    if fallback_tools_needed:
        state["fallback_tools_suggested"] = fallback_tools_needed
        state["empty_sources"] = empty_sources
        add_trace_event(state, "aggregate_results", "fallback_triggered",
                       {
                           "empty_sources": empty_sources,
                           "suggested_fallbacks": fallback_tools_needed
                       })

    # Update confidence based on data quality
    if data_quality["completeness"] < 0.8:
        update_confidence(state, 0.9, "Some data sources unavailable")

    # If ALL sources returned empty/error, trigger low confidence for retry
    if empty_sources and len(empty_sources) == len(tools_executed):
        update_confidence(state, 0.4, "All selected tools returned empty or error results")
        state["all_sources_empty"] = True

    add_trace_event(
        state,
        "aggregate_results",
        "aggregation_complete",
        {
            "metrics_sources": len(aggregated["metrics"]),
            "knowledge_results": len(aggregated["knowledge"]),
            "data_quality": data_quality
        }
    )

    duration_ms = (time.time() - start_time) * 1000
    state["node_durations"]["aggregate_results"] = duration_ms

    add_trace_event(state, "aggregate_results", "node_end", {"duration_ms": duration_ms})

    return state


# ============================================================================
# NODE 5: Perform Inference
# ============================================================================

def perform_inference(state: AgentState) -> AgentState:
    """
    Analyze aggregated data and make inferences.

    Performs analysis based on query intent:
    - Threshold checks (is metric above/below threshold?)
    - Comparisons (which service is better?)
    - Trend analysis (is performance improving/degrading?)
    - Recommendations (what actions to take?)

    Updates:
        - state["inference_result"]
        - state["inference_type"]
        - state["findings"]
        - state["recommendations"]
        - state["trace"]
    """
    start_time = time.time()
    aggregated = state["aggregated_data"]
    query = state["query"]
    intent = state["intent"]

    add_trace_event(state, "perform_inference", "node_start", {"intent": intent})

    findings = []
    recommendations = []
    inference_type = INFERENCE_NONE
    inference_result = {}

    # Perform inference based on intent and available data
    if intent == INTENT_METRICS_LOOKUP:
        # Check for threshold violations
        metrics_data = aggregated.get("metrics", {})

        if "api" in metrics_data:
            api_data = metrics_data["api"]

            # Latency threshold check
            if "metrics" in api_data and "p95" in api_data["metrics"]:
                p95 = api_data["metrics"]["p95"]
                if p95 > 100:
                    findings.append(f"High P95 latency detected: {p95}ms (threshold: 100ms)")
                    recommendations.append("Investigate slow endpoints and consider caching")
                    inference_type = INFERENCE_THRESHOLD
                else:
                    findings.append(f"P95 latency is healthy: {p95}ms")

            # Error rate check
            if "error_rate" in api_data:
                error_rate = api_data["error_rate"]
                if error_rate > 0.05:
                    findings.append(f"High error rate: {error_rate*100:.2f}% (threshold: 5%)")
                    recommendations.append("Review error logs and implement error handling")
                    inference_type = INFERENCE_THRESHOLD
                else:
                    findings.append(f"Error rate is acceptable: {error_rate*100:.2f}%")

            # Health status
            if "status" in api_data:
                status = api_data["status"]
                if status != "healthy":
                    findings.append(f"Service health: {status}")
                    recommendations.append("Check service health dependencies")

        # Database comparison
        if "database" in metrics_data:
            db_data = metrics_data["database"]
            if "data" in db_data and len(db_data["data"]) > 0:
                # Analyze trends if we have time-series data
                findings.append(f"Historical data available: {db_data['row_count']} records")
                inference_type = INFERENCE_TREND

        inference_result = {
            "thresholds_checked": True,
            "violations": [f for f in findings if "High" in f or "error" in f.lower()]
        }

    elif intent == INTENT_KNOWLEDGE_LOOKUP:
        # Summarize knowledge base results
        knowledge = aggregated.get("knowledge", [])
        if knowledge:
            findings.append(f"Found {len(knowledge)} relevant documentation sections")
            top_result = knowledge[0]
            findings.append(f"Most relevant: {top_result['filename']} (score: {top_result['score']:.3f})")
            inference_type = INFERENCE_AGGREGATION
        else:
            findings.append("No relevant documentation found")
            recommendations.append("Try rephrasing your question or check if documentation exists")

        inference_result = {
            "sources_found": len(knowledge),
            "top_score": knowledge[0]["score"] if knowledge else 0.0
        }

    elif intent == INTENT_CALCULATION:
        # Present calculation results
        calc = aggregated.get("calculations", {})
        if "result" in calc:
            findings.append(f"Calculation result: {calc['result']}")
            inference_type = INFERENCE_AGGREGATION
        else:
            findings.append("Calculation could not be performed")

        inference_result = calc

    elif intent == INTENT_MIXED:
        # Combine findings from multiple sources
        findings.append("Analyzed multiple data sources")
        inference_type = INFERENCE_AGGREGATION

        if aggregated.get("metrics"):
            findings.append("Metrics data available")
        if aggregated.get("knowledge"):
            findings.append(f"Found {len(aggregated['knowledge'])} documentation references")

        inference_result = {
            "sources": list(aggregated.keys()),
            "multi_source": True
        }

    # If no findings, add default
    if not findings:
        findings.append("Analysis complete - no issues detected")

    state["findings"] = findings
    state["recommendations"] = recommendations
    state["inference_type"] = inference_type
    state["inference_result"] = inference_result

    add_trace_event(
        state,
        "perform_inference",
        "inference_complete",
        {
            "inference_type": inference_type,
            "findings_count": len(findings),
            "recommendations_count": len(recommendations)
        }
    )

    duration_ms = (time.time() - start_time) * 1000
    state["node_durations"]["perform_inference"] = duration_ms

    add_trace_event(state, "perform_inference", "node_end", {"duration_ms": duration_ms})

    return state


# ============================================================================
# NODE 6: Check Feedback
# ============================================================================

def check_feedback(state: AgentState) -> AgentState:
    """
    Evaluate confidence and decide if retry/clarification is needed.

    Checks:
    - Overall confidence score
    - Data completeness
    - Tool failures
    - Retry count (max retries = 2)

    Decisions:
    - confidence >= 0.8: Proceed to response
    - confidence 0.6-0.8: Proceed with caveats
    - confidence < 0.6: Retry with different tools OR ask for clarification

    Updates:
        - state["feedback_needed"]
        - state["retry_reason"]
        - state["retry_count"]
        - state["clarification_question"]
        - state["trace"]
    """
    start_time = time.time()
    confidence = state.get("confidence", 0.0)
    retry_count = state.get("retry_count", 0)
    data_quality = state.get("data_quality", {})
    tool_errors = state.get("tool_errors", {})

    add_trace_event(
        state,
        "check_feedback",
        "node_start",
        {
            "confidence": confidence,
            "retry_count": retry_count
        }
    )

    feedback_needed = False
    retry_reason = None
    clarification_question = None

    # Check if we've exceeded max retries
    if retry_count >= MAX_RETRIES:
        feedback_needed = False
        add_trace_event(
            state,
            "check_feedback",
            "max_retries_reached",
            {"retry_count": retry_count}
        )

    # High confidence - BUT check if tools were actually executed
    elif confidence >= CONFIDENCE_HIGH:
        # CRITICAL CHECK: If no tools were executed (e.g., clarification intent), lower confidence
        tools_executed = state.get("tools_executed", [])
        intent = state.get("intent")

        if not tools_executed and intent == INTENT_CLARIFICATION:
            # Clarification intent with no tools - should have triggered clarification, not proceeded
            feedback_needed = False  # Don't retry, but acknowledge the issue
            state["confidence"] = 0.5  # Lower confidence to medium
            add_trace_event(
                state,
                "check_feedback",
                "clarification_without_user_interaction",
                {"note": "Clarification intent detected but proceeding anyway - should ask user for clarification"}
            )
        else:
            feedback_needed = False
            add_trace_event(
                state,
                "check_feedback",
                "high_confidence",
                {"confidence": confidence}
            )

    # Medium confidence - proceed with note
    elif confidence >= CONFIDENCE_MEDIUM:
        feedback_needed = False
        add_trace_event(
            state,
            "check_feedback",
            "medium_confidence",
            {"confidence": confidence, "note": "Proceeding with some uncertainty"}
        )

    # Low confidence - need feedback
    else:
        # Analyze why confidence is low and apply intelligent fallback

        # Check if fallback tools are suggested (from aggregate_results)
        fallback_tools = state.get("fallback_tools_suggested", [])
        all_sources_empty = state.get("all_sources_empty", False)

        if fallback_tools and retry_count < MAX_RETRIES:
            # INTELLIGENT FALLBACK: Retry with suggested alternative tools
            feedback_needed = True
            retry_reason = "empty_results_fallback"
            state["retry_count"] = retry_count + 1

            # Override tools_to_use with fallback suggestions
            state["tools_to_use"] = fallback_tools
            state["tool_selection_reasoning"] = f"Auto-fallback: Previous tools returned empty, trying: {', '.join(fallback_tools)}"

            add_trace_event(
                state,
                "check_feedback",
                "retry_with_fallback_tools",
                {
                    "original_tools": state.get("tools_executed", []),
                    "fallback_tools": fallback_tools,
                    "reason": "Previous tools returned empty/error results"
                }
            )

        elif all_sources_empty and retry_count < MAX_RETRIES:
            # ALL sources empty - try a different approach or answer from LLM knowledge
            feedback_needed = False  # Don't retry, but flag for LLM-only answer
            state["answer_from_llm_knowledge"] = True

            add_trace_event(
                state,
                "check_feedback",
                "all_sources_empty",
                {"decision": "Will answer from LLM general knowledge with explicit disclaimer"}
            )

        elif tool_errors:
            # Tools failed - maybe try alternative approach
            if retry_count < MAX_RETRIES:
                feedback_needed = True
                retry_reason = "tool_failures"
                state["retry_count"] = retry_count + 1

                add_trace_event(
                    state,
                    "check_feedback",
                    "retry_due_to_tool_failures",
                    {"failed_tools": list(tool_errors.keys())}
                )
            else:
                # Max retries reached, proceed with what we have
                feedback_needed = False

        elif data_quality.get("completeness", 1.0) < 0.7:
            # Missing data
            if retry_count < MAX_RETRIES:
                feedback_needed = True
                retry_reason = "incomplete_data"
                state["retry_count"] = retry_count + 1

                add_trace_event(
                    state,
                    "check_feedback",
                    "retry_due_to_incomplete_data",
                    {"completeness": data_quality["completeness"]}
                )
            else:
                feedback_needed = False

        elif state.get("intent") == INTENT_UNKNOWN:
            # Unclear intent - ask for clarification
            clarification_question = "I'm not sure I understand. Are you asking about:\n1. Service metrics (performance, errors, latency)\n2. Documentation or how-to guides\n3. Calculations or comparisons"
            feedback_needed = True
            retry_reason = "unclear_intent"

            add_trace_event(
                state,
                "check_feedback",
                "clarification_needed",
                {"reason": "unclear intent"}
            )

        else:
            # Low confidence but not sure why - proceed anyway
            feedback_needed = False
            add_trace_event(
                state,
                "check_feedback",
                "low_confidence_proceeding",
                {"confidence": confidence}
            )

    state["feedback_needed"] = feedback_needed
    state["retry_reason"] = retry_reason
    state["clarification_question"] = clarification_question

    duration_ms = (time.time() - start_time) * 1000
    state["node_durations"]["check_feedback"] = duration_ms

    add_trace_event(
        state,
        "check_feedback",
        "node_end",
        {
            "duration_ms": duration_ms,
            "feedback_needed": feedback_needed,
            "retry_reason": retry_reason
        }
    )

    return state


# ============================================================================
# HELPER: Synthesize Knowledge Answer
# ============================================================================

def _synthesize_knowledge_answer(query: str, knowledge: List[Dict[str, Any]]) -> str:
    """
    Use LLM to synthesize a coherent answer from RAG chunks.

    Args:
        query: User's original question
        knowledge: List of RAG results with filename, content, score

    Returns:
        Synthesized answer with source citations
    """
    try:
        # Build context from knowledge chunks
        context_parts = []
        for i, doc in enumerate(knowledge[:3], 1):
            context_parts.append(f"Source {i}: {doc['filename']}")
            context_parts.append(f"{doc['content']}\n")

        context = "\n".join(context_parts)

        # Use OpenAI to synthesize answer
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant that answers questions using provided documentation and your expertise.

TASK:
1. Read the documentation excerpts provided
2. Answer the user's question using the documentation as PRIMARY source
3. If documentation is incomplete, supplement with general best practices (clearly labeled)
4. Quote relevant parts with source attribution
5. Be concise but complete

GUIDELINES (in priority order):
1. PRIMARY SOURCE: Always prioritize information from the provided documentation
2. SUPPLEMENTARY INFO: If docs are incomplete, add general best practices with clear disclaimer
3. SOURCE ATTRIBUTION: Always cite documentation sources (e.g., "According to api_guide.md...")
4. LABEL SUPPLEMENTS: Clearly mark supplementary info as "General best practice (not in docs):"
5. HONESTY: If you don't know something, say so - don't make up specifics

OUTPUT EXAMPLES (study the format):

Example 1 - Complete Answer:
Question: "How do I deploy the service?"
Answer:
According to **deployment.md**, the deployment process involves three main steps:

1. **Build the Docker image**: Run `docker build -t myservice:latest .`
2. **Push to registry**: Use `docker push myservice:latest` to upload the image
3. **Deploy to Kubernetes**: Apply the deployment with `kubectl apply -f deployment.yaml`

The documentation also notes that you should verify the deployment status with `kubectl get pods` to ensure all pods are running.

**Source**: deployment.md

Example 2 - Answer with Supplementary Information:
Question: "How do I improve database performance?"
Answer:
**From documentation** (architecture.md):
- Use connection pooling with appropriate limits
- Implement caching layers to reduce database load

**General best practices** (supplementary):
- Add indexes on frequently queried columns
- Monitor and optimize slow queries using query logs
- Consider read replicas for high-traffic applications
- Use database connection pooling (already mentioned in docs)

**Note**: The supplementary practices are industry-standard recommendations not covered in the provided documentation.

**Sources**: architecture.md (primary), general best practices (supplementary)

Example 3 - Multiple Sources:
Question: "What causes high latency?"
Answer:
Based on the documentation, high latency can be caused by several factors:

**Common Causes** (from troubleshooting.md):
- Database connection pool exhaustion
- Inefficient query patterns
- Network congestion

**Recommended Solutions** (from architecture.md):
- Implement caching layers to reduce database load
- Use connection pooling with appropriate limits
- Add CDN for static assets

**Sources**: troubleshooting.md, architecture.md

Format your answer in markdown with proper citations like these examples."""
                },
                {
                    "role": "user",
                    "content": f"""Question: {query}

Documentation:
{context}

Please provide a comprehensive answer based on the documentation above, with proper citations."""
                }
            ],
            temperature=0.3,
            max_tokens=800
        )

        return response.choices[0].message.content

    except Exception as e:
        # Fallback to simple formatting if LLM fails
        fallback = "Based on the documentation:\n\n"
        for i, doc in enumerate(knowledge[:3], 1):
            fallback += f"**From {doc['filename']}** (relevance: {doc['score']:.2f}):\n"
            fallback += f"{doc['content'][:300]}...\n\n"
        return fallback


# ============================================================================
# HELPER: Generate LLM Fallback Answer
# ============================================================================

def _generate_llm_fallback_answer(query: str) -> str:
    """
    Generate answer from LLM general knowledge when no documentation is found.

    This is used as a last resort when:
    - Knowledge base returns no results
    - All data sources are empty
    - User still needs an answer

    Args:
        query: User's original question

    Returns:
        Answer from LLM's general knowledge with clear disclaimers
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant providing general guidance on software engineering and system operations.

IMPORTANT CONTEXT:
- The user's question could not be answered from their project documentation
- You are providing general best practices and industry knowledge
- BE HONEST about what you don't know - don't make up specific details

GUIDELINES:
1. Provide helpful general guidance based on industry best practices
2. Use phrases like "Generally...", "Common approaches include...", "Best practices suggest..."
3. DO NOT make up specific commands, configurations, or project details
4. If the question requires project-specific knowledge, say so explicitly
5. Keep answer concise (3-5 bullet points max)

FORMAT:
- Start with context acknowledgment
- Provide 3-5 general recommendations
- End with suggestion to check project documentation"""
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nProvide general guidance since no project documentation was found."
                }
            ],
            temperature=0.5,
            max_tokens=400
        )

        return response.choices[0].message.content

    except Exception as e:
        return (f"I apologize, but I couldn't find relevant documentation for your question, "
               f"and I'm unable to provide general guidance at this time.\n\n"
               f"**Suggestion**: Please check if documentation exists for this topic in your project.")


# ============================================================================
# NODE 7: Format Response
# ============================================================================

def format_response(state: AgentState) -> AgentState:
    """
    Create final formatted answer with trace.

    Combines:
    - Findings from inference
    - Recommendations
    - Data from tools
    - Execution trace
    - Confidence level

    Updates:
        - state["final_answer"]
        - state["answer_format"]
        - state["end_time"]
        - state["total_duration_ms"]
        - state["trace"]
    """
    start_time = time.time()

    add_trace_event(state, "format_response", "node_start", {})

    # CRITICAL: Check if we need to ask for clarification
    clarification_question = state.get("clarification_question")
    if clarification_question:
        # Return the clarification question directly as the answer
        state["final_answer"] = clarification_question
        state["answer_format"] = "markdown"
        state["end_time"] = datetime.now().isoformat()
        start_time_dt = datetime.fromisoformat(state["start_time"])
        end_time_dt = datetime.fromisoformat(state["end_time"])
        total_duration = (end_time_dt - start_time_dt).total_seconds() * 1000
        state["total_duration_ms"] = total_duration

        duration_ms = (time.time() - start_time) * 1000
        state["node_durations"]["format_response"] = duration_ms

        add_trace_event(state, "format_response", "clarification_returned",
                       {"clarification": clarification_question})
        add_trace_event(state, "format_response", "node_end", {"duration_ms": duration_ms})

        return state

    # Build answer sections
    answer_parts = []

    # Check if answering from general knowledge due to no data
    answer_from_llm = state.get("answer_from_llm_knowledge", False)
    all_sources_empty = state.get("all_sources_empty", False)
    empty_sources = state.get("empty_sources", [])

    # 1. Data source disclaimer (if applicable)
    if all_sources_empty and not answer_from_llm:
        answer_parts.append("⚠️ **Data Source Status**: All queried sources returned no results\n")
        if empty_sources:
            answer_parts.append(f"_Empty sources: {', '.join(empty_sources)}_\n")
        answer_parts.append("")

    # 2. Main findings
    findings = state.get("findings", [])
    if findings:
        answer_parts.append("## Findings\n")
        for i, finding in enumerate(findings, 1):
            answer_parts.append(f"{i}. {finding}")
        answer_parts.append("")

    # 2. Recommendations (if any)
    recommendations = state.get("recommendations", [])
    if recommendations:
        answer_parts.append("## Recommendations\n")
        for i, rec in enumerate(recommendations, 1):
            answer_parts.append(f"{i}. {rec}")
        answer_parts.append("")

    # 3. Synthesized answer for knowledge lookup
    if state.get("intent") == INTENT_KNOWLEDGE_LOOKUP:
        knowledge = state.get("aggregated_data", {}).get("knowledge", [])
        answer_from_llm = state.get("answer_from_llm_knowledge", False)

        if knowledge:
            # Generate synthesized answer using LLM with documentation
            synthesized = _synthesize_knowledge_answer(state["query"], knowledge)
            if synthesized:
                answer_parts.append("## Answer\n")
                answer_parts.append(synthesized)
                answer_parts.append("")
        elif answer_from_llm:
            # NO DOCS FOUND - Answer from LLM general knowledge with explicit disclaimer
            answer_parts.append("## Answer\n")
            answer_parts.append("⚠️ **Note: No relevant documentation found in knowledge base**\n")
            answer_parts.append("_The following answer is based on general best practices and AI knowledge, "
                              "NOT from your project's documentation._\n\n")

            # Generate LLM-only answer
            llm_answer = _generate_llm_fallback_answer(state["query"])
            answer_parts.append(llm_answer)
            answer_parts.append("\n**Source**: General AI knowledge (not from project documentation)\n")

    # 4. Confidence note
    confidence = state.get("confidence", 0.0)
    if confidence < CONFIDENCE_MEDIUM:
        answer_parts.append(f"_Note: This answer has lower confidence ({confidence:.2f}). Some data may be incomplete._\n")

    # 5. Trace summary
    answer_parts.append("## Execution Trace\n")

    trace_events = state.get("trace", [])
    intent = state.get("intent", "unknown")
    tools_executed = state.get("tools_executed", [])

    answer_parts.append(f"- **Intent classified**: {intent}")
    answer_parts.append(f"- **Tools used**: {', '.join(tools_executed) if tools_executed else 'none'}")

    node_durations = state.get("node_durations", {})
    if node_durations:
        total_node_time = sum(node_durations.values())
        answer_parts.append(f"- **Processing time**: {total_node_time:.0f}ms")

    answer_parts.append(f"- **Confidence**: {confidence:.2f}")
    answer_parts.append(f"- **Trace events**: {len(trace_events)}")

    # Combine all parts
    final_answer = "\n".join(answer_parts)

    state["final_answer"] = final_answer
    state["answer_format"] = "markdown"

    # Set end time and calculate total duration
    state["end_time"] = datetime.now().isoformat()
    start_time_dt = datetime.fromisoformat(state["start_time"])
    end_time_dt = datetime.fromisoformat(state["end_time"])
    total_duration = (end_time_dt - start_time_dt).total_seconds() * 1000
    state["total_duration_ms"] = total_duration

    duration_ms = (time.time() - start_time) * 1000
    state["node_durations"]["format_response"] = duration_ms

    add_trace_event(
        state,
        "format_response",
        "node_end",
        {
            "duration_ms": duration_ms,
            "answer_length": len(final_answer)
        }
    )

    return state
