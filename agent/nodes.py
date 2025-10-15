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

    add_trace_event(state, "classify_intent", "node_start", {"query": query})

    try:
        # Use OpenAI to classify intent
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an intent classifier for a metrics monitoring agent.

Your task: Analyze user queries and classify them into the correct intent category.

INTENT DEFINITIONS:
- metrics_lookup: Questions about service metrics, performance, latency, errors, throughput, health status
- knowledge_lookup: Questions about documentation, how-to guides, best practices, configuration
- calculation: Requests for calculations, comparisons, percentages, mathematical operations
- mixed: Requires both metrics AND knowledge/calculation
- clarification: Query is too vague or ambiguous
- unknown: Cannot determine intent

GUARDRAILS:
1. ONLY respond with valid JSON - no additional text, no markdown
2. Intent MUST be exactly one of: metrics_lookup, knowledge_lookup, calculation, mixed, clarification, unknown
3. Confidence MUST be a number between 0.0 and 1.0
4. Reasoning MUST be a concise string (max 50 words)

OUTPUT EXAMPLES (study these carefully):

Example 1 - Metrics Lookup:
Query: "What is the current latency for api-gateway?"
Response:
{
    "intent": "metrics_lookup",
    "confidence": 0.95,
    "reasoning": "Clear request for specific metric (latency) about a named service (api-gateway)."
}

Example 2 - Knowledge Lookup:
Query: "How do I troubleshoot high latency issues?"
Response:
{
    "intent": "knowledge_lookup",
    "confidence": 0.90,
    "reasoning": "Asking for procedural knowledge about troubleshooting, not requesting actual metric data."
}

Example 3 - Calculation:
Query: "Calculate the average of 150, 200, and 250"
Response:
{
    "intent": "calculation",
    "confidence": 1.0,
    "reasoning": "Explicit calculation request with numerical values provided."
}

Example 4 - Mixed Intent:
Query: "What's the latency for api-gateway and how can I reduce it?"
Response:
{
    "intent": "mixed",
    "confidence": 0.92,
    "reasoning": "Requires both metrics lookup (current latency) and knowledge (how to reduce it)."
}

Example 5 - Clarification Needed:
Query: "stuff about things"
Response:
{
    "intent": "clarification",
    "confidence": 0.3,
    "reasoning": "Query is too vague to determine what information or action is needed."
}

YOUR RESPONSE MUST BE VALID JSON ONLY - no extra text."""
                },
                {
                    "role": "user",
                    "content": f"Classify this query: {query}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )

        result = json.loads(response.choices[0].message.content)

        intent = result.get("intent", INTENT_UNKNOWN)
        intent_confidence = result.get("confidence", 0.5)
        reasoning = result.get("reasoning", "")

        state["intent"] = intent
        state["intent_confidence"] = intent_confidence
        state["confidence"] = intent_confidence

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
        query_lower = query.lower()

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
    query = state["query"]

    add_trace_event(state, "select_tools", "node_start", {"intent": intent})

    tools_to_use = []
    reasoning = ""

    # Intent-based tool mapping
    if intent == INTENT_METRICS_LOOKUP:
        # Determine if we need real-time (API) or historical (DB) data
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["current", "now", "real-time", "live", "latest"]):
            tools_to_use = [TOOL_METRICS_API]
            reasoning = "Query asks for current/real-time metrics - using REST API"
        elif any(kw in query_lower for kw in ["history", "historical", "past", "trend", "over time"]):
            tools_to_use = [TOOL_SQL_DATABASE]
            reasoning = "Query asks for historical data - using SQL database"
        else:
            # Use both for comprehensive answer
            tools_to_use = [TOOL_METRICS_API, TOOL_SQL_DATABASE]
            reasoning = "Using both API (current) and database (historical) for complete metrics"

    elif intent == INTENT_KNOWLEDGE_LOOKUP:
        tools_to_use = [TOOL_KNOWLEDGE_RAG]
        reasoning = "Documentation/knowledge question - using RAG search"

    elif intent == INTENT_CALCULATION:
        # Check if we need to fetch data first
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["metric", "latency", "error", "throughput"]):
            tools_to_use = [TOOL_METRICS_API, TOOL_CALCULATOR]
            reasoning = "Calculation requires metrics data - fetching from API then calculating"
        else:
            tools_to_use = [TOOL_CALCULATOR]
            reasoning = "Direct calculation - using calculator only"

    elif intent == INTENT_MIXED:
        # Use OpenAI to determine which tools
        tools_to_use = [TOOL_METRICS_API, TOOL_KNOWLEDGE_RAG, TOOL_SQL_DATABASE]
        reasoning = "Complex query requiring multiple tool types"

    elif intent == INTENT_CLARIFICATION:
        tools_to_use = []
        reasoning = "Query needs clarification before tool selection"
        state["feedback_needed"] = True
        state["clarification_question"] = "Could you please provide more details about what you're looking for?"

    else:  # INTENT_UNKNOWN
        # Try a broad approach
        tools_to_use = [TOOL_KNOWLEDGE_RAG]
        reasoning = "Intent unclear - starting with knowledge base search"

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
    query = state["query"]

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
    services = ["api-gateway", "auth-service", "business-logic", "data-processor", "cache-service"]
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

    # Aggregate metrics API data
    if TOOL_METRICS_API in tools_executed:
        api_output = tool_outputs.get(TOOL_METRICS_API, {})
        if "error" not in api_output:
            aggregated["metrics"]["api"] = api_output
        else:
            data_quality["completeness"] *= 0.8
            data_quality["issues"].append("API data unavailable")

    # Aggregate SQL database data
    if TOOL_SQL_DATABASE in tools_executed:
        db_output = tool_outputs.get(TOOL_SQL_DATABASE, {})
        if "error" not in db_output:
            aggregated["metrics"]["database"] = db_output
        else:
            data_quality["completeness"] *= 0.8
            data_quality["issues"].append("Database query failed")

    # Aggregate knowledge base results
    if TOOL_KNOWLEDGE_RAG in tools_executed:
        rag_output = tool_outputs.get(TOOL_KNOWLEDGE_RAG, {})
        if "results" in rag_output:
            aggregated["knowledge"] = rag_output["results"]
        else:
            data_quality["completeness"] *= 0.9
            data_quality["issues"].append("No knowledge base results")

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

    # Update confidence based on data quality
    if data_quality["completeness"] < 0.8:
        update_confidence(state, 0.9, "Some data sources unavailable")

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

    # High confidence - proceed
    elif confidence >= CONFIDENCE_HIGH:
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
        # Analyze why confidence is low
        if tool_errors:
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
                    "content": """You are a helpful assistant that answers questions using provided documentation.

TASK:
1. Read the documentation excerpts provided
2. Answer the user's question based on ONLY the provided content
3. Quote relevant parts with source attribution
4. Be concise but complete
5. If the documentation doesn't fully answer the question, say so

GUARDRAILS:
- ONLY use information from the provided documentation - NO outside knowledge
- Do NOT make up information or fill gaps with assumptions
- If unsure or information is missing, explicitly state this limitation
- ALWAYS cite sources for all claims (e.g., "According to api_guide.md...")
- Keep answer focused and relevant to the question

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

Example 2 - Partial Answer with Limitation:
Question: "What's the database backup schedule?"
Answer:
According to **monitoring.md**, the system includes automated health checks and alerting, but specific information about database backup schedules is not covered in the provided documentation.

I can confirm that monitoring is set up for database health, but for backup schedule details, you may need to consult the database administration guide.

**Source**: monitoring.md (partial information)

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

    # Build answer sections
    answer_parts = []

    # 1. Main findings
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
        if knowledge:
            # Generate synthesized answer using LLM
            synthesized = _synthesize_knowledge_answer(state["query"], knowledge)
            if synthesized:
                answer_parts.append("## Answer\n")
                answer_parts.append(synthesized)
                answer_parts.append("")

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
