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
# GUARDRAILS: Intent Classification Validation
# ============================================================================

# Monitoring-related keywords that indicate IN-DOMAIN queries
MONITORING_KEYWORDS = [
    'latency', 'error', 'metric', 'health', 'status', 'performance',
    'cpu', 'memory', 'request', 'response', 'throughput', 'availability',
    'service', 'api', 'database', 'cache', 'queue', 'server', 'gateway',
    'auth', 'payment', 'business', 'processor', 'log', 'trace', 'monitor',
    'p50', 'p95', 'p99', 'rps', '4xx', '5xx', 'uptime', 'downtime',
    'degraded', 'unhealthy', 'healthy', 'trend', 'spike', 'anomaly'
]

# Clarification patterns (vague but in-domain)
CLARIFICATION_PATTERNS = [
    'what is the', 'show me', 'how is', 'is the', 'check', 'tell me about',
    'what about', 'give me', 'display', 'any', 'some'
]


def normalize_intent(raw_intent: str) -> str:
    """
    Normalize intent string to match official intent constants using fuzzy matching.

    This handles variations that LLMs might return:
    - "metrics" or "metrics_lookup" → INTENT_METRICS_LOOKUP
    - "knowledge" or "knowledge_lookup" → INTENT_KNOWLEDGE_LOOKUP
    - "calculation" or "calculate" or "calc" → INTENT_CALCULATION
    - "clarify" or "clarification" or "clarify_intent" → INTENT_CLARIFICATION
    - "mixed" or "multi" or "multiple" → INTENT_MIXED
    - "unknown" or "unclear" or "off_topic" → INTENT_UNKNOWN

    Args:
        raw_intent: Intent string from LLM (may be variation)

    Returns:
        Normalized intent matching one of the official constants
    """
    # Normalize to lowercase and remove common separators
    normalized = raw_intent.lower().strip().replace('-', '_').replace(' ', '_')

    # METRICS_LOOKUP variations
    if any(keyword in normalized for keyword in ['metric', 'metrics']):
        return INTENT_METRICS_LOOKUP

    # KNOWLEDGE_LOOKUP variations
    if any(keyword in normalized for keyword in ['knowledge', 'doc', 'documentation', 'rag']):
        return INTENT_KNOWLEDGE_LOOKUP

    # CALCULATION variations
    if any(keyword in normalized for keyword in ['calc', 'calculation', 'compute', 'math']):
        return INTENT_CALCULATION

    # CLARIFICATION variations (need more info from user)
    if any(keyword in normalized for keyword in ['clarif', 'clarify', 'clarification', 'need_clarif']):
        return INTENT_CLARIFICATION

    # MIXED variations
    if any(keyword in normalized for keyword in ['mixed', 'multi', 'multiple', 'hybrid']):
        return INTENT_MIXED

    # UNKNOWN variations (out of domain / unclear intent)
    if any(keyword in normalized for keyword in ['unknown', 'unclear', 'off_topic', 'unrelated', 'out_of_domain']):
        return INTENT_UNKNOWN

    # If no match, try exact match with constants
    if normalized == 'metrics_lookup':
        return INTENT_METRICS_LOOKUP
    elif normalized == 'knowledge_lookup':
        return INTENT_KNOWLEDGE_LOOKUP
    elif normalized == 'calculation':
        return INTENT_CALCULATION
    elif normalized == 'clarify':
        return INTENT_CLARIFICATION
    elif normalized == 'mixed':
        return INTENT_MIXED
    elif normalized == 'unknown':
        return INTENT_UNKNOWN

    # Default to unknown if no match
    return INTENT_UNKNOWN


def apply_intent_guardrails(query: str, intent: str, confidence: float, reasoning: str) -> tuple:
    """
    Apply lightweight guardrails as a safety net for intent classification.

    The LLM prompt now contains comprehensive guidance about distribution boundaries,
    so these guardrails are just a backup to catch obvious errors.

    Simple checks:
    1. Has monitoring keywords + classified as unknown → Suggest clarify
    2. No monitoring keywords + classified as clarify → Suggest unknown

    Args:
        query: User's query (original case)
        intent: Classified intent
        confidence: Classification confidence (0-1)
        reasoning: Original classification reasoning

    Returns:
        (corrected_intent, correction_reason or None)
    """
    query_lower = query.lower().strip()

    # Check for monitoring keywords (basic check)
    matched_keywords = [kw for kw in MONITORING_KEYWORDS if kw in query_lower]
    has_monitoring_keywords = len(matched_keywords) > 0

    # Safety check 1: Has keywords but classified as unknown → Should probably be clarify
    if has_monitoring_keywords and intent == INTENT_UNKNOWN:
        return (
            INTENT_CLARIFICATION,
            f"GUARDRAIL: Query contains monitoring keywords {matched_keywords[:3]} "
            f"but was classified as unknown. Corrected to clarify (in-domain but vague)."
        )

    # Safety check 2: No keywords but classified as clarify → Should probably be unknown
    if not has_monitoring_keywords and intent == INTENT_CLARIFICATION:
        return (
            INTENT_UNKNOWN,
            f"GUARDRAIL: Query has no monitoring keywords but was classified as clarify. "
            f"Corrected to unknown (out-of-distribution)."
        )

    # No correction needed - trust the LLM's classification
    return (intent, None)


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
                    "content": """You are an expert intent classifier for a metrics monitoring agent.

YOUR TASK: Classify user queries into the correct intent based on what data sources are needed.

═══════════════════════════════════════════════════════════════════════
🎯 CRITICAL CLASSIFICATION PRINCIPLE (READ THIS FIRST!)
═══════════════════════════════════════════════════════════════════════

**MAKE REASONABLE DEFAULTS - DON'T BE PEDANTIC!**

The agent is designed to be HELPFUL, not RIGID. When minor details are missing:
✅ Make reasonable assumptions (default service, default time range)
❌ Don't ask for clarification unless CRITICAL information is missing

**When to DEFAULT vs CLARIFY:**
- Missing service name + has metric → DEFAULT to first service (api-gateway)
- Missing time range + has metric/service → DEFAULT to "recent" or "current"
- Missing metric type + has service → DEFAULT to common metrics (latency, errors)
- NO monitoring keywords at all → CLARIFY or UNKNOWN

═══════════════════════════════════════════════════════════════════════
📊 DATA SOURCES - What You Can Query
═══════════════════════════════════════════════════════════════════════

**1. REST API** (Real-time metrics, last 7 days)
   Services Available:
   • api-gateway
   • auth-service
   • business-logic
   • data-processor
   • payment-service

   Metrics Available:
   • Latency: p50, p95, p99 (milliseconds)
   • Throughput: requests/sec, total requests
   • Errors: 4xx, 5xx counts, error rate
   • Health: healthy/degraded/unhealthy status

   Use For: "current", "now", "latest", "real-time", "is service healthy"

**2. SQL DATABASE** (Historical data, 7 days hourly)
   Same 5 Services × 168 hours = 840 data points

   Metrics Available (historical only):
   • CPU usage % over time
   • Memory usage % over time
   • Request counts per hour
   • Error counts per hour
   • Latency averages per hour
   • Status history

   Use For: "average", "trend", "over time", "last week", "compare", "historical", "CPU", "memory"

   ⚠️ CRITICAL: CPU and memory are ONLY in SQL database, NOT in REST API

**3. KNOWLEDGE BASE** (Documentation)
   Files: architecture.md, api_guide.md, troubleshooting.md, deployment.md, monitoring.md

   Contains:
   • How-to guides (setup, configuration)
   • Troubleshooting steps
   • Best practices
   • Architecture explanations

   Use For: "how to", "how do I", "explain", "why", "configure", "best practice"

**4. CALCULATOR** (Math operations)
   Operations: +, -, *, /, %, <, >, ==
   Functions: abs, min, max, round, sum

   Use For: "calculate", "compute", "average of [numbers]", "percentage"

═══════════════════════════════════════════════════════════════════════
📋 INTENT TYPES - Complete Definitions
═══════════════════════════════════════════════════════════════════════

**1. metrics_lookup** - Fetching metric data (80% of queries)
   Definition: Query asks for ACTUAL METRIC VALUES or STATUS
   Data Sources: REST API (current) OR SQL database (historical)
   Keywords: "what is", "show", "current", "latency", "errors", "CPU", "memory", "healthy"

   Includes:
   • Current metrics: "What's the latency for api-gateway?"
   • Historical metrics: "Show CPU usage last week"
   • Comparisons: "Compare memory between services"
   • Status checks: "Is auth-service healthy?"

**2. knowledge_lookup** - Documentation/how-to queries
   Definition: Query asks HOW TO do something or needs EXPLANATION
   Data Source: Knowledge base (documentation files)
   Keywords: "how to", "how do I", "explain", "why", "configure", "best practice", "guide"

   Includes:
   • Procedures: "How do I deploy?"
   • Explanations: "Why is latency high?"
   • Best practices: "What's the recommended setup?"

**3. calculation** - Math operations
   Definition: Query asks to COMPUTE something with explicit numbers
   Data Source: Calculator tool
   Keywords: "calculate", "compute", "average of", "sum", "percentage"

   Includes:
   • Direct math: "Calculate 100 + 200"
   • Averages: "Average of 50, 75, 100"

**4. mixed** - Multiple data sources needed
   Definition: Query explicitly needs BOTH metrics AND knowledge (or metrics AND calculations)
   Data Sources: Multiple tools required
   Keywords: "and", "also", "both", "as well as"

   Includes:
   • "Show latency AND how to improve it" (metrics + knowledge)
   • "Get error rate and explain causes" (metrics + knowledge)

**5. clarify** - In-domain but vague (RARE - use sparingly!)
   Definition: Has monitoring keywords BUT query is nonsensical or completely ambiguous
   Trigger: ONLY if you genuinely cannot make any reasonable assumption

   ⚠️ USE RARELY! Only when query is truly incomprehensible:
   • "Show me the service thing" - what thing?
   • "What about the stuff?" - what stuff?

   ✅ DON'T USE for minor missing details - make defaults instead!

**6. unknown** - Out-of-distribution (greetings, off-topic)
   Definition: ZERO monitoring/system keywords - completely different domain
   Trigger: NO connection to monitoring/metrics/services at all

   Examples:
   • Greetings: "Hello", "Hi there"
   • Weather: "What's the weather?"
   • General: "Tell me a joke"

═══════════════════════════════════════════════════════════════════════
🚨 WHEN TO USE CLARIFY vs UNKNOWN (Critical Decision!)
═══════════════════════════════════════════════════════════════════════

This is where most mistakes happen. Follow this decision tree:

**DECISION TREE:**

1. Does query have ANY monitoring/service keywords?
   • YES → Continue to step 2 (could be metrics_lookup, clarify, or other in-domain intent)
   • NO → UNKNOWN (off-topic)

2. Can you make reasonable assumptions about missing details?
   • YES → Use metrics_lookup with defaults (be helpful!)
   • NO → Only then consider clarify

**Monitoring Keywords (if ANY present → NOT unknown):**
latency, error, metric, health, status, CPU, memory, throughput, request, response,
service, api, gateway, auth, payment, processor, monitor, availability, uptime

**CLARIFY vs metrics_lookup Decision:**

Query: "Show me latency"
• Has keyword: ✅ "latency"
• Missing: service name
• Decision: metrics_lookup (default to api-gateway)
• Reasoning: Can make reasonable default

Query: "Compare memory between api-gateway and auth-service"
• Has keywords: ✅ "memory", "api-gateway", "auth-service"
• Missing: time range
• Decision: metrics_lookup (default to "recent" or "last 24 hours")
• Reasoning: Services specified, time range can default

Query: "Show me the thing"
• Has keyword: ❌ "thing" is too vague
• Missing: everything
• Decision: unknown (no monitoring context)
• Reasoning: Cannot infer what user wants

Query: "What about that service issue?"
• Has keywords: ✅ "service", "issue"
• Missing: which service, what issue
• Decision: clarify (monitoring context but too ambiguous)
• Reasoning: In-domain but genuinely unclear what user wants

═══════════════════════════════════════════════════════════════════════
🎓 5 COMPREHENSIVE CLASSIFICATION EXAMPLES (Study These!)
═══════════════════════════════════════════════════════════════════════

These examples demonstrate the complete decision-making process. MEMORIZE THESE PATTERNS!

**EXAMPLE 1: Memory comparison query (User's actual failing query!)**
Query: "Compare memory usage between api-gateway and auth-service"

Analysis:
• Keywords present: ✅ "memory", "api-gateway", "auth-service" (3 monitoring keywords!)
• Missing details: time range (when? current? historical?)
• Service names: ✅ BOTH specified explicitly
• Metric type: ✅ "memory" clearly stated
• Can we make defaults? ✅ YES - default to "recent" or "last 24 hours"

Decision:
{
    "intent": "metrics_lookup",
    "confidence": 0.93,
    "reasoning": "Query asks to compare memory usage between two specific services. Both service names provided. Missing time range can default to 'recent' or 'last 24 hours'. Use SQL database for historical memory data.",
    "missing_params": []
}

Why metrics_lookup and NOT clarify? We have enough information! Both services specified, metric is clear (memory), we can assume a reasonable time frame.

**EXAMPLE 2: Vague query with confusing wording**
Query: "memory usage for latency service"

Analysis:
• Keywords present: ✅ "memory", "latency" (monitoring keywords)
• Service names: ❌ "latency service" doesn't exist - "latency" is a METRIC, not a service name
• Confusion: User seems confused about terminology
• Can we make defaults? ⚠️ MAYBE - could default to api-gateway, but "latency service" is nonsensical

Decision:
{
    "intent": "metrics_lookup",
    "confidence": 0.75,
    "reasoning": "Query asks for memory usage but 'latency service' is not a valid service name. Defaulting to api-gateway for service. Use SQL database for memory metrics.",
    "missing_params": []
}

Why metrics_lookup? Has monitoring keywords, clear intent to get memory metric. Even though wording is confusing, we can be helpful by defaulting to a service.

**EXAMPLE 3: Historical trend request**
Query: "Show me services where CPU exceeded 75% in the last 4 days and rank by frequency"

Analysis:
• Keywords present: ✅ "CPU", "exceeded", "last 4 days" (monitoring keywords + time range!)
• Service names: Multiple services (asking for ALL that match criteria)
• Metric type: ✅ "CPU" clearly stated
• Time range: ✅ "last 4 days" explicitly provided
• Aggregation: ✅ "rank by frequency" = COUNT + ORDER BY

Decision:
{
    "intent": "metrics_lookup",
    "confidence": 0.96,
    "reasoning": "Historical threshold query with aggregation. SQL database can filter (WHERE CPU > 75), group by service, count occurrences, and rank. All SQL capabilities - single data source.",
    "missing_params": []
}

Why metrics_lookup and NOT mixed? SQL handles filtering, counting, and ranking - no need for separate calculator tool.

**EXAMPLE 4: Mixed query needing multiple sources**
Query: "What's the current error rate and how do I reduce it?"

Analysis:
• Keywords present: ✅ "error rate", "how do I", "reduce"
• Two-part query: (1) get metric data, (2) get procedural knowledge
• Part 1: "current error rate" → REST API (metrics_lookup)
• Part 2: "how do I reduce it" → Knowledge base (knowledge_lookup)
• Explicitly asks for BOTH data AND guidance

Decision:
{
    "intent": "mixed",
    "confidence": 0.97,
    "reasoning": "Two-part query: (1) Fetch current error rate from REST API, (2) Search troubleshooting documentation for error reduction strategies. Requires both metrics and knowledge sources.",
    "missing_params": []
}

Why mixed? User explicitly asks for BOTH "what's the rate" (metrics) AND "how do I reduce it" (knowledge).

**EXAMPLE 5: Off-topic query (out of distribution)**
Query: "What's the weather like today?"

Analysis:
• Keywords present: ❌ ZERO monitoring/system keywords
• Domain: Weather (meteorology) - completely unrelated to monitoring
• If user provided more details, could it become monitoring query? NO - weather will never be about service metrics
• No amount of clarification makes this about monitoring

Decision:
{
    "intent": "unknown",
    "confidence": 0.98,
    "reasoning": "Weather query - completely unrelated to monitoring/metrics domain. No monitoring keywords present. Will use LLM fallback to provide friendly response.",
    "missing_params": []
}

Why unknown and NOT clarify? NO monitoring keywords at all. This is a different domain entirely (meteorology vs monitoring).

═══════════════════════════════════════════════════════════════════════
📋 JSON OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════

CRITICAL RULES:
1. ONLY respond with valid JSON - no markdown, no explanations outside JSON
2. Intent MUST be exactly one of: metrics_lookup, knowledge_lookup, calculation, mixed, clarify, unknown
3. Confidence MUST be between 0.0 and 1.0
4. Reasoning MUST explain which data source(s) are needed and why
5. For "clarify" intent, ALWAYS include "missing_params" array
6. For all other intents, missing_params should be an empty array []

═══════════════════════════════════════════════════════════════════════
EXACT JSON OUTPUT EXAMPLES (Copy these formats exactly!):
═══════════════════════════════════════════════════════════════════════

Example 1 - Metrics Query:
Query: "What is the latency for api-gateway?"
{
    "intent": "metrics_lookup",
    "confidence": 0.95,
    "reasoning": "Query asks for metric value (latency) with service specified - use REST API",
    "missing_params": []
}

Example 2 - Knowledge Query:
Query: "How do I reduce latency?"
{
    "intent": "knowledge_lookup",
    "confidence": 0.92,
    "reasoning": "Query asks HOW TO (procedural knowledge) - search documentation",
    "missing_params": []
}

Example 3 - Calculation Query:
Query: "Calculate the average of 10, 20, 30"
{
    "intent": "calculation",
    "confidence": 0.98,
    "reasoning": "Direct arithmetic request with explicit numbers - use calculator",
    "missing_params": []
}

Example 4 - Mixed Query:
Query: "What's the error rate for api-gateway and how do I fix it?"
{
    "intent": "mixed",
    "confidence": 0.96,
    "reasoning": "Two-part query: metrics (error rate) + knowledge (how to fix) - use both REST API and RAG",
    "missing_params": []
}

Example 5 - Clarify Intent (VAGUE but IN-DOMAIN):
Query: "What's the latency?"
{
    "intent": "clarify",
    "confidence": 0.85,
    "reasoning": "Has monitoring keyword 'latency' but missing which service - need clarification",
    "missing_params": ["service_name"]
}

Example 6 - Unknown Intent (OUT-OF-DOMAIN):
Query: "Hello! How are you?"
{
    "intent": "unknown",
    "confidence": 0.95,
    "reasoning": "Greeting with no monitoring keywords - completely unrelated to monitoring/metrics domain",
    "missing_params": []
}

IMPORTANT NOTES:
- For "clarify" intent, ALWAYS include "missing_params" array with specific missing fields
- Possible missing_params values: "service_name", "metric_type", "time_range", "aggregation_type"
- For all other intents, missing_params should be an empty array []
- The "intent" field accepts variations: "metrics" = "metrics_lookup", "knowledge" = "knowledge_lookup", "calc" = "calculation"
- But prefer using exact values: metrics_lookup, knowledge_lookup, calculation, mixed, clarify, unknown

YOUR RESPONSE MUST BE VALID JSON ONLY - NO MARKDOWN, NO EXPLANATIONS."""
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

        # Get raw intent and normalize it (handles variations like "metrics", "clarify", "clarification", etc.)
        raw_intent = result.get("intent", INTENT_UNKNOWN)
        intent = normalize_intent(raw_intent)
        intent_confidence = result.get("confidence", 0.5)
        reasoning = result.get("reasoning", "")
        missing_params = result.get("missing_params", [])

        # Log normalization if intent was changed
        if raw_intent != intent:
            add_trace_event(
                state,
                "classify_intent",
                "intent_normalized",
                {
                    "raw_intent": raw_intent,
                    "normalized_intent": intent,
                    "note": f"Intent normalized from '{raw_intent}' to '{intent}' for consistency"
                }
            )

        # APPLY GUARDRAILS to prevent misclassification
        corrected_intent, correction_reason = apply_intent_guardrails(
            current_query, intent, intent_confidence, reasoning
        )

        if correction_reason:
            # Guardrail triggered - log the correction
            add_trace_event(
                state,
                "classify_intent",
                "guardrail_correction",
                {
                    "original_intent": intent,
                    "corrected_intent": corrected_intent,
                    "reason": correction_reason
                }
            )
            intent = corrected_intent
            reasoning = f"{reasoning}\n\n{correction_reason}"

        state["intent"] = intent
        state["intent_confidence"] = intent_confidence
        state["confidence"] = intent_confidence
        state["current_query"] = current_query  # Store extracted query for tool execution

        # Store missing_params for clarify intent (used in tool selection)
        if intent == INTENT_CLARIFICATION and missing_params:
            state["missing_params"] = missing_params

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

        # Build clarification question based on missing_params
        missing_params = state.get("missing_params", [])

        if missing_params:
            # Specific missing parameters identified
            param_questions = []
            if "service_name" in missing_params:
                param_questions.append("- Which **service**? (e.g., api-gateway, auth-service, payment-service)")
            if "metric_type" in missing_params:
                param_questions.append("- Which **metric**? (e.g., latency, errors, CPU, memory, throughput)")
            if "time_range" in missing_params:
                param_questions.append("- What **time range**? (e.g., current, last hour, last 24 hours, last week)")
            if "aggregation_type" in missing_params:
                param_questions.append("- What **aggregation**? (e.g., average, sum, max, count)")

            clarification = (
                f"I understand you're asking about monitoring/metrics, but I need more details:\n\n"
                f"{''.join(param_questions)}\n\n"
                f"**Example**: \"What is the latency for api-gateway in the last 24 hours?\""
            )
        else:
            # Generic clarification (fallback)
            clarification = (
                "Could you please clarify what you're looking for?\n\n"
                "Are you asking about:\n"
                "1. **Current service metrics** (latency, errors, throughput, health)?\n"
                "2. **Historical trends** (CPU usage over time, error patterns, etc.)?\n"
                "3. **Documentation** (how to configure, troubleshoot, or optimize)?\n"
                "4. **Calculations** (comparing numbers, computing averages, etc.)?"
            )

        state["clarification_question"] = clarification

    else:  # INTENT_UNKNOWN
        # Query is off-topic or unrelated to monitoring/metrics - no tools needed
        tools_to_use = []
        reasoning = "Query is unrelated to monitoring/metrics - no tools needed (will provide friendly guidance message)"

        # Flag for format_response to return helpful guidance
        state["off_topic_query"] = True

    state["tools_to_use"] = tools_to_use
    state["tool_selection_reasoning"] = reasoning

    # LOG ORCHESTRATION DECISION (for visibility)
    if "orchestration_log" not in state:
        state["orchestration_log"] = []

    orchestration_decision = {
        "stage": "tool_selection",
        "intent": intent,
        "decision": f"Selected {len(tools_to_use)} tool(s): {', '.join(tools_to_use) if tools_to_use else 'none'}",
        "reasoning": reasoning,
        "retry_iteration": state.get("retry_count", 0),
        "timestamp": datetime.now().isoformat()
    }
    state["orchestration_log"].append(orchestration_decision)

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
                update_confidence(state, 0.7, f"Tool {tool_name} returned error")
            else:
                tools_executed.append(tool_name)
                # Don't inflate confidence - successful tool execution maintains current confidence
                # Only increase slightly if we're already low
                if state.get("confidence", 1.0) < 0.6:
                    update_confidence(state, 1.05, f"Tool {tool_name} executed successfully (confidence boost)")
                # Otherwise just log success without changing confidence

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
        update_confidence(state, 0.85, "Some data sources unavailable")

    if data_quality["completeness"] < 0.5:
        update_confidence(state, 0.7, "Majority of data sources unavailable")

    # If ALL sources returned empty/error, trigger low confidence for retry
    if empty_sources and len(empty_sources) == len(tools_executed):
        update_confidence(state, 0.35, "All selected tools returned empty or error results")
        state["all_sources_empty"] = True

    # Partial results penalty - if some tools returned empty but not all
    elif empty_sources and len(empty_sources) > 0:
        penalty = 0.9 - (0.1 * len(empty_sources))  # Each empty source reduces by 0.1
        update_confidence(state, penalty, f"{len(empty_sources)} tool(s) returned empty results")

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
        # Analyze why confidence is low and determine retry strategy

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

        elif tool_errors and retry_count < MAX_RETRIES:
            # Tools failed - retry with alternative approach
            feedback_needed = True
            retry_reason = "tool_failures"
            state["retry_count"] = retry_count + 1

            add_trace_event(
                state,
                "check_feedback",
                "retry_due_to_tool_failures",
                {"failed_tools": list(tool_errors.keys())}
            )

        elif data_quality.get("completeness", 1.0) < 0.7 and retry_count < MAX_RETRIES:
            # Missing data - retry
            feedback_needed = True
            retry_reason = "incomplete_data"
            state["retry_count"] = retry_count + 1

            add_trace_event(
                state,
                "check_feedback",
                "retry_due_to_incomplete_data",
                {"completeness": data_quality["completeness"]}
            )

        elif state.get("intent") == INTENT_UNKNOWN and retry_count < MAX_RETRIES:
            # Unclear intent - DON'T retry, this is off-topic
            # Just mark as unknown and proceed to LLM fallback
            feedback_needed = False
            state["off_topic_query"] = True

            add_trace_event(
                state,
                "check_feedback",
                "unknown_intent_skip_retry",
                {"reason": "Unknown intent - will use LLM fallback instead of retry"}
            )

        elif confidence < 0.5 and retry_count < MAX_RETRIES:
            # CRITICAL FIX: Very low confidence (<0.5) - trigger retry even without specific failure
            feedback_needed = True
            retry_reason = "low_confidence_general"
            state["retry_count"] = retry_count + 1

            # Try using different tools or adding more tools
            current_tools = state.get("tools_executed", [])
            intent = state.get("intent")

            # Suggest adding complementary tools
            suggested_tools = []
            if intent == INTENT_METRICS_LOOKUP:
                # If only used API, try adding SQL; if only SQL, try adding API
                if TOOL_METRICS_API in current_tools and TOOL_SQL_DATABASE not in current_tools:
                    suggested_tools.append(TOOL_SQL_DATABASE)
                elif TOOL_SQL_DATABASE in current_tools and TOOL_METRICS_API not in current_tools:
                    suggested_tools.append(TOOL_METRICS_API)
                elif not current_tools:
                    # No tools executed at all - try both
                    suggested_tools = [TOOL_METRICS_API, TOOL_SQL_DATABASE]

            if suggested_tools:
                state["tools_to_use"] = suggested_tools
                state["tool_selection_reasoning"] = f"Low confidence retry: trying {', '.join(suggested_tools)}"

            add_trace_event(
                state,
                "check_feedback",
                "retry_due_to_low_confidence",
                {
                    "confidence": confidence,
                    "retry_reason": "General low confidence - trying alternative approach",
                    "suggested_tools": suggested_tools
                }
            )

        else:
            # Either:
            # 1. Confidence is low but retry_count >= MAX_RETRIES (exhausted retries)
            # 2. Confidence is between 0.5-0.6 (acceptable but not great)
            feedback_needed = False

            if retry_count >= MAX_RETRIES:
                add_trace_event(
                    state,
                    "check_feedback",
                    "max_retries_exhausted",
                    {"confidence": confidence, "retries_used": retry_count}
                )
            else:
                add_trace_event(
                    state,
                    "check_feedback",
                    "acceptable_low_confidence",
                    {"confidence": confidence, "note": "Between 0.5-0.6, proceeding with caveats"}
                )

    state["feedback_needed"] = feedback_needed
    state["retry_reason"] = retry_reason

    # CRITICAL: Only set clarification_question if we're creating a NEW one
    # Don't overwrite existing clarification_question from select_tools
    if clarification_question is not None:
        state["clarification_question"] = clarification_question

    # LOG FEEDBACK ITERATION (for visibility into retry decisions)
    if feedback_needed and retry_reason:
        if "feedback_iterations" not in state:
            state["feedback_iterations"] = []

        feedback_iteration = {
            "iteration": retry_count + 1,
            "reason": retry_reason,
            "confidence_at_retry": confidence,
            "fallback_tools": state.get("fallback_tools_suggested", []),
            "timestamp": datetime.now().isoformat()
        }
        state["feedback_iterations"].append(feedback_iteration)

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
    Generate answer from LLM general knowledge when specialized tools can't help.

    This is a FEEDBACK LOOP mechanism used when:
    - Unknown intent (off-topic queries)
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
                    "content": """You are a helpful AI assistant providing general guidance.

IMPORTANT CONTEXT:
- The user's question could not be answered using specialized monitoring/metrics tools
- You are providing general knowledge or best practices
- BE HONEST about what you don't know - don't make up specific details
- If the question is a casual greeting, respond in a friendly but professional way

GUIDELINES:
1. If it's a greeting ("hi", "hello"), respond warmly but briefly, mentioning you're an AI assistant
2. If it's a general question, provide helpful guidance based on general knowledge
3. Use phrases like "Generally...", "Common approaches include...", "From a general perspective..."
4. DO NOT make up specific technical details, commands, or configurations
5. Keep answer concise (3-5 sentences max)

FORMAT:
- For greetings: Warm acknowledgment + brief self-introduction
- For questions: Helpful general guidance
- For off-topic: Polite acknowledgment + gentle redirect"""
                },
                {
                    "role": "user",
                    "content": f"{query}"
                }
            ],
            temperature=0.7,  # Higher temp for more natural responses
            max_tokens=300
        )

        return response.choices[0].message.content

    except Exception as e:
        return (f"I apologize, but I'm unable to provide a response at this time. "
               f"Please try rephrasing your question or ask about monitoring, metrics, or documentation.")


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
        # LOG AS FEEDBACK ITERATION: Clarification is an interactive feedback loop
        if "feedback_iterations" not in state:
            state["feedback_iterations"] = []

        # Count this as a feedback iteration (clarification request)
        feedback_iteration = {
            "iteration": len(state["feedback_iterations"]) + 1,
            "reason": "clarification_required",
            "intent": state.get("intent", "clarify"),
            "confidence_at_feedback": state.get("confidence", 0.5),
            "action": "asking_user_for_clarification",
            "missing_params": state.get("missing_params", []),
            "timestamp": datetime.now().isoformat()
        }
        state["feedback_iterations"].append(feedback_iteration)

        add_trace_event(state, "format_response", "clarification_feedback_logged",
                       {"reason": "In-domain query requires clarification - interactive feedback loop"})

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

    # Check for off-topic queries (unknown intent) - FEEDBACK LOOP: LLM FALLBACK
    if state.get("off_topic_query"):
        # LOG AS FEEDBACK ITERATION: Unknown intent uses LLM fallback (automatic feedback loop)
        if "feedback_iterations" not in state:
            state["feedback_iterations"] = []

        # Count this as a feedback iteration (LLM fallback)
        feedback_iteration = {
            "iteration": len(state["feedback_iterations"]) + 1,
            "reason": "unknown_intent_llm_fallback",
            "intent": state.get("intent", "unknown"),
            "confidence_at_feedback": state.get("confidence", 0.5),
            "action": "using_llm_general_knowledge",
            "tools_used": [],  # No specialized tools - out of domain
            "timestamp": datetime.now().isoformat()
        }
        state["feedback_iterations"].append(feedback_iteration)

        add_trace_event(state, "format_response", "unknown_feedback_logged",
                       {"reason": "Out-of-domain query using LLM fallback - automatic feedback loop"})

        # FEEDBACK LOOP: Tools can't help → Try LLM general knowledge
        query = state.get("current_query", state["query"])

        add_trace_event(state, "format_response", "llm_fallback_triggered",
                       {"reason": "Query unrelated to monitoring/metrics - attempting LLM general knowledge"})

        try:
            # Generate answer from LLM general knowledge
            llm_answer = _generate_llm_fallback_answer(query)

            state["final_answer"] = f"""⚠️ **Note: Query appears outside monitoring/metrics domain**

_Attempting to answer using general AI knowledge (no specialized tools used)_

---

{llm_answer}

---

**💡 For monitoring-specific queries, I can help with:**
- Service Metrics (latency, errors, throughput, health)
- Historical Data (CPU, memory, trends over time)
- Documentation (how-to guides, troubleshooting)
- Calculations (averages, comparisons, percentages)
"""
            state["answer_format"] = "markdown"

            add_trace_event(state, "format_response", "llm_fallback_success",
                           {"answer_length": len(llm_answer)})

        except Exception as e:
            # If LLM fallback fails, provide guidance message
            state["final_answer"] = """I'm a monitoring and metrics agent designed to help with:

1. **Service Metrics** - Current latency, errors, throughput, health status
2. **Historical Data** - CPU usage, memory trends, error patterns over time
3. **Documentation** - How-to guides, troubleshooting procedures, best practices
4. **Calculations** - Computing averages, comparisons, percentages

Your query appears to be outside my area of expertise. Could you rephrase it to focus on one of these areas?

**Example queries:**
- "What is the latency for api-gateway?"
- "How do I reduce error rates?"
- "Show me CPU usage for the last week"
- "Calculate the average of 150, 200, and 250"
"""
            add_trace_event(state, "format_response", "llm_fallback_failed",
                           {"error": str(e)})

        state["end_time"] = datetime.now().isoformat()
        start_time_dt = datetime.fromisoformat(state["start_time"])
        end_time_dt = datetime.fromisoformat(state["end_time"])
        total_duration = (end_time_dt - start_time_dt).total_seconds() * 1000
        state["total_duration_ms"] = total_duration

        duration_ms = (time.time() - start_time) * 1000
        state["node_durations"]["format_response"] = duration_ms

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
