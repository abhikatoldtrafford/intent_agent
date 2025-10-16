"""
Agent Tools

Provides 4 tools for the agent to use:
1. REST API Tool - Query metrics API (HTTP)
2. Knowledge RAG Tool - Search documentation (direct Python)
3. SQL Database Tool - Query metrics database (direct Python)
4. Calculator Tool - Perform calculations (utility)
"""

import os
import requests
import re
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from langchain_core.tools import tool
from services.rag_service import RAGService
from services.db_service import DatabaseService

# Load environment variables (override existing)
load_dotenv(override=True)

# Configuration
API_BASE_URL = os.getenv("METRICS_API_URL", "http://127.0.0.1:8001")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize services for direct Python calls
_rag_service = None
_db_service = None


def get_rag_service() -> RAGService:
    """Lazy initialization of RAG service."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(
            docs_path="data/docs",
            embeddings_path="data/embeddings",
            openai_api_key=OPENAI_API_KEY,
            use_semantic_chunking=True
        )
        # Load from cache or initialize
        _rag_service.initialize(force_rebuild=False)
    return _rag_service


def get_db_service() -> DatabaseService:
    """Lazy initialization of DB service."""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService(
            db_path="data/metrics.db",
            openai_api_key=OPENAI_API_KEY
        )
    return _db_service


# ============================================================================
# TOOL 1: REST API Tool
# ============================================================================

@tool
def query_metrics_api(
    metric_type: str,
    service: Optional[str] = None,
    period: str = "1h",
    **kwargs
) -> Dict[str, Any]:
    """
    Query the local metrics REST API for real-time metrics data.

    This tool makes HTTP calls to the FastAPI service running on localhost:8001.
    Use this for getting current/recent metrics that aren't in the historical database.

    Args:
        metric_type: Type of metric to query. Options:
            - "latency": Get latency metrics (p50, p95, p99)
            - "throughput": Get throughput/RPS data
            - "errors": Get error rates and breakdown
            - "health": Get service health status
            - "services": List all available services
        service: Service name (required for latency, throughput, errors, health)
        period: Time period for metrics (1h, 6h, 24h, 7d). Default: 1h
        **kwargs: Additional parameters specific to metric type

    Returns:
        Dictionary containing the API response with metrics data

    Examples:
        query_metrics_api("latency", service="api-gateway", period="1h")
        query_metrics_api("throughput", service="auth-service", period="6h")
        query_metrics_api("errors", service="business-logic", period="24h")
        query_metrics_api("health", service="data-processor")
        query_metrics_api("services")
    """
    try:
        if metric_type == "latency":
            if not service:
                return {"error": "Service name required for latency metrics"}

            response = requests.get(
                f"{API_BASE_URL}/metrics/latency",
                params={"service": service, "period": period},
                timeout=5
            )
            response.raise_for_status()
            return response.json()

        elif metric_type == "throughput":
            if not service:
                return {"error": "Service name required for throughput metrics"}

            interval = kwargs.get("interval", "5m")
            response = requests.get(
                f"{API_BASE_URL}/metrics/throughput",
                params={"service": service, "period": period, "interval": interval},
                timeout=5
            )
            response.raise_for_status()
            return response.json()

        elif metric_type == "errors":
            if not service:
                return {"error": "Service name required for error metrics"}

            response = requests.get(
                f"{API_BASE_URL}/metrics/errors",
                params={"service": service, "period": period},
                timeout=5
            )
            response.raise_for_status()
            return response.json()

        elif metric_type == "health":
            service = service or "api-gateway"
            response = requests.get(
                f"{API_BASE_URL}/health",
                params={"service": service},
                timeout=5
            )
            response.raise_for_status()
            return response.json()

        elif metric_type == "services":
            response = requests.get(
                f"{API_BASE_URL}/services",
                timeout=5
            )
            response.raise_for_status()
            return response.json()

        else:
            return {
                "error": f"Unknown metric type: {metric_type}",
                "valid_types": ["latency", "throughput", "errors", "health", "services"]
            }

    except requests.exceptions.ConnectionError:
        return {
            "error": "Cannot connect to metrics API",
            "message": "Make sure the API server is running: python start_api_server.py",
            "expected_url": API_BASE_URL
        }
    except requests.exceptions.Timeout:
        return {
            "error": "API request timed out",
            "metric_type": metric_type,
            "service": service
        }
    except requests.exceptions.HTTPError as e:
        return {
            "error": f"HTTP error: {e.response.status_code}",
            "message": e.response.text
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {type(e).__name__}",
            "message": str(e)
        }


# ============================================================================
# TOOL 2: Knowledge RAG Tool
# ============================================================================

@tool
def search_knowledge_base(
    query: str,
    top_k: int = 3,
    search_mode: str = "hybrid"
) -> Dict[str, Any]:
    """
    Search documentation using vector similarity and keyword matching.

    This tool searches through local documentation using RAG (Retrieval Augmented Generation).
    It uses OpenAI embeddings and FAISS for semantic search, plus BM25 for keyword matching.

    Args:
        query: The question or search query
        top_k: Number of results to return (default: 3, max: 10)
        search_mode: Search strategy. Options:
            - "hybrid": Combined vector + BM25 search (recommended)
            - "vector": Semantic vector search only
            - "bm25": Keyword search only

    Returns:
        Dictionary containing:
            - results: List of relevant document chunks
            - query: Original query
            - search_mode: Mode used
            - total_results: Number of results returned

    Examples:
        search_knowledge_base("How do I configure API rate limiting?")
        search_knowledge_base("deployment strategies", top_k=5)
        search_knowledge_base("error handling best practices", search_mode="vector")
    """
    try:
        rag = get_rag_service()

        # Validate parameters
        top_k = min(max(1, top_k), 10)  # Clamp between 1-10

        # Perform search based on mode
        if search_mode == "hybrid":
            results = rag.search_hybrid(query, top_k=top_k)
        elif search_mode == "vector":
            results = rag.search_vector(query, top_k=top_k)
        elif search_mode == "bm25":
            results = rag.search_bm25(query, top_k=top_k)
        else:
            return {
                "error": f"Invalid search mode: {search_mode}",
                "valid_modes": ["hybrid", "vector", "bm25"]
            }

        # Format results for agent consumption
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.content,
                "filename": result.metadata.filename,
                "score": result.score,
                "chunk_id": result.metadata.chunk_id,
                "section": result.metadata.doc_section or "N/A"
            })

        return {
            "results": formatted_results,
            "query": query,
            "search_mode": search_mode,
            "total_results": len(formatted_results)
        }

    except FileNotFoundError:
        return {
            "error": "Documentation not found",
            "message": "RAG service needs to be initialized. Run: python demo_rag.py"
        }
    except Exception as e:
        return {
            "error": f"RAG search failed: {type(e).__name__}",
            "message": str(e)
        }


# ============================================================================
# TOOL 3: SQL Database Tool
# ============================================================================

@tool
def query_sql_database(question: str) -> Dict[str, Any]:
    """
    Query the local SQL database for historical metrics using natural language.

    This tool converts natural language questions into SQL queries and executes them
    against the local SQLite database containing historical service metrics.

    The database contains 840 rows of hourly metrics for 5 services over 7 days:
    - service_name, timestamp, cpu_usage, memory_usage
    - request_count, error_count, avg_latency
    - status, region, instance_id

    Args:
        question: Natural language question about metrics

    Returns:
        Dictionary containing:
            - data: Query results as list of rows
            - columns: Column names
            - row_count: Number of rows returned
            - question: Original question
            - sql_query: Generated SQL (if successful)

    Examples:
        query_sql_database("What is the average CPU usage for api-gateway?")
        query_sql_database("Show me error counts for all services in the last 24 hours")
        query_sql_database("Which service has the highest memory usage?")
        query_sql_database("Compare latency between api-gateway and auth-service")
    """
    try:
        db = get_db_service()

        # Use natural language to SQL conversion
        result = db.natural_language_query(question)

        # Convert rows to list of dicts for easier consumption
        data = result.to_list_of_dicts() if result.rows else []

        return {
            "data": data,
            "columns": result.columns,
            "row_count": result.row_count,
            "question": question,
            "sql_query": result.query,
            "success": True
        }

    except FileNotFoundError:
        return {
            "error": "Database not found",
            "message": "Run the database service setup first"
        }
    except Exception as e:
        return {
            "error": f"Database query failed: {type(e).__name__}",
            "message": str(e),
            "question": question
        }


# ============================================================================
# TOOL 4: Calculator Tool
# ============================================================================

@tool
def calculate(expression: str) -> Dict[str, Any]:
    """
    Perform mathematical calculations and comparisons.

    This tool safely evaluates mathematical expressions including:
    - Basic arithmetic: +, -, *, /, **, %
    - Comparisons: <, >, <=, >=, ==, !=
    - Functions: abs, min, max, round
    - Common calculations: averages, percentages, ratios

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Dictionary containing:
            - result: Calculated value
            - expression: Original expression
            - type: Result type (number, boolean)

    Examples:
        calculate("(150 + 200) / 2")
        calculate("95.5 > 100")
        calculate("(500 - 450) / 450 * 100")  # Percentage change
        calculate("max(45.2, 67.8, 23.1)")

    Security:
        Uses safe evaluation - only allows mathematical operations, no code execution.
    """
    try:
        # Clean the expression
        expression = expression.strip()

        # Define safe namespace for eval
        safe_dict = {
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
            "sum": sum,
            "len": len,
            "__builtins__": {}
        }

        # Validate expression - only allow safe characters and functions
        # Pattern allows: numbers, operators, parentheses, comparisons, and safe functions
        allowed_pattern = r'^[\d\s+\-*/%().,<>=!]+$|^(abs|min|max|round|sum|len)\([\d\s+\-*/%().,<>=!]+\)$'

        # Check if expression contains function calls
        has_function = any(func in expression for func in ['abs', 'min', 'max', 'round', 'sum', 'len'])

        if has_function:
            # More lenient validation for function calls
            if not re.match(r'^[a-z]+\([\d\s+\-*/%().,<>=!]+\)$', expression):
                return {
                    "error": "Invalid function syntax",
                    "expression": expression,
                    "allowed": "Functions: abs(x), min(x,y,...), max(x,y,...), round(x), sum([x,y,...]), len([x,y,...])"
                }
        else:
            # Strict validation for non-function expressions
            if not re.match(r'^[\d\s+\-*/%().,<>=!]+$', expression):
                return {
                    "error": "Invalid characters in expression",
                    "expression": expression,
                    "allowed": "Numbers, operators (+,-,*,/,**,%,<,>,<=,>=,==,!=), and parentheses"
                }

        # Check for dangerous patterns
        dangerous = ["import", "exec", "eval", "compile", "__", "open", "file"]
        if any(d in expression.lower() for d in dangerous):
            return {
                "error": "Potentially unsafe expression",
                "expression": expression
            }

        # Evaluate
        result = eval(expression, safe_dict, {})

        # Determine type
        result_type = "boolean" if isinstance(result, bool) else "number"

        return {
            "result": result,
            "expression": expression,
            "type": result_type,
            "success": True
        }

    except ZeroDivisionError:
        return {
            "error": "Division by zero",
            "expression": expression
        }
    except SyntaxError:
        return {
            "error": "Invalid syntax",
            "expression": expression,
            "message": "Check parentheses and operators"
        }
    except Exception as e:
        return {
            "error": f"Calculation failed: {type(e).__name__}",
            "expression": expression,
            "message": str(e)
        }


# ============================================================================
# Tool Registry
# ============================================================================

ALL_TOOLS = [
    query_metrics_api,
    search_knowledge_base,
    query_sql_database,
    calculate
]

TOOL_DESCRIPTIONS = {
    "query_metrics_api": "Get real-time metrics from the REST API (latency, throughput, errors, health)",
    "search_knowledge_base": "Search documentation and knowledge base using semantic search",
    "query_sql_database": "Query historical metrics from the SQL database using natural language",
    "calculate": "Perform mathematical calculations and comparisons"
}


def get_tool_by_name(name: str):
    """Get a tool by its name."""
    for tool in ALL_TOOLS:
        if tool.name == name:
            return tool
    return None


def list_available_tools() -> List[Dict[str, str]]:
    """List all available tools with descriptions."""
    return [
        {
            "name": tool.name,
            "description": TOOL_DESCRIPTIONS.get(tool.name, tool.description)
        }
        for tool in ALL_TOOLS
    ]
