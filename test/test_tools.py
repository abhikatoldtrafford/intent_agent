"""
Test Suite for Agent Tools

Tests all 4 agent tools:
1. query_metrics_api - REST API tool
2. search_knowledge_base - RAG tool
3. query_sql_database - SQL tool
4. calculate - Calculator tool

Prerequisites:
- API server running on localhost:8001
- RAG service initialized (run demo_rag.py)
- Database created (run validate_db_service.py)
- OPENAI_API_KEY environment variable set
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.tools import (
    query_metrics_api,
    search_knowledge_base,
    query_sql_database,
    calculate,
    list_available_tools
)


def log(message: str, level: str = "INFO"):
    """Enhanced logging with timestamps and levels for Streamlit parsing."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    symbols = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "PROGRESS": "🔄",
        "TEST": "🧪"
    }
    symbol = symbols.get(level, "•")
    print(f"[{timestamp}] {symbol} {level}: {message}", flush=True)


def print_header(title):
    """Print test section header."""
    print("\n" + "="*80, flush=True)
    print(title, flush=True)
    print("="*80, flush=True)
    log(f"Starting test section: {title}", "TEST")


def print_check(message, status=True):
    """Print test result with enhanced logging."""
    symbol = "✓" if status else "✗"
    level = "SUCCESS" if status else "ERROR"
    log(f"{symbol} {message}", level)


def test_calculator():
    """Test calculator tool."""
    print_header("TEST 1: CALCULATOR TOOL")

    log(f"Testing calculator with 6 expressions", "INFO")

    tests = [
        ("(100 + 200) / 2", 150),
        ("95.5 > 100", False),
        ("(500 - 450) / 450 * 100", 11.11),
        ("max(45.2, 67.8, 23.1)", 67.8),
        ("min(10, 20, 30)", 10),
        ("abs(-42)", 42),
    ]

    all_passed = True

    for idx, (expression, expected) in enumerate(tests, 1):
        log(f"Test {idx}/{len(tests)}: Evaluating '{expression}'", "PROGRESS")
        result = calculate.invoke({"expression": expression})

        if "error" in result:
            print_check(f"{expression} - ERROR: {result['error']}", False)
            all_passed = False
        else:
            actual = result["result"]
            # For percentage, allow small difference
            if isinstance(expected, float):
                passed = abs(actual - expected) < 0.1
            else:
                passed = actual == expected

            print_check(f"{expression} = {actual} (expected: {expected})", passed)
            if not passed:
                all_passed = False

    # Test error handling
    log("Testing security and error handling", "PROGRESS")
    result = calculate.invoke({"expression": "import os"})
    print_check("Security test: 'import os' rejected", "error" in result)

    result = calculate.invoke({"expression": "1 / 0"})
    print_check("Error handling: division by zero", "error" in result)

    log(f"Calculator test complete - {'PASSED' if all_passed else 'FAILED'}", "SUCCESS" if all_passed else "ERROR")
    return all_passed


def test_knowledge_base():
    """Test knowledge RAG tool."""
    print_header("TEST 2: KNOWLEDGE BASE TOOL")

    log("Testing knowledge base RAG tool", "INFO")

    # Check if RAG is initialized
    log("Checking RAG prerequisites", "PROGRESS")
    if not os.path.exists("data/embeddings"):
        log("RAG service not initialized", "ERROR")
        print_check("RAG service not initialized - run demo_rag.py first", False)
        return False

    log("RAG service is initialized", "SUCCESS")

    tests = [
        ("How do I configure API rate limiting?", "configuration"),
        ("deployment strategies", "deployment"),
        ("monitoring metrics", "monitoring"),
    ]

    log(f"Testing {len(tests)} knowledge queries", "INFO")

    all_passed = True

    for idx, (query, expected_topic) in enumerate(tests, 1):
        log(f"Query {idx}/{len(tests)}: '{query[:50]}...'", "PROGRESS")
        result = search_knowledge_base.invoke({
            "query": query,
            "top_k": 3
        })

        if "error" in result:
            print_check(f"Query: '{query}' - ERROR: {result['error']}", False)
            all_passed = False
        else:
            results_count = result.get("total_results", 0)
            passed = results_count > 0

            if passed and result.get("results"):
                top_result = result["results"][0]
                print_check(
                    f"Query: '{query}' - Found {results_count} results "
                    f"(top: {top_result['filename']}, score: {top_result['score']:.3f})",
                    True
                )
            else:
                print_check(f"Query: '{query}' - No results found", False)
                all_passed = False

    # Test different search modes
    log("Testing search modes: hybrid, vector, bm25", "PROGRESS")
    for mode in ["hybrid", "vector", "bm25"]:
        result = search_knowledge_base.invoke({
            "query": "API configuration",
            "search_mode": mode
        })
        passed = "error" not in result and result.get("total_results", 0) > 0
        print_check(f"Search mode '{mode}' working", passed)

    log(f"Knowledge base test complete - {'PASSED' if all_passed else 'FAILED'}", "SUCCESS" if all_passed else "ERROR")
    return all_passed


def test_sql_database():
    """Test SQL database tool."""
    print_header("TEST 3: SQL DATABASE TOOL")

    log("Testing SQL database tool with NL-to-SQL", "INFO")

    # Check if database exists
    log("Checking database prerequisites", "PROGRESS")
    if not os.path.exists("data/metrics.db"):
        log("Database not found", "ERROR")
        print_check("Database not found - run validate_db_service.py first", False)
        return False

    log("Database exists", "SUCCESS")

    # Check if OpenAI API key is set (needed for NL-to-SQL)
    if not os.getenv("OPENAI_API_KEY"):
        log("OPENAI_API_KEY not set", "ERROR")
        print_check("OPENAI_API_KEY not set - cannot test NL-to-SQL", False)
        return False

    log("OpenAI API key configured", "SUCCESS")

    tests = [
        "What is the average CPU usage for api-gateway?",
        "Show me error counts for all services",
        "Which service has the highest memory usage?",
        "What is the status distribution across all services?",
    ]

    log(f"Testing {len(tests)} natural language queries", "INFO")

    all_passed = True

    for idx, question in enumerate(tests, 1):
        log(f"Query {idx}/{len(tests)}: '{question[:50]}...'", "PROGRESS")
        result = query_sql_database.invoke({"question": question})

        if "error" in result:
            print_check(f"Question: '{question[:50]}...' - ERROR: {result['error']}", False)
            all_passed = False
        else:
            row_count = result.get("row_count", 0)
            passed = row_count > 0

            if passed:
                print_check(
                    f"Question: '{question[:50]}...' - "
                    f"Returned {row_count} rows",
                    True
                )
            else:
                print_check(f"Question: '{question[:50]}...' - No data returned", False)
                all_passed = False

    log(f"SQL database test complete - {'PASSED' if all_passed else 'FAILED'}", "SUCCESS" if all_passed else "ERROR")
    return all_passed


def test_metrics_api():
    """Test metrics API tool."""
    print_header("TEST 4: METRICS API TOOL")

    log("Testing metrics REST API tool", "INFO")

    # Try to connect to API
    log("Checking API server availability", "PROGRESS")
    import requests
    try:
        response = requests.get("http://127.0.0.1:8001/", timeout=2)
        api_running = response.status_code == 200
    except:
        api_running = False

    if not api_running:
        log("API server not running", "ERROR")
        print_check("API server not running - start with: python start_api_server.py", False)
        return False

    log("API server is running", "SUCCESS")

    tests = [
        ("latency", "api-gateway", "1h"),
        ("throughput", "auth-service", "1h"),
        ("errors", "business-logic", "24h"),
        ("health", "data-processor", None),
        ("services", None, None),
    ]

    log(f"Testing {len(tests)} API endpoints", "INFO")

    all_passed = True

    for idx, (metric_type, service, period) in enumerate(tests, 1):
        log(f"API call {idx}/{len(tests)}: {metric_type}/{service or 'N/A'}", "PROGRESS")
        params = {"metric_type": metric_type}
        if service:
            params["service"] = service
        if period:
            params["period"] = period

        result = query_metrics_api.invoke(params)

        if "error" in result:
            print_check(
                f"API call: {metric_type}/{service or 'N/A'} - "
                f"ERROR: {result['error']}",
                False
            )
            all_passed = False
        else:
            print_check(
                f"API call: {metric_type}/{service or 'N/A'} - Success",
                True
            )

    log(f"Metrics API test complete - {'PASSED' if all_passed else 'FAILED'}", "SUCCESS" if all_passed else "ERROR")
    return all_passed


def test_tool_registry():
    """Test tool registry functions."""
    print_header("TEST 5: TOOL REGISTRY")

    log("Testing tool registry functions", "INFO")

    # List all tools
    log("Retrieving list of available tools", "PROGRESS")
    tools = list_available_tools()
    print_check(f"Found {len(tools)} tools", len(tools) == 4)

    log(f"Tool registry contains {len(tools)} tools", "INFO")
    for tool in tools:
        print(f"    - {tool['name']}: {tool['description']}", flush=True)

    result = len(tools) == 4
    log(f"Tool registry test complete - {'PASSED' if result else 'FAILED'}", "SUCCESS" if result else "ERROR")
    return result


def run_all_tests():
    """Run all tool tests."""
    print("="*80, flush=True)
    print("AGENT TOOLS TEST SUITE", flush=True)
    print("="*80, flush=True)

    log("Starting agent tools test suite", "TEST")
    log(f"Test run initiated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "INFO")

    results = {}

    # Run tests in order
    log("Running 5 tool test suites", "INFO")

    log("Test 1/5: Calculator Tool", "TEST")
    results["Calculator Tool"] = test_calculator()

    log("Test 2/5: Knowledge Base Tool", "TEST")
    results["Knowledge Base Tool"] = test_knowledge_base()

    log("Test 3/5: SQL Database Tool", "TEST")
    results["SQL Database Tool"] = test_sql_database()

    log("Test 4/5: Metrics API Tool", "TEST")
    results["Metrics API Tool"] = test_metrics_api()

    log("Test 5/5: Tool Registry", "TEST")
    results["Tool Registry"] = test_tool_registry()

    # Summary
    print_header("TEST SUMMARY")
    log("Generating test summary", "INFO")

    all_passed = True
    passed_count = 0
    failed_count = 0

    for test_name, passed in results.items():
        print_check(test_name, passed)
        if passed:
            passed_count += 1
        else:
            failed_count += 1
            all_passed = False

    log(f"Test results: {passed_count} passed, {failed_count} failed", "INFO")

    print("\n" + "="*80, flush=True)

    if all_passed:
        log("ALL TESTS PASSED - Agent tools are working correctly!", "SUCCESS")
        print("✓ ALL TESTS PASSED - Agent tools are working correctly!", flush=True)
        print("="*80, flush=True)
        return True
    else:
        log(f"SOME TESTS FAILED - {failed_count} test(s) failed", "ERROR")
        print("✗ SOME TESTS FAILED - Please review errors above", flush=True)
        print("="*80, flush=True)
        print("\nTroubleshooting:")
        print("1. Ensure API server is running: python start_api_server.py")
        print("2. Initialize RAG service: python demo_rag.py")
        print("3. Create database: python tests/validate_db_service.py")
        print("4. Set OPENAI_API_KEY environment variable")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
