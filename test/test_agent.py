"""
Test Suite for Agent Workflow

Tests the complete LangGraph agent workflow including:
- Intent classification
- Tool selection and execution
- Result aggregation
- Inference
- Feedback loop
- Response formatting

Prerequisites:
- All services running (API, RAG, DB)
- OPENAI_API_KEY environment variable set
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import run_agent, get_graph_visualization
from agent.state import (
    INTENT_METRICS_LOOKUP,
    INTENT_KNOWLEDGE_LOOKUP,
    INTENT_CALCULATION,
    TOOL_METRICS_API,
    TOOL_KNOWLEDGE_RAG,
    TOOL_SQL_DATABASE,
    TOOL_CALCULATOR
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


def print_result_summary(result):
    """Print a summary of the agent result."""
    print("\n  Query:", result.get("query"))
    print("  Intent:", result.get("intent"))
    print("  Confidence:", f"{result.get('confidence', 0):.2f}")
    print("  Tools executed:", result.get("tools_executed", []))
    print("  Total duration:", f"{result.get('total_duration_ms', 0):.0f}ms")
    print("  Findings:", len(result.get("findings", [])))
    print("  Trace events:", len(result.get("trace", [])))


def test_metrics_lookup_intent():
    """Test metrics lookup queries."""
    print_header("TEST 1: METRICS LOOKUP INTENT")

    log(f"Testing metrics lookup with 3 queries", "INFO")

    test_queries = [
        "What is the current latency for api-gateway?",
        "Show me error rates for auth-service",
        "Is api-gateway healthy?",
    ]

    all_passed = True

    for idx, query in enumerate(test_queries, 1):
        log(f"Testing query {idx}/{len(test_queries)}", "PROGRESS")
        try:
            log(f"Executing agent with query: '{query[:60]}...'", "INFO")
            result = run_agent(query, verbose=False)

            # Verify intent classification
            log("Verifying intent classification", "PROGRESS")
            intent_correct = result.get("intent") == INTENT_METRICS_LOOKUP
            print_check(f"Query: '{query[:50]}...'", True)
            print_check(f"  Intent classified as metrics_lookup", intent_correct)

            # Verify tools used
            log("Checking tools executed", "PROGRESS")
            tools_executed = result.get("tools_executed", [])
            has_metrics_tool = TOOL_METRICS_API in tools_executed
            print_check(f"  Used metrics tool", has_metrics_tool)

            # Verify we got an answer
            log("Verifying response generation", "PROGRESS")
            has_answer = len(result.get("final_answer", "")) > 0
            print_check(f"  Generated answer", has_answer)

            # Verify findings
            has_findings = len(result.get("findings", [])) > 0
            print_check(f"  Produced findings", has_findings)

            if not (intent_correct and has_metrics_tool and has_answer):
                log(f"Test query {idx} validation failed", "WARNING")
                all_passed = False
            else:
                log(f"Test query {idx} passed all checks", "SUCCESS")

            print_result_summary(result)

        except Exception as e:
            log(f"Exception during query {idx}: {str(e)}", "ERROR")
            print_check(f"Query: '{query}' - ERROR: {e}", False)
            all_passed = False

    log(f"Metrics lookup test complete - {'PASSED' if all_passed else 'FAILED'}", "SUCCESS" if all_passed else "ERROR")
    return all_passed


def test_knowledge_lookup_intent():
    """Test knowledge lookup queries."""
    print_header("TEST 2: KNOWLEDGE LOOKUP INTENT")

    log(f"Testing knowledge lookup with 3 queries", "INFO")

    test_queries = [
        "How do I configure API rate limiting?",
        "Explain the deployment process",
        "What are monitoring best practices?",
    ]

    all_passed = True

    for idx, query in enumerate(test_queries, 1):
        log(f"Testing query {idx}/{len(test_queries)}", "PROGRESS")
        try:
            log(f"Executing agent with query: '{query[:60]}...'", "INFO")
            result = run_agent(query, verbose=False)

            # Verify intent classification
            log("Verifying intent classification", "PROGRESS")
            intent_correct = result.get("intent") == INTENT_KNOWLEDGE_LOOKUP
            print_check(f"Query: '{query[:50]}...'", True)
            print_check(f"  Intent classified as knowledge_lookup", intent_correct)

            # Verify tools used
            log("Checking tools executed", "PROGRESS")
            tools_executed = result.get("tools_executed", [])
            has_rag_tool = TOOL_KNOWLEDGE_RAG in tools_executed
            print_check(f"  Used knowledge base tool", has_rag_tool)

            # Verify we got an answer
            log("Verifying response generation", "PROGRESS")
            has_answer = len(result.get("final_answer", "")) > 0
            print_check(f"  Generated answer", has_answer)

            # Verify knowledge results
            log("Checking knowledge retrieval", "PROGRESS")
            aggregated = result.get("aggregated_data", {})
            has_knowledge = len(aggregated.get("knowledge", [])) > 0
            print_check(f"  Retrieved knowledge", has_knowledge)

            if not (intent_correct and has_rag_tool and has_answer):
                log(f"Test query {idx} validation failed", "WARNING")
                all_passed = False
            else:
                log(f"Test query {idx} passed all checks", "SUCCESS")

            print_result_summary(result)

        except Exception as e:
            log(f"Exception during query {idx}: {str(e)}", "ERROR")
            print_check(f"Query: '{query}' - ERROR: {e}", False)
            all_passed = False

    log(f"Knowledge lookup test complete - {'PASSED' if all_passed else 'FAILED'}", "SUCCESS" if all_passed else "ERROR")
    return all_passed


def test_calculation_intent():
    """Test calculation queries."""
    print_header("TEST 3: CALCULATION INTENT")

    log(f"Testing calculation intent with 3 queries", "INFO")

    test_queries = [
        "Calculate (150 + 200) / 2",
        "What is 95.5 compared to 100?",
        "Compute the average of 10, 20, and 30",
    ]

    all_passed = True

    for idx, query in enumerate(test_queries, 1):
        log(f"Testing query {idx}/{len(test_queries)}", "PROGRESS")
        try:
            log(f"Executing agent with query: '{query[:60]}...'", "INFO")
            result = run_agent(query, verbose=False)

            # Verify intent classification
            log("Verifying intent classification", "PROGRESS")
            intent = result.get("intent")
            intent_correct = intent == INTENT_CALCULATION
            print_check(f"Query: '{query[:50]}...'", True)
            print_check(f"  Intent classified as calculation", intent_correct)

            # Verify tools used
            log("Checking tools executed", "PROGRESS")
            tools_executed = result.get("tools_executed", [])
            has_calc_tool = TOOL_CALCULATOR in tools_executed
            print_check(f"  Used calculator tool", has_calc_tool)

            # Verify we got an answer
            log("Verifying response generation", "PROGRESS")
            has_answer = len(result.get("final_answer", "")) > 0
            print_check(f"  Generated answer", has_answer)

            if not (has_calc_tool and has_answer):
                log(f"Test query {idx} validation failed", "WARNING")
                all_passed = False
            else:
                log(f"Test query {idx} passed all checks", "SUCCESS")

            print_result_summary(result)

        except Exception as e:
            log(f"Exception during query {idx}: {str(e)}", "ERROR")
            print_check(f"Query: '{query}' - ERROR: {e}", False)
            all_passed = False

    log(f"Calculation test complete - {'PASSED' if all_passed else 'FAILED'}", "SUCCESS" if all_passed else "ERROR")
    return all_passed


def test_mixed_intent():
    """Test queries requiring multiple tools."""
    print_header("TEST 4: MIXED INTENT QUERIES")

    log(f"Testing mixed intent with 2 complex queries", "INFO")

    test_queries = [
        "What is the latency for api-gateway and how does it compare to best practices?",
        "Show me error rates and explain how to reduce them",
    ]

    all_passed = True

    for idx, query in enumerate(test_queries, 1):
        log(f"Testing query {idx}/{len(test_queries)}", "PROGRESS")
        try:
            log(f"Executing agent with query: '{query[:60]}...'", "INFO")
            result = run_agent(query, verbose=False)

            print_check(f"Query: '{query[:50]}...'", True)

            # Verify multiple tools used
            log("Checking multi-tool coordination", "PROGRESS")
            tools_executed = result.get("tools_executed", [])
            multiple_tools = len(tools_executed) >= 2
            print_check(f"  Used multiple tools ({len(tools_executed)})", multiple_tools)

            # Verify we got an answer
            log("Verifying response generation", "PROGRESS")
            has_answer = len(result.get("final_answer", "")) > 0
            print_check(f"  Generated answer", has_answer)

            if not (multiple_tools and has_answer):
                log(f"Test query {idx} validation failed", "WARNING")
                all_passed = False
            else:
                log(f"Test query {idx} passed all checks", "SUCCESS")

            print_result_summary(result)

        except Exception as e:
            log(f"Exception during query {idx}: {str(e)}", "ERROR")
            print_check(f"Query: '{query}' - ERROR: {e}", False)
            all_passed = False

    log(f"Mixed intent test complete - {'PASSED' if all_passed else 'FAILED'}", "SUCCESS" if all_passed else "ERROR")
    return all_passed


def test_workflow_components():
    """Test specific workflow components."""
    print_header("TEST 5: WORKFLOW COMPONENTS")

    log("Testing complete workflow with component validation", "INFO")

    query = "What is the latency for api-gateway?"

    try:
        log(f"Executing agent with query: '{query}'", "INFO")
        result = run_agent(query, verbose=False)

        log("Verifying workflow components", "PROGRESS")

        # Check all expected state fields
        checks = {
            "Intent classification": "intent" in result and result["intent"],
            "Tool selection": "tools_to_use" in result and result["tools_to_use"],
            "Tool execution": "tool_outputs" in result and result["tool_outputs"],
            "Aggregation": "aggregated_data" in result,
            "Inference": "inference_result" in result,
            "Confidence score": "confidence" in result and 0 <= result["confidence"] <= 1,
            "Findings": "findings" in result and len(result["findings"]) > 0,
            "Final answer": "final_answer" in result and len(result["final_answer"]) > 0,
            "Trace": "trace" in result and len(result["trace"]) > 0,
            "Node durations": "node_durations" in result and len(result["node_durations"]) > 0,
            "Session ID": "session_id" in result,
            "Timestamps": "start_time" in result and "end_time" in result,
        }

        all_passed = True
        for component, passed in checks.items():
            print_check(component, passed)
            if not passed:
                all_passed = False

        # Print detailed trace
        log("Workflow trace events:", "INFO")
        print("\n  Trace events:")
        for i, event in enumerate(result.get("trace", [])[:5], 1):
            print(f"    {i}. [{event.get('node')}] {event.get('event_type')}")

        # Print node durations
        log("Node execution times:", "INFO")
        print("\n  Node durations:")
        for node, duration in result.get("node_durations", {}).items():
            print(f"    {node}: {duration:.0f}ms")

        if all_passed:
            log("All workflow components validated successfully", "SUCCESS")
        else:
            log("Some workflow components missing or invalid", "WARNING")

        return all_passed

    except Exception as e:
        log(f"Exception during workflow test: {str(e)}", "ERROR")
        print_check(f"Workflow test - ERROR: {e}", False)
        return False


def test_feedback_loop():
    """Test confidence and feedback loop."""
    print_header("TEST 6: FEEDBACK LOOP")

    log("Testing feedback loop mechanism", "INFO")

    # This is tricky to test directly, but we can verify the mechanism exists
    query = "What is the latency for api-gateway?"

    try:
        log(f"Executing agent with query: '{query}'", "INFO")
        result = run_agent(query, verbose=False)

        log("Verifying feedback loop components", "PROGRESS")

        # Check feedback-related fields
        checks = {
            "Confidence calculated": "confidence" in result,
            "Feedback check performed": "feedback_needed" in result,
            "Retry count tracked": "retry_count" in result,
        }

        all_passed = True
        for check, passed in checks.items():
            print_check(check, passed)
            if not passed:
                all_passed = False

        # Print feedback status
        log("Feedback loop state:", "INFO")
        print(f"\n  Confidence: {result.get('confidence', 0):.2f}")
        print(f"  Feedback needed: {result.get('feedback_needed', False)}")
        print(f"  Retry count: {result.get('retry_count', 0)}")
        print(f"  Retry reason: {result.get('retry_reason', 'N/A')}")

        if all_passed:
            log("Feedback loop mechanism validated", "SUCCESS")
        else:
            log("Feedback loop components missing", "WARNING")

        return all_passed

    except Exception as e:
        log(f"Exception during feedback loop test: {str(e)}", "ERROR")
        print_check(f"Feedback loop test - ERROR: {e}", False)
        return False


def test_error_handling():
    """Test error handling in tools."""
    print_header("TEST 7: ERROR HANDLING")

    log("Testing error handling and graceful degradation", "INFO")

    # Query that will cause some tools to fail
    query = "Show me metrics for non-existent-service"

    try:
        log(f"Executing agent with problematic query: '{query}'", "INFO")
        result = run_agent(query, verbose=False)

        log("Verifying graceful error handling", "PROGRESS")

        # Should still get an answer even with errors
        has_answer = len(result.get("final_answer", "")) > 0
        print_check("Generated answer despite errors", has_answer)

        # Check if errors were tracked
        log("Checking error tracking", "PROGRESS")
        has_errors = "tool_errors" in result
        print_check("Tool errors tracked", has_errors)

        # Confidence should be lower
        log("Verifying confidence adjustment", "PROGRESS")
        confidence = result.get("confidence", 1.0)
        lower_confidence = confidence < 0.9
        print_check(f"Confidence reduced (={confidence:.2f})", lower_confidence)

        print_result_summary(result)

        if has_answer:
            log("Error handling test passed - agent recovered gracefully", "SUCCESS")
        else:
            log("Error handling test failed - no answer generated", "WARNING")

        return has_answer

    except Exception as e:
        log(f"Exception during error handling test: {str(e)}", "ERROR")
        print_check(f"Error handling test - ERROR: {e}", False)
        return False


def test_graph_visualization():
    """Test graph visualization."""
    print_header("TEST 8: GRAPH VISUALIZATION")

    log("Testing graph visualization generation", "INFO")

    try:
        log("Generating workflow graph visualization", "PROGRESS")
        viz = get_graph_visualization()
        has_content = len(viz) > 100
        print_check("Graph visualization generated", has_content)

        if has_content:
            log(f"Graph visualization: {len(viz)} characters", "INFO")
            print("\nFirst 500 characters:")
            print(viz[:500])
            log("Graph visualization test passed", "SUCCESS")
        else:
            log("Graph visualization is empty or too small", "WARNING")

        return has_content

    except Exception as e:
        log(f"Exception during visualization test: {str(e)}", "ERROR")
        print_check(f"Visualization test - ERROR: {e}", False)
        return False


def run_all_tests():
    """Run all agent tests."""
    print("="*80, flush=True)
    print("AGENT WORKFLOW TEST SUITE", flush=True)
    print("="*80, flush=True)

    log("Starting agent test suite", "TEST")
    log(f"Test run initiated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "INFO")

    # Check prerequisites
    print("\nChecking prerequisites...", flush=True)
    log("Checking prerequisites", "PROGRESS")

    api_running = False
    try:
        import requests
        response = requests.get("http://127.0.0.1:8001/", timeout=2)
        api_running = response.status_code == 200
    except:
        pass

    prereq_checks = {
        "API server running": api_running,
        "RAG initialized": os.path.exists("data/embeddings"),
        "Database exists": os.path.exists("data/metrics.db"),
        "OpenAI API key set": bool(os.getenv("OPENAI_API_KEY"))
    }

    for check_name, check_result in prereq_checks.items():
        print_check(check_name, check_result)

    all_prereqs = all(prereq_checks.values())

    if not all_prereqs:
        log("Some prerequisites missing - tests may fail", "WARNING")
        print("\n⚠️  Some prerequisites missing - tests may fail")
        print("Run setup:")
        print("  1. python start_api_server.py")
        print("  2. python demo_rag.py")
        print("  3. python tests/validate_db_service.py")
        print("  4. export OPENAI_API_KEY=your_key\n")
    else:
        log("All prerequisites satisfied", "SUCCESS")

    results = {}

    # Run tests
    log("Running 8 test suites", "INFO")

    log("Test 1/8: Metrics Lookup", "TEST")
    results["Metrics Lookup"] = test_metrics_lookup_intent()

    log("Test 2/8: Knowledge Lookup", "TEST")
    results["Knowledge Lookup"] = test_knowledge_lookup_intent()

    log("Test 3/8: Calculation", "TEST")
    results["Calculation"] = test_calculation_intent()

    log("Test 4/8: Mixed Intent", "TEST")
    results["Mixed Intent"] = test_mixed_intent()

    log("Test 5/8: Workflow Components", "TEST")
    results["Workflow Components"] = test_workflow_components()

    log("Test 6/8: Feedback Loop", "TEST")
    results["Feedback Loop"] = test_feedback_loop()

    log("Test 7/8: Error Handling", "TEST")
    results["Error Handling"] = test_error_handling()

    log("Test 8/8: Graph Visualization", "TEST")
    results["Graph Visualization"] = test_graph_visualization()

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
        log("ALL TESTS PASSED - Agent is working correctly!", "SUCCESS")
        print("✓ ALL TESTS PASSED - Agent is working correctly!", flush=True)
        print("="*80, flush=True)
        return True
    else:
        log(f"SOME TESTS FAILED - {failed_count} test(s) failed", "ERROR")
        print("✗ SOME TESTS FAILED - Please review errors above", flush=True)
        print("="*80, flush=True)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
