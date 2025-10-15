#!/usr/bin/env python3
"""
End-to-End Workflow Testing
Tests complete agent workflows for all intent types.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv(override=True)

# Import agent
from agent import run_agent
from agent.state import (
    INTENT_METRICS_LOOKUP,
    INTENT_KNOWLEDGE_LOOKUP,
    INTENT_CALCULATION,
    INTENT_MIXED,
    INTENT_CLARIFICATION,
    INTENT_UNKNOWN
)


def print_result(query, result, test_name):
    """Print formatted result."""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(f"Query: {query}")
    print(f"\nIntent: {result.get('intent', 'unknown')}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    print(f"Tools used: {', '.join(result.get('tools_executed', []))}")
    print(f"Processing time: {result.get('total_duration_ms', 0):.0f}ms")

    # Check for errors
    if result.get('tool_errors'):
        print(f"\n⚠️  Tool errors: {result['tool_errors']}")
        return False

    # Check answer quality
    answer = result.get('final_answer', '')
    if not answer or len(answer) < 50:
        print(f"\n⚠️  Answer too short or missing")
        return False

    print(f"\n✅ PASSED")
    print(f"\nAnswer preview: {answer[:200]}...")
    return True


def test_metrics_lookup():
    """Test 1: Metrics Lookup Intent"""
    query = "What is the current latency for api-gateway?"

    result = run_agent(query, verbose=False)

    # Verify intent
    expected_intent = INTENT_METRICS_LOOKUP
    actual_intent = result.get('intent')

    if actual_intent != expected_intent:
        print(f"\n⚠️  Expected intent '{expected_intent}', got '{actual_intent}'")

    # Verify tools
    expected_tools = ['query_metrics_api']
    actual_tools = result.get('tools_executed', [])

    if not any(tool in actual_tools for tool in expected_tools):
        print(f"\n⚠️  Expected tools {expected_tools}, got {actual_tools}")

    return print_result(query, result, "Metrics Lookup")


def test_knowledge_lookup():
    """Test 2: Knowledge Lookup Intent"""
    query = "How do I troubleshoot high latency issues?"

    result = run_agent(query, verbose=False)

    # Verify intent
    expected_intent = INTENT_KNOWLEDGE_LOOKUP
    actual_intent = result.get('intent')

    if actual_intent != expected_intent:
        print(f"\n⚠️  Expected intent '{expected_intent}', got '{actual_intent}'")

    # Verify tools
    expected_tools = ['search_knowledge_base']
    actual_tools = result.get('tools_executed', [])

    if not any(tool in actual_tools for tool in expected_tools):
        print(f"\n⚠️  Expected tools {expected_tools}, got {actual_tools}")

    return print_result(query, result, "Knowledge Lookup")


def test_calculation():
    """Test 3: Calculation Intent"""
    query = "Calculate the average of 150, 200, and 250"

    result = run_agent(query, verbose=False)

    # Verify intent
    expected_intent = INTENT_CALCULATION
    actual_intent = result.get('intent')

    if actual_intent != expected_intent:
        print(f"\n⚠️  Expected intent '{expected_intent}', got '{actual_intent}'")

    # Verify tools
    expected_tools = ['calculate']
    actual_tools = result.get('tools_executed', [])

    if not any(tool in actual_tools for tool in expected_tools):
        print(f"\n⚠️  Expected tools {expected_tools}, got {actual_tools}")

    return print_result(query, result, "Calculation")


def test_mixed_intent():
    """Test 4: Mixed Intent (Multiple Tools)"""
    query = "What is the latency for api-gateway and how can I improve it?"

    result = run_agent(query, verbose=False)

    # Verify intent
    expected_intent = INTENT_MIXED
    actual_intent = result.get('intent')

    if actual_intent != expected_intent:
        print(f"\n⚠️  Expected intent '{expected_intent}', got '{actual_intent}'")

    # Verify multiple tools used
    actual_tools = result.get('tools_executed', [])

    if len(actual_tools) < 2:
        print(f"\n⚠️  Expected 2+ tools for mixed intent, got {len(actual_tools)}")

    return print_result(query, result, "Mixed Intent")


def test_historical_data():
    """Test 5: Historical Data (SQL Database)"""
    query = "What was the average CPU usage for api-gateway over the last week?"

    result = run_agent(query, verbose=False)

    # Verify tools
    expected_tools = ['query_sql_database']
    actual_tools = result.get('tools_executed', [])

    if not any(tool in actual_tools for tool in expected_tools):
        print(f"\n⚠️  Expected tools {expected_tools}, got {actual_tools}")

    return print_result(query, result, "Historical Data (SQL)")


def test_comparison():
    """Test 6: Service Comparison"""
    query = "Compare memory usage between auth-service and data-processor"

    result = run_agent(query, verbose=False)

    # Verify tools (could be SQL or API)
    actual_tools = result.get('tools_executed', [])

    if not actual_tools:
        print(f"\n⚠️  No tools executed")

    # Verify inference was performed
    inference = result.get('inference_result', {})
    findings = result.get('findings', [])

    if not findings:
        print(f"\n⚠️  No findings generated from comparison")

    return print_result(query, result, "Service Comparison")


def test_error_handling():
    """Test 7: Error Handling (Invalid Service)"""
    query = "What is the latency for nonexistent-service?"

    result = run_agent(query, verbose=False)

    # Should still return an answer even if service doesn't exist
    answer = result.get('final_answer', '')

    if not answer:
        print(f"\n⚠️  No answer returned for invalid service")
        return False

    return print_result(query, result, "Error Handling")


def test_inference_thresholds():
    """Test 8: Inference with Threshold Checks"""
    query = "Show me error rates for all services"

    result = run_agent(query, verbose=False)

    # Verify inference was performed
    inference = result.get('inference_result', {})

    if not inference:
        print(f"\n⚠️  No inference performed")

    # Check if recommendations were generated
    recommendations = result.get('recommendations', [])

    return print_result(query, result, "Inference & Thresholds")


def main():
    """Run all end-to-end workflow tests."""
    print("\n" + "="*80)
    print("END-TO-END WORKFLOW TESTING")
    print("="*80)
    print("\nTesting complete agent workflows for all intent types...")

    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    print(f"✅ OpenAI API Key: Set")

    # Run all tests
    all_results = {}

    try:
        all_results['Metrics Lookup'] = test_metrics_lookup()
    except Exception as e:
        print(f"\n❌ EXCEPTION in Metrics Lookup: {e}")
        all_results['Metrics Lookup'] = False

    try:
        all_results['Knowledge Lookup'] = test_knowledge_lookup()
    except Exception as e:
        print(f"\n❌ EXCEPTION in Knowledge Lookup: {e}")
        all_results['Knowledge Lookup'] = False

    try:
        all_results['Calculation'] = test_calculation()
    except Exception as e:
        print(f"\n❌ EXCEPTION in Calculation: {e}")
        all_results['Calculation'] = False

    try:
        all_results['Mixed Intent'] = test_mixed_intent()
    except Exception as e:
        print(f"\n❌ EXCEPTION in Mixed Intent: {e}")
        all_results['Mixed Intent'] = False

    try:
        all_results['Historical Data'] = test_historical_data()
    except Exception as e:
        print(f"\n❌ EXCEPTION in Historical Data: {e}")
        all_results['Historical Data'] = False

    try:
        all_results['Service Comparison'] = test_comparison()
    except Exception as e:
        print(f"\n❌ EXCEPTION in Service Comparison: {e}")
        all_results['Service Comparison'] = False

    try:
        all_results['Error Handling'] = test_error_handling()
    except Exception as e:
        print(f"\n❌ EXCEPTION in Error Handling: {e}")
        all_results['Error Handling'] = False

    try:
        all_results['Inference & Thresholds'] = test_inference_thresholds()
    except Exception as e:
        print(f"\n❌ EXCEPTION in Inference & Thresholds: {e}")
        all_results['Inference & Thresholds'] = False

    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - END-TO-END WORKFLOWS")
    print("="*80)

    for test_name, passed in all_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    total_passed = sum(all_results.values())
    total_tests = len(all_results)

    print(f"\n{'='*80}")
    print(f"Overall: {total_passed}/{total_tests} workflows passed ({total_passed/total_tests*100:.0f}%)")

    if all(all_results.values()):
        print("\n🎉 ALL END-TO-END WORKFLOWS WORKING PERFECTLY!")
        return 0
    else:
        print("\n⚠️  Some workflows need attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())
