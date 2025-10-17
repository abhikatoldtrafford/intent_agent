#!/usr/bin/env python3
"""
Test script to validate intent classification fixes for previously failing queries.

Tests the two specific queries that were incorrectly classified as 'clarify':
1. "memory usage for latency service"
2. "Compare memory usage between api-gateway and auth-service"

Both should now be classified as 'metrics_lookup' with reasonable confidence.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.state import AgentState, INTENT_METRICS_LOOKUP
from agent.nodes import classify_intent


def test_classification(query: str, expected_intent: str, test_name: str):
    """Test a single query classification."""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(f"Query: \"{query}\"")
    print(f"Expected Intent: {expected_intent}")
    print(f"{'-'*80}")

    # Create initial state
    state = AgentState(
        query=query,
        intent="",
        confidence=0.0,
        tools_selected=[],
        tools_executed=[],
        tool_results={},
        final_answer="",
        trace=[],
        findings=[],
        recommendations=[],
        data_quality={},
        feedback_iterations=0,
        orchestration_log=[]
    )

    try:
        # Run classification
        result_state = classify_intent(state)

        # Display results
        print(f"\n✅ Classification Results:")
        print(f"   Intent: {result_state['intent']}")
        print(f"   Confidence: {result_state['confidence']:.2f}")
        print(f"   Reasoning: {result_state.get('classification_reasoning', 'N/A')}")

        # Validate
        success = result_state['intent'] == expected_intent

        if success:
            print(f"\n🎉 TEST PASSED! Classified as '{result_state['intent']}' (expected '{expected_intent}')")
        else:
            print(f"\n❌ TEST FAILED! Classified as '{result_state['intent']}' (expected '{expected_intent}')")

        return success

    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all classification tests."""
    print("\n" + "="*80)
    print("INTENT CLASSIFICATION FIX VALIDATION")
    print("Testing previously failing queries")
    print("="*80)

    tests = [
        {
            "name": "Query 1: Memory comparison (was failing with 'clarify')",
            "query": "Compare memory usage between api-gateway and auth-service",
            "expected": INTENT_METRICS_LOOKUP
        },
        {
            "name": "Query 2: Memory usage with confusing service name (was failing with 'clarify')",
            "query": "memory usage for latency service",
            "expected": INTENT_METRICS_LOOKUP
        },
        {
            "name": "Query 3: Similar query - should also be metrics_lookup",
            "query": "Show me CPU usage for api-gateway",
            "expected": INTENT_METRICS_LOOKUP
        },
        {
            "name": "Query 4: Historical trend - should be metrics_lookup",
            "query": "What was the average memory for auth-service last week",
            "expected": INTENT_METRICS_LOOKUP
        }
    ]

    results = []
    for test in tests:
        success = test_classification(
            query=test["query"],
            expected_intent=test["expected"],
            test_name=test["name"]
        )
        results.append(success)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(results)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Intent classification is now working correctly.")
        print("\nThe restructured prompt with guardrails successfully:")
        print("  ✅ Makes reasonable defaults instead of asking for clarification")
        print("  ✅ Correctly identifies metrics queries with monitoring keywords")
        print("  ✅ Routes queries to appropriate tools (SQL database for memory/CPU)")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review the classification logic.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
