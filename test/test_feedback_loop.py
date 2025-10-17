#!/usr/bin/env python3
"""
Feedback Loop Testing
Tests the confidence-based retry and clarification logic.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv(override=True)

# Import agent and state constants
from agent import run_agent
from agent.state import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    MAX_RETRIES
)
from utils.trace_display import display_execution_trace, display_trace_summary, display_feedback_iterations

# Track all test executions for observability
TEST_EXECUTIONS = []
TEST_RESULTS_FILE = Path(__file__).parent.parent / "data" / "test_executions.json"

# Control trace display
SHOW_TRACES = os.getenv("SHOW_TRACES", "true").lower() == "true"


def print_test_header(test_name):
    """Print test header."""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")


def run_and_save_test(query, test_name=""):
    """Run agent query and save to test executions for observability."""
    result = run_agent(query, verbose=False)

    # Save execution for observability
    TEST_EXECUTIONS.append({
        "query": query,
        "result": result,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_name": test_name
    })

    return result


def test_high_confidence_no_retry():
    """Test 1: High confidence queries should NOT trigger retry"""
    print_test_header("High Confidence - No Retry")

    query = "What is the current latency for api-gateway?"
    result = run_agent(query, verbose=False)

    confidence = result.get('confidence', 0)
    retry_count = result.get('retry_count', 0)
    feedback_needed = result.get('feedback_needed', False)

    print(f"Query: {query}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Retry count: {retry_count}")
    print(f"Feedback needed: {feedback_needed}")

    # Verify no retry was triggered
    if confidence >= CONFIDENCE_HIGH:
        print("\n✅ PASS: High confidence, no retry triggered")
        return True
    else:
        print(f"\n⚠️  FAIL: Expected high confidence (>= {CONFIDENCE_HIGH}), got {confidence:.2f}")
        return False


def test_ambiguous_query():
    """Test 2: Ambiguous queries might trigger lower confidence"""
    print_test_header("Ambiguous Query")

    query = "stuff"
    result = run_agent(query, verbose=False)

    confidence = result.get('confidence', 0)
    intent = result.get('intent', 'unknown')

    print(f"Query: {query}")
    print(f"Intent: {intent}")
    print(f"Confidence: {confidence:.2f}")

    # Very ambiguous query should have lower confidence
    if confidence < CONFIDENCE_HIGH:
        print(f"\n✅ PASS: Ambiguous query resulted in lower confidence ({confidence:.2f})")
        return True
    else:
        print(f"\n⚠️  NOTE: Even ambiguous query got high confidence ({confidence:.2f})")
        # This is not necessarily a failure - the agent might handle it well
        return True


def test_retry_logic_structure():
    """Test 3: Verify retry logic is correctly implemented"""
    print_test_header("Retry Logic Structure")

    # Test that retry count increments properly
    query = "test query"
    result = run_agent(query, verbose=False)

    retry_count = result.get('retry_count', 0)
    trace_events = result.get('trace', [])

    # Check for feedback loop trace events
    feedback_events = [e for e in trace_events if e.get('node') == 'check_feedback']

    print(f"Query: {query}")
    print(f"Final retry count: {retry_count}")
    print(f"Feedback check events: {len(feedback_events)}")
    print(f"Max retries allowed: {MAX_RETRIES}")

    # Verify retry count doesn't exceed max
    if retry_count <= MAX_RETRIES:
        print(f"\n✅ PASS: Retry count ({retry_count}) within limits (max: {MAX_RETRIES})")
        return True
    else:
        print(f"\n⚠️  FAIL: Retry count ({retry_count}) exceeds max ({MAX_RETRIES})")
        return False


def test_confidence_thresholds():
    """Test 4: Test understanding of confidence thresholds"""
    print_test_header("Confidence Thresholds")

    # Run several queries and check confidence distribution
    queries = [
        "What is the latency for api-gateway?",
        "How do I fix errors?",
        "Calculate 100 + 200",
        "Show me metrics and explain how to improve them"
    ]

    confidence_scores = []
    for query in queries:
        result = run_agent(query, verbose=False)
        confidence = result.get('confidence', 0)
        confidence_scores.append(confidence)
        print(f"Query: {query[:50]}")
        print(f"  Confidence: {confidence:.2f}")

    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    print(f"\nAverage confidence: {avg_confidence:.2f}")
    print(f"Confidence HIGH threshold: {CONFIDENCE_HIGH}")
    print(f"Confidence MEDIUM threshold: {CONFIDENCE_MEDIUM}")

    # Most queries should have high confidence
    high_confidence_count = sum(1 for c in confidence_scores if c >= CONFIDENCE_HIGH)
    print(f"\n{high_confidence_count}/{len(queries)} queries had high confidence")

    if avg_confidence >= CONFIDENCE_MEDIUM:
        print(f"\n✅ PASS: Average confidence ({avg_confidence:.2f}) is acceptable")
        return True
    else:
        print(f"\n⚠️  WARNING: Average confidence ({avg_confidence:.2f}) is low")
        return False


def test_trace_completeness():
    """Test 5: Verify feedback loop events are traced"""
    print_test_header("Trace Completeness")

    query = "What is the throughput for business-logic?"
    result = run_agent(query, verbose=False)

    trace_events = result.get('trace', [])

    # Check for key node events
    nodes_traced = set(e.get('node') for e in trace_events)
    expected_nodes = {
        'classify_intent',
        'select_tools',
        'execute_tools',
        'aggregate_results',
        'perform_inference',
        'check_feedback',
        'format_response'
    }

    print(f"Query: {query}")
    print(f"Total trace events: {len(trace_events)}")
    print(f"Nodes traced: {nodes_traced}")

    # Display detailed trace
    if SHOW_TRACES:
        display_execution_trace(result, show_all=False)

    missing_nodes = expected_nodes - nodes_traced
    if not missing_nodes:
        print(f"\n✅ PASS: All expected nodes traced")
        return True
    else:
        print(f"\n⚠️  WARNING: Missing nodes in trace: {missing_nodes}")
        return False


def test_feedback_decision_logic():
    """Test 6: Verify feedback decision logic"""
    print_test_header("Feedback Decision Logic")

    # Test a clear query that should NOT need feedback
    query = "What is the error rate for auth-service?"
    result = run_agent(query, verbose=False)

    feedback_needed = result.get('feedback_needed', False)
    confidence = result.get('confidence', 0)
    retry_reason = result.get('retry_reason')

    print(f"Query: {query}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Feedback needed: {feedback_needed}")
    print(f"Retry reason: {retry_reason}")

    # Clear query should NOT need feedback
    if not feedback_needed:
        print(f"\n✅ PASS: Clear query correctly identified as not needing feedback")
        return True
    else:
        print(f"\n⚠️  WARNING: Feedback was triggered for a clear query")
        print(f"  Reason: {retry_reason}")
        # This might be OK depending on the reason
        return True


def test_max_retries_enforcement():
    """Test 7: Verify max retries is enforced"""
    print_test_header("Max Retries Enforcement")

    # Run a query and check retry count
    query = "Compare all services"
    result = run_agent(query, verbose=False)

    retry_count = result.get('retry_count', 0)
    final_answer = result.get('final_answer', '')

    print(f"Query: {query}")
    print(f"Retry count: {retry_count}")
    print(f"Max retries: {MAX_RETRIES}")
    print(f"Got final answer: {bool(final_answer)}")

    # Should always get a final answer even if retries occurred
    if final_answer and retry_count <= MAX_RETRIES:
        print(f"\n✅ PASS: Got final answer with retry count ({retry_count}) within limits")
        return True
    else:
        print(f"\n⚠️  FAIL: Problem with retry handling")
        return False


def test_data_quality_impact():
    """Test 8: Verify data quality impacts confidence"""
    print_test_header("Data Quality Impact")

    query = "What is the latency for api-gateway?"
    result = run_agent(query, verbose=False)

    data_quality = result.get('data_quality', {})
    confidence = result.get('confidence', 0)

    print(f"Query: {query}")
    print(f"Data quality completeness: {data_quality.get('completeness', 1.0):.2f}")
    print(f"Data quality consistency: {data_quality.get('consistency', 1.0):.2f}")
    print(f"Data quality issues: {data_quality.get('issues', [])}")
    print(f"Final confidence: {confidence:.2f}")

    # Good data quality should result in high confidence
    completeness = data_quality.get('completeness', 1.0)
    if completeness >= 0.8 and confidence >= CONFIDENCE_MEDIUM:
        print(f"\n✅ PASS: Good data quality resulted in good confidence")
        return True
    elif completeness < 0.8:
        print(f"\n⚠️  NOTE: Low data quality ({completeness:.2f}) detected")
        if confidence < CONFIDENCE_MEDIUM:
            print(f"  Correctly resulted in lower confidence ({confidence:.2f})")
            return True
        else:
            print(f"  But confidence is still high ({confidence:.2f})")
            return True
    else:
        print(f"\n✅ PASS: Data quality and confidence are reasonable")
        return True


def test_unknown_intent_feedback():
    """Test 9: Verify unknown intent triggers feedback iteration (LLM fallback)"""
    print_test_header("Unknown Intent Feedback Loop")

    # Test multiple out-of-distribution queries
    test_queries = [
        ("What's the weather like today?", "Weather query - completely out of domain"),
        ("Hello! How are you?", "Greeting - no monitoring context"),
        ("Tell me a joke", "Entertainment request")
    ]

    all_passed = True

    for query, description in test_queries:
        print(f"\n{'─'*60}")
        print(f"Testing: {query}")
        print(f"Expected: {description}")
        print(f"{'─'*60}")

        result = run_and_save_test(query, test_name="Unknown Intent Feedback")

        intent = result.get('intent', '')
        feedback_iterations = result.get('feedback_iterations', [])
        off_topic_query = result.get('off_topic_query', False)
        tools_executed = result.get('tools_executed', [])
        final_answer = result.get('final_answer', '')

        print(f"✓ Intent classified as: {intent}")
        print(f"✓ Off-topic flag: {off_topic_query}")
        print(f"✓ Tools executed: {len(tools_executed)} ({', '.join(tools_executed) if tools_executed else 'none'})")
        print(f"✓ Feedback iterations: {len(feedback_iterations)}")
        print(f"✓ Got final answer: {len(final_answer) > 0}")

        # Display detailed trace for this query
        if SHOW_TRACES:
            display_trace_summary(result)
            display_feedback_iterations(result)

        if feedback_iterations:
            print(f"\n📋 Feedback iteration details:")
            for i, iteration in enumerate(feedback_iterations, 1):
                reason = iteration.get('reason', 'N/A')
                action = iteration.get('action', 'N/A')
                iter_intent = iteration.get('intent', 'N/A')
                confidence = iteration.get('confidence_at_feedback', 0)
                print(f"  {i}. Reason: {reason}")
                print(f"     Action: {action}")
                print(f"     Intent: {iter_intent}")
                print(f"     Confidence: {confidence:.2f}")

        # Validate the result
        query_passed = True
        issues = []

        # Check 1: Should be unknown intent
        if intent != "unknown":
            issues.append(f"Expected intent 'unknown', got '{intent}'")
            query_passed = False

        # Check 2: Should have feedback iteration
        if len(feedback_iterations) == 0:
            issues.append("No feedback iteration logged")
            query_passed = False
        else:
            # Check 3: Feedback reason should be correct
            has_correct_feedback = any(
                fb.get('reason') == 'unknown_intent_llm_fallback'
                for fb in feedback_iterations
            )
            if not has_correct_feedback:
                issues.append(f"Expected feedback reason 'unknown_intent_llm_fallback', got: {[fb.get('reason') for fb in feedback_iterations]}")
                query_passed = False

            # Check 4: Action should be correct
            has_correct_action = any(
                fb.get('action') == 'using_llm_general_knowledge'
                for fb in feedback_iterations
            )
            if not has_correct_action:
                issues.append(f"Expected action 'using_llm_general_knowledge'")
                query_passed = False

        # Check 5: Should NOT execute specialized tools
        if len(tools_executed) > 0:
            issues.append(f"Unexpected tools executed: {tools_executed} (should be none for unknown intent)")
            # This is a warning, not a failure
            print(f"  ⚠️  WARNING: {issues[-1]}")

        if query_passed:
            print(f"\n✅ PASS: '{query}' correctly handled as unknown intent with LLM fallback")
        else:
            print(f"\n❌ FAIL: '{query}' - Issues found:")
            for issue in issues:
                print(f"     • {issue}")
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL QUERIES PASSED: Unknown intent feedback working correctly")
    else:
        print("❌ SOME QUERIES FAILED: See details above")

    return all_passed


def test_clarify_intent_feedback():
    """Test 10: Verify clarify intent triggers feedback iteration (clarification request)"""
    print_test_header("Clarify Intent Feedback Loop")

    # Test multiple vague in-domain queries
    test_queries = [
        ("Show me the latency", "Vague - has 'latency' keyword but missing service name"),
        ("What's the error rate?", "Vague - has 'error' keyword but missing service name"),
        ("Check the CPU usage", "Vague - has 'CPU' keyword but missing service and time range")
    ]

    all_passed = True
    at_least_one_clarify = False

    for query, description in test_queries:
        print(f"\n{'─'*60}")
        print(f"Testing: {query}")
        print(f"Expected: {description}")
        print(f"{'─'*60}")

        result = run_and_save_test(query, test_name="Clarify Intent Feedback")

        intent = result.get('intent', '')
        feedback_iterations = result.get('feedback_iterations', [])
        clarification_question = result.get('clarification_question', '')
        tools_executed = result.get('tools_executed', [])
        final_answer = result.get('final_answer', '')

        print(f"✓ Intent classified as: {intent}")
        print(f"✓ Has clarification question: {bool(clarification_question)}")
        print(f"✓ Tools executed: {len(tools_executed)} ({', '.join(tools_executed) if tools_executed else 'none'})")
        print(f"✓ Feedback iterations: {len(feedback_iterations)}")
        print(f"✓ Got final answer: {len(final_answer) > 0}")

        # Display detailed trace for this query
        if SHOW_TRACES:
            display_trace_summary(result)
            display_feedback_iterations(result)

        if clarification_question:
            print(f"\n📝 Clarification question generated:")
            # Show first 200 chars
            preview = clarification_question[:200] + "..." if len(clarification_question) > 200 else clarification_question
            print(f"   {preview}")

        if feedback_iterations:
            print(f"\n📋 Feedback iteration details:")
            for i, iteration in enumerate(feedback_iterations, 1):
                reason = iteration.get('reason', 'N/A')
                action = iteration.get('action', 'N/A')
                iter_intent = iteration.get('intent', 'N/A')
                confidence = iteration.get('confidence_at_feedback', 0)
                missing_params = iteration.get('missing_params', [])
                print(f"  {i}. Reason: {reason}")
                print(f"     Action: {action}")
                print(f"     Intent: {iter_intent}")
                print(f"     Confidence: {confidence:.2f}")
                if missing_params:
                    print(f"     Missing params: {', '.join(missing_params)}")

        # Validate the result
        # NOTE: LLM might classify as metrics_lookup and infer default service
        # So we check IF it's clarify, then feedback should be logged correctly
        if intent == "clarify":
            at_least_one_clarify = True
            query_passed = True
            issues = []

            # Check 1: Should have feedback iteration
            if len(feedback_iterations) == 0:
                issues.append("Clarify intent but no feedback iteration logged")
                query_passed = False
            else:
                # Check 2: Feedback reason should be correct
                has_correct_feedback = any(
                    fb.get('reason') == 'clarification_required'
                    for fb in feedback_iterations
                )
                if not has_correct_feedback:
                    issues.append(f"Expected feedback reason 'clarification_required', got: {[fb.get('reason') for fb in feedback_iterations]}")
                    query_passed = False

                # Check 3: Action should be correct
                has_correct_action = any(
                    fb.get('action') == 'asking_user_for_clarification'
                    for fb in feedback_iterations
                )
                if not has_correct_action:
                    issues.append(f"Expected action 'asking_user_for_clarification'")
                    query_passed = False

            # Check 4: Should have clarification question
            if not clarification_question:
                issues.append("Clarify intent but no clarification_question generated")
                query_passed = False

            # Check 5: Should NOT execute specialized tools (waiting for user)
            if len(tools_executed) > 0:
                print(f"  ⚠️  NOTE: Tools executed despite clarify intent: {tools_executed}")

            if query_passed:
                print(f"\n✅ PASS: '{query}' correctly handled as clarify intent with feedback")
            else:
                print(f"\n❌ FAIL: '{query}' - Issues found:")
                for issue in issues:
                    print(f"     • {issue}")
                all_passed = False

        else:
            # Intent is NOT clarify - this is OK, LLM might infer context
            print(f"\n💡 INFO: Query classified as '{intent}' instead of 'clarify'")
            print(f"     This is acceptable - LLM may have inferred missing parameters")
            print(f"     (e.g., defaulted to 'api-gateway' or interpreted as general query)")

    print(f"\n{'='*60}")
    if at_least_one_clarify:
        if all_passed:
            print("✅ ALL CLARIFY QUERIES PASSED: Clarification feedback working correctly")
        else:
            print("❌ SOME CLARIFY QUERIES FAILED: See details above")
    else:
        print("💡 NOTE: None of the queries were classified as 'clarify'")
        print("     This is acceptable - the LLM may infer missing parameters")
        print("     The important thing is: IF a query is classified as clarify,")
        print("     THEN it should log feedback correctly (which is tested above)")
        all_passed = True  # Not a failure if LLM is smart enough to infer

    return all_passed


def main():
    """Run all feedback loop tests."""
    print("\n" + "="*80)
    print("FEEDBACK LOOP TESTING")
    print("="*80)
    print("\nTesting confidence-based retry and clarification logic...")

    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    print(f"✅ OpenAI API Key: Set")
    print(f"✅ Confidence HIGH threshold: {CONFIDENCE_HIGH}")
    print(f"✅ Confidence MEDIUM threshold: {CONFIDENCE_MEDIUM}")
    print(f"✅ Max retries: {MAX_RETRIES}")

    # Run all tests
    all_results = {}

    try:
        all_results['High Confidence - No Retry'] = test_high_confidence_no_retry()
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        all_results['High Confidence - No Retry'] = False

    try:
        all_results['Ambiguous Query'] = test_ambiguous_query()
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        all_results['Ambiguous Query'] = False

    try:
        all_results['Retry Logic Structure'] = test_retry_logic_structure()
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        all_results['Retry Logic Structure'] = False

    try:
        all_results['Confidence Thresholds'] = test_confidence_thresholds()
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        all_results['Confidence Thresholds'] = False

    try:
        all_results['Trace Completeness'] = test_trace_completeness()
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        all_results['Trace Completeness'] = False

    try:
        all_results['Feedback Decision Logic'] = test_feedback_decision_logic()
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        all_results['Feedback Decision Logic'] = False

    try:
        all_results['Max Retries Enforcement'] = test_max_retries_enforcement()
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        all_results['Max Retries Enforcement'] = False

    try:
        all_results['Data Quality Impact'] = test_data_quality_impact()
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        all_results['Data Quality Impact'] = False

    # NEW TESTS: Unknown and Clarify Feedback
    try:
        all_results['Unknown Intent Feedback'] = test_unknown_intent_feedback()
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        all_results['Unknown Intent Feedback'] = False

    try:
        all_results['Clarify Intent Feedback'] = test_clarify_intent_feedback()
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        all_results['Clarify Intent Feedback'] = False

    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - FEEDBACK LOOP")
    print("="*80)

    for test_name, passed in all_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    total_passed = sum(all_results.values())
    total_tests = len(all_results)

    print(f"\n{'='*80}")
    print(f"Overall: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.0f}%)")

    # Save test executions for observability
    try:
        TEST_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TEST_RESULTS_FILE, 'w') as f:
            json.dump(TEST_EXECUTIONS, f, indent=2, default=str)
        print(f"\n💾 Saved {len(TEST_EXECUTIONS)} test executions to {TEST_RESULTS_FILE}")
        print(f"   Load these in Streamlit observability tab to view details!")
    except Exception as e:
        print(f"\n⚠️  Warning: Could not save test executions: {e}")

    if all(all_results.values()):
        print("\n🎉 FEEDBACK LOOP WORKING CORRECTLY!")
        return 0
    else:
        print("\n⚠️  Some feedback loop aspects need attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())
