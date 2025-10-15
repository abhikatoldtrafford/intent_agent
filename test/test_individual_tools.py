#!/usr/bin/env python3
"""
Individual Tool Testing Script
Tests each of the 4 agent tools in isolation.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv(override=True)

# Import tools
from agent.tools import (
    query_metrics_api,
    search_knowledge_base,
    query_sql_database,
    calculate
)


def test_rest_api_tool():
    """Test Tool 1: REST API"""
    print("\n" + "="*80)
    print("TOOL 1: REST API (query_metrics_api)")
    print("="*80)

    tests = [
        ('Services List', {'metric_type': 'services'}),
        ('Latency Metrics', {'metric_type': 'latency', 'service': 'api-gateway'}),
        ('Health Check', {'metric_type': 'health', 'service': 'api-gateway'}),
        ('Throughput', {'metric_type': 'throughput', 'service': 'auth-service'}),
        ('Error Metrics', {'metric_type': 'errors', 'service': 'business-logic'}),
    ]

    results = []
    for name, params in tests:
        print(f"\n[Test: {name}]")
        print(f"Parameters: {params}")

        try:
            result = query_metrics_api.invoke(params)

            if 'error' in result:
                print(f"❌ FAILED: {result['error']}")
                results.append(False)
            else:
                print(f"✅ PASSED")
                print(f"Response keys: {list(result.keys())}")
                results.append(True)
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            results.append(False)

    success_rate = sum(results) / len(results) * 100
    print(f"\n{'='*80}")
    print(f"REST API Tool: {sum(results)}/{len(results)} tests passed ({success_rate:.0f}%)")
    return all(results)


def test_knowledge_rag_tool():
    """Test Tool 2: Knowledge RAG"""
    print("\n" + "="*80)
    print("TOOL 2: KNOWLEDGE RAG (search_knowledge_base)")
    print("="*80)

    queries = [
        'How do I fix high latency?',
        'What is the deployment process?',
        'How do I monitor errors?',
        'What is the system architecture?'
    ]

    results = []
    for query in queries:
        print(f"\n[Query: {query}]")

        try:
            result = search_knowledge_base.invoke({
                'query': query,
                'top_k': 3,
                'search_mode': 'hybrid'
            })

            if 'error' in result:
                print(f"❌ FAILED: {result['error']}")
                results.append(False)
            else:
                print(f"✅ PASSED")
                print(f"Found {result['total_results']} results")
                if result['results']:
                    print(f"Top result: {result['results'][0]['filename']} (score: {result['results'][0]['score']:.3f})")
                results.append(True)
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            results.append(False)

    success_rate = sum(results) / len(results) * 100
    print(f"\n{'='*80}")
    print(f"Knowledge RAG Tool: {sum(results)}/{len(results)} tests passed ({success_rate:.0f}%)")
    return all(results)


def test_sql_database_tool():
    """Test Tool 3: SQL Database"""
    print("\n" + "="*80)
    print("TOOL 3: SQL DATABASE (query_sql_database)")
    print("="*80)

    questions = [
        'What is the average CPU usage for api-gateway?',
        'Show me error counts for all services',
        'Which service has the highest memory usage?',
        'Compare latency between api-gateway and auth-service'
    ]

    results = []
    for question in questions:
        print(f"\n[Question: {question}]")

        try:
            result = query_sql_database.invoke({'question': question})

            if 'error' in result:
                print(f"❌ FAILED: {result['error']}")
                results.append(False)
            else:
                print(f"✅ PASSED")
                print(f"Returned {result['row_count']} rows")
                print(f"SQL: {result['sql_query'][:80]}...")
                results.append(True)
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            results.append(False)

    success_rate = sum(results) / len(results) * 100
    print(f"\n{'='*80}")
    print(f"SQL Database Tool: {sum(results)}/{len(results)} tests passed ({success_rate:.0f}%)")
    return all(results)


def test_calculator_tool():
    """Test Tool 4: Calculator"""
    print("\n" + "="*80)
    print("TOOL 4: CALCULATOR (calculate)")
    print("="*80)

    tests = [
        ('Simple addition', '(150 + 200) / 2'),
        ('Comparison', '95 < 100'),
        ('Percentage', '(500 - 450) / 450 * 100'),
        ('Max function', 'max(45.2, 67.8, 23.1)'),
        ('Complex expression', '(100 * 0.95) + (50 * 0.05)')
    ]

    results = []
    for name, expression in tests:
        print(f"\n[Test: {name}]")
        print(f"Expression: {expression}")

        try:
            result = calculate.invoke({'expression': expression})

            if 'error' in result:
                print(f"❌ FAILED: {result['error']}")
                results.append(False)
            else:
                print(f"✅ PASSED")
                print(f"Result: {result['result']} (type: {result['type']})")
                results.append(True)
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            results.append(False)

    success_rate = sum(results) / len(results) * 100
    print(f"\n{'='*80}")
    print(f"Calculator Tool: {sum(results)}/{len(results)} tests passed ({success_rate:.0f}%)")
    return all(results)


def main():
    """Run all individual tool tests."""
    print("\n" + "="*80)
    print("INDIVIDUAL TOOL TESTING")
    print("="*80)
    print("\nTesting each of the 4 agent tools in isolation...")

    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    print(f"✅ OpenAI API Key: Set")

    # Run tests
    all_results = {}

    all_results['REST API'] = test_rest_api_tool()
    all_results['Knowledge RAG'] = test_knowledge_rag_tool()
    all_results['SQL Database'] = test_sql_database_tool()
    all_results['Calculator'] = test_calculator_tool()

    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    for tool, passed in all_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {tool} Tool")

    total_passed = sum(all_results.values())
    total_tools = len(all_results)

    print(f"\n{'='*80}")
    print(f"Overall: {total_passed}/{total_tools} tools passed all tests")

    if all(all_results.values()):
        print("\n🎉 ALL TOOLS WORKING PERFECTLY!")
        return 0
    else:
        print("\n⚠️  Some tools need attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())
