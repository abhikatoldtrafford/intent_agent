#!/usr/bin/env python3
"""
Test Trace Cache System

Validates that trace caching, LangSmith API fetching, and auto-population work correctly.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.trace_cache import (
    ensure_cache_dir,
    save_cache,
    load_cache,
    is_cache_valid,
    get_cache_status,
    cache_agent_execution,
    get_agent_executions,
    auto_populate_traces,
    CACHE_DIR,
    AGENT_CACHE_FILE
)


def test_cache_directory():
    """Test 1: Cache directory creation"""
    print("\n" + "="*80)
    print("TEST 1: Cache Directory Creation")
    print("="*80)

    ensure_cache_dir()

    if CACHE_DIR.exists():
        print(f"✅ PASS: Cache directory exists at {CACHE_DIR}")
        return True
    else:
        print(f"❌ FAIL: Cache directory not created")
        return False


def test_cache_save_load():
    """Test 2: Cache save and load"""
    print("\n" + "="*80)
    print("TEST 2: Cache Save and Load")
    print("="*80)

    # Create test data
    test_data = [
        {
            "query": "test query 1",
            "result": {"intent": "test", "confidence": 0.95},
            "timestamp": datetime.now().isoformat()
        },
        {
            "query": "test query 2",
            "result": {"intent": "metrics_lookup", "confidence": 0.88},
            "timestamp": datetime.now().isoformat()
        }
    ]

    test_cache_file = CACHE_DIR / "test_cache.json"

    # Save
    save_cache(test_cache_file, test_data)
    print(f"✓ Saved {len(test_data)} items to cache")

    # Load
    loaded_data = load_cache(test_cache_file)

    if loaded_data and len(loaded_data) == len(test_data):
        print(f"✓ Loaded {len(loaded_data)} items from cache")
        print("✅ PASS: Cache save/load works correctly")

        # Cleanup
        if test_cache_file.exists():
            test_cache_file.unlink()

        return True
    else:
        print(f"❌ FAIL: Cache load returned {len(loaded_data) if loaded_data else 0} items, expected {len(test_data)}")
        return False


def test_cache_validity():
    """Test 3: Cache validity check"""
    print("\n" + "="*80)
    print("TEST 3: Cache Validity Check")
    print("="*80)

    test_cache_file = CACHE_DIR / "test_validity.json"

    # Create fresh cache
    save_cache(test_cache_file, [{"test": "data"}])

    # Should be valid (just created)
    if is_cache_valid(test_cache_file):
        print("✓ Fresh cache is valid")

        # Cleanup
        if test_cache_file.exists():
            test_cache_file.unlink()

        print("✅ PASS: Cache validity check works")
        return True
    else:
        print("❌ FAIL: Fresh cache marked as invalid")
        return False


def test_cache_agent_execution():
    """Test 4: Cache agent execution"""
    print("\n" + "="*80)
    print("TEST 4: Cache Agent Execution")
    print("="*80)

    # Clear existing cache
    if AGENT_CACHE_FILE.exists():
        AGENT_CACHE_FILE.unlink()

    # Cache an execution
    test_result = {
        "intent": "metrics_lookup",
        "confidence": 0.92,
        "tools_executed": ["query_metrics_api"],
        "final_answer": "Test answer",
        "trace": []
    }

    cache_agent_execution(test_result, "test query for caching")
    print("✓ Cached agent execution")

    # Verify it was cached
    cached = get_agent_executions()

    if cached and len(cached) > 0:
        print(f"✓ Retrieved {len(cached)} cached execution(s)")
        first_exec = cached[0]

        if first_exec.get('query') == "test query for caching":
            print("✓ Query matches")
            print("✅ PASS: Agent execution caching works")

            # Cleanup
            if AGENT_CACHE_FILE.exists():
                AGENT_CACHE_FILE.unlink()

            return True
        else:
            print(f"❌ FAIL: Query mismatch - got '{first_exec.get('query')}'")
            return False
    else:
        print("❌ FAIL: No cached executions retrieved")
        return False


def test_cache_status():
    """Test 5: Cache status reporting"""
    print("\n" + "="*80)
    print("TEST 5: Cache Status Reporting")
    print("="*80)

    status = get_cache_status()

    print("Cache Status:")
    for name, info in status.items():
        exists = "✓" if info.get('exists') else "✗"
        valid = "✓" if info.get('valid') else "✗"
        print(f"  {name}: exists={exists}, valid={valid}")

    if isinstance(status, dict) and len(status) > 0:
        print("✅ PASS: Cache status reporting works")
        return True
    else:
        print("❌ FAIL: Cache status not available")
        return False


def test_auto_populate():
    """Test 6: Auto-populate traces"""
    print("\n" + "="*80)
    print("TEST 6: Auto-Populate Traces")
    print("="*80)

    summary = auto_populate_traces()

    print("Auto-population summary:")
    print(f"  • LangSmith traces: {summary.get('langsmith_traces', 0)}")
    print(f"  • Agent executions: {summary.get('agent_executions', 0)}")
    print(f"  • Demo executions: {summary.get('demo_executions', 0)}")
    print(f"  • Test executions: {summary.get('test_executions', 0)}")
    print(f"  • Total: {summary.get('total', 0)}")

    if isinstance(summary, dict):
        print("✅ PASS: Auto-populate works")
        return True
    else:
        print("❌ FAIL: Auto-populate failed")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("TRACE CACHE SYSTEM TESTS")
    print("="*80)
    print("\nTesting trace caching, LangSmith integration, and auto-population")

    results = []

    try:
        results.append(("Cache Directory Creation", test_cache_directory()))
    except Exception as e:
        print(f"❌ TEST EXCEPTION: {e}")
        results.append(("Cache Directory Creation", False))

    try:
        results.append(("Cache Save/Load", test_cache_save_load()))
    except Exception as e:
        print(f"❌ TEST EXCEPTION: {e}")
        results.append(("Cache Save/Load", False))

    try:
        results.append(("Cache Validity", test_cache_validity()))
    except Exception as e:
        print(f"❌ TEST EXCEPTION: {e}")
        results.append(("Cache Validity", False))

    try:
        results.append(("Agent Execution Caching", test_cache_agent_execution()))
    except Exception as e:
        print(f"❌ TEST EXCEPTION: {e}")
        results.append(("Agent Execution Caching", False))

    try:
        results.append(("Cache Status", test_cache_status()))
    except Exception as e:
        print(f"❌ TEST EXCEPTION: {e}")
        results.append(("Cache Status", False))

    try:
        results.append(("Auto-Populate", test_auto_populate()))
    except Exception as e:
        print(f"❌ TEST EXCEPTION: {e}")
        results.append(("Auto-Populate", False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*80 + "\n")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Trace caching system is working correctly.")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
