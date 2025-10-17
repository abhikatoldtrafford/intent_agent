#!/usr/bin/env python3
"""
Quick test to verify all API endpoints work with prefilled parameters
"""

import requests
import sys

def test_endpoint(name, url):
    """Test a single endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    print(f"{'='*60}")

    try:
        response = requests.get(url, timeout=5)
        duration_ms = response.elapsed.total_seconds() * 1000

        print(f"✅ Status: {response.status_code}")
        print(f"⏱️  Duration: {duration_ms:.2f}ms")

        if response.headers.get('content-type', '').startswith('application/json'):
            data = response.json()
            print(f"📦 Response keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")

            # Show first few fields
            if isinstance(data, dict):
                for key, value in list(data.items())[:3]:
                    if isinstance(value, (str, int, float)):
                        print(f"   • {key}: {value}")
                    elif isinstance(value, dict):
                        print(f"   • {key}: {list(value.keys())}")
        else:
            print(f"📦 Response: {response.text[:100]}")

        return response.status_code == 200

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    """Test all default endpoints"""
    print("\n" + "="*80)
    print("API ENDPOINTS TEST - Prefilled Parameters")
    print("="*80)
    print("\nTesting with default prefilled values (same as Streamlit app)")

    base_url = "http://127.0.0.1:8001"

    # Test cases matching the Streamlit prefilled defaults
    tests = [
        ("Latency (default)", f"{base_url}/metrics/latency?service=api-gateway&period=1h"),
        ("Throughput (default)", f"{base_url}/metrics/throughput?service=api-gateway&period=1h&interval=5m"),
        ("Errors (default)", f"{base_url}/metrics/errors?service=api-gateway&period=1h"),
        ("Health (default)", f"{base_url}/health?service=api-gateway"),
        ("Services List", f"{base_url}/services"),
    ]

    results = []
    for name, url in tests:
        success = test_endpoint(name, url)
        results.append((name, success))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\nResults: {passed}/{total} endpoints working ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 All prefilled endpoints work! Ready to use in Streamlit.")
        return 0
    else:
        print("\n⚠️  Some endpoints failed. Check if API server is running:")
        print("    python start_api_server.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
