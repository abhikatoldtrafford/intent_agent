"""
API Service Validation Script

Tests all API endpoints to ensure they're working correctly.
Requires the API server to be running on localhost:8001
"""

import sys
import requests
import time
from pathlib import Path


API_BASE_URL = "http://127.0.0.1:8001"


def print_header(title):
    """Print section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def print_check(message, status=True):
    """Print check result."""
    symbol = "✓" if status else "✗"
    print(f"  {symbol} {message}")


def check_server_running():
    """Check if API server is running."""
    print_header("CHECK 1: SERVER STATUS")

    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        if response.status_code == 200:
            print_check("API server is running", True)
            data = response.json()
            print(f"    Service: {data.get('service')}")
            print(f"    Version: {data.get('version')}")
            return True
        else:
            print_check(f"Server returned status {response.status_code}", False)
            return False
    except requests.exceptions.ConnectionError:
        print_check("Cannot connect to API server", False)
        print("\n  Please start the server first:")
        print("    python start_api_server.py")
        print("  Or:")
        print("    python -m services.api_service")
        return False
    except Exception as e:
        print_check(f"Error: {e}", False)
        return False


def test_latency_endpoint():
    """Test latency metrics endpoint."""
    print_header("TEST 2: LATENCY METRICS")

    try:
        response = requests.get(
            f"{API_BASE_URL}/metrics/latency",
            params={"service": "api-gateway", "period": "1h"}
        )

        if response.status_code == 200:
            data = response.json()
            print_check("GET /metrics/latency - Success", True)
            print(f"    Service: {data['service']}")
            print(f"    P50: {data['metrics']['p50']}ms")
            print(f"    P95: {data['metrics']['p95']}ms")
            print(f"    P99: {data['metrics']['p99']}ms")
            print(f"    Samples: {data['sample_count']}")
            return True
        else:
            print_check(f"Status {response.status_code}", False)
            return False

    except Exception as e:
        print_check(f"Error: {e}", False)
        return False


def test_throughput_endpoint():
    """Test throughput metrics endpoint."""
    print_header("TEST 3: THROUGHPUT METRICS")

    try:
        response = requests.get(
            f"{API_BASE_URL}/metrics/throughput",
            params={"service": "auth-service", "period": "1h", "interval": "5m"}
        )

        if response.status_code == 200:
            data = response.json()
            print_check("GET /metrics/throughput - Success", True)
            print(f"    Service: {data['service']}")
            print(f"    Data points: {len(data['data_points'])}")
            print(f"    Avg RPS: {data['summary']['avg_rps']}")
            print(f"    Peak RPS: {data['summary']['peak_rps']}")
            print(f"    Total requests: {data['summary']['total_requests']}")
            return True
        else:
            print_check(f"Status {response.status_code}", False)
            return False

    except Exception as e:
        print_check(f"Error: {e}", False)
        return False


def test_errors_endpoint():
    """Test error metrics endpoint."""
    print_header("TEST 4: ERROR METRICS")

    try:
        response = requests.get(
            f"{API_BASE_URL}/metrics/errors",
            params={"service": "business-logic", "period": "24h"}
        )

        if response.status_code == 200:
            data = response.json()
            print_check("GET /metrics/errors - Success", True)
            print(f"    Service: {data['service']}")
            print(f"    Error rate: {data['error_rate']*100:.2f}%")
            print(f"    Total requests: {data['total_requests']:,}")
            print(f"    Total errors: {data['total_errors']:,}")
            print(f"    4xx errors: {data['error_breakdown']['4xx']['count']:,}")
            print(f"    5xx errors: {data['error_breakdown']['5xx']['count']:,}")
            return True
        else:
            print_check(f"Status {response.status_code}", False)
            return False

    except Exception as e:
        print_check(f"Error: {e}", False)
        return False


def test_query_endpoint():
    """Test metrics query endpoint."""
    print_header("TEST 5: METRICS QUERY (POST)")

    try:
        payload = {
            "services": ["api-gateway", "auth-service"],
            "metrics": ["latency", "throughput", "errors"],
            "time_range": {
                "start": "2025-10-14T00:00:00Z",
                "end": "2025-10-15T00:00:00Z"
            },
            "aggregation": "5m"
        }

        response = requests.post(
            f"{API_BASE_URL}/metrics/query",
            json=payload
        )

        if response.status_code == 200:
            data = response.json()
            print_check("POST /metrics/query - Success", True)
            print(f"    Query ID: {data['query_id']}")
            print(f"    Results: {len(data['results'])} services")

            for result in data['results']:
                print(f"\n    {result['service']}:")
                if 'latency' in result:
                    print(f"      Latency P95: {result['latency']['p95']}ms")
                if 'throughput' in result:
                    print(f"      Avg RPS: {result['throughput']['avg_rps']}")
                if 'errors' in result:
                    print(f"      Error rate: {result['errors']['rate']*100:.2f}%")

            return True
        else:
            print_check(f"Status {response.status_code}", False)
            return False

    except Exception as e:
        print_check(f"Error: {e}", False)
        return False


def test_health_endpoint():
    """Test health check endpoint."""
    print_header("TEST 6: HEALTH CHECK")

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            params={"service": "data-processor"}
        )

        if response.status_code == 200:
            data = response.json()
            print_check("GET /health - Success", True)
            print(f"    Service: {data['service']}")
            print(f"    Status: {data['status']}")
            print(f"    Version: {data['version']}")
            print(f"    Uptime: {data['uptime']} seconds")

            print(f"\n    Component health:")
            for component, details in data['checks'].items():
                if isinstance(details, dict) and 'status' in details:
                    print(f"      {component}: {details['status']}")

            return True
        else:
            print_check(f"Status {response.status_code}", False)
            return False

    except Exception as e:
        print_check(f"Error: {e}", False)
        return False


def test_services_list():
    """Test services list endpoint."""
    print_header("TEST 7: SERVICES LIST")

    try:
        response = requests.get(f"{API_BASE_URL}/services")

        if response.status_code == 200:
            data = response.json()
            print_check("GET /services - Success", True)
            print(f"    Total services: {data['total_services']}")
            print(f"    Healthy services: {data['healthy_services']}")

            print(f"\n    Services:")
            for service in data['services']:
                print(f"      - {service['name']}: {service['instances']} instances (v{service['version']})")

            return True
        else:
            print_check(f"Status {response.status_code}", False)
            return False

    except Exception as e:
        print_check(f"Error: {e}", False)
        return False


def test_performance():
    """Test response time performance."""
    print_header("TEST 8: PERFORMANCE")

    endpoints = [
        ("/metrics/latency?service=api-gateway&period=1h", "Latency"),
        ("/metrics/throughput?service=auth-service&period=1h", "Throughput"),
        ("/metrics/errors?service=business-logic&period=1h", "Errors"),
        ("/health?service=data-processor", "Health"),
    ]

    all_passed = True

    for endpoint, name in endpoints:
        try:
            start = time.time()
            response = requests.get(f"{API_BASE_URL}{endpoint}")
            elapsed = time.time() - start

            if response.status_code == 200:
                print_check(f"{name}: {elapsed*1000:.2f}ms", elapsed < 1.0)
                if elapsed >= 1.0:
                    all_passed = False
            else:
                print_check(f"{name}: Failed", False)
                all_passed = False

        except Exception as e:
            print_check(f"{name}: Error - {e}", False)
            all_passed = False

    return all_passed


def run_all_tests():
    """Run all validation tests."""
    print("="*80)
    print("API SERVICE VALIDATION")
    print("="*80)

    results = {}

    # Check server first
    if not check_server_running():
        print("\n" + "="*80)
        print("✗ SERVER NOT RUNNING - Cannot proceed with tests")
        print("="*80)
        return False

    # Run tests
    results["Latency Endpoint"] = test_latency_endpoint()
    results["Throughput Endpoint"] = test_throughput_endpoint()
    results["Errors Endpoint"] = test_errors_endpoint()
    results["Query Endpoint"] = test_query_endpoint()
    results["Health Endpoint"] = test_health_endpoint()
    results["Services List"] = test_services_list()
    results["Performance"] = test_performance()

    # Summary
    print_header("VALIDATION SUMMARY")

    all_passed = True
    for test, passed in results.items():
        print_check(test, passed)
        if not passed:
            all_passed = False

    print("\n" + "="*80)

    if all_passed:
        print("✓ ALL TESTS PASSED - API Service is working correctly!")
        print("="*80)
        print("\nAPI is ready for agent integration")
        print(f"Base URL: {API_BASE_URL}")
        print("Documentation: http://127.0.0.1:8001/docs")
        return True
    else:
        print("✗ SOME TESTS FAILED - Please review errors above")
        print("="*80)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
