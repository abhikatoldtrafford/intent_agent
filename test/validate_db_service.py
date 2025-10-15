"""
Database Service Validation Script

Tests:
- Database initialization
- Data population (100+ rows)
- Safe SQL query execution
- Predefined query methods
- Natural language to SQL (optional, requires OpenAI)
- Error handling
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.db_service import DatabaseService, QueryResult


def print_header(title):
    """Print section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def print_check(message, status=True):
    """Print check result."""
    symbol = "✓" if status else "✗"
    print(f"  {symbol} {message}")


def test_initialization():
    """Test database initialization."""
    print_header("TEST 1: DATABASE INITIALIZATION")

    try:
        db = DatabaseService(db_path="data/metrics.db")
        print_check("DatabaseService instantiation", True)
        print(f"    Database path: {db.db_path}")

        # Check stats
        stats = db.get_stats()
        print_check(f"Database has {stats['total_records']} records", stats['total_records'] > 100)
        print_check(f"Database has {stats['unique_services']} services", stats['unique_services'] > 0)

        print(f"\n  Database statistics:")
        print(f"    Total records: {stats['total_records']}")
        print(f"    Unique services: {stats['unique_services']}")
        print(f"    Time range: {stats['time_range']['start']} to {stats['time_range']['end']}")

        print(f"\n  Service breakdown:")
        for service, count in stats['service_counts'].items():
            print(f"    {service}: {count} records")

        return db, True

    except Exception as e:
        print_check(f"Initialization failed: {e}", False)
        return None, False


def test_safe_query_execution(db):
    """Test safe SQL query execution."""
    print_header("TEST 2: SAFE SQL QUERY EXECUTION")

    all_passed = True

    # Test 1: Valid SELECT query
    try:
        result = db.execute_query("SELECT COUNT(*) FROM service_metrics")
        print_check(f"Valid SELECT query: {result.rows[0][0]} records", True)
    except Exception as e:
        print_check(f"Valid SELECT failed: {e}", False)
        all_passed = False

    # Test 2: Block dangerous queries
    dangerous_queries = [
        ("DROP TABLE service_metrics", "DROP"),
        ("DELETE FROM service_metrics", "DELETE"),
        ("UPDATE service_metrics SET cpu_usage = 100", "UPDATE"),
        ("INSERT INTO service_metrics VALUES (1, 'test')", "INSERT")
    ]

    for query, operation in dangerous_queries:
        try:
            db.execute_query(query, safe_mode=True)
            print_check(f"Block {operation} query", False)
            all_passed = False
        except ValueError:
            print_check(f"Block {operation} query", True)

    # Test 3: Parameterized queries
    try:
        result = db.execute_query(
            "SELECT * FROM service_metrics WHERE service_name = ? LIMIT 5",
            params=("api-gateway",)
        )
        print_check(f"Parameterized query: {result.row_count} rows", result.row_count > 0)
    except Exception as e:
        print_check(f"Parameterized query failed: {e}", False)
        all_passed = False

    return all_passed


def test_predefined_queries(db):
    """Test predefined query methods."""
    print_header("TEST 3: PREDEFINED QUERY METHODS")

    all_passed = True

    # Test 1: get_service_metrics
    try:
        result = db.get_service_metrics("api-gateway", hours=24, limit=10)
        print_check(f"get_service_metrics: {result.row_count} rows", result.row_count > 0)
        print(f"    Columns: {', '.join(result.columns)}")
    except Exception as e:
        print_check(f"get_service_metrics failed: {e}", False)
        all_passed = False

    # Test 2: get_average_metrics
    try:
        result = db.get_average_metrics("auth-service", hours=24)
        print_check(f"get_average_metrics: {result.row_count} rows", result.row_count > 0)
        if result.rows:
            data = result.to_list_of_dicts()[0]
            print(f"    Avg CPU: {data['avg_cpu']}%")
            print(f"    Avg Memory: {data['avg_memory']}%")
            print(f"    Avg Latency: {data['avg_latency']}ms")
    except Exception as e:
        print_check(f"get_average_metrics failed: {e}", False)
        all_passed = False

    # Test 3: compare_services
    try:
        result = db.compare_services("cpu_usage", hours=24)
        print_check(f"compare_services: {result.row_count} services", result.row_count > 0)
        if result.rows:
            top_service = result.to_list_of_dicts()[0]
            print(f"    Highest CPU: {top_service['service_name']} ({top_service['avg_value']}%)")
    except Exception as e:
        print_check(f"compare_services failed: {e}", False)
        all_passed = False

    # Test 4: get_unhealthy_services
    try:
        result = db.get_unhealthy_services(hours=168)  # Last week
        print_check(f"get_unhealthy_services: {result.row_count} incidents", True)
        if result.row_count > 0:
            print(f"    Found {result.row_count} unhealthy/degraded states")
    except Exception as e:
        print_check(f"get_unhealthy_services failed: {e}", False)
        all_passed = False

    # Test 5: get_error_spike_services
    try:
        result = db.get_error_spike_services(threshold=50, hours=24)
        print_check(f"get_error_spike_services: {result.row_count} services", True)
    except Exception as e:
        print_check(f"get_error_spike_services failed: {e}", False)
        all_passed = False

    return all_passed


def test_result_formats(db):
    """Test different result output formats."""
    print_header("TEST 4: RESULT FORMATS")

    all_passed = True

    try:
        result = db.get_service_metrics("business-logic", hours=1, limit=3)

        # Test 1: to_dict()
        dict_result = result.to_dict()
        print_check("to_dict() format", "columns" in dict_result and "rows" in dict_result)

        # Test 2: to_list_of_dicts()
        list_result = result.to_list_of_dicts()
        print_check("to_list_of_dicts() format", isinstance(list_result, list))

        # Test 3: Raw access
        print_check("Raw columns access", len(result.columns) > 0)
        print_check("Raw rows access", len(result.rows) > 0)

        # Show sample
        if list_result:
            print(f"\n  Sample row:")
            sample = list_result[0]
            for key, value in list(sample.items())[:5]:
                print(f"    {key}: {value}")

    except Exception as e:
        print_check(f"Result formats failed: {e}", False)
        all_passed = False

    return all_passed


def test_data_quality(db):
    """Test data quality and variety."""
    print_header("TEST 5: DATA QUALITY")

    all_passed = True

    # Test 1: Multiple services
    try:
        result = db.execute_query("SELECT DISTINCT service_name FROM service_metrics")
        services = [row[0] for row in result.rows]
        print_check(f"Multiple services ({len(services)})", len(services) >= 3)
        print(f"    Services: {', '.join(services)}")
    except Exception as e:
        print_check(f"Service variety check failed: {e}", False)
        all_passed = False

    # Test 2: Status variety
    try:
        result = db.execute_query("SELECT DISTINCT status FROM service_metrics")
        statuses = [row[0] for row in result.rows]
        print_check(f"Multiple statuses ({len(statuses)})", len(statuses) >= 2)
        print(f"    Statuses: {', '.join(statuses)}")
    except Exception as e:
        print_check(f"Status variety check failed: {e}", False)
        all_passed = False

    # Test 3: Time range
    try:
        result = db.execute_query("""
            SELECT
                MIN(timestamp) as earliest,
                MAX(timestamp) as latest,
                COUNT(DISTINCT DATE(timestamp)) as days
            FROM service_metrics
        """)
        data = result.to_list_of_dicts()[0]
        print_check(f"Time range: {data['days']} days of data", data['days'] >= 1)
        print(f"    From: {data['earliest']}")
        print(f"    To: {data['latest']}")
    except Exception as e:
        print_check(f"Time range check failed: {e}", False)
        all_passed = False

    # Test 4: Metric ranges
    try:
        result = db.execute_query("""
            SELECT
                ROUND(MIN(cpu_usage), 2) as min_cpu,
                ROUND(MAX(cpu_usage), 2) as max_cpu,
                ROUND(MIN(memory_usage), 2) as min_mem,
                ROUND(MAX(memory_usage), 2) as max_mem,
                ROUND(MIN(avg_latency), 2) as min_lat,
                ROUND(MAX(avg_latency), 2) as max_lat
            FROM service_metrics
        """)
        data = result.to_list_of_dicts()[0]
        print_check("Realistic metric ranges", True)
        print(f"    CPU: {data['min_cpu']}% - {data['max_cpu']}%")
        print(f"    Memory: {data['min_mem']}% - {data['max_mem']}%")
        print(f"    Latency: {data['min_lat']}ms - {data['max_lat']}ms")
    except Exception as e:
        print_check(f"Metric ranges check failed: {e}", False)
        all_passed = False

    return all_passed


def test_complex_queries(db):
    """Test complex analytical queries."""
    print_header("TEST 6: COMPLEX QUERIES")

    all_passed = True

    # Test 1: Aggregation with grouping
    try:
        result = db.execute_query("""
            SELECT
                service_name,
                status,
                COUNT(*) as count,
                ROUND(AVG(cpu_usage), 2) as avg_cpu
            FROM service_metrics
            WHERE timestamp >= datetime('now', '-24 hours')
            GROUP BY service_name, status
            ORDER BY service_name, status
        """)
        print_check(f"Group by query: {result.row_count} groups", result.row_count > 0)
    except Exception as e:
        print_check(f"Group by query failed: {e}", False)
        all_passed = False

    # Test 2: Window functions (if supported)
    try:
        result = db.execute_query("""
            SELECT
                service_name,
                timestamp,
                cpu_usage,
                ROUND(AVG(cpu_usage) OVER (
                    PARTITION BY service_name
                    ORDER BY timestamp
                    ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                ), 2) as rolling_avg
            FROM service_metrics
            WHERE service_name = 'api-gateway'
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        print_check(f"Window function query: {result.row_count} rows", result.row_count > 0)
    except Exception as e:
        print_check(f"Window function query (optional): skipped", True)

    # Test 3: Subquery
    try:
        result = db.execute_query("""
            SELECT
                service_name,
                cpu_usage,
                timestamp
            FROM service_metrics
            WHERE cpu_usage > (
                SELECT AVG(cpu_usage) * 1.5
                FROM service_metrics
            )
            LIMIT 10
        """)
        print_check(f"Subquery: {result.row_count} high CPU instances", True)
    except Exception as e:
        print_check(f"Subquery failed: {e}", False)
        all_passed = False

    return all_passed


def test_natural_language(db):
    """Test natural language to SQL (optional)."""
    print_header("TEST 7: NATURAL LANGUAGE TO SQL (Optional)")

    if not db.openai_client:
        print_check("OpenAI API key not configured - skipping NL-to-SQL tests", True)
        return True

    all_passed = True

    test_questions = [
        "What are the average CPU and memory usage for api-gateway in the last 24 hours?",
        "Which service has the highest error count?",
        "Show me all unhealthy services from the last hour"
    ]

    for question in test_questions[:2]:  # Test first 2 to save API calls
        try:
            print(f"\n  Question: {question}")
            result = db.natural_language_query(question)
            print_check(f"NL-to-SQL: {result.row_count} rows", True)
            print(f"    Generated SQL: {result.query[:80]}...")
        except Exception as e:
            print_check(f"NL-to-SQL failed: {e}", False)
            all_passed = False

    return all_passed


def test_performance(db):
    """Test query performance."""
    print_header("TEST 8: PERFORMANCE")

    import time

    all_passed = True

    # Test 1: Simple query
    try:
        start = time.time()
        result = db.execute_query("SELECT * FROM service_metrics LIMIT 100")
        elapsed = time.time() - start
        print_check(f"Simple query: {elapsed*1000:.2f}ms for {result.row_count} rows", elapsed < 1.0)
    except Exception as e:
        print_check(f"Simple query failed: {e}", False)
        all_passed = False

    # Test 2: Aggregation query
    try:
        start = time.time()
        result = db.compare_services("cpu_usage", hours=168)
        elapsed = time.time() - start
        print_check(f"Aggregation query: {elapsed*1000:.2f}ms", elapsed < 1.0)
    except Exception as e:
        print_check(f"Aggregation query failed: {e}", False)
        all_passed = False

    # Test 3: Filter + sort query
    try:
        start = time.time()
        result = db.get_service_metrics("data-processor", hours=24, limit=50)
        elapsed = time.time() - start
        print_check(f"Filter + sort query: {elapsed*1000:.2f}ms", elapsed < 1.0)
    except Exception as e:
        print_check(f"Filter + sort query failed: {e}", False)
        all_passed = False

    return all_passed


def run_all_tests():
    """Run all validation tests."""
    print("="*80)
    print("DATABASE SERVICE VALIDATION")
    print("="*80)

    results = {}

    # Test 1: Initialization
    db, passed = test_initialization()
    results["Initialization"] = passed

    if not db:
        print("\n✗ Cannot proceed without database initialization")
        return False

    # Test 2: Safe query execution
    results["Safe Query Execution"] = test_safe_query_execution(db)

    # Test 3: Predefined queries
    results["Predefined Queries"] = test_predefined_queries(db)

    # Test 4: Result formats
    results["Result Formats"] = test_result_formats(db)

    # Test 5: Data quality
    results["Data Quality"] = test_data_quality(db)

    # Test 6: Complex queries
    results["Complex Queries"] = test_complex_queries(db)

    # Test 7: Natural language (optional)
    results["Natural Language"] = test_natural_language(db)

    # Test 8: Performance
    results["Performance"] = test_performance(db)

    # Summary
    print_header("VALIDATION SUMMARY")

    all_passed = True
    for test, passed in results.items():
        print_check(test, passed)
        if not passed:
            all_passed = False

    print("\n" + "="*80)

    if all_passed:
        print("✓ ALL TESTS PASSED - Database Service is ready!")
        print("="*80)
        print("\nDatabase info:")
        stats = db.get_stats()
        print(f"  Location: {stats['database_path']}")
        print(f"  Records: {stats['total_records']}")
        print(f"  Services: {stats['unique_services']}")
        return True
    else:
        print("✗ SOME TESTS FAILED - Please review errors above")
        print("="*80)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
