"""
Database Service: Local SQLite database for service metrics.

Features:
- SQLite database with service metrics
- 100+ rows of realistic time-series data
- Safe SQL query execution with sanitization
- Predefined query templates
- Natural language to SQL conversion
- Connection pooling and error handling
"""

# Load environment variables from .env file (override system env)
from dotenv import load_dotenv
load_dotenv(override=True)

import sqlite3
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import random
from contextlib import contextmanager


@dataclass
class QueryResult:
    """Result from database query."""
    columns: List[str]
    rows: List[Tuple]
    row_count: int
    query: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "columns": self.columns,
            "rows": [dict(zip(self.columns, row)) for row in self.rows],
            "row_count": self.row_count,
            "query": self.query
        }

    def to_list_of_dicts(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries."""
        return [dict(zip(self.columns, row)) for row in self.rows]


class DatabaseService:
    """
    Local SQLite database service for service metrics.

    Provides:
    - Service metrics storage
    - Safe SQL query execution
    - Predefined query templates
    - Natural language to SQL (with OpenAI)
    """

    def __init__(
        self,
        db_path: str = "data/metrics.db",
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize database service.

        Args:
            db_path: Path to SQLite database file
            openai_api_key: OpenAI API key for NL-to-SQL
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # OpenAI client for NL-to-SQL
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=self.api_key)
        else:
            self.openai_client = None

        # Initialize database
        self._initialize_database()

    @contextmanager
    def get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Access columns by name
        try:
            yield conn
        finally:
            conn.close()

    def _initialize_database(self):
        """Initialize database schema and populate with data."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Create table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS service_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    request_count INTEGER,
                    error_count INTEGER,
                    avg_latency REAL,
                    status TEXT,
                    region TEXT,
                    instance_id TEXT
                )
            """)

            # Create indices for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_service_name
                ON service_metrics(service_name)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON service_metrics(timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON service_metrics(status)
            """)

            # Check if data already exists
            cursor.execute("SELECT COUNT(*) FROM service_metrics")
            count = cursor.fetchone()[0]

            if count == 0:
                print("Populating database with sample data...")
                self._populate_sample_data(conn)
            else:
                print(f"Database already contains {count} records")

            conn.commit()

    def _populate_sample_data(self, conn):
        """Populate database with 100+ rows of realistic data."""
        cursor = conn.cursor()

        # Services to track
        services = [
            "api-gateway",
            "auth-service",
            "business-logic",
            "data-processor",
            "payment-service"
        ]

        # Regions
        regions = ["us-east-1", "us-west-2", "eu-west-1"]

        # Statuses
        statuses = ["healthy", "degraded", "unhealthy"]

        # Generate time series data (last 30 days, hourly)
        now = datetime.utcnow()
        data = []

        for service in services:
            # Each service has different baseline metrics
            if service == "api-gateway":
                base_cpu = 45.0
                base_memory = 62.0
                base_requests = 5000
                base_latency = 45.0
            elif service == "auth-service":
                base_cpu = 35.0
                base_memory = 55.0
                base_requests = 2000
                base_latency = 80.0
            elif service == "business-logic":
                base_cpu = 60.0
                base_memory = 70.0
                base_requests = 3000
                base_latency = 150.0
            elif service == "data-processor":
                base_cpu = 75.0
                base_memory = 80.0
                base_requests = 10000
                base_latency = 25.0
            else:  # payment-service
                base_cpu = 40.0
                base_memory = 50.0
                base_requests = 1000
                base_latency = 200.0

            # Generate hourly data for last 7 days (168 hours)
            for hours_ago in range(168):
                timestamp = now - timedelta(hours=hours_ago)

                # Add some realistic variation and patterns
                hour_of_day = timestamp.hour

                # Traffic pattern (higher during business hours)
                if 9 <= hour_of_day <= 17:
                    traffic_multiplier = 1.5
                else:
                    traffic_multiplier = 0.7

                # Simulate some incidents/degradation
                is_incident = random.random() < 0.02  # 2% chance of incident

                if is_incident:
                    cpu = min(95.0, base_cpu * random.uniform(1.5, 2.0))
                    memory = min(95.0, base_memory * random.uniform(1.3, 1.8))
                    requests = int(base_requests * traffic_multiplier * random.uniform(0.5, 1.5))
                    errors = int(requests * random.uniform(0.05, 0.15))  # 5-15% errors
                    latency = base_latency * random.uniform(2.0, 5.0)
                    status = "unhealthy" if random.random() < 0.6 else "degraded"
                else:
                    cpu = base_cpu * random.uniform(0.8, 1.2)
                    memory = base_memory * random.uniform(0.9, 1.1)
                    requests = int(base_requests * traffic_multiplier * random.uniform(0.8, 1.2))
                    errors = int(requests * random.uniform(0.001, 0.01))  # 0.1-1% errors
                    latency = base_latency * random.uniform(0.8, 1.2)
                    status = "healthy" if random.random() < 0.95 else "degraded"

                # Round values
                cpu = round(cpu, 2)
                memory = round(memory, 2)
                latency = round(latency, 2)

                # Select region and instance
                region = random.choice(regions)
                instance_id = f"{service}-{random.randint(1, 5)}"

                data.append((
                    service,
                    timestamp.isoformat(),
                    cpu,
                    memory,
                    requests,
                    errors,
                    latency,
                    status,
                    region,
                    instance_id
                ))

        # Insert all data
        cursor.executemany("""
            INSERT INTO service_metrics (
                service_name, timestamp, cpu_usage, memory_usage,
                request_count, error_count, avg_latency, status,
                region, instance_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)

        print(f"Inserted {len(data)} records into database")

    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        safe_mode: bool = True
    ) -> QueryResult:
        """
        Execute SQL query safely.

        Args:
            query: SQL query to execute
            params: Query parameters (for parameterized queries)
            safe_mode: If True, only allow SELECT queries

        Returns:
            QueryResult object
        """
        # Safety check
        if safe_mode:
            query_upper = query.strip().upper()
            if not query_upper.startswith("SELECT"):
                raise ValueError("Only SELECT queries allowed in safe mode")

            # Block dangerous operations
            dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
            for keyword in dangerous_keywords:
                if keyword in query_upper:
                    raise ValueError(f"Query contains forbidden keyword: {keyword}")

        with self.get_connection() as conn:
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # Fetch results
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Convert sqlite3.Row to tuples
            rows_as_tuples = [tuple(row) for row in rows]

            return QueryResult(
                columns=columns,
                rows=rows_as_tuples,
                row_count=len(rows_as_tuples),
                query=query
            )

    def get_service_metrics(
        self,
        service_name: Optional[str] = None,
        status: Optional[str] = None,
        hours: int = 24,
        limit: int = 100
    ) -> QueryResult:
        """
        Get service metrics with filters.

        Args:
            service_name: Filter by service name
            status: Filter by status
            hours: Look back hours
            limit: Max results

        Returns:
            QueryResult
        """
        query = """
            SELECT * FROM service_metrics
            WHERE timestamp >= datetime('now', ? || ' hours')
        """
        params = [f"-{hours}"]

        if service_name:
            query += " AND service_name = ?"
            params.append(service_name)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        return self.execute_query(query, tuple(params))

    def get_average_metrics(
        self,
        service_name: str,
        hours: int = 24
    ) -> QueryResult:
        """Get average metrics for a service."""
        query = """
            SELECT
                service_name,
                COUNT(*) as sample_count,
                ROUND(AVG(cpu_usage), 2) as avg_cpu,
                ROUND(AVG(memory_usage), 2) as avg_memory,
                SUM(request_count) as total_requests,
                SUM(error_count) as total_errors,
                ROUND(AVG(avg_latency), 2) as avg_latency,
                ROUND(100.0 * SUM(error_count) / SUM(request_count), 4) as error_rate
            FROM service_metrics
            WHERE service_name = ?
                AND timestamp >= datetime('now', ? || ' hours')
            GROUP BY service_name
        """
        params = (service_name, f"-{hours}")
        return self.execute_query(query, params)

    def compare_services(
        self,
        metric: str = "cpu_usage",
        hours: int = 24
    ) -> QueryResult:
        """Compare metric across all services."""
        valid_metrics = ["cpu_usage", "memory_usage", "avg_latency", "error_count"]
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric. Choose from: {valid_metrics}")

        query = f"""
            SELECT
                service_name,
                ROUND(AVG({metric}), 2) as avg_value,
                ROUND(MIN({metric}), 2) as min_value,
                ROUND(MAX({metric}), 2) as max_value
            FROM service_metrics
            WHERE timestamp >= datetime('now', ? || ' hours')
            GROUP BY service_name
            ORDER BY avg_value DESC
        """
        params = (f"-{hours}",)
        return self.execute_query(query, params)

    def get_unhealthy_services(self, hours: int = 1) -> QueryResult:
        """Get services that are currently unhealthy."""
        query = """
            SELECT
                service_name,
                status,
                cpu_usage,
                memory_usage,
                error_count,
                avg_latency,
                timestamp
            FROM service_metrics
            WHERE status IN ('unhealthy', 'degraded')
                AND timestamp >= datetime('now', ? || ' hours')
            ORDER BY timestamp DESC
        """
        params = (f"-{hours}",)
        return self.execute_query(query, params)

    def get_error_spike_services(self, threshold: int = 100, hours: int = 1) -> QueryResult:
        """Get services with error count spikes."""
        query = """
            SELECT
                service_name,
                SUM(error_count) as total_errors,
                SUM(request_count) as total_requests,
                ROUND(100.0 * SUM(error_count) / SUM(request_count), 2) as error_rate
            FROM service_metrics
            WHERE timestamp >= datetime('now', ? || ' hours')
            GROUP BY service_name
            HAVING total_errors > ?
            ORDER BY total_errors DESC
        """
        params = (f"-{hours}", threshold)
        return self.execute_query(query, params)

    def natural_language_query(self, question: str) -> QueryResult:
        """
        Convert natural language question to SQL and execute.

        Args:
            question: Natural language question

        Returns:
            QueryResult
        """
        if not self.openai_client:
            raise ValueError("OpenAI API key required for natural language queries")

        # Get schema info
        schema = self._get_schema_info()

        # Create prompt for NL-to-SQL
        prompt = f"""You are a SQL expert. Convert natural language questions to SQLite queries.

DATABASE SCHEMA:
Table: service_metrics

Columns (name | type | description):
  id              INTEGER   Primary key, auto-increment
  service_name    TEXT      Service identifier (NOT NULL)
  timestamp       TEXT      ISO format timestamp (NOT NULL)
  cpu_usage       REAL      CPU usage percentage (0-100)
  memory_usage    REAL      Memory usage percentage (0-100)
  request_count   INTEGER   Number of requests
  error_count     INTEGER   Number of errors
  avg_latency     REAL      Average latency in milliseconds
  status          TEXT      Service health status
  region          TEXT      Deployment region
  instance_id     TEXT      Instance identifier

VALID VALUES:
  service_name: 'api-gateway', 'auth-service', 'business-logic', 'data-processor', 'payment-service'
  status: 'healthy', 'degraded', 'unhealthy'
  region: 'us-east-1', 'us-west-2', 'eu-west-1'

DATA CHARACTERISTICS:
  - 840 rows covering 7 days of hourly data
  - Timestamps range from 2025-10-08 to 2025-10-15
  - Each service has 168 hourly records

STRICT RULES:
1. Return ONLY the SQL query - no markdown, no explanations, no comments
2. ONLY SELECT queries allowed (no INSERT, UPDATE, DELETE, DROP, ALTER, CREATE)
3. Column names MUST match schema exactly (case-sensitive)
4. Service names MUST match valid values exactly
5. Use datetime('now', '-N hours') for relative time queries
6. Always use ROUND() for floating point aggregates
7. Always use proper GROUP BY with aggregate functions
8. Use LIMIT to prevent excessive results
9. If ambiguous, return: SELECT 'Error: Ambiguous query' as message

SQL EXAMPLES (based on actual schema and data):

Example 1 - Recent metrics for specific service:
Question: "What is the CPU usage for api-gateway?"
SQL:
SELECT service_name, timestamp, cpu_usage, memory_usage, request_count, status
FROM service_metrics
WHERE service_name = 'api-gateway'
ORDER BY timestamp DESC
LIMIT 10

Example 2 - Average metrics over time:
Question: "What was the average CPU usage for auth-service over the last 24 hours?"
SQL:
SELECT service_name,
       ROUND(AVG(cpu_usage), 2) as avg_cpu,
       ROUND(AVG(memory_usage), 2) as avg_memory,
       ROUND(AVG(avg_latency), 2) as avg_latency_ms,
       COUNT(*) as sample_count
FROM service_metrics
WHERE service_name = 'auth-service'
  AND timestamp >= datetime('now', '-24 hours')
GROUP BY service_name

Example 3 - Service comparison:
Question: "Compare memory usage between api-gateway and data-processor"
SQL:
SELECT service_name,
       ROUND(AVG(memory_usage), 2) as avg_memory,
       ROUND(MIN(memory_usage), 2) as min_memory,
       ROUND(MAX(memory_usage), 2) as max_memory,
       COUNT(*) as samples
FROM service_metrics
WHERE service_name IN ('api-gateway', 'data-processor')
  AND timestamp >= datetime('now', '-24 hours')
GROUP BY service_name
ORDER BY avg_memory DESC

Example 4 - Status filtering:
Question: "Show me all unhealthy services in the last hour"
SQL:
SELECT service_name, status, cpu_usage, memory_usage, error_count, avg_latency, timestamp
FROM service_metrics
WHERE status = 'unhealthy'
  AND timestamp >= datetime('now', '-1 hours')
ORDER BY timestamp DESC
LIMIT 50

Example 5 - Error rate analysis:
Question: "Which service has the most errors?"
SQL:
SELECT service_name,
       SUM(error_count) as total_errors,
       SUM(request_count) as total_requests,
       ROUND(100.0 * SUM(error_count) / NULLIF(SUM(request_count), 0), 2) as error_rate_pct,
       COUNT(*) as measurements
FROM service_metrics
WHERE timestamp >= datetime('now', '-24 hours')
GROUP BY service_name
HAVING total_errors > 0
ORDER BY total_errors DESC
LIMIT 10

Example 6 - All services status:
Question: "Show me the status of all services"
SQL:
SELECT service_name,
       status,
       ROUND(AVG(cpu_usage), 2) as avg_cpu,
       ROUND(AVG(memory_usage), 2) as avg_memory,
       MAX(timestamp) as last_check
FROM service_metrics
WHERE timestamp >= datetime('now', '-1 hours')
GROUP BY service_name, status
ORDER BY service_name, timestamp DESC

YOUR TASK:
Convert this question to SQL following ALL rules and examples above.

Question: {question}

SQL Query:"""

        # Get SQL from OpenAI
        response = self.openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a SQL expert specializing in SQLite queries for the service_metrics table.

CRITICAL RULES:
1. Output ONLY the SQL query - no explanations, no markdown, no comments
2. Column names are case-sensitive: service_name, timestamp, cpu_usage, memory_usage, request_count, error_count, avg_latency, status, region, instance_id
3. Service names are exact: 'api-gateway', 'auth-service', 'business-logic', 'data-processor', 'payment-service'
4. Use ROUND() for all floating point aggregates
5. Use NULLIF() to prevent division by zero
6. Use datetime('now', '-N hours') for time filters
7. Always include ORDER BY and LIMIT for safety

Study the 6 examples carefully and follow their patterns exactly."""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )

        sql_query = response.choices[0].message.content.strip()

        # Clean up query (remove markdown code blocks if present)
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

        print(f"Generated SQL: {sql_query}")

        # Execute query
        return self.execute_query(sql_query, safe_mode=True)

    def _get_schema_info(self) -> str:
        """Get database schema information."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(service_metrics)")
            columns = cursor.fetchall()

            schema = "Table: service_metrics\nColumns:\n"
            for col in columns:
                schema += f"  - {col[1]} ({col[2]})\n"

            # Add sample row
            cursor.execute("SELECT * FROM service_metrics LIMIT 1")
            sample = cursor.fetchone()
            if sample:
                schema += "\nSample row:\n"
                for i, col in enumerate(columns):
                    schema += f"  {col[1]}: {sample[i]}\n"

            return schema

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Total records
            cursor.execute("SELECT COUNT(*) FROM service_metrics")
            total_records = cursor.fetchone()[0]

            # Unique services
            cursor.execute("SELECT COUNT(DISTINCT service_name) FROM service_metrics")
            unique_services = cursor.fetchone()[0]

            # Time range
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM service_metrics")
            min_ts, max_ts = cursor.fetchone()

            # Service breakdown
            cursor.execute("""
                SELECT service_name, COUNT(*) as count
                FROM service_metrics
                GROUP BY service_name
                ORDER BY count DESC
            """)
            service_counts = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "total_records": total_records,
                "unique_services": unique_services,
                "time_range": {
                    "start": min_ts,
                    "end": max_ts
                },
                "service_counts": service_counts,
                "database_path": str(self.db_path)
            }

    def reset_database(self):
        """Drop and recreate database (for testing)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS service_metrics")
            conn.commit()

        self._initialize_database()


def main():
    """Example usage of database service."""

    print("="*80)
    print("DATABASE SERVICE DEMO")
    print("="*80)

    # Initialize database
    db = DatabaseService(db_path="data/metrics.db")

    # Show stats
    print("\n1. Database Statistics:")
    print("-" * 80)
    stats = db.get_stats()
    for key, value in stats.items():
        if key != "service_counts":
            print(f"  {key}: {value}")

    print("\n  Service breakdown:")
    for service, count in stats["service_counts"].items():
        print(f"    {service}: {count} records")

    # Example queries
    print("\n2. Recent Metrics for API Gateway:")
    print("-" * 80)
    result = db.get_service_metrics("api-gateway", hours=1, limit=5)
    for row in result.to_list_of_dicts():
        print(f"  {row['timestamp']}: CPU={row['cpu_usage']}%, "
              f"Memory={row['memory_usage']}%, Latency={row['avg_latency']}ms")

    print("\n3. Average Metrics (Last 24h):")
    print("-" * 80)
    result = db.get_average_metrics("auth-service", hours=24)
    if result.rows:
        data = result.to_list_of_dicts()[0]
        print(f"  Service: {data['service_name']}")
        print(f"  Avg CPU: {data['avg_cpu']}%")
        print(f"  Avg Memory: {data['avg_memory']}%")
        print(f"  Avg Latency: {data['avg_latency']}ms")
        print(f"  Error Rate: {data['error_rate']}%")

    print("\n4. Compare CPU Usage Across Services:")
    print("-" * 80)
    result = db.compare_services("cpu_usage", hours=24)
    for row in result.to_list_of_dicts():
        print(f"  {row['service_name']}: avg={row['avg_value']}%, "
              f"min={row['min_value']}%, max={row['max_value']}%")

    print("\n5. Unhealthy Services:")
    print("-" * 80)
    result = db.get_unhealthy_services(hours=1)
    if result.rows:
        for row in result.to_list_of_dicts()[:5]:
            print(f"  {row['service_name']} ({row['status']}): "
                  f"CPU={row['cpu_usage']}%, Errors={row['error_count']}")
    else:
        print("  No unhealthy services found")

    print("\n6. Custom SQL Query:")
    print("-" * 80)
    custom_query = """
        SELECT service_name, status, COUNT(*) as count
        FROM service_metrics
        WHERE timestamp >= datetime('now', '-24 hours')
        GROUP BY service_name, status
        ORDER BY service_name, status
    """
    result = db.execute_query(custom_query)
    for row in result.to_list_of_dicts():
        print(f"  {row['service_name']} - {row['status']}: {row['count']} records")

    print("\n" + "="*80)
    print("✓ DATABASE SERVICE DEMO COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
