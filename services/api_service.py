"""
API Service: FastAPI-based metrics REST API.

This service simulates an external metrics API that the agent can query.
It provides endpoints for latency, throughput, errors, and health checks.

Features:
- FastAPI REST endpoints
- Realistic metrics data generation
- Optional integration with DB service for historical data
- CORS enabled for cross-origin requests
- Comprehensive error handling
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import random
import uvicorn


# ============================================================================
# Pydantic Models
# ============================================================================

class ServiceName(str, Enum):
    """Available services."""
    API_GATEWAY = "api-gateway"
    AUTH_SERVICE = "auth-service"
    BUSINESS_LOGIC = "business-logic"
    DATA_PROCESSOR = "data-processor"
    PAYMENT_SERVICE = "payment-service"


class TimePeriod(str, Enum):
    """Time period options."""
    HOUR_1 = "1h"
    HOUR_6 = "6h"
    HOUR_24 = "24h"
    DAY_7 = "7d"


class HealthStatus(str, Enum):
    """Health status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class LatencyMetrics(BaseModel):
    """Latency metrics response."""
    service: str
    period: str
    timestamp: str
    metrics: Dict[str, float] = Field(..., description="Latency percentiles in milliseconds")
    unit: str = "milliseconds"
    sample_count: int


class ThroughputMetrics(BaseModel):
    """Throughput metrics response."""
    service: str
    period: str
    interval: str
    data_points: List[Dict[str, Any]]
    summary: Dict[str, float]


class ErrorMetrics(BaseModel):
    """Error metrics response."""
    service: str
    period: str
    error_rate: float
    total_requests: int
    total_errors: int
    error_breakdown: Dict[str, Any]


class MetricsQuery(BaseModel):
    """Request model for metrics query."""
    services: List[str] = Field(..., description="List of service names")
    metrics: List[str] = Field(..., description="List of metric types: latency, throughput, errors")
    time_range: Dict[str, str] = Field(..., description="Start and end timestamps")
    filters: Optional[Dict[str, str]] = Field(None, description="Additional filters")
    aggregation: Optional[str] = Field("5m", description="Aggregation interval")


class MetricsQueryResponse(BaseModel):
    """Response model for metrics query."""
    query_id: str
    results: List[Dict[str, Any]]


class HealthCheck(BaseModel):
    """Health check response."""
    service: str
    status: str
    timestamp: str
    checks: Dict[str, Any]
    uptime: int
    version: str


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Metrics API",
    description="REST API for service metrics and health monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================

def generate_latency_data(service: str, period: str) -> LatencyMetrics:
    """Generate realistic latency metrics."""
    # Base latency per service
    base_latencies = {
        "api-gateway": {"p50": 25, "p95": 65, "p99": 150},
        "auth-service": {"p50": 40, "p95": 110, "p99": 220},
        "business-logic": {"p50": 80, "p95": 200, "p99": 450},
        "data-processor": {"p50": 15, "p95": 35, "p99": 85},
        "payment-service": {"p50": 120, "p95": 280, "p99": 650}
    }

    base = base_latencies.get(service, {"p50": 50, "p95": 150, "p99": 350})

    # Add variation based on time period
    period_multipliers = {"1h": 1.0, "6h": 1.05, "24h": 1.1, "7d": 1.15}
    multiplier = period_multipliers.get(period, 1.0)

    # Add random variation
    variation = random.uniform(0.85, 1.15)

    metrics = {
        "p50": round(base["p50"] * multiplier * variation, 2),
        "p95": round(base["p95"] * multiplier * variation, 2),
        "p99": round(base["p99"] * multiplier * variation, 2),
        "avg": round(base["p50"] * 1.3 * multiplier * variation, 2),
        "max": round(base["p99"] * 2.5 * multiplier * variation, 2),
        "min": round(base["p50"] * 0.4 * multiplier * variation, 2)
    }

    # Sample count based on period
    sample_counts = {"1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800}

    return LatencyMetrics(
        service=service,
        period=period,
        timestamp=datetime.utcnow().isoformat() + "Z",
        metrics=metrics,
        sample_count=sample_counts.get(period, 3600)
    )


def generate_throughput_data(service: str, period: str, interval: str = "5m") -> ThroughputMetrics:
    """Generate realistic throughput metrics."""
    # Base RPS per service
    base_rps = {
        "api-gateway": 1500,
        "auth-service": 600,
        "business-logic": 800,
        "data-processor": 3000,
        "payment-service": 300
    }

    base = base_rps.get(service, 1000)

    # Generate data points
    intervals = {"1m": 60, "5m": 12, "15m": 4, "1h": 1}
    num_points = intervals.get(interval, 12)

    data_points = []
    total_requests = 0

    now = datetime.utcnow()
    for i in range(num_points):
        timestamp = now - timedelta(minutes=i * 5)

        # Add time-of-day variation
        hour = timestamp.hour
        if 9 <= hour <= 17:
            traffic_mult = random.uniform(1.3, 1.7)  # Business hours
        else:
            traffic_mult = random.uniform(0.6, 0.9)  # Off hours

        rps = int(base * traffic_mult * random.uniform(0.9, 1.1))
        requests = rps * 300  # 5 minutes worth

        data_points.append({
            "timestamp": timestamp.isoformat() + "Z",
            "requests_per_second": rps,
            "total_requests": requests
        })
        total_requests += requests

    # Reverse to get chronological order
    data_points.reverse()

    # Calculate summary
    rps_values = [dp["requests_per_second"] for dp in data_points]
    summary = {
        "avg_rps": round(sum(rps_values) / len(rps_values), 2),
        "peak_rps": max(rps_values),
        "total_requests": total_requests
    }

    return ThroughputMetrics(
        service=service,
        period=period,
        interval=interval,
        data_points=data_points,
        summary=summary
    )


def generate_error_data(service: str, period: str) -> ErrorMetrics:
    """Generate realistic error metrics."""
    # Base error rate per service
    base_error_rates = {
        "api-gateway": 0.008,
        "auth-service": 0.015,
        "business-logic": 0.012,
        "data-processor": 0.005,
        "payment-service": 0.020
    }

    base_rate = base_error_rates.get(service, 0.01)

    # Add random spike occasionally
    if random.random() < 0.1:  # 10% chance of spike
        error_rate = base_rate * random.uniform(2.0, 5.0)
    else:
        error_rate = base_rate * random.uniform(0.8, 1.2)

    error_rate = min(error_rate, 0.15)  # Cap at 15%

    # Calculate totals based on period
    period_requests = {
        "1h": 500000,
        "6h": 3000000,
        "24h": 12000000,
        "7d": 84000000
    }

    total_requests = period_requests.get(period, 500000)
    total_errors = int(total_requests * error_rate)

    # Error breakdown
    error_4xx = int(total_errors * random.uniform(0.75, 0.85))
    error_5xx = total_errors - error_4xx

    error_breakdown = {
        "4xx": {
            "count": error_4xx,
            "percentage": round(100 * error_4xx / total_errors, 2) if total_errors > 0 else 0,
            "codes": {
                "400": int(error_4xx * 0.15),
                "401": int(error_4xx * 0.40),
                "403": int(error_4xx * 0.20),
                "404": int(error_4xx * 0.25)
            }
        },
        "5xx": {
            "count": error_5xx,
            "percentage": round(100 * error_5xx / total_errors, 2) if total_errors > 0 else 0,
            "codes": {
                "500": int(error_5xx * 0.60),
                "502": int(error_5xx * 0.20),
                "503": int(error_5xx * 0.15),
                "504": int(error_5xx * 0.05)
            }
        }
    }

    return ErrorMetrics(
        service=service,
        period=period,
        error_rate=round(error_rate, 4),
        total_requests=total_requests,
        total_errors=total_errors,
        error_breakdown=error_breakdown
    )


def generate_health_data(service: str) -> HealthCheck:
    """Generate health check data."""
    # Randomly determine status (mostly healthy)
    rand = random.random()
    if rand < 0.90:
        status = HealthStatus.HEALTHY
        db_status = "healthy"
        cache_status = "healthy"
        dependencies_healthy = True
    elif rand < 0.97:
        status = HealthStatus.DEGRADED
        db_status = random.choice(["healthy", "degraded"])
        cache_status = random.choice(["healthy", "degraded"])
        dependencies_healthy = random.choice([True, False])
    else:
        status = HealthStatus.UNHEALTHY
        db_status = "unhealthy"
        cache_status = random.choice(["degraded", "unhealthy"])
        dependencies_healthy = False

    checks = {
        "database": {
            "status": db_status,
            "latency": random.randint(2, 10) if db_status == "healthy" else random.randint(50, 200),
            "details": "Connection pool: 45/100" if db_status == "healthy" else "Connection pool exhausted"
        },
        "cache": {
            "status": cache_status,
            "latency": random.randint(1, 5) if cache_status == "healthy" else random.randint(20, 100),
            "details": "Redis cluster: all nodes up" if cache_status == "healthy" else "Redis: degraded performance"
        },
        "dependencies": {
            "auth-service": "healthy" if dependencies_healthy else random.choice(["degraded", "unhealthy"]),
            "business-logic": "healthy" if dependencies_healthy else random.choice(["healthy", "degraded"])
        }
    }

    return HealthCheck(
        service=service,
        status=status.value,
        timestamp=datetime.utcnow().isoformat() + "Z",
        checks=checks,
        uptime=random.randint(86400, 2592000),  # 1 day to 30 days
        version=f"{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 20)}"
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Metrics API",
        "version": "1.0.0",
        "endpoints": [
            "/metrics/latency",
            "/metrics/throughput",
            "/metrics/errors",
            "/metrics/query",
            "/health",
            "/docs"
        ]
    }


@app.get("/metrics/latency", response_model=LatencyMetrics)
async def get_latency_metrics(
    service: str = Query(..., description="Service name"),
    period: str = Query("1h", description="Time period (1h, 6h, 24h, 7d)"),
    aggregation: Optional[str] = Query(None, description="Aggregation method")
):
    """
    Get latency metrics for a service.

    Returns percentiles (p50, p95, p99) and other latency statistics.
    """
    try:
        return generate_latency_data(service, period)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/throughput", response_model=ThroughputMetrics)
async def get_throughput_metrics(
    service: str = Query(..., description="Service name"),
    period: str = Query("1h", description="Time period (1h, 6h, 24h, 7d)"),
    interval: str = Query("5m", description="Data point interval (1m, 5m, 15m, 1h)")
):
    """
    Get throughput metrics for a service.

    Returns requests per second over time with summary statistics.
    """
    try:
        return generate_throughput_data(service, period, interval)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/errors", response_model=ErrorMetrics)
async def get_error_metrics(
    service: str = Query(..., description="Service name"),
    period: str = Query("1h", description="Time period (1h, 6h, 24h, 7d)")
):
    """
    Get error metrics for a service.

    Returns error rates and breakdown by status code.
    """
    try:
        return generate_error_data(service, period)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/metrics/query", response_model=MetricsQueryResponse)
async def query_metrics(query: MetricsQuery):
    """
    Query metrics with complex filters.

    Accepts multiple services and metric types with time range filtering.
    """
    import uuid

    results = []

    for service in query.services:
        service_results = {"service": service}

        for metric_type in query.metrics:
            if metric_type == "latency":
                latency_data = generate_latency_data(service, "1h")
                service_results["latency"] = {"p95": latency_data.metrics["p95"]}

            elif metric_type == "throughput":
                throughput_data = generate_throughput_data(service, "1h")
                service_results["throughput"] = {"avg_rps": throughput_data.summary["avg_rps"]}

            elif metric_type == "errors":
                error_data = generate_error_data(service, "1h")
                service_results["errors"] = {"rate": error_data.error_rate}

        results.append(service_results)

    return MetricsQueryResponse(
        query_id=f"q_{uuid.uuid4().hex[:8]}",
        results=results
    )


@app.get("/health", response_model=HealthCheck)
async def health_check(
    service: str = Query("api-gateway", description="Service to check")
):
    """
    Check health status of a service.

    Returns overall status and component health checks.
    """
    try:
        return generate_health_data(service)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/services")
async def list_services():
    """List all available services."""
    return {
        "services": [
            {
                "name": "api-gateway",
                "status": "running",
                "instances": 5,
                "version": "2.3.1",
                "health": "healthy"
            },
            {
                "name": "auth-service",
                "status": "running",
                "instances": 3,
                "version": "1.8.2",
                "health": "healthy"
            },
            {
                "name": "business-logic",
                "status": "running",
                "instances": 4,
                "version": "3.2.0",
                "health": "healthy"
            },
            {
                "name": "data-processor",
                "status": "running",
                "instances": 10,
                "version": "2.1.5",
                "health": "healthy"
            },
            {
                "name": "payment-service",
                "status": "running",
                "instances": 2,
                "version": "1.5.0",
                "health": "healthy"
            }
        ],
        "total_services": 5,
        "healthy_services": 5
    }


# ============================================================================
# Server Functions
# ============================================================================

def run_server(host: str = "127.0.0.1", port: int = 8001, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "services.api_service:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    print("="*80)
    print("METRICS API SERVICE")
    print("="*80)
    print(f"\nStarting server at http://127.0.0.1:8001")
    print(f"API Docs: http://127.0.0.1:8001/docs")
    print(f"ReDoc: http://127.0.0.1:8001/redoc")
    print("\nEndpoints:")
    print("  GET  /metrics/latency")
    print("  GET  /metrics/throughput")
    print("  GET  /metrics/errors")
    print("  POST /metrics/query")
    print("  GET  /health")
    print("  GET  /services")
    print("\nPress Ctrl+C to stop")
    print("="*80 + "\n")

    run_server(host="127.0.0.1", port=8001, reload=False)
