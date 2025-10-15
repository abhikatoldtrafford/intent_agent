# API Service Documentation

## Overview

The API Service is a FastAPI-based REST API that provides metrics endpoints for service monitoring. It simulates an external metrics API that the agent can query via HTTP.

## Architecture Decision

**Why REST API for Metrics?**

The agent uses **three different service access patterns**:

1. **RAG Service** → Direct Python calls (local, fast)
2. **DB Service** → Direct Python calls (local, fast)
3. **API Service** → HTTP REST calls (demonstrates external API integration)

This design:
- Shows both local and remote service access patterns
- Demonstrates HTTP/REST tool usage
- Simulates real-world external API interactions
- Keeps RAG/DB fast with direct calls

## Features

- **FastAPI** framework with automatic OpenAPI docs
- **6 endpoints** for metrics and health checks
- **Realistic data generation** with variation and patterns
- **CORS enabled** for cross-origin requests
- **Pydantic models** for request/response validation
- **Auto-generated docs** at `/docs` and `/redoc`

## Installation

```bash
# Install dependencies
pip install -r requirements-api.txt

# Or install individually
pip install fastapi uvicorn pydantic requests
```

## Starting the Server

### Option 1: Using the startup script
```bash
python start_api_server.py
```

### Option 2: Direct execution
```bash
python -m services.api_service
```

### Option 3: Using uvicorn
```bash
uvicorn services.api_service:app --host 127.0.0.1 --port 8001 --reload
```

The server will start at:
- **Base URL**: http://127.0.0.1:8001
- **API Docs**: http://127.0.0.1:8001/docs
- **ReDoc**: http://127.0.0.1:8001/redoc

## API Endpoints

### 1. GET /metrics/latency

Get latency metrics for a service.

**Parameters:**
- `service` (required): Service name
- `period` (optional): Time period (1h, 6h, 24h, 7d), default: 1h
- `aggregation` (optional): Aggregation method

**Example:**
```bash
curl "http://127.0.0.1:8001/metrics/latency?service=api-gateway&period=1h"
```

**Response:**
```json
{
  "service": "api-gateway",
  "period": "1h",
  "timestamp": "2025-10-15T10:30:00Z",
  "metrics": {
    "p50": 25.3,
    "p95": 68.7,
    "p99": 156.2,
    "avg": 34.8,
    "max": 425.5,
    "min": 10.2
  },
  "unit": "milliseconds",
  "sample_count": 3600
}
```

### 2. GET /metrics/throughput

Get throughput metrics for a service.

**Parameters:**
- `service` (required): Service name
- `period` (optional): Time period, default: 1h
- `interval` (optional): Data point interval (1m, 5m, 15m, 1h), default: 5m

**Example:**
```bash
curl "http://127.0.0.1:8001/metrics/throughput?service=auth-service&period=1h&interval=5m"
```

**Response:**
```json
{
  "service": "auth-service",
  "period": "1h",
  "interval": "5m",
  "data_points": [
    {
      "timestamp": "2025-10-15T09:00:00Z",
      "requests_per_second": 645,
      "total_requests": 193500
    },
    ...
  ],
  "summary": {
    "avg_rps": 612.5,
    "peak_rps": 892,
    "total_requests": 2205000
  }
}
```

### 3. GET /metrics/errors

Get error metrics for a service.

**Parameters:**
- `service` (required): Service name
- `period` (optional): Time period, default: 1h

**Example:**
```bash
curl "http://127.0.0.1:8001/metrics/errors?service=business-logic&period=24h"
```

**Response:**
```json
{
  "service": "business-logic",
  "period": "24h",
  "error_rate": 0.0124,
  "total_requests": 12000000,
  "total_errors": 148800,
  "error_breakdown": {
    "4xx": {
      "count": 119040,
      "percentage": 80.0,
      "codes": {
        "400": 17856,
        "401": 47616,
        "403": 23808,
        "404": 29760
      }
    },
    "5xx": {
      "count": 29760,
      "percentage": 20.0,
      "codes": {
        "500": 17856,
        "502": 5952,
        "503": 4464,
        "504": 1488
      }
    }
  }
}
```

### 4. POST /metrics/query

Query metrics with complex filters.

**Request Body:**
```json
{
  "services": ["api-gateway", "auth-service"],
  "metrics": ["latency", "throughput", "errors"],
  "time_range": {
    "start": "2025-10-14T00:00:00Z",
    "end": "2025-10-15T00:00:00Z"
  },
  "filters": {
    "region": "us-east-1"
  },
  "aggregation": "5m"
}
```

**Example:**
```bash
curl -X POST "http://127.0.0.1:8001/metrics/query" \
  -H "Content-Type: application/json" \
  -d '{
    "services": ["api-gateway"],
    "metrics": ["latency", "errors"],
    "time_range": {
      "start": "2025-10-14T00:00:00Z",
      "end": "2025-10-15T00:00:00Z"
    }
  }'
```

**Response:**
```json
{
  "query_id": "q_abc12345",
  "results": [
    {
      "service": "api-gateway",
      "latency": {"p95": 68.7},
      "errors": {"rate": 0.008}
    }
  ]
}
```

### 5. GET /health

Check health status of a service.

**Parameters:**
- `service` (optional): Service name, default: api-gateway

**Example:**
```bash
curl "http://127.0.0.1:8001/health?service=data-processor"
```

**Response:**
```json
{
  "service": "data-processor",
  "status": "healthy",
  "timestamp": "2025-10-15T10:30:00Z",
  "checks": {
    "database": {
      "status": "healthy",
      "latency": 5,
      "details": "Connection pool: 45/100"
    },
    "cache": {
      "status": "healthy",
      "latency": 2,
      "details": "Redis cluster: all nodes up"
    },
    "dependencies": {
      "auth-service": "healthy",
      "business-logic": "healthy"
    }
  },
  "uptime": 604800,
  "version": "2.1.5"
}
```

### 6. GET /services

List all available services.

**Example:**
```bash
curl "http://127.0.0.1:8001/services"
```

**Response:**
```json
{
  "services": [
    {
      "name": "api-gateway",
      "status": "running",
      "instances": 5,
      "version": "2.3.1",
      "health": "healthy"
    },
    ...
  ],
  "total_services": 5,
  "healthy_services": 5
}
```

## Data Generation

The API generates realistic metrics with:

**Latency Patterns:**
- Different baseline per service
- Time period variations
- Random fluctuations (±15%)

**Throughput Patterns:**
- Business hours (9-17): Higher traffic (1.3-1.7x)
- Off hours: Lower traffic (0.6-0.9x)
- Random variations

**Error Patterns:**
- Base error rates per service (0.5-2%)
- Occasional spikes (2-5x)
- 4xx/5xx breakdown (~80%/20%)

**Health Patterns:**
- 90% healthy
- 7% degraded
- 3% unhealthy

## Python Client Usage

```python
import requests

BASE_URL = "http://127.0.0.1:8001"

# Get latency metrics
response = requests.get(
    f"{BASE_URL}/metrics/latency",
    params={"service": "api-gateway", "period": "1h"}
)
data = response.json()
print(f"P95 Latency: {data['metrics']['p95']}ms")

# Query multiple metrics
response = requests.post(
    f"{BASE_URL}/metrics/query",
    json={
        "services": ["api-gateway", "auth-service"],
        "metrics": ["latency", "throughput"],
        "time_range": {
            "start": "2025-10-14T00:00:00Z",
            "end": "2025-10-15T00:00:00Z"
        }
    }
)
results = response.json()
for result in results["results"]:
    print(f"{result['service']}: {result}")
```

## Testing

### Start the server
```bash
python start_api_server.py
```

### In another terminal, run validation
```bash
python tests/validate_api_service.py
```

### Manual testing
Visit http://127.0.0.1:8001/docs for interactive API documentation.

## Integration with Agent

The agent's REST API tool will call these endpoints:

```python
import requests
from langchain.tools import tool

@tool
def query_metrics_api(query_type: str, service: str = None) -> dict:
    """Query the local metrics REST API for real-time metrics data."""
    base_url = "http://127.0.0.1:8001"

    if query_type == "latency":
        response = requests.get(
            f"{base_url}/metrics/latency",
            params={"service": service, "period": "1h"}
        )
    elif query_type == "throughput":
        response = requests.get(
            f"{base_url}/metrics/throughput",
            params={"service": service, "period": "1h"}
        )
    elif query_type == "errors":
        response = requests.get(
            f"{base_url}/metrics/errors",
            params={"service": service, "period": "1h"}
        )

    return response.json()
```

## OpenAPI Schema

The API automatically generates OpenAPI 3.0 schema available at:
- JSON format: http://127.0.0.1:8001/openapi.json
- Interactive docs: http://127.0.0.1:8001/docs

## Performance

- Response time: <50ms for all endpoints
- Concurrent requests: Supports high concurrency (FastAPI + uvicorn)
- Memory usage: ~50MB

## Configuration

### Change Port

```python
# In api_service.py or when running
uvicorn services.api_service:app --port 8080
```

### Enable CORS for specific origins

```python
# In api_service.py, modify CORSMiddleware
allow_origins=["http://localhost:3000", "http://localhost:8080"]
```

## Troubleshooting

### Port already in use
```bash
# Find process using port 8001
lsof -i :8001

# Kill it
kill -9 <PID>
```

### Server not starting
```bash
# Check if dependencies installed
pip install fastapi uvicorn

# Check Python version (requires 3.8+)
python --version
```

### Connection refused
- Make sure server is running
- Check firewall settings
- Verify port 8001 is not blocked

## Summary

The API Service provides:
- ✅ 6 REST endpoints for metrics and health
- ✅ FastAPI with automatic OpenAPI docs
- ✅ Realistic data generation with patterns
- ✅ <50ms response time
- ✅ CORS enabled
- ✅ Pydantic validation
- ✅ Ready for agent integration

**Runs locally on http://127.0.0.1:8001**
