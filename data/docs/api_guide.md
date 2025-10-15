# API Usage Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
4. [Request/Response Format](#requestresponse-format)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Best Practices](#best-practices)
8. [Code Examples](#code-examples)

## Introduction

Welcome to the API Usage Guide. This document provides comprehensive information about our RESTful API, including authentication, available endpoints, request/response formats, and best practices.

### Base URL

All API requests should be made to:
```
Production: https://api.example.com/v1
Staging: https://api-staging.example.com/v1
Development: http://localhost:8001/v1
```

### API Versioning

We use URL-based versioning. The current version is `v1`. When breaking changes are introduced, we'll release a new version (v2) while maintaining backward compatibility for at least 6 months.

### Content Type

All requests and responses use JSON format:
```
Content-Type: application/json
```

## Authentication

### API Keys

Every request must include a valid API key in the headers:

```http
X-API-Key: your_api_key_here
```

**Obtaining an API Key**:
1. Log into the developer portal at https://developer.example.com
2. Navigate to "API Keys" section
3. Click "Generate New Key"
4. Copy and securely store your key

**Security Notes**:
- Never expose API keys in client-side code
- Rotate keys every 90 days
- Use different keys for development, staging, and production
- Revoke compromised keys immediately

### JWT Authentication

For user-specific operations, you must include a JWT token:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Obtaining a JWT Token**:

```bash
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure_password"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

**Token Refresh**:

When your access token expires (after 1 hour), use the refresh token:

```bash
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "your_refresh_token_here"
}
```

## API Endpoints

### Metrics Endpoints

#### Get Latency Metrics

Retrieve latency metrics for a specific service.

```http
GET /metrics/latency?service={service_name}&period={time_period}
```

**Query Parameters**:
- `service` (required): Service name (e.g., api-gateway, auth-service)
- `period` (optional): Time period (1h, 6h, 24h, 7d). Default: 1h
- `aggregation` (optional): Aggregation method (avg, p50, p95, p99). Default: all

**Example Request**:
```bash
curl -X GET "https://api.example.com/v1/metrics/latency?service=api-gateway&period=1h" \
  -H "X-API-Key: your_api_key" \
  -H "Authorization: Bearer your_jwt_token"
```

**Example Response**:
```json
{
  "service": "api-gateway",
  "period": "1h",
  "timestamp": "2025-10-15T10:00:00Z",
  "metrics": {
    "p50": 45.2,
    "p95": 120.5,
    "p99": 250.3,
    "avg": 68.7,
    "max": 1250.0,
    "min": 12.3
  },
  "unit": "milliseconds",
  "sample_count": 15420
}
```

**Response Codes**:
- `200 OK`: Success
- `400 Bad Request`: Invalid parameters
- `401 Unauthorized`: Missing or invalid authentication
- `404 Not Found`: Service not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

#### Get Throughput Metrics

Retrieve request throughput metrics.

```http
GET /metrics/throughput?service={service_name}&period={time_period}
```

**Query Parameters**:
- `service` (required): Service name
- `period` (optional): Time period (1h, 6h, 24h, 7d). Default: 1h
- `interval` (optional): Data point interval (1m, 5m, 15m, 1h). Default: 5m

**Example Response**:
```json
{
  "service": "api-gateway",
  "period": "1h",
  "interval": "5m",
  "data_points": [
    {
      "timestamp": "2025-10-15T09:00:00Z",
      "requests_per_second": 1250,
      "total_requests": 375000
    },
    {
      "timestamp": "2025-10-15T09:05:00Z",
      "requests_per_second": 1320,
      "total_requests": 396000
    }
  ],
  "summary": {
    "avg_rps": 1285,
    "peak_rps": 1650,
    "total_requests": 4626000
  }
}
```

#### Get Error Rates

Retrieve error rate metrics for a service.

```http
GET /metrics/errors?service={service_name}&period={time_period}
```

**Example Response**:
```json
{
  "service": "auth-service",
  "period": "24h",
  "error_rate": 0.023,
  "total_requests": 5000000,
  "total_errors": 115000,
  "error_breakdown": {
    "4xx": {
      "count": 95000,
      "percentage": 82.6,
      "codes": {
        "400": 15000,
        "401": 55000,
        "403": 12000,
        "404": 13000
      }
    },
    "5xx": {
      "count": 20000,
      "percentage": 17.4,
      "codes": {
        "500": 12000,
        "502": 5000,
        "503": 2500,
        "504": 500
      }
    }
  }
}
```

#### Post Metrics Query

Query metrics with complex filters.

```http
POST /metrics/query
Content-Type: application/json
```

**Request Body**:
```json
{
  "services": ["api-gateway", "auth-service"],
  "metrics": ["latency", "throughput", "errors"],
  "time_range": {
    "start": "2025-10-15T00:00:00Z",
    "end": "2025-10-15T23:59:59Z"
  },
  "filters": {
    "region": "us-east-1",
    "environment": "production"
  },
  "aggregation": "5m"
}
```

**Response**:
```json
{
  "query_id": "q_abc123",
  "results": [
    {
      "service": "api-gateway",
      "latency": { "p95": 120.5 },
      "throughput": { "avg_rps": 1285 },
      "errors": { "rate": 0.015 }
    }
  ]
}
```

### Health Check Endpoints

#### Service Health

Check the health status of a specific service.

```http
GET /health?service={service_name}
```

**Example Response**:
```json
{
  "service": "api-gateway",
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
  "version": "2.3.1"
}
```

### Service Management Endpoints

#### List Services

Get a list of all available services.

```http
GET /services
```

**Example Response**:
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
    {
      "name": "auth-service",
      "status": "running",
      "instances": 3,
      "version": "1.8.2",
      "health": "healthy"
    },
    {
      "name": "data-processor",
      "status": "running",
      "instances": 10,
      "version": "3.1.0",
      "health": "degraded"
    }
  ],
  "total_services": 3,
  "healthy_services": 2
}
```

#### Get Service Details

Get detailed information about a specific service.

```http
GET /services/{service_name}
```

**Example Response**:
```json
{
  "name": "auth-service",
  "description": "Authentication and authorization service",
  "status": "running",
  "instances": [
    {
      "id": "auth-svc-1",
      "host": "10.0.1.23",
      "port": 8080,
      "status": "healthy",
      "cpu": 45.2,
      "memory": 62.3,
      "uptime": 432000
    },
    {
      "id": "auth-svc-2",
      "host": "10.0.1.24",
      "port": 8080,
      "status": "healthy",
      "cpu": 38.7,
      "memory": 58.1,
      "uptime": 432000
    }
  ],
  "configuration": {
    "jwt_expiry": 3600,
    "refresh_token_expiry": 604800,
    "max_sessions_per_user": 5
  },
  "dependencies": [
    "postgresql-primary",
    "redis-cluster"
  ],
  "endpoints": [
    "POST /auth/login",
    "POST /auth/logout",
    "POST /auth/refresh",
    "GET /auth/validate"
  ]
}
```

### Configuration Endpoints

#### Get Configuration

Retrieve service configuration.

```http
GET /config/{service_name}
```

**Example Response**:
```json
{
  "service": "api-gateway",
  "configuration": {
    "upstream_timeout": 30000,
    "max_request_size": 10485760,
    "rate_limit": {
      "enabled": true,
      "requests_per_minute": 1000,
      "burst": 50
    },
    "circuit_breaker": {
      "enabled": true,
      "failure_threshold": 0.5,
      "recovery_timeout": 30000
    },
    "cors": {
      "enabled": true,
      "allowed_origins": ["https://app.example.com"],
      "allowed_methods": ["GET", "POST", "PUT", "DELETE"]
    }
  },
  "last_updated": "2025-10-10T15:30:00Z"
}
```

## Request/Response Format

### Standard Request Headers

All requests should include:

```http
X-API-Key: your_api_key
Authorization: Bearer your_jwt_token
Content-Type: application/json
Accept: application/json
X-Request-ID: unique_request_id (optional but recommended)
```

### Standard Response Format

All successful responses follow this structure:

```json
{
  "data": { /* response data */ },
  "metadata": {
    "request_id": "req_abc123",
    "timestamp": "2025-10-15T10:00:00Z",
    "processing_time_ms": 45
  }
}
```

### Pagination

For endpoints returning lists, pagination is supported:

**Request**:
```http
GET /services?page=2&page_size=20&sort=name&order=asc
```

**Response**:
```json
{
  "data": [ /* array of items */ ],
  "pagination": {
    "page": 2,
    "page_size": 20,
    "total_items": 156,
    "total_pages": 8,
    "has_next": true,
    "has_previous": true
  }
}
```

**Pagination Parameters**:
- `page` (default: 1): Page number
- `page_size` (default: 50, max: 100): Items per page
- `sort` (optional): Field to sort by
- `order` (default: asc): Sort order (asc or desc)

## Error Handling

### Error Response Format

All error responses follow this structure:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid service name parameter",
    "details": "Service 'invalid-service' does not exist",
    "request_id": "req_abc123",
    "timestamp": "2025-10-15T10:00:00Z"
  }
}
```

### Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | INVALID_REQUEST | Malformed request or invalid parameters |
| 401 | UNAUTHORIZED | Missing or invalid authentication credentials |
| 403 | FORBIDDEN | Authenticated but not authorized for this resource |
| 404 | NOT_FOUND | Resource not found |
| 409 | CONFLICT | Request conflicts with current state |
| 422 | VALIDATION_FAILED | Request validation failed |
| 429 | RATE_LIMIT_EXCEEDED | Too many requests |
| 500 | INTERNAL_ERROR | Internal server error |
| 502 | BAD_GATEWAY | Upstream service unavailable |
| 503 | SERVICE_UNAVAILABLE | Service temporarily unavailable |
| 504 | GATEWAY_TIMEOUT | Upstream service timeout |

### Error Handling Best Practices

1. **Always check HTTP status code** before processing response
2. **Log error details** including request_id for troubleshooting
3. **Implement exponential backoff** for 429 and 5xx errors
4. **Display user-friendly messages** based on error codes
5. **Don't retry 4xx errors** (except 429) without fixing the request

## Rate Limiting

### Rate Limit Headers

Every response includes rate limit information:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 847
X-RateLimit-Reset: 1634567890
```

- `X-RateLimit-Limit`: Maximum requests allowed per window
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Unix timestamp when the limit resets

### Rate Limit Tiers

| Tier | Requests/Minute | Burst |
|------|-----------------|-------|
| Free | 100 | 10 |
| Basic | 1,000 | 50 |
| Pro | 5,000 | 200 |
| Enterprise | 20,000 | 1000 |

### Handling Rate Limits

When you receive a 429 response:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "retry_after": 60
  }
}
```

**Recommended approach**:
```python
import time

response = make_request()
if response.status_code == 429:
    retry_after = int(response.headers.get('Retry-After', 60))
    time.sleep(retry_after)
    response = make_request()  # Retry
```

## Best Practices

### 1. Use Appropriate HTTP Methods

- **GET**: Retrieve resources (idempotent)
- **POST**: Create new resources
- **PUT**: Update entire resource (idempotent)
- **PATCH**: Partial update
- **DELETE**: Remove resource (idempotent)

### 2. Include Request IDs

Always include a unique request ID for tracking:

```http
X-Request-ID: req_abc123xyz
```

This helps with:
- Troubleshooting issues
- Correlating logs across services
- Tracking request flow

### 3. Handle Timeouts

Set appropriate timeouts:
- Connection timeout: 5 seconds
- Read timeout: 30 seconds

```python
import requests

response = requests.get(
    url,
    timeout=(5, 30)  # (connect, read)
)
```

### 4. Implement Retry Logic

Use exponential backoff for transient failures:

```python
import time

max_retries = 3
base_delay = 1

for attempt in range(max_retries):
    try:
        response = make_request()
        if response.status_code < 500:
            break
    except Exception:
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
```

### 5. Cache Responses

Cache responses when appropriate:
- Use ETags for conditional requests
- Cache GET requests with stable data
- Respect Cache-Control headers

### 6. Use Compression

Enable gzip compression for large responses:

```http
Accept-Encoding: gzip, deflate
```

### 7. Monitor API Usage

Track your API usage:
- Monitor rate limit headers
- Log response times
- Track error rates
- Set up alerts for anomalies

## Code Examples

### Python (requests)

```python
import requests

API_KEY = "your_api_key"
JWT_TOKEN = "your_jwt_token"
BASE_URL = "https://api.example.com/v1"

headers = {
    "X-API-Key": API_KEY,
    "Authorization": f"Bearer {JWT_TOKEN}",
    "Content-Type": "application/json"
}

# Get latency metrics
response = requests.get(
    f"{BASE_URL}/metrics/latency",
    headers=headers,
    params={
        "service": "api-gateway",
        "period": "1h"
    }
)

if response.status_code == 200:
    data = response.json()
    print(f"P95 Latency: {data['metrics']['p95']}ms")
else:
    print(f"Error: {response.json()['error']['message']}")
```

### JavaScript (fetch)

```javascript
const API_KEY = "your_api_key";
const JWT_TOKEN = "your_jwt_token";
const BASE_URL = "https://api.example.com/v1";

async function getLatencyMetrics(service, period = "1h") {
  const response = await fetch(
    `${BASE_URL}/metrics/latency?service=${service}&period=${period}`,
    {
      headers: {
        "X-API-Key": API_KEY,
        "Authorization": `Bearer ${JWT_TOKEN}`,
        "Content-Type": "application/json"
      }
    }
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error.message);
  }

  return response.json();
}

// Usage
getLatencyMetrics("api-gateway", "1h")
  .then(data => console.log("P95 Latency:", data.metrics.p95))
  .catch(error => console.error("Error:", error.message));
```

### cURL

```bash
# Get latency metrics
curl -X GET "https://api.example.com/v1/metrics/latency?service=api-gateway&period=1h" \
  -H "X-API-Key: your_api_key" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json"

# Post metrics query
curl -X POST "https://api.example.com/v1/metrics/query" \
  -H "X-API-Key: your_api_key" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "services": ["api-gateway"],
    "metrics": ["latency"],
    "time_range": {
      "start": "2025-10-15T00:00:00Z",
      "end": "2025-10-15T23:59:59Z"
    }
  }'
```

## Webhooks

### Subscribing to Events

You can subscribe to events via webhooks:

```http
POST /webhooks/subscribe
Content-Type: application/json

{
  "url": "https://your-app.com/webhooks/metrics",
  "events": ["metrics.threshold_exceeded", "service.health_changed"],
  "secret": "your_webhook_secret"
}
```

### Webhook Payload

When an event occurs, we'll POST to your URL:

```json
{
  "event": "metrics.threshold_exceeded",
  "timestamp": "2025-10-15T10:00:00Z",
  "data": {
    "service": "api-gateway",
    "metric": "latency_p95",
    "value": 150.5,
    "threshold": 100,
    "severity": "warning"
  },
  "signature": "sha256=abc123..."
}
```

### Verifying Webhook Signatures

```python
import hmac
import hashlib

def verify_signature(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

## Support

For API support:
- Documentation: https://docs.example.com
- Email: api-support@example.com
- Status Page: https://status.example.com
- Community Forum: https://forum.example.com
