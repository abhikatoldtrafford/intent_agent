# Monitoring and Alerting Guide

## Table of Contents
1. [Overview](#overview)
2. [Metrics Collection](#metrics-collection)
3. [Key Performance Indicators](#key-performance-indicators)
4. [Alerting Rules](#alerting-rules)
5. [Dashboards](#dashboards)
6. [Log Aggregation](#log-aggregation)
7. [Distributed Tracing](#distributed-tracing)
8. [APM Tools](#apm-tools)
9. [On-Call Procedures](#on-call-procedures)
10. [Incident Response](#incident-response)

## Overview

Effective monitoring is critical for maintaining system reliability and performance. Our monitoring strategy follows the **three pillars of observability**:

1. **Metrics**: Quantitative data about system behavior
2. **Logs**: Detailed event records
3. **Traces**: Request flow through distributed systems

### Monitoring Stack

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Loki**: Log aggregation
- **Jaeger**: Distributed tracing
- **AlertManager**: Alert routing and deduplication
- **PagerDuty**: On-call notification and escalation

## Metrics Collection

### Instrumentation

All services expose metrics at `/metrics` endpoint in Prometheus format:

```python
from prometheus_client import Counter, Histogram, Gauge

# Request counter
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

# Request duration histogram
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# Active connections gauge
active_connections = Gauge(
    'active_connections',
    'Number of active connections'
)

# Usage in code
@app.get("/api/orders")
async def get_orders():
    http_requests_total.labels(method='GET', endpoint='/api/orders', status=200).inc()

    with http_request_duration_seconds.labels(method='GET', endpoint='/api/orders').time():
        # Process request
        return orders
```

### Standard Metrics

Every service must expose these metrics:

#### Request Metrics
- `http_requests_total`: Total number of HTTP requests
- `http_request_duration_seconds`: Request latency histogram
- `http_requests_in_flight`: Current number of requests being processed

#### Resource Metrics
- `process_cpu_seconds_total`: CPU time consumed
- `process_resident_memory_bytes`: Memory usage
- `process_open_fds`: Number of open file descriptors

#### Application Metrics
- `database_connections_total`: Total DB connections
- `database_connections_active`: Active DB connections
- `cache_hits_total`: Cache hit count
- `cache_misses_total`: Cache miss count

#### Custom Business Metrics
- `orders_created_total`: Number of orders created
- `payments_processed_total`: Payments processed
- `user_signups_total`: New user registrations

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'production'
    region: 'us-east-1'

scrape_configs:
  - job_name: 'api-gateway'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - production
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: api-gateway
        action: keep
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: instance

  - job_name: 'auth-service'
    static_configs:
      - targets: ['auth-service:8080']
        labels:
          service: 'auth-service'
          env: 'production'

  - job_name: 'data-processor'
    static_configs:
      - targets: ['data-processor:8080']
```

### Querying Metrics

**Request rate**:
```promql
rate(http_requests_total[5m])
```

**P95 latency**:
```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

**Error rate**:
```promql
sum(rate(http_requests_total{status=~"5.."}[5m])) /
sum(rate(http_requests_total[5m]))
```

**CPU usage**:
```promql
rate(process_cpu_seconds_total[5m])
```

**Memory usage**:
```promql
process_resident_memory_bytes / 1024 / 1024  # Convert to MB
```

## Key Performance Indicators

### Service-Level Indicators (SLIs)

#### API Gateway

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Availability | 99.99% | 99.95% | 99.9% |
| P50 Latency | 20ms | 30ms | 50ms |
| P95 Latency | 50ms | 75ms | 100ms |
| P99 Latency | 100ms | 150ms | 200ms |
| Error Rate | 0.1% | 0.5% | 1% |

#### Auth Service

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Availability | 99.95% | 99.9% | 99.5% |
| P95 Latency | 100ms | 150ms | 200ms |
| Error Rate | 0.5% | 1% | 2% |
| Token Validation | <10ms | <20ms | <30ms |

#### Data Processor

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Processing Rate | 50k/s | 40k/s | 30k/s |
| Processing Lag | <1min | <5min | <10min |
| Error Rate | 0.5% | 1% | 2% |
| Dead Letter Queue | 0 | <100 | <1000 |

#### Business Logic Service

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Availability | 99.9% | 99.5% | 99% |
| P95 Latency | 200ms | 300ms | 500ms |
| Error Rate | 1% | 2% | 5% |
| DB Query Time | <20ms | <50ms | <100ms |

### Service-Level Objectives (SLOs)

**Monthly Availability SLOs**:
- API Gateway: 99.99% (4.3 minutes downtime/month)
- Auth Service: 99.95% (21.6 minutes downtime/month)
- Business Logic: 99.9% (43.2 minutes downtime/month)
- Data Processor: 99.9% (43.2 minutes downtime/month)

**Error Budget**:
- API Gateway: 0.01% error budget (4,320 failed requests per 43.2M requests)
- Calculate remaining budget: `(1 - actual_uptime) / (1 - target_uptime)`

## Alerting Rules

### Critical Alerts (Page On-Call)

#### High Error Rate
```yaml
- alert: HighErrorRate
  expr: |
    sum(rate(http_requests_total{status=~"5.."}[5m])) by (service) /
    sum(rate(http_requests_total[5m])) by (service) > 0.05
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High error rate on {{ $labels.service }}"
    description: "Error rate is {{ $value | humanizePercentage }} on {{ $labels.service }}"
```

#### Service Down
```yaml
- alert: ServiceDown
  expr: up == 0
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Service {{ $labels.job }} is down"
    description: "{{ $labels.instance }} has been down for more than 2 minutes"
```

#### High Latency
```yaml
- alert: HighLatency
  expr: |
    histogram_quantile(0.95,
      rate(http_request_duration_seconds_bucket[5m])
    ) > 0.2
  for: 10m
  labels:
    severity: critical
  annotations:
    summary: "High latency on {{ $labels.service }}"
    description: "P95 latency is {{ $value }}s on {{ $labels.service }}"
```

#### Database Connection Pool Exhausted
```yaml
- alert: DatabaseConnectionPoolExhausted
  expr: |
    database_connections_active / database_connections_total > 0.9
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Database connection pool nearly exhausted"
    description: "{{ $labels.service }} is using {{ $value | humanizePercentage }} of connection pool"
```

#### Disk Space Low
```yaml
- alert: DiskSpaceLow
  expr: |
    (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Disk space low on {{ $labels.instance }}"
    description: "Only {{ $value | humanizePercentage }} disk space remaining"
```

### Warning Alerts (No Page)

#### Elevated Error Rate
```yaml
- alert: ElevatedErrorRate
  expr: |
    sum(rate(http_requests_total{status=~"5.."}[5m])) by (service) /
    sum(rate(http_requests_total[5m])) by (service) > 0.01
  for: 15m
  labels:
    severity: warning
  annotations:
    summary: "Elevated error rate on {{ $labels.service }}"
```

#### High Memory Usage
```yaml
- alert: HighMemoryUsage
  expr: |
    process_resident_memory_bytes / 1024 / 1024 / 1024 > 1.5
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "High memory usage on {{ $labels.service }}"
    description: "{{ $labels.service }} is using {{ $value }}GB of memory"
```

#### Slow Database Queries
```yaml
- alert: SlowDatabaseQueries
  expr: |
    histogram_quantile(0.95,
      rate(database_query_duration_seconds_bucket[5m])
    ) > 0.1
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Slow database queries detected"
    description: "P95 query time is {{ $value }}s"
```

## Dashboards

### Service Overview Dashboard

**Panels**:
1. **Request Rate**: Requests per second across all services
2. **Error Rate**: Percentage of failed requests
3. **Latency**: P50, P95, P99 latency
4. **Service Health**: Status of all services (up/down)
5. **Active Instances**: Number of running instances per service

**Grafana Query Examples**:

```promql
# Request rate by service
sum(rate(http_requests_total[5m])) by (service)

# Error rate percentage
100 * (
  sum(rate(http_requests_total{status=~"5.."}[5m])) by (service) /
  sum(rate(http_requests_total[5m])) by (service)
)

# P95 latency
histogram_quantile(0.95,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service)
)
```

### Infrastructure Dashboard

**Panels**:
1. **CPU Usage**: Per node and per pod
2. **Memory Usage**: Used vs available
3. **Network I/O**: Bytes sent/received
4. **Disk I/O**: Read/write operations
5. **Pod Status**: Running, pending, failed pods

### Business Metrics Dashboard

**Panels**:
1. **Orders Created**: Total orders per hour
2. **Revenue**: Total revenue per hour
3. **User Signups**: New registrations
4. **Payment Success Rate**: Successful payments / total attempts
5. **Active Users**: Currently active users

### Custom Dashboard Example

```json
{
  "dashboard": {
    "title": "API Gateway Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{service='api-gateway'}[5m]))"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{service='api-gateway',status=~'5..'}[5m])) / sum(rate(http_requests_total{service='api-gateway'}[5m]))"
          }
        ]
      }
    ]
  }
}
```

## Log Aggregation

### Structured Logging

All services use structured JSON logging:

```python
import logging
import json

class StructuredLogger:
    def __init__(self, service_name):
        self.service = service_name
        self.logger = logging.getLogger(service_name)

    def log(self, level, message, **kwargs):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service,
            "level": level,
            "message": message,
            **kwargs
        }
        self.logger.log(level, json.dumps(log_entry))

# Usage
logger = StructuredLogger("api-gateway")
logger.log(
    logging.INFO,
    "Request processed",
    request_id="req_123",
    method="GET",
    path="/api/orders",
    duration_ms=45,
    status=200
)
```

**Output**:
```json
{
  "timestamp": "2025-10-15T10:30:45.123Z",
  "service": "api-gateway",
  "level": "INFO",
  "message": "Request processed",
  "request_id": "req_123",
  "method": "GET",
  "path": "/api/orders",
  "duration_ms": 45,
  "status": 200
}
```

### Log Levels

Use appropriate log levels:
- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages for potentially harmful situations
- **ERROR**: Error messages for error events
- **CRITICAL**: Critical messages for severe error events

### LogQL Queries (Loki)

**Find all errors in last hour**:
```logql
{service="api-gateway"} |= "ERROR" | json | level="ERROR"
```

**Request duration > 1s**:
```logql
{service="business-logic"} | json | duration_ms > 1000
```

**Count 5xx errors**:
```logql
sum(count_over_time({service="api-gateway"} |= "status=5" [1h]))
```

**Top error messages**:
```logql
topk(10,
  sum by (message) (
    count_over_time({service="auth-service"} | json | level="ERROR" [1h])
  )
)
```

## Distributed Tracing

### Trace Implementation

Using OpenTelemetry:

```python
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# Configure tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Manual span creation
@app.get("/api/orders")
async def get_orders():
    with tracer.start_as_current_span("get_orders") as span:
        span.set_attribute("user_id", user_id)

        with tracer.start_as_current_span("database_query"):
            orders = await db.query("SELECT * FROM orders")

        with tracer.start_as_current_span("process_results"):
            processed = process_orders(orders)

        return processed
```

### Trace Context Propagation

Ensure trace context is propagated across services:

```python
import requests
from opentelemetry.propagate import inject

# Client side - inject trace context
headers = {}
inject(headers)

response = requests.get(
    "http://business-logic/api/process",
    headers=headers
)

# Server side - extract trace context automatically by instrumentation
```

### Analyzing Traces

**Common Patterns to Look For**:
1. **Long Database Queries**: Spans with >100ms duration
2. **N+1 Queries**: Multiple sequential database calls
3. **Slow External APIs**: Long spans for external service calls
4. **High Latency Services**: Services contributing most to total latency

**Jaeger UI**:
- Search traces by service, operation, duration
- View waterfall diagram of spans
- Compare traces to identify bottlenecks
- Track error rates per operation

## APM Tools

### Application Performance Monitoring

We use custom APM implementation with:
- Request/response time tracking
- Error rate monitoring
- Resource utilization tracking
- Dependency mapping

### Transaction Tracking

```python
from apm_client import APMClient

apm = APMClient(service_name="api-gateway")

@app.get("/api/orders")
async def get_orders():
    transaction = apm.begin_transaction("get_orders")

    try:
        # Database query
        with apm.capture_span("database_query", span_type="db"):
            orders = await db.query("SELECT * FROM orders")

        # External API call
        with apm.capture_span("payment_api", span_type="external"):
            payments = await payment_api.get_payments()

        transaction.result = "success"
        return orders

    except Exception as e:
        apm.capture_exception(e)
        transaction.result = "error"
        raise
    finally:
        apm.end_transaction()
```

## On-Call Procedures

### On-Call Rotation

- **Schedule**: 1-week rotations
- **Primary**: First responder
- **Secondary**: Backup for primary
- **Manager**: Escalation point

### Alert Response Time

| Severity | Response Time | Resolution Time |
|----------|--------------|-----------------|
| Critical | 5 minutes | 1 hour |
| High | 15 minutes | 4 hours |
| Warning | Next business day | 1 week |

### On-Call Checklist

When paged:
1. **Acknowledge alert** in PagerDuty (within 5 minutes)
2. **Check dashboard** for service health
3. **Review alert details** and metrics
4. **Investigate logs** for errors
5. **Identify root cause**
6. **Take corrective action**
7. **Monitor for recovery**
8. **Document incident**
9. **Follow up with postmortem** if needed

### Common Actions

**Service Down**:
```bash
# Check pod status
kubectl get pods -n production -l app=api-gateway

# Restart if needed
kubectl rollout restart deployment/api-gateway -n production
```

**High Latency**:
```bash
# Scale up instances
kubectl scale deployment api-gateway --replicas=10 -n production

# Check database connections
kubectl exec -it postgresql-primary-0 -- psql -c "SELECT count(*) FROM pg_stat_activity"
```

**High Error Rate**:
```bash
# Check recent logs
kubectl logs -n production -l app=auth-service --tail=100 | grep ERROR

# Rollback if due to recent deployment
kubectl rollout undo deployment/auth-service -n production
```

## Incident Response

### Incident Severity Levels

**Severity 1 (Critical)**:
- Complete service outage
- Data loss or corruption
- Security breach
- Financial impact >$10k/hour

**Severity 2 (High)**:
- Major feature unavailable
- Severe performance degradation
- Partial outage affecting >20% users

**Severity 3 (Medium)**:
- Minor feature unavailable
- Performance degradation
- Workaround available

**Severity 4 (Low)**:
- Cosmetic issues
- No user impact
- Feature request

### Incident Response Process

1. **Detection**: Alert fired or user report
2. **Acknowledgment**: On-call acknowledges (5 min)
3. **Assessment**: Determine severity and impact
4. **Response**: Take corrective action
5. **Communication**: Update stakeholders
6. **Resolution**: Issue resolved
7. **Postmortem**: Document lessons learned

### Communication During Incidents

**Status Page Updates**:
```bash
# Update status page
./scripts/update_status.sh --status investigating --message "Investigating elevated error rates on API Gateway"

# Post resolution
./scripts/update_status.sh --status resolved --message "Issue resolved. Service operating normally."
```

**Slack Notifications**:
```
#incidents channel:
[INCIDENT] Sev-1: API Gateway Down
Start Time: 2025-10-15 10:30 UTC
Impact: All API requests failing
Status: Investigating
Lead: @john
```

### Postmortem Template

```markdown
# Incident Postmortem: API Gateway Outage

## Incident Summary
- **Date**: 2025-10-15
- **Duration**: 10:30 - 11:15 UTC (45 minutes)
- **Severity**: Sev-1
- **Impact**: 100% of API requests failed

## Timeline
- 10:30: Alert fired for API Gateway down
- 10:35: On-call acknowledged, began investigation
- 10:40: Identified root cause (OOM kills)
- 10:45: Increased memory limits
- 10:50: Deployed fix
- 11:00: Service recovered
- 11:15: Confirmed stable

## Root Cause
Memory leak in v2.3.1 caused OOM kills of all pods.

## Resolution
Increased memory limits and rolled back to v2.3.0.

## Action Items
- [ ] Fix memory leak in caching layer
- [ ] Add memory usage alerts
- [ ] Improve staging testing for memory leaks
- [ ] Document memory profiling procedures

## Lessons Learned
- Need better memory usage monitoring
- Staging tests should include long-running tests
- Consider gradual rollout (canary) for major versions
```

## Best Practices

### Monitoring Best Practices

1. **Monitor outcomes, not outputs**: Focus on user-facing metrics
2. **Set actionable alerts**: Every alert should require action
3. **Avoid alert fatigue**: Tune thresholds to reduce noise
4. **Use runbooks**: Document response procedures
5. **Practice incident response**: Regular fire drills
6. **Review and improve**: Continuous improvement of monitoring

### Alerting Best Practices

1. **Alert on symptoms, not causes**: Alert on user impact
2. **Use appropriate severity**: Not everything is critical
3. **Include context**: Add helpful annotations
4. **Aggregate related alerts**: Reduce alert spam
5. **Test alerts**: Verify alerts fire correctly
6. **Document response**: Every alert needs a runbook

### Dashboard Best Practices

1. **Keep it simple**: Don't overcrowd dashboards
2. **Focus on key metrics**: 5-7 panels per dashboard
3. **Use appropriate visualizations**: Graphs for trends, gauges for current values
4. **Add context**: Include SLO targets as reference lines
5. **Organize logically**: Related metrics together
6. **Mobile-friendly**: Accessible on mobile devices

## Monitoring Resources

### Documentation
- Prometheus: https://prometheus.io/docs
- Grafana: https://grafana.com/docs
- Jaeger: https://www.jaegertracing.io/docs

### Internal Resources
- Runbook: https://wiki.example.com/runbooks
- Dashboards: https://grafana.example.com
- Alerts: https://alertmanager.example.com
- Status Page: https://status.example.com

### Support
- Slack: #monitoring
- Email: observability@example.com
- On-call: PagerDuty
