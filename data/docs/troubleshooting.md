# Troubleshooting Guide

## Table of Contents
1. [Common Issues](#common-issues)
2. [Performance Problems](#performance-problems)
3. [Authentication Errors](#authentication-errors)
4. [Service Health Issues](#service-health-issues)
5. [Database Problems](#database-problems)
6. [Network and Connectivity](#network-and-connectivity)
7. [Monitoring and Debugging](#monitoring-and-debugging)
8. [Emergency Procedures](#emergency-procedures)

## Common Issues

### High Latency

**Symptom**: API responses are slow, P95 latency exceeds 200ms.

**Possible Causes**:
1. Database query performance
2. External API dependencies
3. Memory pressure or CPU saturation
4. Network congestion
5. Insufficient connection pooling

**Diagnosis Steps**:

1. **Check current metrics**:
```bash
curl -X GET "http://localhost:8001/metrics/latency?service=api-gateway&period=1h" \
  -H "X-API-Key: your_key"
```

2. **Review service health**:
```bash
curl -X GET "http://localhost:8001/health?service=api-gateway"
```

3. **Check database connection pool**:
   - Navigate to service dashboard
   - Look for connection pool exhaustion
   - Normal: 45/100 connections
   - Warning: 85/100 connections
   - Critical: 95/100 connections

4. **Analyze slow queries**:
```sql
-- Check for slow queries in PostgreSQL
SELECT pid, now() - pg_stat_activity.query_start AS duration, query
FROM pg_stat_activity
WHERE state = 'active' AND now() - pg_stat_activity.query_start > interval '5 seconds'
ORDER BY duration DESC;
```

**Solutions**:

**Quick Fix (Immediate)**:
1. Scale up service instances:
```bash
kubectl scale deployment api-gateway --replicas=8
```

2. Clear cache if stale:
```bash
redis-cli FLUSHDB
```

3. Restart underperforming instances:
```bash
kubectl rollout restart deployment/api-gateway
```

**Long-term Fix**:
1. **Add database indices** on frequently queried columns
2. **Implement query caching** for expensive operations
3. **Enable connection pooling** with optimal settings:
   ```yaml
   pool_size: 100
   max_overflow: 50
   pool_timeout: 30
   pool_recycle: 3600
   ```

4. **Add read replicas** for read-heavy queries
5. **Implement circuit breakers** to prevent cascading failures

**Prevention**:
- Set up alerts for P95 latency > 150ms
- Regular performance testing
- Query optimization reviews
- Capacity planning based on growth projections

### High Error Rates

**Symptom**: Error rate exceeds 5%, increased 5xx responses.

**Possible Causes**:
1. Service dependency failures
2. Database connectivity issues
3. Memory leaks or OOM errors
4. Configuration errors
5. Bug in recent deployment

**Diagnosis Steps**:

1. **Check error breakdown**:
```bash
curl -X GET "http://localhost:8001/metrics/errors?service=auth-service&period=1h"
```

2. **Review recent logs**:
```bash
kubectl logs -l app=auth-service --tail=1000 | grep ERROR
```

3. **Check service dependencies**:
```bash
curl -X GET "http://localhost:8001/health?service=auth-service"
```

4. **Analyze error patterns**:
   - Are errors random or clustered?
   - Do they correlate with specific endpoints?
   - Are they related to specific users or requests?

**Solutions**:

**For 5xx Errors**:
1. Check service health and restart if needed
2. Verify database connectivity
3. Check memory usage (OOM errors)
4. Review recent deployments (rollback if needed)

**For 4xx Errors**:
1. Review API documentation
2. Check authentication/authorization logic
3. Validate request parameters
4. Review rate limiting settings

**Rollback Procedure**:
```bash
# Check deployment history
kubectl rollout history deployment/auth-service

# Rollback to previous version
kubectl rollout undo deployment/auth-service

# Verify rollback
kubectl rollout status deployment/auth-service
```

### Service Unavailable (503)

**Symptom**: Service returns 503 Service Unavailable.

**Possible Causes**:
1. All instances down or unhealthy
2. Service overwhelmed with requests
3. Circuit breaker is open
4. Deployment in progress

**Diagnosis Steps**:

1. **Check service status**:
```bash
kubectl get pods -l app=data-processor
```

2. **Check pod logs**:
```bash
kubectl logs data-processor-<pod-id> --tail=100
```

3. **Check resource utilization**:
```bash
kubectl top pods -l app=data-processor
```

**Solutions**:

1. **If no healthy instances**:
```bash
# Check why pods are failing
kubectl describe pod data-processor-<pod-id>

# Common issues:
# - Image pull errors
# - OOM kills
# - Failed health checks
# - Insufficient resources
```

2. **If overwhelmed**:
```bash
# Scale up immediately
kubectl scale deployment data-processor --replicas=15

# Check if scaling helped
kubectl get hpa data-processor
```

3. **If circuit breaker open**:
   - Wait for recovery timeout (30 seconds)
   - Check downstream dependencies
   - Fix root cause before resetting

## Performance Problems

### Memory Leaks

**Symptom**: Memory usage continuously increases, eventually leading to OOM kills.

**Diagnosis Steps**:

1. **Monitor memory over time**:
```bash
# Check current memory usage
kubectl top pods -l app=business-logic

# Watch memory trends
watch -n 5 'kubectl top pods -l app=business-logic'
```

2. **Check for OOM kills in logs**:
```bash
kubectl get events --sort-by='.lastTimestamp' | grep OOM
```

3. **Analyze heap dumps** (if available):
```bash
# Trigger heap dump
kubectl exec business-logic-<pod-id> -- jmap -dump:live,format=b,file=/tmp/heap.hprof <pid>

# Copy heap dump locally
kubectl cp business-logic-<pod-id>:/tmp/heap.hprof ./heap.hprof

# Analyze with VisualVM or Eclipse MAT
```

**Solutions**:

**Immediate**:
1. Restart affected instances:
```bash
kubectl delete pod business-logic-<pod-id>
```

2. Increase memory limits (temporary):
```yaml
resources:
  limits:
    memory: 4Gi  # Increased from 2Gi
```

**Long-term**:
1. **Fix memory leaks in code**:
   - Unclosed database connections
   - Event listener leaks
   - Large object retention in cache
   - Circular references

2. **Implement proper cleanup**:
```python
# Example: Proper connection cleanup
try:
    conn = get_db_connection()
    result = conn.execute(query)
finally:
    conn.close()  # Always close
```

3. **Configure garbage collection** (Java):
```bash
JAVA_OPTS="-XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:+PrintGCDetails"
```

### CPU Saturation

**Symptom**: CPU usage consistently above 80%, slow response times.

**Diagnosis Steps**:

1. **Check CPU usage**:
```bash
kubectl top pods -l app=data-processor
```

2. **Profile CPU usage**:
```bash
# For Python services
kubectl exec data-processor-<pod-id> -- py-spy top --pid <pid>

# For Node.js services
kubectl exec api-gateway-<pod-id> -- node --prof app.js
```

3. **Identify hot paths**:
   - Review APM traces
   - Analyze flame graphs
   - Check for inefficient algorithms

**Solutions**:

1. **Optimize hot code paths**:
   - Use profiling to identify bottlenecks
   - Optimize loops and data structures
   - Add caching for expensive computations

2. **Scale horizontally**:
```bash
kubectl scale deployment data-processor --replicas=20
```

3. **Implement rate limiting**:
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/expensive-operation")
@limiter.limit("10/minute")
async def expensive_operation():
    # Process
    pass
```

### Slow Database Queries

**Symptom**: Database queries taking >1 second, high DB CPU usage.

**Diagnosis Steps**:

1. **Identify slow queries**:
```sql
-- PostgreSQL
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

2. **Analyze query plans**:
```sql
EXPLAIN ANALYZE
SELECT * FROM orders
WHERE customer_id = 123
AND created_at > '2025-01-01';
```

3. **Check missing indices**:
```sql
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
AND tablename = 'orders';
```

**Solutions**:

1. **Add missing indices**:
```sql
-- For single column
CREATE INDEX idx_orders_customer_id ON orders(customer_id);

-- For composite index
CREATE INDEX idx_orders_customer_date ON orders(customer_id, created_at);

-- For partial index
CREATE INDEX idx_active_orders ON orders(status) WHERE status = 'active';
```

2. **Optimize queries**:
```sql
-- Bad: SELECT *
SELECT id, name, email FROM users WHERE active = true;

-- Good: Select only needed columns
SELECT id, name FROM users WHERE active = true;

-- Bad: N+1 query
-- Multiple SELECT * FROM orders WHERE customer_id = ?

-- Good: JOIN
SELECT c.*, o.*
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE c.active = true;
```

3. **Implement query caching**:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_customer_orders(customer_id):
    return db.query("SELECT * FROM orders WHERE customer_id = ?", customer_id)
```

4. **Use connection pooling**:
```python
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:pass@localhost/db',
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True
)
```

## Authentication Errors

### Invalid JWT Token (401)

**Symptom**: Requests fail with 401 Unauthorized, error message "Invalid JWT token".

**Possible Causes**:
1. Expired token (>1 hour old)
2. Invalid signature
3. Token format incorrect
4. Token blacklisted (after logout)

**Solutions**:

1. **Refresh expired token**:
```bash
curl -X POST "http://localhost:8001/auth/refresh" \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "your_refresh_token"}'
```

2. **Verify token format**:
```javascript
// JWT should have three parts separated by dots
const parts = token.split('.');
if (parts.length !== 3) {
  console.error('Invalid JWT format');
}

// Decode and inspect
const payload = JSON.parse(atob(parts[1]));
console.log('Token expires at:', new Date(payload.exp * 1000));
```

3. **Check token expiration**:
```python
import jwt
from datetime import datetime

try:
    payload = jwt.decode(token, secret_key, algorithms=['HS256'])
    exp_time = datetime.fromtimestamp(payload['exp'])
    print(f"Token expires at: {exp_time}")
except jwt.ExpiredSignatureError:
    print("Token has expired, need to refresh")
```

### Permission Denied (403)

**Symptom**: Authenticated but request fails with 403 Forbidden.

**Possible Causes**:
1. Insufficient permissions
2. Resource access restricted
3. IP whitelist restrictions
4. Role-based access control (RBAC) denial

**Solutions**:

1. **Check user roles and permissions**:
```bash
curl -X GET "http://localhost:8001/auth/user/me" \
  -H "Authorization: Bearer your_jwt_token"
```

2. **Verify required permissions**:
   - Consult API documentation for required permissions
   - Contact admin to grant necessary roles

3. **Review RBAC policies**:
```yaml
# Example RBAC policy
roles:
  - name: viewer
    permissions: [read:orders, read:customers]
  - name: editor
    permissions: [read:*, write:orders, write:customers]
  - name: admin
    permissions: [*:*]
```

## Service Health Issues

### Failed Health Checks

**Symptom**: Service marked as unhealthy, not receiving traffic.

**Diagnosis Steps**:

1. **Check health endpoint directly**:
```bash
kubectl exec api-gateway-<pod-id> -- curl http://localhost:8080/health
```

2. **Review health check configuration**:
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

3. **Check service logs**:
```bash
kubectl logs api-gateway-<pod-id> | grep health
```

**Possible Issues**:

1. **Database connectivity**:
   - Health check includes DB ping
   - DB connection pool exhausted
   - DB credentials incorrect

2. **Dependency unavailable**:
   - Required service is down
   - Network partition
   - DNS resolution failure

3. **Slow startup**:
   - Service not ready within initialDelaySeconds
   - Increase initialDelaySeconds to 60s

**Solutions**:

1. **Fix underlying issue** (DB, dependencies, etc.)

2. **Adjust health check timing**:
```yaml
livenessProbe:
  initialDelaySeconds: 60  # Increased
  periodSeconds: 15        # Less frequent
  timeoutSeconds: 10       # More time to respond
  failureThreshold: 5      # More tolerance
```

3. **Implement graceful degradation**:
```python
@app.get("/health")
def health_check():
    health = {
        "status": "healthy",
        "checks": {}
    }

    # Check database (non-blocking)
    try:
        db.execute("SELECT 1")
        health["checks"]["database"] = "healthy"
    except Exception:
        health["checks"]["database"] = "degraded"
        health["status"] = "degraded"  # Still return 200

    return health
```

### Service Crashes

**Symptom**: Service repeatedly restarting, CrashLoopBackOff status.

**Diagnosis Steps**:

1. **Check pod status**:
```bash
kubectl get pod data-processor-<pod-id>
```

2. **Review crash logs**:
```bash
kubectl logs data-processor-<pod-id> --previous
```

3. **Check events**:
```bash
kubectl describe pod data-processor-<pod-id>
```

**Common Causes**:

1. **OOM Kills**:
```
Last State:     Terminated
  Reason:       OOMKilled
  Exit Code:    137
```
**Solution**: Increase memory limits or fix memory leak

2. **Unhandled Exceptions**:
```python
# Check logs for:
# - Uncaught exceptions
# - Segmentation faults
# - Assertion errors
```
**Solution**: Add proper error handling

3. **Missing Dependencies**:
```
Error: Cannot connect to database
```
**Solution**: Check environment variables, secrets

4. **Port Already in Use**:
```
Error: Address already in use: 8080
```
**Solution**: Change port or fix port conflict

## Database Problems

### Connection Pool Exhaustion

**Symptom**: Errors like "FATAL: sorry, too many clients".

**Diagnosis**:
```sql
-- Check current connections
SELECT count(*) FROM pg_stat_activity;

-- Check by application
SELECT application_name, count(*)
FROM pg_stat_activity
GROUP BY application_name;

-- Check idle connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'idle';
```

**Solutions**:

1. **Increase max connections** (PostgreSQL):
```sql
ALTER SYSTEM SET max_connections = 300;
SELECT pg_reload_conf();
```

2. **Configure connection pooling**:
```python
from sqlalchemy import create_engine

engine = create_engine(
    'postgresql://user:pass@localhost/db',
    pool_size=20,           # Base pool size
    max_overflow=30,        # Additional connections
    pool_timeout=30,        # Wait time for connection
    pool_recycle=3600,      # Recycle connections after 1 hour
    pool_pre_ping=True      # Verify connection before using
)
```

3. **Close idle connections**:
```sql
-- Kill idle connections older than 1 hour
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle'
AND state_change < now() - interval '1 hour';
```

### Replication Lag

**Symptom**: Read replicas showing stale data, lag > 5 seconds.

**Diagnosis**:
```sql
-- Check replication lag (on primary)
SELECT
  client_addr,
  state,
  sent_lsn,
  write_lsn,
  flush_lsn,
  replay_lsn,
  sync_state,
  pg_wal_lsn_diff(sent_lsn, replay_lsn) AS lag_bytes
FROM pg_stat_replication;
```

**Solutions**:

1. **Identify bottleneck**:
   - Network bandwidth
   - Disk I/O on replica
   - Large transactions on primary

2. **Optimize replication**:
```sql
-- On primary
ALTER SYSTEM SET wal_compression = on;
ALTER SYSTEM SET max_wal_senders = 10;

-- On replica
ALTER SYSTEM SET hot_standby_feedback = on;
```

3. **Temporary workaround**:
   - Route reads to primary (reduced scalability)
   - Implement read-your-writes consistency
   - Add eventual consistency warnings to UI

## Network and Connectivity

### DNS Resolution Failures

**Symptom**: "Could not resolve host" errors.

**Diagnosis**:
```bash
# Test DNS resolution
kubectl exec api-gateway-<pod-id> -- nslookup auth-service

# Check DNS config
kubectl exec api-gateway-<pod-id> -- cat /etc/resolv.conf
```

**Solutions**:
1. Check service names and namespaces
2. Verify DNS service is running
3. Use fully qualified names: `service-name.namespace.svc.cluster.local`

### Timeout Errors

**Symptom**: Requests timeout after 30 seconds.

**Solutions**:

1. **Increase timeout values**:
```python
import requests

response = requests.get(
    url,
    timeout=(10, 60)  # (connect, read) in seconds
)
```

2. **Implement circuit breaker**:
```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
def call_external_api():
    return requests.get(external_url, timeout=10)
```

## Monitoring and Debugging

### Enable Debug Logging

**Temporarily enable debug logs**:
```bash
kubectl set env deployment/api-gateway LOG_LEVEL=DEBUG
```

**Check logs**:
```bash
kubectl logs -f api-gateway-<pod-id> | grep DEBUG
```

### Distributed Tracing

**View trace for specific request**:
1. Get request ID from error response
2. Query tracing system (Jaeger/Zipkin)
3. Analyze spans for bottlenecks

## Emergency Procedures

### Emergency Rollback

```bash
# Rollback deployment
kubectl rollout undo deployment/api-gateway

# Verify rollback
kubectl rollout status deployment/api-gateway

# Check if issues resolved
curl http://localhost:8001/health
```

### Emergency Scale Down

```bash
# If service causing issues
kubectl scale deployment problematic-service --replicas=0

# Monitor other services
kubectl get pods --watch
```

### Emergency Maintenance Mode

```bash
# Enable maintenance mode at load balancer
# Returns 503 with custom message
kubectl apply -f maintenance-mode.yaml
```

## Getting Help

When contacting support, include:
1. Service name and version
2. Error messages and stack traces
3. Request ID for failed requests
4. Recent changes or deployments
5. Relevant logs and metrics

**Support Channels**:
- Email: support@example.com
- Slack: #ops-support
- On-call: PagerDuty escalation
- Emergency: Call on-call engineer directly
