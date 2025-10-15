# Deployment Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Deployment Environments](#deployment-environments)
4. [Deployment Process](#deployment-process)
5. [Rollback Procedures](#rollback-procedures)
6. [Database Migrations](#database-migrations)
7. [Configuration Management](#configuration-management)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [Blue-Green Deployment](#blue-green-deployment)
10. [Post-Deployment Verification](#post-deployment-verification)

## Overview

This guide covers the deployment process for all microservices in our distributed system. We follow a structured deployment process to ensure reliability, minimize downtime, and enable quick rollbacks if issues arise.

### Deployment Strategy

We use a **rolling deployment** strategy with automated health checks:
- Deploy to one instance at a time
- Verify health before proceeding
- Automatic rollback on failure
- Zero-downtime deployments

### Deployment Philosophy

**Key Principles**:
1. **Automate Everything**: Manual deployments lead to errors
2. **Deploy Frequently**: Small, incremental changes reduce risk
3. **Monitor Continuously**: Watch metrics during and after deployment
4. **Rollback Quickly**: Don't hesitate to rollback if issues detected
5. **Test Thoroughly**: Every deployment tested in staging first

## Prerequisites

### Required Tools

Before deploying, ensure you have:

1. **kubectl** (v1.28+): Kubernetes CLI
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Verify installation
kubectl version --client
```

2. **Helm** (v3.12+): Kubernetes package manager
```bash
# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify installation
helm version
```

3. **Docker** (v24+): Container runtime
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Verify installation
docker version
```

4. **AWS CLI** (v2): For ECR access
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure credentials
aws configure
```

### Access Requirements

You need:
- AWS credentials with ECR read access
- Kubernetes cluster access (RBAC role: deployer)
- VPN access for production deployments
- PagerDuty access for on-call notifications

### Pre-Deployment Checklist

- [ ] All tests passing in CI/CD
- [ ] Code reviewed and approved
- [ ] Staging deployment successful
- [ ] Database migrations tested
- [ ] Feature flags configured
- [ ] Rollback plan documented
- [ ] On-call engineer notified
- [ ] Deployment window scheduled

## Deployment Environments

### Development Environment

**Purpose**: Local development and testing

**Characteristics**:
- Single-node Kubernetes cluster (minikube/kind)
- Local database instances
- Mock external services
- Debug logging enabled
- No resource limits

**Access**:
```bash
kubectl config use-context dev
kubectl get nodes
```

### Staging Environment

**Purpose**: Pre-production testing and validation

**Characteristics**:
- Multi-node Kubernetes cluster (3 nodes)
- Replicated database instances
- Real external service integrations
- Production-like configuration
- Resource limits enforced

**Access**:
```bash
kubectl config use-context staging
kubectl get nodes
```

**Deployment Command**:
```bash
./deploy.sh staging api-gateway v2.3.1
```

### Production Environment

**Purpose**: Live customer-facing services

**Characteristics**:
- Multi-AZ Kubernetes cluster (9 nodes)
- Highly available database setup
- All integrations live
- Strict resource limits
- Enhanced monitoring and alerting

**Access**:
```bash
kubectl config use-context production
kubectl get nodes
```

**Deployment Command**:
```bash
# Requires approval
./deploy.sh production api-gateway v2.3.1 --approve
```

## Deployment Process

### Step 1: Build Docker Image

**Build locally**:
```bash
# Navigate to service directory
cd services/api-gateway

# Build image
docker build -t api-gateway:v2.3.1 .

# Tag for ECR
docker tag api-gateway:v2.3.1 123456789.dkr.ecr.us-east-1.amazonaws.com/api-gateway:v2.3.1

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/api-gateway:v2.3.1
```

**Or use CI/CD** (recommended):
```bash
# Trigger build via git tag
git tag v2.3.1
git push origin v2.3.1

# CI/CD automatically builds and pushes image
```

### Step 2: Update Kubernetes Manifests

**Update deployment YAML**:
```yaml
# k8s/api-gateway/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: production
spec:
  replicas: 5
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
        version: v2.3.1
    spec:
      containers:
      - name: api-gateway
        image: 123456789.dkr.ecr.us-east-1.amazonaws.com/api-gateway:v2.3.1
        ports:
        - containerPort: 8080
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: host
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 2Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
```

### Step 3: Apply Configuration Updates

**Update ConfigMaps** (if needed):
```bash
kubectl apply -f k8s/api-gateway/configmap.yaml -n production
```

**Update Secrets** (if needed):
```bash
# Create secret from file
kubectl create secret generic api-keys \
  --from-file=api-key=./secrets/api-key.txt \
  -n production \
  --dry-run=client -o yaml | kubectl apply -f -
```

### Step 4: Deploy Application

**Using kubectl**:
```bash
# Apply deployment
kubectl apply -f k8s/api-gateway/deployment.yaml -n production

# Watch rollout status
kubectl rollout status deployment/api-gateway -n production

# Expected output:
# Waiting for deployment "api-gateway" rollout to finish: 1 out of 5 new replicas have been updated...
# Waiting for deployment "api-gateway" rollout to finish: 2 out of 5 new replicas have been updated...
# ...
# deployment "api-gateway" successfully rolled out
```

**Using Helm**:
```bash
# Update Helm chart
helm upgrade api-gateway ./charts/api-gateway \
  --namespace production \
  --set image.tag=v2.3.1 \
  --set replicaCount=5 \
  --wait \
  --timeout 10m
```

**Using Custom Deploy Script**:
```bash
# Our deploy script with safety checks
./deploy.sh production api-gateway v2.3.1
```

### Step 5: Monitor Deployment

**Watch pod rollout**:
```bash
watch kubectl get pods -n production -l app=api-gateway
```

**Check logs**:
```bash
# Tail logs from new pods
kubectl logs -f deployment/api-gateway -n production --tail=100
```

**Monitor metrics**:
```bash
# Check latency during deployment
curl -X GET "http://api.example.com/v1/metrics/latency?service=api-gateway&period=15m"
```

**Watch for errors**:
```bash
# Monitor error rates
kubectl logs -n production -l app=api-gateway | grep -i error
```

### Step 6: Verify Deployment

**Check service health**:
```bash
curl -X GET "http://api.example.com/v1/health?service=api-gateway"
```

**Run smoke tests**:
```bash
# Execute automated smoke tests
./tests/smoke/run_smoke_tests.sh production api-gateway
```

**Verify version**:
```bash
# Check deployed version
kubectl get deployment api-gateway -n production -o jsonpath='{.spec.template.spec.containers[0].image}'
```

**Check metrics**:
- Latency: Should remain stable or improve
- Error rate: Should not increase
- Throughput: Should handle expected load
- Memory/CPU: Should be within normal ranges

## Rollback Procedures

### When to Rollback

Rollback immediately if:
- Error rate increases by >50%
- P95 latency increases by >100ms
- Critical functionality broken
- Security vulnerability introduced
- Data corruption detected

### Quick Rollback

**Using kubectl**:
```bash
# Check rollout history
kubectl rollout history deployment/api-gateway -n production

# Rollback to previous version
kubectl rollout undo deployment/api-gateway -n production

# Verify rollback
kubectl rollout status deployment/api-gateway -n production
```

**Rollback to specific revision**:
```bash
# View revision details
kubectl rollout history deployment/api-gateway -n production --revision=3

# Rollback to specific revision
kubectl rollout undo deployment/api-gateway -n production --to-revision=3
```

**Using Helm**:
```bash
# View release history
helm history api-gateway -n production

# Rollback to previous release
helm rollback api-gateway -n production

# Rollback to specific revision
helm rollback api-gateway 5 -n production
```

### Post-Rollback Actions

1. **Notify team** of rollback via Slack/PagerDuty
2. **Create incident report** documenting issue
3. **Analyze root cause** of deployment failure
4. **Update tests** to catch issue in future
5. **Plan fix** and schedule re-deployment

## Database Migrations

### Migration Strategy

We use **forward-only migrations** with backward compatibility:
1. Deploy schema changes that are backward compatible
2. Deploy application code
3. Remove deprecated columns/tables in future release

### Running Migrations

**Pre-deployment migration** (backward compatible):
```bash
# Connect to database
kubectl exec -it postgresql-primary-0 -n production -- psql -U postgres

# Or use migration tool
./migrate.sh production up

# Example migration
CREATE TABLE new_orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    amount DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT NOW()
);

# Add new column (with default)
ALTER TABLE orders ADD COLUMN status VARCHAR(50) DEFAULT 'pending';

# Create index
CREATE INDEX idx_orders_status ON orders(status);
```

**Post-deployment cleanup** (next release):
```sql
-- Remove deprecated column (after all code updated)
ALTER TABLE orders DROP COLUMN old_status;

-- Remove deprecated table
DROP TABLE IF EXISTS legacy_orders;
```

### Migration Best Practices

1. **Always test migrations** on staging first
2. **Take database backup** before migration
3. **Use transactions** where possible
4. **Monitor query performance** during migration
5. **Keep migrations small** and incremental
6. **Add indexes after data load** for large tables

### Zero-Downtime Migration Example

**Scenario**: Rename column from `user_name` to `username`

**Step 1** (Release N):
```sql
-- Add new column
ALTER TABLE users ADD COLUMN username VARCHAR(255);

-- Copy data
UPDATE users SET username = user_name;

-- Deploy application code that writes to both columns
```

**Step 2** (Release N+1):
```sql
-- Deploy application code that reads from new column only
-- (Still writes to both for safety)
```

**Step 3** (Release N+2):
```sql
-- Remove old column
ALTER TABLE users DROP COLUMN user_name;

-- Deploy application code that only uses new column
```

## Configuration Management

### Environment Variables

**Deployment-specific configuration**:
```yaml
env:
- name: ENVIRONMENT
  value: "production"
- name: LOG_LEVEL
  value: "INFO"
- name: MAX_CONNECTIONS
  value: "100"
```

**Secrets**:
```yaml
env:
- name: DB_PASSWORD
  valueFrom:
    secretKeyRef:
      name: db-credentials
      key: password
- name: API_KEY
  valueFrom:
    secretKeyRef:
      name: api-keys
      key: openai-key
```

**ConfigMaps**:
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: api-gateway-config
data:
  nginx.conf: |
    upstream backend {
      server business-logic:8080;
    }
    server {
      listen 8080;
      location / {
        proxy_pass http://backend;
      }
    }
```

### Feature Flags

**Using environment variables**:
```python
ENABLE_NEW_FEATURE = os.getenv('ENABLE_NEW_FEATURE', 'false').lower() == 'true'

if ENABLE_NEW_FEATURE:
    # New code path
    process_with_new_logic()
else:
    # Old code path
    process_with_old_logic()
```

**Using external service** (LaunchDarkly, etc.):
```python
from ldclient import get

ld_client = get()

if ld_client.variation('new-checkout-flow', user, False):
    # New checkout flow
    pass
```

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Login to ECR
        run: |
          aws ecr get-login-password --region us-east-1 | \
          docker login --username AWS --password-stdin $ECR_REGISTRY

      - name: Build and push image
        run: |
          docker build -t api-gateway:${{ github.ref_name }} .
          docker tag api-gateway:${{ github.ref_name }} $ECR_REGISTRY/api-gateway:${{ github.ref_name }}
          docker push $ECR_REGISTRY/api-gateway:${{ github.ref_name }}

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Staging
        run: |
          kubectl set image deployment/api-gateway \
            api-gateway=$ECR_REGISTRY/api-gateway:${{ github.ref_name }} \
            -n staging

      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/api-gateway -n staging

      - name: Run smoke tests
        run: ./tests/smoke/run_smoke_tests.sh staging

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to Production
        run: |
          kubectl set image deployment/api-gateway \
            api-gateway=$ECR_REGISTRY/api-gateway:${{ github.ref_name }} \
            -n production

      - name: Monitor deployment
        run: |
          kubectl rollout status deployment/api-gateway -n production
          ./scripts/monitor_deployment.sh api-gateway 300

      - name: Notify Slack
        if: always()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "Deployment ${{ job.status }}: api-gateway ${{ github.ref_name }}"
            }
```

### Pipeline Stages

1. **Build**: Compile code, run unit tests
2. **Test**: Run integration tests
3. **Package**: Build Docker image
4. **Push**: Push image to ECR
5. **Deploy Staging**: Deploy to staging environment
6. **Smoke Test**: Run automated smoke tests
7. **Manual Approval**: Require approval for production
8. **Deploy Production**: Rolling deployment to production
9. **Verify**: Post-deployment verification
10. **Notify**: Send notifications to team

## Blue-Green Deployment

### When to Use

Use blue-green deployment for:
- Major version upgrades
- Database schema changes
- High-risk deployments
- Instant rollback requirement

### Setup

**Step 1: Deploy Green Environment**:
```bash
# Create green deployment
kubectl apply -f k8s/api-gateway/deployment-green.yaml

# Wait for green to be ready
kubectl wait --for=condition=available deployment/api-gateway-green -n production

# Test green environment
curl http://api-gateway-green.production.svc.cluster.local/health
```

**Step 2: Switch Traffic**:
```bash
# Update service selector to point to green
kubectl patch service api-gateway -n production -p '{"spec":{"selector":{"version":"green"}}}'

# Verify traffic switched
kubectl get service api-gateway -n production -o yaml
```

**Step 3: Monitor**:
```bash
# Monitor metrics for 10 minutes
./scripts/monitor_deployment.sh api-gateway 600
```

**Step 4: Cleanup or Rollback**:

If successful:
```bash
# Remove blue deployment
kubectl delete deployment api-gateway-blue -n production
```

If issues detected:
```bash
# Switch back to blue
kubectl patch service api-gateway -n production -p '{"spec":{"selector":{"version":"blue"}}}'
```

## Post-Deployment Verification

### Automated Checks

**Health check**:
```bash
curl http://api.example.com/v1/health
# Expected: {"status": "healthy"}
```

**Version check**:
```bash
curl http://api.example.com/v1/version
# Expected: {"version": "v2.3.1"}
```

**Smoke tests**:
```bash
./tests/smoke/run_smoke_tests.sh production
```

### Manual Verification

1. **Check key user flows**:
   - User login
   - Create order
   - Process payment
   - Generate report

2. **Verify metrics**:
   - Check Grafana dashboards
   - Compare with pre-deployment baseline
   - Verify no anomalies

3. **Review logs**:
   - Check for error spikes
   - Verify no unexpected warnings
   - Confirm normal operation

### Success Criteria

Deployment is successful if:
- [ ] All health checks passing
- [ ] Error rate < 1%
- [ ] P95 latency < 200ms
- [ ] No critical alerts fired
- [ ] All smoke tests passing
- [ ] Key user flows working
- [ ] No rollback required after 1 hour

## Deployment Checklist

### Pre-Deployment

- [ ] All tests passing in CI/CD
- [ ] Code reviewed and approved
- [ ] Deployment plan documented
- [ ] Rollback plan prepared
- [ ] Database migrations tested
- [ ] Staging deployment successful
- [ ] Feature flags configured
- [ ] Monitoring alerts configured
- [ ] On-call engineer notified
- [ ] Change ticket created
- [ ] Deployment window scheduled
- [ ] Stakeholders informed

### During Deployment

- [ ] Backup database
- [ ] Run database migrations
- [ ] Deploy application
- [ ] Monitor rollout progress
- [ ] Watch error rates
- [ ] Check latency metrics
- [ ] Verify health checks
- [ ] Run smoke tests

### Post-Deployment

- [ ] All pods running
- [ ] Health checks passing
- [ ] Metrics within normal ranges
- [ ] Smoke tests passed
- [ ] User flows verified
- [ ] No critical alerts
- [ ] Documentation updated
- [ ] Change ticket closed
- [ ] Team notified of completion

## Troubleshooting Deployments

### Pod Not Starting

**Check pod status**:
```bash
kubectl describe pod api-gateway-<pod-id> -n production
```

**Common issues**:
- Image pull errors: Check ECR credentials
- OOM kills: Increase memory limits
- Failed health checks: Increase initialDelaySeconds

### Deployment Stuck

**Check rollout status**:
```bash
kubectl rollout status deployment/api-gateway -n production
```

**Force restart**:
```bash
kubectl rollout restart deployment/api-gateway -n production
```

### High Error Rate After Deployment

**Immediate actions**:
1. Check error logs
2. Verify configuration
3. Rollback if critical

**Rollback**:
```bash
kubectl rollout undo deployment/api-gateway -n production
```

## Best Practices

1. **Deploy during low-traffic periods**
2. **Always deploy to staging first**
3. **Monitor for at least 1 hour post-deployment**
4. **Keep deployments small and frequent**
5. **Use feature flags for risky changes**
6. **Automate everything possible**
7. **Document every deployment**
8. **Have a rollback plan ready**
9. **Communicate with stakeholders**
10. **Learn from every deployment**

## Support

For deployment issues:
- Slack: #deployments
- Email: devops@example.com
- On-call: PagerDuty escalation
- Documentation: https://docs.example.com/deployments
