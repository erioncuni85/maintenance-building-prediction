# Komuniteti Predictive Maintenance - Deployment Guide

This guide covers the deployment of the Komuniteti Predictive Maintenance pipeline in various environments.

## 🚀 Quick Deploy

### Local Development

```bash
# 1. Clone and setup
git clone <repository-url>
cd komuniteti-maintenance-prediction
cp environment_template.env .env
# Edit .env with your database credentials

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialize ZenML
zenml init

# 4. Create sample data and train model
python -m src.pipelines.training_pipeline --create-sample-data
python -m src.pipelines.training_pipeline --data-source csv

# 5. Start API server
python -m src.api.serve
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

## 🏗️ Production Deployment

### 1. Environment Setup

Create production environment file:
```bash
# Production configuration
DB_HOST=production-mysql-host
DB_PORT=3306
DB_DATABASE=komuniteti_production
DB_USERNAME=ml_service_user
DB_PASSWORD=secure_password

LARAVEL_API_URL=https://api.komuniteti.com/api
LARAVEL_API_TOKEN=production_api_token

ENVIRONMENT=production
DEBUG=false
```

### 2. Docker Production Build

```dockerfile
# Dockerfile.production
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser
RUN chown -R mluser:mluser /app
USER mluser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["python", "-m", "src.api.serve", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. Kubernetes Deployment

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: komuniteti-ml

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-config
  namespace: komuniteti-ml
data:
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  ENVIRONMENT: "production"
  DEBUG: "false"

---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ml-secrets
  namespace: komuniteti-ml
type: Opaque
stringData:
  DB_HOST: "mysql-service"
  DB_PORT: "3306"
  DB_DATABASE: "komuniteti"
  DB_USERNAME: "ml_user"
  DB_PASSWORD: "secure_password"
  LARAVEL_API_TOKEN: "production_token"

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: komuniteti-ml-api
  namespace: komuniteti-ml
spec:
  replicas: 3
  selector:
    matchLabels:
      app: komuniteti-ml-api
  template:
    metadata:
      labels:
        app: komuniteti-ml-api
    spec:
      containers:
      - name: ml-api
        image: komuniteti/ml-api:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: ml-config
        - secretRef:
            name: ml-secrets
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: komuniteti-ml-service
  namespace: komuniteti-ml
spec:
  selector:
    app: komuniteti-ml-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: komuniteti-ml-ingress
  namespace: komuniteti-ml
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: ml-api.komuniteti.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: komuniteti-ml-service
            port:
              number: 80

---
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
  namespace: komuniteti-ml
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

### 4. Deployment Script

```bash
#!/bin/bash
# deploy.sh

set -e

echo "🚀 Deploying Komuniteti ML API..."

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -f Dockerfile.production -t komuniteti/ml-api:latest .
docker tag komuniteti/ml-api:latest komuniteti/ml-api:$(git rev-parse --short HEAD)

# Push to registry
echo "⬆️ Pushing to registry..."
docker push komuniteti/ml-api:latest
docker push komuniteti/ml-api:$(git rev-parse --short HEAD)

# Deploy to Kubernetes
echo "☸️ Deploying to Kubernetes..."
kubectl apply -f k8s/

# Wait for deployment
echo "⏳ Waiting for deployment..."
kubectl rollout status deployment/komuniteti-ml-api -n komuniteti-ml

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n komuniteti-ml
curl -f https://ml-api.komuniteti.com/health

echo "🎉 Deployment complete!"
```

## 🔄 CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy ML API

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    
    - name: Run tests
      run: pytest tests/ -v
    
    - name: Lint code
      run: |
        black --check src/
        isort --check src/

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -f Dockerfile.production -t komuniteti/ml-api:${{ github.sha }} .
        docker tag komuniteti/ml-api:${{ github.sha }} komuniteti/ml-api:latest
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push komuniteti/ml-api:${{ github.sha }}
        docker push komuniteti/ml-api:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Kubernetes
      run: |
        echo "${{ secrets.KUBECONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
        kubectl set image deployment/komuniteti-ml-api ml-api=komuniteti/ml-api:${{ github.sha }} -n komuniteti-ml
        kubectl rollout status deployment/komuniteti-ml-api -n komuniteti-ml
```

## 📊 Monitoring Setup

### Prometheus Monitoring

```yaml
# monitoring/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'komuniteti-ml-api'
      static_configs:
      - targets: ['komuniteti-ml-service:80']
      metrics_path: /metrics
      scrape_interval: 30s

---
# monitoring/grafana-dashboard.json
{
  "dashboard": {
    "title": "Komuniteti ML API",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "http_request_duration_seconds{job=\"komuniteti-ml-api\"}"
          }
        ]
      },
      {
        "title": "Prediction Requests",
        "type": "graph", 
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"komuniteti-ml-api\",endpoint=\"/predict\"}[5m])"
          }
        ]
      },
      {
        "title": "Model Performance",
        "type": "singlestat",
        "targets": [
          {
            "expr": "model_accuracy{job=\"komuniteti-ml-api\"}"
          }
        ]
      }
    ]
  }
}
```

### Health Checks

```bash
#!/bin/bash
# health-check.sh

API_URL="${API_URL:-http://localhost:8000}"

echo "🏥 Checking API health..."

# Basic health check
response=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/health)
if [ $response -eq 200 ]; then
    echo "✅ API is healthy"
else
    echo "❌ API health check failed (HTTP $response)"
    exit 1
fi

# Model health check
model_info=$(curl -s $API_URL/model/info)
if echo "$model_info" | jq -e '.model_version' > /dev/null; then
    echo "✅ Model is loaded"
    echo "📊 Model version: $(echo "$model_info" | jq -r '.model_version')"
else
    echo "❌ Model not available"
    exit 1
fi

# Performance test
echo "🏃 Running performance test..."
start_time=$(date +%s%N)
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "building_id": 123,
    "building_type": "residential",
    "asset_type": "elevator",
    "city": "Tirana",
    "country": "Albania",
    "building_area": 2500.0,
    "building_floors": 10
  }' > /dev/null
end_time=$(date +%s%N)
duration=$(( (end_time - start_time) / 1000000 ))

if [ $duration -lt 1000 ]; then
    echo "✅ Response time: ${duration}ms (Good)"
else
    echo "⚠️ Response time: ${duration}ms (Slow)"
fi

echo "🎉 Health check complete!"
```

## 🔒 Security Considerations

### Network Security

1. **API Authentication**
```python
# Add to src/api/serve.py
from fastapi.security import HTTPBearer
from fastapi import Security, HTTPException

security = HTTPBearer()

async def verify_token(credentials: HTTPCredentials = Security(security)):
    if credentials.credentials != os.getenv("API_TOKEN"):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials

# Protect endpoints
@app.post("/predict")
async def predict_maintenance(
    request: PredictionRequest,
    credentials: HTTPCredentials = Security(verify_token)
):
    # ... prediction logic
```

2. **Rate Limiting**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("100/minute")
async def predict_maintenance(request: Request, prediction_request: PredictionRequest):
    # ... prediction logic
```

### Data Security

1. **Environment Variables**: Never commit secrets to version control
2. **Database Encryption**: Use encrypted connections to database
3. **API HTTPS**: Always use HTTPS in production
4. **Input Validation**: Validate all inputs using Pydantic schemas

## 🔄 Backup and Recovery

### Model Backup

```bash
#!/bin/bash
# backup-models.sh

BACKUP_DIR="/backups/models/$(date +%Y-%m-%d)"
mkdir -p $BACKUP_DIR

# Backup model artifacts
cp -r models/ $BACKUP_DIR/
tar -czf $BACKUP_DIR/models.tar.gz -C $BACKUP_DIR models/

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR/models.tar.gz s3://komuniteti-ml-backups/

echo "✅ Model backup completed: $BACKUP_DIR"
```

### Disaster Recovery

```bash
#!/bin/bash
# restore-models.sh

BACKUP_DATE=${1:-$(date +%Y-%m-%d)}
BACKUP_FILE="s3://komuniteti-ml-backups/models-$BACKUP_DATE.tar.gz"

echo "🔄 Restoring models from $BACKUP_DATE..."

# Download backup
aws s3 cp $BACKUP_FILE /tmp/models.tar.gz

# Extract models
tar -xzf /tmp/models.tar.gz -C /app/

# Restart API
kubectl rollout restart deployment/komuniteti-ml-api -n komuniteti-ml

echo "✅ Model restoration completed"
```

## 📋 Maintenance Schedule

### Daily
- [ ] Check API health and performance
- [ ] Monitor prediction accuracy
- [ ] Review error logs

### Weekly  
- [ ] Model performance evaluation
- [ ] Data quality assessment
- [ ] Security updates

### Monthly
- [ ] Model retraining with new data
- [ ] Performance benchmarking
- [ ] Capacity planning review

### Quarterly
- [ ] Full system backup
- [ ] Disaster recovery testing
- [ ] Architecture review

## 🚨 Troubleshooting

### Common Issues

**Deployment Fails:**
```bash
# Check pod status
kubectl get pods -n komuniteti-ml

# Check logs
kubectl logs deployment/komuniteti-ml-api -n komuniteti-ml

# Check events
kubectl get events -n komuniteti-ml --sort-by='.lastTimestamp'
```

**Model Not Loading:**
```bash
# Check model files
kubectl exec -it deployment/komuniteti-ml-api -n komuniteti-ml -- ls -la /app/models/

# Force model reload
curl -X POST https://ml-api.komuniteti.com/model/reload
```

**Performance Issues:**
```bash
# Check resource usage
kubectl top pods -n komuniteti-ml

# Scale up replicas
kubectl scale deployment komuniteti-ml-api --replicas=5 -n komuniteti-ml
```

## 📞 Support Contacts

- **DevOps Team**: devops@komuniteti.com
- **ML Team**: ml-team@komuniteti.com
- **On-Call**: +383-xx-xxx-xxx

---

**Deployment Guide v1.0 - Built for Komuniteti Predictive Maintenance** 