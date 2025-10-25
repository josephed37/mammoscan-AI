# Deployment Guide

This document provides comprehensive instructions for deploying Mammoscan AI to various environments.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Local Deployment with Docker Compose](#local-deployment-with-docker-compose)
- [Google Cloud Platform Deployment](#google-cloud-platform-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Environment Variables](#environment-variables)
- [Monitoring and Health Checks](#monitoring-and-health-checks)
- [Troubleshooting](#troubleshooting)

## Overview

Mammoscan AI consists of three main components:

1. **Backend API** (Go): Handles image preprocessing and ONNX model inference
2. **Frontend Web App** (Streamlit): Interactive user interface for image upload and analysis
3. **ML Model** (ONNX): 99MB champion model stored in Google Cloud Storage

The application is designed to be cloud-native and can be deployed locally using Docker Compose or to Google Cloud Platform using Cloud Run.

## Prerequisites

### For Local Deployment

- Docker Engine 20.10+
- Docker Compose 2.0+
- 2GB of available RAM
- Internet connection (to download model from GCS)

### For GCP Deployment

- Google Cloud Platform account
- `gcloud` CLI installed and configured
- Project with billing enabled
- Required GCP APIs enabled:
  - Cloud Run API
  - Cloud Build API
  - Container Registry API
  - Cloud Storage API
- Service account with appropriate permissions

### For CI/CD

- GitHub repository
- Workload Identity Federation configured
- GitHub Actions enabled

## Local Deployment with Docker Compose

### Quick Start

1. **Clone the repository**:

   ```bash
   git clone https://github.com/josephed37/mammoscan-AI.git
   cd mammoscan-AI
   ```

2. **Start the services**:

   ```bash
   make docker-up
   ```

   Or using Docker Compose directly:

   ```bash
   docker compose -f deployments/docker-compose.yml up
   ```

3. **Access the application**:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8080
   - Health check: http://localhost:8080/healthy

### Configuration

The `deployments/docker-compose.yml` file defines two services:

```yaml
services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.api
    ports:
      - "8080:8080"
    environment:
      - MODEL_GCS_BUCKET=mammoscan-ai-models
      - MODEL_GCS_OBJECT=champion_model.onnx
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: ./web
      dockerfile: Dockerfile.web
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://backend:8080/api/v1/predict
    depends_on:
      backend:
        condition: service_healthy
```

### Development Mode

For development with local model file (faster startup):

```bash
docker build -f backend/Dockerfile.dev -t mammoscan-backend:dev backend/
docker run -p 8080:8080 mammoscan-backend:dev
```

### Useful Commands

```bash
# View logs
make docker-logs

# Stop services
make docker-down

# Rebuild from scratch
make docker-rebuild

# View service status
docker compose -f deployments/docker-compose.yml ps
```

## Google Cloud Platform Deployment

### Architecture

The application is deployed to GCP Cloud Run with the following architecture:

- **Backend Service**: Serverless container running Go API
- **Frontend Service**: Serverless container running Streamlit app
- **Model Storage**: Google Cloud Storage bucket
- **Container Registry**: Google Container Registry (GCR) for Docker images

### Setup GCP Project

1. **Create a new GCP project**:

   ```bash
   export PROJECT_ID="mammoscan-ai"
   export REGION="europe-west1"

   gcloud projects create $PROJECT_ID
   gcloud config set project $PROJECT_ID
   ```

2. **Enable required APIs**:

   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   gcloud services enable storage.googleapis.com
   ```

3. **Create GCS bucket for model storage**:

   ```bash
   gsutil mb -l $REGION gs://mammoscan-ai-models
   ```

4. **Upload the ONNX model**:

   ```bash
   gsutil cp models/saved_models/champion_model.onnx gs://mammoscan-ai-models/
   ```

### Manual Deployment

#### Step 1: Build and Push Container Images

```bash
# Build images using Cloud Build
gcloud builds submit --config=deployments/cloudbuild.yaml .
```

This creates:
- `gcr.io/mammoscan-ai/mammoscan-backend:latest`
- `gcr.io/mammoscan-ai/mammoscan-frontend:latest`

#### Step 2: Deploy Backend Service

```bash
gcloud run deploy mammoscan-backend \
  --image gcr.io/$PROJECT_ID/mammoscan-backend:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --set-env-vars MODEL_GCS_BUCKET=mammoscan-ai-models,MODEL_GCS_OBJECT=champion_model.onnx
```

#### Step 3: Get Backend URL

```bash
BACKEND_URL=$(gcloud run services describe mammoscan-backend \
  --region $REGION \
  --format 'value(status.url)')
```

#### Step 4: Deploy Frontend Service

```bash
gcloud run deploy mammoscan-frontend \
  --image gcr.io/$PROJECT_ID/mammoscan-frontend:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars API_URL=${BACKEND_URL}/api/v1/predict
```

#### Step 5: Access the Application

```bash
gcloud run services describe mammoscan-frontend \
  --region $REGION \
  --format 'value(status.url)'
```

### Cloud Run Configuration

#### Backend Service

- **Memory**: 1GB (sufficient for ONNX model)
- **CPU**: 1 vCPU (allows parallel request handling)
- **Concurrency**: 80 (Cloud Run default)
- **Timeout**: 300 seconds
- **Min Instances**: 0 (scales to zero when idle)
- **Max Instances**: 10 (can be adjusted based on load)

#### Frontend Service

- **Memory**: 512MB (default, adequate for Streamlit)
- **CPU**: 1 vCPU
- **Concurrency**: 80
- **Timeout**: 300 seconds
- **Min Instances**: 0
- **Max Instances**: 10

### IAM Permissions

The backend service needs access to GCS to download the model:

```bash
# Grant Cloud Run service account access to GCS bucket
gcloud storage buckets add-iam-policy-binding gs://mammoscan-ai-models \
  --member=serviceAccount:$(gcloud run services describe mammoscan-backend \
  --region $REGION \
  --format 'value(spec.template.spec.serviceAccountName)') \
  --role=roles/storage.objectViewer
```

## CI/CD Pipeline

The project uses GitHub Actions for continuous deployment to GCP Cloud Run.

### Workflow Configuration

The workflow is defined in `.github/workflows/ci.yml` and triggers on pushes to the `main` branch.

### Workload Identity Federation Setup

This allows GitHub Actions to authenticate to GCP without storing service account keys.

1. **Create Workload Identity Pool**:

   ```bash
   gcloud iam workload-identity-pools create github-pool \
     --location=global \
     --display-name="GitHub Actions Pool"
   ```

2. **Create Workload Identity Provider**:

   ```bash
   gcloud iam workload-identity-pools providers create-oidc github-provider \
     --location=global \
     --workload-identity-pool=github-pool \
     --issuer-uri=https://token.actions.githubusercontent.com \
     --attribute-mapping=google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository \
     --attribute-condition="assertion.repository=='josephed37/mammoscan-AI'"
   ```

3. **Create Service Account**:

   ```bash
   gcloud iam service-accounts create github-actions-sa \
     --display-name="GitHub Actions Service Account"
   ```

4. **Grant Permissions**:

   ```bash
   # Cloud Run Admin
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member=serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com \
     --role=roles/run.admin

   # Cloud Build Editor
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member=serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com \
     --role=roles/cloudbuild.builds.editor

   # Storage Admin (for GCR)
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member=serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com \
     --role=roles/storage.admin

   # Service Account User
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member=serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com \
     --role=roles/iam.serviceAccountUser
   ```

5. **Bind Workload Identity**:

   ```bash
   gcloud iam service-accounts add-iam-policy-binding \
     github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com \
     --role=roles/iam.workloadIdentityUser \
     --member="principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/josephed37/mammoscan-AI"
   ```

### GitHub Secrets Configuration

Add the following secrets to your GitHub repository (Settings > Secrets and variables > Actions):

- `GCP_PROJECT_ID`: Your GCP project ID (e.g., `mammoscan-ai`)
- `GCP_WORKLOAD_IDENTITY_PROVIDER`: Full provider resource name

  ```
  projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider
  ```

- `GCP_SERVICE_ACCOUNT`: Service account email

  ```
  github-actions-sa@mammoscan-ai.iam.gserviceaccount.com
  ```

### Workflow Steps

The CI/CD workflow performs the following steps:

1. **Checkout**: Clone the repository
2. **Authenticate to GCP**: Use Workload Identity Federation
3. **Set up Cloud SDK**: Install and configure `gcloud` CLI
4. **Build and Push Images**: Execute Cloud Build
5. **Deploy Backend**: Deploy to Cloud Run with environment variables
6. **Deploy Frontend**: Deploy to Cloud Run with backend URL

### Triggering Deployments

Deployments are automatically triggered when code is pushed to `main`:

```bash
git add .
git commit -m "Deploy changes"
git push origin main
```

## Environment Variables

### Backend

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `MODEL_GCS_BUCKET` | GCS bucket containing the ONNX model | `mammoscan-ai-models` | Yes |
| `MODEL_GCS_OBJECT` | Object path within the bucket | `champion_model.onnx` | Yes |
| `PORT` | Port the server listens on | `8080` | No (default: 8080) |

### Frontend

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `API_URL` | Backend prediction endpoint URL | `http://backend:8080/api/v1/predict` | Yes |
| `PORT` | Port Streamlit runs on | `8501` | No (default: 8501) |

## Monitoring and Health Checks

### Health Check Endpoint

The backend provides a health check endpoint:

```bash
curl http://localhost:8080/healthy
```

Response:

```json
{
  "status": "OK"
}
```

### Cloud Run Monitoring

GCP provides built-in monitoring:

1. **View logs**:

   ```bash
   gcloud run services logs read mammoscan-backend --region $REGION
   gcloud run services logs read mammoscan-frontend --region $REGION
   ```

2. **View metrics in Cloud Console**:
   - Request count
   - Request latency
   - Error rate
   - Container instance count

3. **Set up alerting**:

   ```bash
   # Example: Alert on high error rate
   gcloud alpha monitoring policies create \
     --notification-channels=CHANNEL_ID \
     --display-name="High Error Rate" \
     --condition-display-name="Error rate > 5%" \
     --condition-threshold-value=0.05 \
     --condition-threshold-duration=60s
   ```

### Application Performance

The backend is designed for:
- **Inference latency**: <10ms per prediction
- **Total request latency**: <500ms (including preprocessing)
- **Throughput**: 100+ requests/second (with auto-scaling)

## Troubleshooting

### Common Issues

#### 1. Model Download Failures

**Symptom**: Backend fails to start with "failed to download model" error

**Solutions**:
- Verify the GCS bucket and object exist:

  ```bash
  gsutil ls gs://mammoscan-ai-models/champion_model.onnx
  ```

- Check IAM permissions for the service account
- Ensure the Cloud Storage API is enabled

#### 2. Frontend Cannot Connect to Backend

**Symptom**: "Connection error" when uploading images

**Solutions**:
- Verify `API_URL` environment variable is correct
- Check backend health endpoint
- Review Cloud Run logs for errors
- Ensure backend service allows unauthenticated requests

#### 3. Image Upload Failures

**Symptom**: "Invalid image format" errors

**Solutions**:
- Ensure image is JPEG or PNG format
- Check image file size (Cloud Run has 32MB request limit)
- Verify image is not corrupted

#### 4. High Memory Usage

**Symptom**: Backend instances restarting due to OOM

**Solutions**:
- Increase Cloud Run memory allocation to 2Gi
- Reduce concurrency to limit parallel requests
- Check for memory leaks in inference code

#### 5. Slow Cold Starts

**Symptom**: First request takes >30 seconds

**Solutions**:
- Set minimum instances to 1 to keep one instance warm:

  ```bash
  gcloud run services update mammoscan-backend \
    --region $REGION \
    --min-instances=1
  ```

- Consider using Cloud Run's always-on CPU allocation
- Optimize Docker image size (currently ~150MB for backend)

### Debug Mode

To enable verbose logging:

```bash
# Backend (modify main.go)
gin.SetMode(gin.DebugMode)

# Frontend (modify app.py)
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Rollback Deployments

To rollback to a previous revision:

```bash
# List revisions
gcloud run revisions list --service mammoscan-backend --region $REGION

# Rollback to specific revision
gcloud run services update-traffic mammoscan-backend \
  --to-revisions=REVISION_NAME=100 \
  --region $REGION
```

## Security Considerations

### Production Checklist

- [ ] Enable VPC connector for private networking
- [ ] Implement authentication (IAM, API keys, or OAuth)
- [ ] Set up Cloud Armor for DDoS protection
- [ ] Enable Cloud Logging and Cloud Monitoring
- [ ] Configure secrets using Secret Manager (not environment variables)
- [ ] Set up backup and disaster recovery for GCS bucket
- [ ] Implement rate limiting
- [ ] Enable HTTPS only (Cloud Run default)
- [ ] Review and minimize IAM permissions
- [ ] Set up vulnerability scanning for container images

### HIPAA Compliance

For healthcare applications handling PHI:

- Sign Google Cloud HIPAA BAA
- Enable audit logging
- Encrypt data at rest and in transit
- Implement access controls and authentication
- Use VPC Service Controls
- Regular security audits

## Cost Optimization

### Estimated Costs (GCP)

For moderate usage (1000 requests/day):
- Cloud Run: ~$5-10/month
- Cloud Storage: <$1/month
- Cloud Build: ~$2-5/month
- **Total**: ~$10-15/month

### Cost-Saving Tips

1. **Use minimum instances wisely**: Set to 0 for development, 1+ for production
2. **Optimize container size**: Smaller images = faster deployments = lower costs
3. **Set appropriate memory limits**: Don't over-provision
4. **Use Cloud Build caching**: Speeds up builds and reduces costs
5. **Monitor and set budget alerts**: Prevent unexpected charges

## Additional Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Workload Identity Federation](https://cloud.google.com/iam/docs/workload-identity-federation)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## Support

For issues or questions:
- GitHub Issues: https://github.com/josephed37/mammoscan-AI/issues
- Project Wiki: https://github.com/josephed37/mammoscan-AI/wiki