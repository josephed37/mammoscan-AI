# MammoScan AI 🩺

[![CI](https://github.com/josephed37/mammoscan-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/josephed37/mammoscan-AI/actions/workflows/ci.yml)
[![CD](https://github.com/josephed37/mammoscan-AI/actions/workflows/cd.yml/badge.svg)](https://github.com/josephed37/mammoscan-AI/actions/workflows/cd.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered application for breast cancer detection from mammogram images, featuring a high-performance Go backend and modern MLOps pipeline. Built in public to demonstrate end-to-end ML system development.

**🔗 [Live Demo](https://mammoscan-frontend-15175657305.europe-west1.run.app)** | **📖 [Documentation](https://github.com/josephed37/mammoscan-AI/wiki)**

---

## ✨ Features

- **AI-Powered Detection:** CNN-based classification with 99MB ONNX model
- **High-Performance Backend:** Go API with <10ms inference latency
- **Interactive Frontend:** Streamlit web application for easy image upload
- **Production-Ready MLOps:** Full CI/CD pipeline with automated deployment
- **Cloud-Native:** Deployed on Google Cloud Run with auto-scaling
- **Secure:** Workload Identity Federation for GitHub Actions

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   GitHub    │─────▶│ Cloud Build  │─────▶│   GCR/GAR   │
│   Actions   │      │   (CI/CD)    │      │  (Images)   │
└─────────────┘      └──────────────┘      └─────────────┘
                                                    │
                     ┌──────────────────────────────┘
                     ▼
              ┌──────────────┐
              │  Cloud Run   │
              │  (Backend)   │◀────┐
              └──────────────┘     │
                     │             │
                     ▼             │
              ┌──────────────┐     │
              │     GCS      │     │
              │   (Model)    │     │
              └──────────────┘     │
                                   │
              ┌──────────────┐     │
              │  Cloud Run   │─────┘
              │  (Frontend)  │
              └──────────────┘
```

## 🛠️ Tech Stack

**ML/AI:**

- Python 3.11, TensorFlow/Keras
- ONNX Runtime for inference
- DVC for data versioning

**Backend:**

- Go 1.24 with Gin framework
- ONNX Runtime Go bindings
- Docker multi-stage builds

**Frontend:**

- Streamlit
- Python requests

**Infrastructure:**

- Google Cloud Run (serverless containers)
- Google Cloud Storage (model storage)
- Google Container Registry
- GitHub Actions (CI/CD)
- Workload Identity Federation

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Go 1.24+
- Python 3.11+
- Google Cloud CLI (for deployment)

### Local Development

1. **Clone the repository**

```bash
git clone https://github.com/josephed37/mammoscan-AI.git
cd mammoscan-AI
```

2. **Run with Docker Compose**

```bash
# Using Makefile (recommended)
make docker-up

# Or using Docker Compose directly
docker compose -f deployments/docker-compose.yml up --build
```

3. **Access the application**
   - Frontend: <http://localhost:8501>
   - Backend API: <http://localhost:8080>
   - Health check: <http://localhost:8080/healthy>

4. **Stop services**

```bash
make docker-down
```

For detailed setup instructions, see the [Local Development Guide](docs/LOCAL_DEVELOPMENT.md).

### Manual Setup

**Backend:**

```bash
cd backend
go mod download
go run cmd/api/main.go
```

**Frontend:**

```bash
cd web
pip install -r requirements.txt
streamlit run app.py
```

**ML Training:**

```bash
# Using Makefile (recommended)
make run-pipeline  # Full pipeline: preprocess → train → evaluate

# Or manually
cd ml
pip install -r requirements.txt
python -m ml.scripts.train --model baseline --epochs 20
python -m ml.scripts.evaluate
```

See the [Model Training Guide](docs/MODEL_TRAINING.md) for comprehensive instructions.

## 📊 Model Performance

**Current Champion Model:** Baseline CNN v2

- **Accuracy:** 87.5%
- **Sensitivity (Recall):** 94.7% - High cancer detection rate
- **Precision:** 58.1% - Optimized for minimal false negatives
- **Decision Threshold:** 0.110593 (optimized for medical screening)
- **Model Size:** 99MB ONNX format
- **Inference Latency:** <10ms

The model prioritizes catching cancer cases (high recall) over reducing false positives, which is appropriate for medical screening where missed cancers are more critical than false alarms.

**Upcoming:** Transfer learning with EfficientNet/ResNet for improved precision and specificity.

## 🔧 Configuration

**Environment Variables:**

Backend:

- `PORT`: Server port (default: 8080)
- `MODEL_GCS_BUCKET`: GCS bucket for model
- `MODEL_GCS_OBJECT`: Model file name
- `MODEL_PATH`: Local model path

Frontend:

- `PORT`: Streamlit port (default: 8501)
- `API_URL`: Backend API endpoint

## 🚢 Deployment

The project uses GitHub Actions for automated deployment to Google Cloud Run.

**Manual Deployment:**

```bash
# Build and push images
gcloud builds submit --config=deployments/cloudbuild.yaml .

# Deploy backend
gcloud run deploy mammoscan-backend \
  --image gcr.io/mammoscan-ai/mammoscan-backend:latest \
  --region europe-west1 \
  --memory 1Gi

# Deploy frontend
gcloud run deploy mammoscan-frontend \
  --image gcr.io/mammoscan-ai/mammoscan-frontend:latest \
  --region europe-west1
```

See [Deployment Guide](docs/DEPLOYMENT.md) for detailed instructions.

## 📚 Documentation

- **[Architecture Details](docs/ARCHITECTURE.md)** - System design, component architecture, and technology stack
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Local and cloud deployment instructions
- **[Model Training Guide](docs/MODEL_TRAINING.md)** - ML pipeline, training, and evaluation
- **[Local Development Setup](docs/LOCAL_DEVELOPMENT.md)** - Development environment setup
- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute to the project

### API Endpoints

**Health Check:**

```
GET /healthy
Response: {"status": "OK"}
```

**Prediction:**

```
POST /api/v1/predict
Content-Type: multipart/form-data
Body: image file (JPEG/PNG)

Response:
{
  "prediction": "Cancer" | "Non-Cancer",
  "confidence_score": 0.0-1.0,
  "model_name": "baseline_cnn_v2",
  "model_threshold": 0.110593
}
```

## 🗺️ Roadmap

### Completed ✅

- [x] Basic CNN model training with MLflow tracking
- [x] Go backend with ONNX inference (<10ms latency)
- [x] Streamlit frontend with interactive UI
- [x] Docker containerization
- [x] CI/CD pipeline with GitHub Actions
- [x] Cloud Run deployment with Workload Identity Federation
- [x] Data versioning with DVC
- [x] Model export to ONNX format
- [x] GCS model storage (prevents image bloat)
- [x] Comprehensive documentation

### In Progress 🚧

- [ ] Transfer learning implementation (EfficientNet/ResNet)
- [ ] Comprehensive testing suite (unit, integration, E2E)
- [ ] Model performance optimization

### Planned 📋

- [ ] Model versioning and registry system
- [ ] A/B testing framework for model comparison
- [ ] Advanced monitoring and observability (OpenTelemetry)
- [ ] Grad-CAM visualization for explainability
- [ ] Batch prediction API
- [ ] Real-time feedback loop and automated retraining
- [ ] HIPAA compliance enhancements

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Joseph Edjeani**

- GitHub: [@josephed37](https://github.com/josephed37)
- LinkedIn: [Connect with me](www.linkedin.com/in/joseph-edjeani)

## 🙏 Acknowledgments

- Breast cancer dataset providers
- Open source ML community
- Google Cloud Platform for free tier

---

**Note:** This is an educational project. Always consult healthcare professionals for medical diagnoses.
