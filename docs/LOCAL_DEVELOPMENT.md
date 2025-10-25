# Local Development Guide

This guide provides detailed instructions for setting up and running Mammoscan AI locally for development purposes.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Initial Setup](#initial-setup)
- [Development Workflow](#development-workflow)
- [Running the Application](#running-the-application)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Debugging](#debugging)
- [Common Development Tasks](#common-development-tasks)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

1. **Git** (2.30+)
   ```bash
   git --version
   ```

2. **Docker** (20.10+) and **Docker Compose** (2.0+)
   ```bash
   docker --version
   docker compose version
   ```

3. **Python** (3.11)
   ```bash
   python --version
   # or
   python3 --version
   ```

4. **Go** (1.24+) - Optional, only if developing backend
   ```bash
   go version
   ```

5. **Make** - For using Makefile commands
   ```bash
   make --version
   ```

### Optional Tools

- **MLflow** - For experiment tracking UI
- **Jupyter** - For notebook development
- **VSCode** or **GoLand** - Recommended IDEs

### System Requirements

- **RAM**: Minimum 4GB, recommended 8GB+
- **Disk Space**: 5GB for data, models, and containers
- **OS**: Linux, macOS, or Windows (with WSL2)

## Initial Setup

### 1. Clone the Repository

```bash
git clone https://github.com/josephed37/mammoscan-AI.git
cd mammoscan-AI
```

### 2. Set Up Python Environment

#### Using venv (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install ML pipeline dependencies
pip install --upgrade pip
pip install -r ml/requirements.txt

# Install frontend dependencies
pip install -r web/requirements.txt
```

#### Using conda

```bash
# Create conda environment
conda create -n mammoscan python=3.11
conda activate mammoscan

# Install dependencies
pip install -r ml/requirements.txt
pip install -r web/requirements.txt
```

### 3. Set Up Data

The project uses DVC for data version control.

```bash
# Install DVC (if not already installed)
pip install dvc

# Pull data from remote storage
dvc pull

# This will download:
# - data/raw/ (original mammogram images)
# - data/processed/ (train/val/test splits)
```

**Note**: If you don't have access to the DVC remote, you'll need to provide your own dataset. See [Data Requirements](#data-requirements) below.

### 4. Set Up Go Backend (Optional)

If you're developing the backend:

```bash
cd backend

# Download Go dependencies
go mod download

# Verify build works
go build -o bin/api cmd/api/main.go
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Model Configuration
MODEL_GCS_BUCKET=mammoscan-ai-models
MODEL_GCS_OBJECT=champion_model.onnx

# API Configuration
BACKEND_PORT=8080
FRONTEND_PORT=8501
API_URL=http://localhost:8080/api/v1/predict

# ML Training Configuration
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=baseline_training

# Data Paths
RAW_DATA_PATH=./data/raw
PROCESSED_DATA_PATH=./data/processed
MODEL_CHECKPOINT_PATH=./models/checkpoints
MODEL_EXPORT_PATH=./models/saved_models
```

## Development Workflow

### Recommended Branch Strategy

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, commit frequently
git add .
git commit -m "Description of changes"

# Push to remote
git push origin feature/your-feature-name

# Create pull request on GitHub
```

### Project Structure Overview

```
mammoscan-AI/
├── backend/              # Go backend API
│   ├── cmd/api/         # Application entry point
│   ├── internal/        # Internal packages
│   │   ├── handlers/    # HTTP handlers
│   │   ├── inference/   # ONNX inference
│   │   ├── preprocess/  # Image preprocessing
│   │   └── models/      # Data structures
│   ├── Dockerfile.api   # Production container
│   └── Dockerfile.dev   # Development container
├── web/                 # Streamlit frontend
│   ├── app.py          # Main application
│   └── requirements.txt
├── ml/                  # ML training pipeline
│   ├── src/            # Source modules
│   │   ├── model.py
│   │   ├── data_utils.py
│   │   └── preprocess_utils.py
│   ├── scripts/        # Training scripts
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── export.py
│   ├── notebooks/      # Jupyter notebooks
│   └── requirements.txt
├── data/               # Dataset (DVC tracked)
├── models/             # Trained models
├── deployments/        # Infrastructure configs
├── Makefile           # Development commands
└── README.md
```

## Running the Application

### Option 1: Docker Compose (Recommended)

This is the easiest way to run the full stack locally.

```bash
# Start all services
make docker-up
# or
docker compose -f deployments/docker-compose.yml up

# Access:
# - Frontend: http://localhost:8501
# - Backend API: http://localhost:8080
# - Health check: http://localhost:8080/healthy

# View logs
make docker-logs
# or
docker compose -f deployments/docker-compose.yml logs -f

# Stop services
make docker-down
# or
docker compose -f deployments/docker-compose.yml down
```

**Rebuild after code changes**:

```bash
make docker-rebuild
# or
docker compose -f deployments/docker-compose.yml up --build
```

### Option 2: Running Services Individually

#### Run Backend Locally

**Method 1: Using Docker (with local model)**

```bash
cd backend

# Build development image (includes model)
docker build -f Dockerfile.dev -t mammoscan-backend:dev .

# Run container
docker run -p 8080:8080 mammoscan-backend:dev
```

**Method 2: Using Go directly**

```bash
cd backend

# Ensure model exists locally
# Download from GCS or use local ONNX file
export MODEL_PATH=../models/saved_models/champion_model.onnx

# Run the application
go run cmd/api/main.go

# Or build and run
go build -o bin/api cmd/api/main.go
./bin/api
```

#### Run Frontend Locally

```bash
cd web

# Set API URL
export API_URL=http://localhost:8080/api/v1/predict

# Run Streamlit
streamlit run app.py --server.port 8501

# Or using Docker
docker build -f Dockerfile.web -t mammoscan-frontend:dev .
docker run -p 8501:8501 -e API_URL=http://host.docker.internal:8080/api/v1/predict mammoscan-frontend:dev
```

### Option 3: Development with Hot Reload

#### Backend with Air (Go hot reload)

```bash
# Install Air
go install github.com/cosmtrek/air@latest

# Run with hot reload
cd backend
air

# Air watches for file changes and automatically rebuilds
```

#### Frontend with Streamlit (Built-in hot reload)

```bash
cd web
streamlit run app.py

# Streamlit automatically reloads when app.py changes
```

## Testing

### Backend Tests

```bash
cd backend

# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Run specific package
go test ./internal/handlers

# Verbose output
go test -v ./...

# Run with race detector
go test -race ./...
```

### Frontend Tests

```bash
cd web

# Install test dependencies
pip install pytest pytest-streamlit

# Run tests (if test files exist)
pytest tests/
```

### ML Pipeline Tests

```bash
cd ml

# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Integration Tests

```bash
# Start services
make docker-up

# Run integration tests (example)
python tests/integration/test_api.py

# Test health endpoint
curl http://localhost:8080/healthy

# Test prediction endpoint
curl -X POST http://localhost:8080/api/v1/predict \
  -F "image=@path/to/test_image.jpg"
```

### Manual Testing

```bash
# Health check
curl http://localhost:8080/healthy

# Prediction with sample image
cd tests/sample_images
curl -X POST http://localhost:8080/api/v1/predict \
  -F "image=@cancer_sample.jpg" \
  -o response.json

# View response
cat response.json | jq
```

## Code Quality

### Go Code Quality

```bash
cd backend

# Format code
go fmt ./...

# Run linter (install golangci-lint first)
golangci-lint run

# Run vet
go vet ./...

# Check for common issues
staticcheck ./...
```

### Python Code Quality

```bash
# Format with black
black ml/ web/

# Sort imports
isort ml/ web/

# Lint with flake8
flake8 ml/ web/ --max-line-length=88

# Type checking with mypy
mypy ml/src/

# Check code complexity
radon cc ml/src/ -a
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

## Debugging

### Backend Debugging

#### Using Delve (Go debugger)

```bash
# Install Delve
go install github.com/go-delve/delve/cmd/dlv@latest

# Run with debugger
cd backend
dlv debug cmd/api/main.go

# Set breakpoint
(dlv) break main.main
(dlv) break handlers.Predict

# Continue execution
(dlv) continue

# Inspect variables
(dlv) print variableName
```

#### VSCode Debug Configuration

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Backend",
      "type": "go",
      "request": "launch",
      "mode": "debug",
      "program": "${workspaceFolder}/backend/cmd/api",
      "env": {
        "MODEL_PATH": "${workspaceFolder}/models/saved_models/champion_model.onnx"
      }
    }
  ]
}
```

### Frontend Debugging

```bash
# Run Streamlit with debug logging
streamlit run app.py --logger.level=debug

# Add debug prints in app.py
import logging
logging.basicConfig(level=logging.DEBUG)
logging.debug("Debug message here")
```

### ML Pipeline Debugging

```python
# In training scripts, use pdb
import pdb

# Set breakpoint
pdb.set_trace()

# Or use ipdb for better interface
import ipdb
ipdb.set_trace()
```

## Common Development Tasks

### Train a New Model

```bash
# Full pipeline: preprocess → train → evaluate
make run-pipeline

# Or step by step:
make preprocess
make train MODEL=baseline EPOCHS=20
make evaluate
```

### Export Model to ONNX

```bash
cd ml

# Export latest checkpoint
python scripts/export.py \
  --keras-model ../models/checkpoints/baseline_model_v2.keras \
  --output-path ../models/saved_models/new_model.onnx
```

### Upload Model to GCS

```bash
# Authenticate
gcloud auth login

# Upload model
gsutil cp models/saved_models/champion_model.onnx \
  gs://mammoscan-ai-models/
```

### View MLflow UI

```bash
# Start MLflow server
mlflow ui --port 5000

# Access at http://localhost:5000
```

### Run Jupyter Notebooks

```bash
# Start Jupyter
cd ml/notebooks
jupyter notebook

# Or use JupyterLab
jupyter lab
```

### Add New Dependencies

**Python**:

```bash
# Add to requirements.txt
echo "new-package==1.0.0" >> ml/requirements.txt

# Install
pip install -r ml/requirements.txt

# Freeze all dependencies
pip freeze > ml/requirements.txt
```

**Go**:

```bash
cd backend

# Add dependency
go get github.com/package/name

# Tidy dependencies
go mod tidy
```

### Database Migrations (Future)

When database is added:

```bash
# Example with migrate tool
migrate -path migrations -database "postgres://..." up
```

## Troubleshooting

### Issue: Port Already in Use

```bash
# Find process using port 8080
lsof -i :8080

# Kill process
kill -9 <PID>

# Or use different port
docker run -p 8081:8080 mammoscan-backend:dev
```

### Issue: Docker Build Fails

```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker compose -f deployments/docker-compose.yml build --no-cache
```

### Issue: Model Not Found

```bash
# Check model exists
ls -lh models/saved_models/champion_model.onnx

# Download from GCS
gsutil cp gs://mammoscan-ai-models/champion_model.onnx models/saved_models/

# Or use development Dockerfile with bundled model
cd backend
docker build -f Dockerfile.dev -t mammoscan-backend:dev .
```

### Issue: Python Import Errors

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install --upgrade -r ml/requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"

# Run from project root
cd /path/to/mammoscan-AI
python -m ml.scripts.train
```

### Issue: Go Module Errors

```bash
cd backend

# Clean module cache
go clean -modcache

# Download dependencies
go mod download

# Verify dependencies
go mod verify

# Update dependencies
go get -u ./...
go mod tidy
```

### Issue: Out of Memory

```bash
# Increase Docker memory
# Docker Desktop → Settings → Resources → Memory → 8GB

# Or reduce batch size in training
python ml/scripts/train.py --batch-size 16

# Monitor memory usage
docker stats
```

### Issue: Data Not Found

```bash
# Check DVC status
dvc status

# Pull data
dvc pull

# If DVC remote not configured, check data directory
ls -R data/

# Manual data setup (if needed)
mkdir -p data/raw/Cancer
mkdir -p data/raw/Non-Cancer
# Add images...

# Run preprocessing
make preprocess
```

### Issue: CUDA/GPU Not Available

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install GPU support (if CUDA installed)
pip install tensorflow[and-cuda]

# Or train on CPU (slower)
export CUDA_VISIBLE_DEVICES=""
python ml/scripts/train.py
```

## Data Requirements

If you're using your own dataset:

### Expected Directory Structure

```
data/raw/
├── Cancer/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── Non-Cancer/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

### Data Specifications

- **Format**: JPEG or PNG
- **Size**: Ideally 224×224 or larger (will be resized)
- **Color**: RGB (3 channels)
- **Labels**: Directory name determines label
- **Balance**: Aim for ~1:5 ratio (Cancer:Non-Cancer) to reflect real-world distribution

### Running Preprocessing

```bash
make preprocess
# or
python -m ml.scripts.preprocess \
  --input-dir data/raw \
  --output-dir data/processed \
  --train-split 0.7 \
  --val-split 0.15 \
  --test-split 0.15
```

## Performance Optimization Tips

### 1. Backend

- Use production build: `go build -ldflags="-s -w"` (reduces binary size)
- Enable Go CPU profiling: `import _ "net/http/pprof"`
- Use connection pooling for database (when added)

### 2. Frontend

- Cache API responses in session state
- Use `@st.cache_data` for expensive computations
- Optimize image loading

### 3. ML Training

- Use mixed precision training: `tf.keras.mixed_precision.set_global_policy('mixed_float16')`
- Enable XLA compilation: `tf.config.optimizer.set_jit(True)`
- Use data augmentation on GPU: `tf.data.experimental.AUTOTUNE`

## Development Best Practices

### Code Style

- **Go**: Follow [Effective Go](https://go.dev/doc/effective_go)
- **Python**: Follow [PEP 8](https://pep8.org/) and [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- **Commits**: Use [Conventional Commits](https://www.conventionalcommits.org/)

### Testing Guidelines

- Write unit tests for all new functions
- Maintain >80% code coverage
- Add integration tests for API endpoints
- Test edge cases and error conditions

### Documentation

- Add docstrings to all functions
- Update README when adding features
- Keep docs/ synchronized with code
- Add inline comments for complex logic

## Additional Resources

- [Go Documentation](https://go.dev/doc/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## Getting Help

- **GitHub Issues**: https://github.com/josephed37/mammoscan-AI/issues
- **Project Wiki**: https://github.com/josephed37/mammoscan-AI/wiki
- **Discussions**: https://github.com/josephed37/mammoscan-AI/discussions

## Next Steps

After setting up your local environment:

1. Review the [Architecture Documentation](ARCHITECTURE.md)
2. Try training a model following [Model Training Guide](MODEL_TRAINING.md)
3. Read the [Contributing Guidelines](../CONTRIBUTING.md)
4. Check the [Deployment Guide](DEPLOYMENT.md) for production deployment
