# Architecture Documentation

This document provides a comprehensive overview of the Mammoscan AI system architecture, component interactions, data flows, and design decisions.

## Table of Contents

- [System Overview](#system-overview)
- [High-Level Architecture](#high-level-architecture)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Design Decisions](#design-decisions)
- [Security Architecture](#security-architecture)
- [Scalability and Performance](#scalability-and-performance)
- [Future Architecture Considerations](#future-architecture-considerations)

## System Overview

Mammoscan AI is an end-to-end machine learning system for breast cancer detection from mammogram images. The system is designed following MLOps best practices with clear separation of concerns:

- **ML Pipeline**: Python-based training and evaluation pipeline
- **Backend API**: High-performance Go service for inference
- **Frontend**: Interactive Streamlit web application
- **Infrastructure**: Cloud-native deployment on Google Cloud Platform

### Key Design Principles

1. **Separation of Concerns**: ML development, model serving, and UI are independent
2. **Cloud-Native**: Stateless, containerized, auto-scaling services
3. **Performance**: Sub-10ms inference latency with efficient ONNX runtime
4. **Reproducibility**: Version control for code, data (DVC), and models (MLflow)
5. **Security**: No credentials in code, Workload Identity Federation for GCP
6. **Observability**: Health checks, logging, and metrics throughout

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User / Browser                          │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTPS
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend (Streamlit)                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  - Image Upload Interface                                  │ │
│  │  - Result Visualization                                    │ │
│  │  - Project Information Pages                               │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP POST /api/v1/predict
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend API (Go)                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Gin Router                                                │ │
│  │  ├─ GET  /healthy         → Health Check                  │ │
│  │  └─ POST /api/v1/predict  → Prediction Handler            │ │
│  └──────────────────┬──────────────────────────────────────────┘ │
│                     │                                            │
│  ┌──────────────────▼──────────────────────────────────────────┐ │
│  │  Preprocessing Pipeline                                    │ │
│  │  ├─ Image Decode (JPEG/PNG)                                │ │
│  │  ├─ Resize (Lanczos3 → 224x224)                            │ │
│  │  └─ Tensorization (4D float32)                             │ │
│  └──────────────────┬──────────────────────────────────────────┘ │
│                     │                                            │
│  ┌──────────────────▼──────────────────────────────────────────┐ │
│  │  ONNX Inference Engine                                     │ │
│  │  ├─ Model: champion_model.onnx (99MB)                      │ │
│  │  ├─ Runtime: Gorgonia backend                              │ │
│  │  └─ Output: Probability [0.0-1.0]                          │ │
│  └──────────────────┬──────────────────────────────────────────┘ │
│                     │                                            │
│  ┌──────────────────▼──────────────────────────────────────────┐ │
│  │  Classification Logic                                      │ │
│  │  ├─ Threshold: 0.110593                                    │ │
│  │  └─ Output: "Cancer" / "Non-Cancer"                        │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │ Model Download (on startup)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Google Cloud Storage (GCS)                          │
│  gs://mammoscan-ai-models/champion_model.onnx                   │
└─────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                    ML Training Pipeline                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Data Preprocessing                                        │ │
│  │  ├─ DVC: Version control for datasets                      │ │
│  │  ├─ Raw images → Processed (224x224)                       │ │
│  │  └─ Train/Val/Test splits                                  │ │
│  └──────────────────┬──────────────────────────────────────────┘ │
│                     │                                            │
│  ┌──────────────────▼──────────────────────────────────────────┐ │
│  │  Model Training (TensorFlow/Keras)                         │ │
│  │  ├─ Architecture: CNN / Transfer Learning                  │ │
│  │  ├─ Tracking: MLflow                                       │ │
│  │  └─ Output: .keras checkpoint                              │ │
│  └──────────────────┬──────────────────────────────────────────┘ │
│                     │                                            │
│  ┌──────────────────▼──────────────────────────────────────────┐ │
│  │  Model Evaluation                                          │ │
│  │  ├─ Metrics: Accuracy, Precision, Recall                   │ │
│  │  ├─ Threshold Optimization                                 │ │
│  │  └─ Report: JSON metrics file                              │ │
│  └──────────────────┬──────────────────────────────────────────┘ │
│                     │                                            │
│  ┌──────────────────▼──────────────────────────────────────────┐ │
│  │  Model Export                                              │ │
│  │  ├─ Keras → ONNX conversion                                │ │
│  │  ├─ Upload to GCS                                          │ │
│  │  └─ Version tracking                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Backend API (Go)

#### File Structure

```
backend/
├── cmd/api/main.go              # Application entry point
├── internal/
│   ├── handlers/                # HTTP request handlers
│   │   └── handlers.go
│   ├── inference/               # ML inference engine
│   │   └── onnx.go
│   ├── preprocess/              # Image preprocessing
│   │   └── image.go
│   └── models/                  # Data structures
│       └── types.go
├── Dockerfile.api               # Production container
├── Dockerfile.dev               # Development container
└── go.mod                       # Dependencies
```

#### Component Responsibilities

**Main Application** (`cmd/api/main.go:1-100`)

```go
func main() {
    // 1. Download model from GCS on startup
    modelPath := downloadModelFromGCS()

    // 2. Initialize ONNX inference engine
    engine := inference.NewONNXEngine(modelPath)

    // 3. Set up Gin router
    router := gin.Default()
    router.GET("/healthy", handlers.HealthCheck)
    router.POST("/api/v1/predict", handlers.NewHandler(engine).Predict)

    // 4. Start server with graceful shutdown
    server.ListenAndServe()
}
```

**Handlers** (`internal/handlers/handlers.go:1-150`)

- `HealthCheck`: Simple liveness probe returning 200 OK
- `Predict`: Orchestrates the prediction pipeline
  1. Validate multipart form data
  2. Decode uploaded image
  3. Preprocess image to tensor
  4. Run inference
  5. Apply threshold
  6. Return JSON response

**Inference Engine** (`internal/inference/onnx.go:1-100`)

```go
type ONNXEngine struct {
    model *onnx.Model
}

func (e *ONNXEngine) Predict(inputTensor tensor.Tensor) (float32, error) {
    // Execute ONNX model with Gorgonia backend
    output := e.model.Run(inputTensor)
    return output[0], nil
}
```

**Preprocessing** (`internal/preprocess/image.go:1-200`)

Pipeline stages:
1. **Decode**: Read multipart file, decode JPEG/PNG
2. **Resize**: Lanczos3 resampling to 224×224
3. **Convert**: RGBA → RGB, 16-bit → 8-bit normalization
4. **Tensorize**: Create 4D tensor [1, 224, 224, 3] in float32

**Data Models** (`internal/models/types.go:1-50`)

```go
type PredictionResponse struct {
    Prediction     string  `json:"prediction"`      // "Cancer" | "Non-Cancer"
    ConfidenceScore float32 `json:"confidence_score"` // 0.0-1.0
    ModelName      string  `json:"model_name"`      // "baseline_cnn_v2"
    ModelThreshold float32 `json:"model_threshold"` // 0.110593
}

type ErrorResponse struct {
    Error string `json:"error"`
}
```

### 2. Frontend Application (Streamlit)

#### File Structure

```
web/
├── app.py                   # Main application
├── Dockerfile.web           # Container definition
├── docker-entrypoint.sh     # Startup script
└── requirements.txt         # Python dependencies
```

#### Page Architecture

**Navigation** (`app.py:1-50`)

```python
pages = {
    "Project Overview": show_overview_page,
    "Interactive Demo": show_demo_page
}
```

**Overview Page** (`app.py:50-150`)

- Project introduction
- Technical architecture diagram
- Key objectives and metrics
- Technology stack
- Medical disclaimer

**Demo Page** (`app.py:150-350`)

- File uploader (JPEG/PNG)
- Image display with preview
- Analyze button with progress indicator
- Result card with:
  - Prediction (Cancer/Non-Cancer)
  - Confidence score
  - Visual confidence meter
  - Model metadata
- Error handling UI

**API Integration** (`app.py:200-250`)

```python
def call_api(image_bytes):
    response = requests.post(
        API_URL,
        files={"image": image_bytes},
        timeout=30
    )
    return response.json()
```

### 3. ML Pipeline (Python)

#### Architecture Layers

**Data Layer** (`ml/src/data_utils.py`)

```python
def load_image_paths(data_dir):
    # Scan directory tree
    # Assign labels from folder structure
    # Return DataFrame: [image_path, label]
```

**Preprocessing Layer** (`ml/src/preprocess_utils.py`)

```python
def process_and_save_image(img_path, output_path):
    # Load image
    # Resize to 224x224
    # Save to output directory
```

**Model Layer** (`ml/src/model.py`)

```python
def create_baseline_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def create_transfer_model(base='EfficientNetB0'):
    base_model = EfficientNetB0(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=output)
```

**Training Layer** (`ml/scripts/train.py`)

```python
def train_model(model_type, epochs, batch_size, learning_rate):
    # 1. Load data
    train_ds, val_ds = load_datasets()

    # 2. Build model
    model = MODEL_REGISTRY[model_type]()

    # 3. Configure MLflow
    mlflow.set_experiment(f"{model_type}_training")
    mlflow.autolog()

    # 4. Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_callback]
    )

    # 5. Save checkpoint
    model.save(f"models/checkpoints/{model_type}_v2.keras")

    return history
```

**Evaluation Layer** (`ml/scripts/evaluate.py`)

```python
def evaluate_model(model_path, test_data_path):
    # 1. Load model and test data
    model = tf.keras.models.load_model(model_path)
    test_ds = load_test_dataset(test_data_path)

    # 2. Generate predictions
    y_pred_proba = model.predict(test_ds)

    # 3. Find optimal threshold
    threshold = optimize_threshold(y_true, y_pred_proba)

    # 4. Compute metrics
    y_pred = (y_pred_proba > threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred),
        "threshold": threshold
    }

    # 5. Save report
    with open(f"reports/{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f)
```

**Export Layer** (`ml/scripts/export.py`)

```python
def export_to_onnx(keras_path, onnx_path):
    # 1. Load Keras model
    model = tf.keras.models.load_model(keras_path)

    # 2. Create inference-only model (remove augmentation)
    input_layer = Input(shape=(224, 224, 3))
    output = model.layers[1:](input_layer)  # Skip augmentation layer
    inference_model = Model(input_layer, output)

    # 3. Convert to ONNX
    onnx_model = tf2onnx.convert.from_keras(inference_model)

    # 4. Save
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
```

## Data Flow

### Training Data Flow

```
Raw Mammogram Images
    │
    ├─ DVC Tracking (data/raw.dvc)
    │
    ▼
Preprocessing Script
    │
    ├─ Resize: 224x224
    ├─ Format: PNG
    └─ Organize: train/val/test splits
    │
    ├─ DVC Tracking (data/processed.dvc)
    │
    ▼
TensorFlow Data Pipeline
    │
    ├─ ImageDataGenerator (on-the-fly augmentation)
    ├─ Batch: 32 images
    └─ Normalize: [0-255] → [0-1]
    │
    ▼
Model Training
    │
    ├─ Forward pass
    ├─ Loss computation: Binary crossentropy
    ├─ Backward pass: Adam optimizer
    └─ MLflow logging
    │
    ▼
Model Checkpoint (.keras)
    │
    ├─ Best validation loss
    └─ Saved to: models/checkpoints/
    │
    ▼
Model Evaluation
    │
    ├─ Test set predictions
    ├─ Threshold optimization
    └─ Metrics report
    │
    ▼
ONNX Export
    │
    ├─ Keras → ONNX conversion
    ├─ Validation: Compare outputs
    └─ Upload to GCS
```

### Inference Data Flow

```
User Upload (Browser)
    │
    ├─ File: mammogram.jpg
    ├─ Format: JPEG/PNG
    └─ Size: <32MB
    │
    ▼
Frontend (Streamlit)
    │
    ├─ Display preview
    ├─ Convert to bytes
    └─ HTTP POST to backend
    │
    ▼
Backend API Handler
    │
    ├─ Validate: Content-Type, file presence
    └─ Extract: multipart form data
    │
    ▼
Preprocessing Pipeline
    │
    ├─ Decode: image.Decode(file)
    ├─ Resize: resize.Resize(img, 224, 224, Lanczos3)
    ├─ Convert: RGBA → RGB, uint16 → uint8
    └─ Tensorize: [1, 224, 224, 3] float32
    │
    ▼
ONNX Inference Engine
    │
    ├─ Input: tensor [1, 224, 224, 3]
    ├─ Model: champion_model.onnx
    ├─ Runtime: Gorgonia backend
    └─ Output: probability float32
    │
    ▼
Classification Logic
    │
    ├─ Compare: probability vs 0.110593
    ├─ Decision: "Cancer" if > threshold else "Non-Cancer"
    └─ Package: JSON response
    │
    ▼
Frontend Display
    │
    ├─ Show prediction
    ├─ Confidence meter
    └─ Model metadata
```

## Technology Stack

### Backend Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Language | Go | 1.24.4 | High performance, low latency |
| Web Framework | Gin | 1.10.1 | HTTP routing, middleware |
| ML Runtime | onnx-go | 0.5.0 | ONNX model execution |
| Tensor Library | Gorgonia | 0.9.24 | Tensor operations |
| Image Processing | resize | 0.0.0 | High-quality image resizing |
| Cloud SDK | GCP Go SDK | 1.57.0 | GCS integration |

**Why Go for Backend?**
- **Performance**: 10-100x faster than Python for inference
- **Concurrency**: Native goroutines for parallel requests
- **Memory**: Lower memory footprint than Python
- **Deployment**: Single binary, no runtime dependencies
- **Latency**: <10ms inference time (vs 50-100ms Python)

### ML Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Language | Python | 3.11 | ML ecosystem support |
| Deep Learning | TensorFlow/Keras | 2.16.1 | Model training |
| Experiment Tracking | MLflow | 3.3.2 | Reproducibility |
| Data Versioning | DVC | Latest | Dataset version control |
| Model Export | tf2onnx | Latest | Keras → ONNX conversion |
| Cloud Storage | google-cloud-storage | 3.3.0 | Model artifact storage |

### Frontend Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | Streamlit | Rapid UI development |
| Image Handling | PIL | Image preview |
| HTTP Client | requests | API calls |
| Styling | Custom CSS | Medical theme |

### Infrastructure Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Containers | Docker | Application packaging |
| Orchestration | Docker Compose | Local development |
| Cloud Platform | Google Cloud Platform | Production deployment |
| Compute | Cloud Run | Serverless containers |
| Storage | Cloud Storage | Model artifacts |
| CI/CD | GitHub Actions | Automated deployment |
| Registry | GCR | Container images |

## Design Decisions

### 1. ONNX for Model Deployment

**Decision**: Export Keras model to ONNX format for production inference

**Rationale**:
- Cross-platform compatibility (Python → Go)
- Optimized inference runtime
- No TensorFlow dependency in production
- Smaller deployment footprint
- Better performance (optimized graph)

**Trade-offs**:
- Additional export step in pipeline
- Must validate ONNX output matches Keras
- Limited to ONNX-supported operations

### 2. Go Backend vs Python

**Decision**: Use Go for serving API instead of Python (Flask/FastAPI)

**Rationale**:
- **Latency**: <10ms inference vs 50-100ms Python
- **Throughput**: 10x higher requests/second
- **Memory**: 50MB vs 200MB+ Python
- **Concurrency**: Native goroutines vs GIL limitations
- **Deployment**: Single binary vs Python runtime + dependencies

**Trade-offs**:
- Less mature ML ecosystem in Go
- ONNX runtime required (vs native TensorFlow/PyTorch)
- Smaller developer community for ML in Go

### 3. Cloud Storage for Model

**Decision**: Download model from GCS at startup instead of bundling in Docker image

**Rationale**:
- **Image Size**: 150MB vs 250MB with bundled model
- **Deployment Speed**: Faster image push/pull
- **Model Updates**: Deploy new model without rebuilding container
- **Versioning**: Centralized model versioning in GCS
- **Cost**: Faster builds reduce Cloud Build costs

**Trade-offs**:
- Longer cold start (download on first instance)
- Dependency on GCS availability
- Network latency on startup

### 4. Streamlit vs React

**Decision**: Use Streamlit for frontend instead of React/Vue

**Rationale**:
- **Development Speed**: 10x faster than React for MVP
- **Python Integration**: Native data science tooling
- **Deployment**: Single container, no build step
- **Maintenance**: Less code to maintain

**Trade-offs**:
- Limited customization vs React
- Less interactive (full page reloads)
- Heavier than static sites

### 5. Cloud Run vs GKE/VMs

**Decision**: Deploy to Cloud Run instead of Kubernetes or VMs

**Rationale**:
- **Simplicity**: No cluster management
- **Cost**: Pay only for requests, scales to zero
- **Auto-scaling**: Built-in, no configuration
- **HTTPS**: Automatic SSL certificates
- **Deployment**: Single command

**Trade-offs**:
- Less control over infrastructure
- Cold start latency (mitigated with min instances)
- Limited to HTTP/HTTPS

### 6. Threshold Optimization

**Decision**: Use 0.110593 threshold (vs default 0.5)

**Rationale**:
- **Medical Context**: False negatives worse than false positives
- **High Recall**: 94.7% catch rate for cancer
- **Acceptable Precision**: 58% precision acceptable for screening
- **Follow-up**: Human experts review all positive cases

**Trade-offs**:
- More false positives (42% of predicted cancers)
- Increased workload for radiologists
- Better safe than sorry for cancer detection

## Security Architecture

### Authentication and Authorization

**Current State**: Unauthenticated public API (demo only)

**Production Recommendations**:

1. **API Authentication**:
   - JWT tokens for session management
   - API keys for programmatic access
   - OAuth2 for third-party integrations

2. **Identity and Access Management**:
   ```
   User → Cloud Identity-Aware Proxy → Cloud Run
   ```
   - IAM for service-to-service auth
   - Workload Identity for GCP resources

3. **Rate Limiting**:
   - Cloud Armor for DDoS protection
   - API Gateway for rate limiting
   - Per-user quotas

### Data Security

**In Transit**:
- HTTPS only (Cloud Run enforced)
- TLS 1.3
- No PHI in logs

**At Rest**:
- GCS encryption (default)
- Secret Manager for credentials
- No sensitive data in containers

### Network Security

```
Internet
    │
    ├─ Cloud Armor (WAF, DDoS protection)
    │
    ▼
Cloud Load Balancer
    │
    ├─ SSL Termination
    │
    ▼
Cloud Run Services
    │
    ├─ VPC Connector (optional)
    │
    ▼
GCS / Cloud SQL (if needed)
```

## Scalability and Performance

### Performance Characteristics

| Metric | Target | Achieved | Notes |
|--------|--------|----------|-------|
| Inference Latency | <10ms | ~5ms | ONNX model only |
| Total Request Latency | <500ms | ~200ms | Including preprocessing |
| Throughput | 100 req/s | 150 req/s | Single instance |
| Cold Start | <30s | ~15s | Model download time |
| Memory Usage | <1GB | ~600MB | Backend with model |
| Image Size | <200MB | ~150MB | Backend container |

### Scaling Strategy

**Horizontal Scaling** (Cloud Run):
- Auto-scales based on request volume
- Max concurrency: 80 requests per instance
- Max instances: 10 (configurable)
- Min instances: 0 (scales to zero)

**Optimization Techniques**:

1. **Request-Level**:
   - ONNX runtime optimization
   - Efficient image resizing (Lanczos3)
   - Connection pooling
   - Response caching (if appropriate)

2. **Instance-Level**:
   - Pre-loaded model in memory
   - Reuse of goroutines
   - Minimal memory allocation

3. **System-Level**:
   - CDN for static assets (if needed)
   - Database connection pooling (if added)
   - Async processing for non-critical tasks

### Load Testing Results

**Methodology**: 1000 concurrent requests with 224x224 images

| Percentile | Latency |
|------------|---------|
| P50 | 180ms |
| P95 | 350ms |
| P99 | 600ms |
| P99.9 | 1200ms |

**Bottlenecks**:
1. Image resizing (40% of latency)
2. Network I/O (30%)
3. ONNX inference (20%)
4. JSON serialization (10%)

## Future Architecture Considerations

### Planned Improvements

1. **Model Versioning**:
   - Blue-green deployment for models
   - A/B testing framework
   - Canary releases

2. **Observability**:
   - OpenTelemetry integration
   - Custom metrics (inference distribution)
   - Distributed tracing
   - Real-time dashboards

3. **Data Pipeline**:
   - Real-time feedback loop
   - Automated retraining pipeline
   - Data drift detection
   - Model performance monitoring

4. **Advanced Features**:
   - Multi-model serving (ensemble)
   - Batch prediction API
   - Async processing with Pub/Sub
   - Explainability (Grad-CAM visualization)

### Scaling Beyond Current Architecture

**For High Traffic (>1000 req/s)**:

1. **Add Caching Layer**:
   ```
   Client → Cloud CDN → Cloud Run → Redis → GCS
   ```

2. **Database for Logging**:
   ```
   Predictions → Pub/Sub → Cloud Functions → BigQuery
   ```

3. **Separate Inference Service**:
   ```
   API Gateway → Auth Service → Inference Service → Model Server
   ```

**For HIPAA Compliance**:

1. **VPC Service Controls**:
   ```
   VPC Network → Private Cloud Run → Private GCS → Cloud SQL
   ```

2. **Audit Logging**:
   - All API requests logged
   - BigQuery for compliance queries
   - Cloud Logging for real-time monitoring

3. **Encryption**:
   - Customer-managed encryption keys (CMEK)
   - Field-level encryption for PHI
   - Encrypted backups

## Conclusion

The Mammoscan AI architecture is designed for:

- **Performance**: Sub-second inference with Go + ONNX
- **Scalability**: Cloud-native auto-scaling on Cloud Run
- **Maintainability**: Clear separation of concerns, modular design
- **Reproducibility**: Version control for code, data, and models
- **Cost-Effectiveness**: Serverless, scales to zero when idle

The architecture balances simplicity with production-readiness, making it suitable for both demonstration and real-world deployment with appropriate security enhancements.
