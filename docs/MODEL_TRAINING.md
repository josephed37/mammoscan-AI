# Model Training Guide

This guide provides comprehensive instructions for training, evaluating, and deploying machine learning models for Mammoscan AI.

## Table of Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Model Architectures](#model-architectures)
- [Training Pipeline](#training-pipeline)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Model Export](#model-export)
- [MLflow Integration](#mlflow-integration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

Mammoscan AI uses a supervised learning approach for binary classification of mammogram images:

- **Task**: Binary classification (Cancer vs Non-Cancer)
- **Input**: RGB mammogram images (224×224 pixels)
- **Output**: Probability score [0.0-1.0]
- **Framework**: TensorFlow/Keras 2.16.1
- **Tracking**: MLflow for experiment management
- **Versioning**: DVC for data and model versioning

### Training Workflow

```
Raw Data → Preprocessing → Model Training → Evaluation → Export → Deployment
   ↓            ↓              ↓              ↓           ↓          ↓
  DVC      data/processed   Checkpoints   Metrics     ONNX       GCS
```

## Data Preparation

### Dataset Structure

The expected directory structure for training data:

```
data/
├── raw/                          # Original images (DVC tracked)
│   ├── Cancer/
│   │   ├── image001.jpg
│   │   ├── image002.png
│   │   └── ...
│   └── Non-Cancer/
│       ├── image001.jpg
│       ├── image002.png
│       └── ...
└── processed/                    # Preprocessed splits (DVC tracked)
    ├── train/
    │   ├── Cancer/
    │   └── Non-Cancer/
    ├── val/
    │   ├── Cancer/
    │   └── Non-Cancer/
    └── test/
        ├── Cancer/
        └── Non-Cancer/
```

### Data Specifications

- **Image Format**: JPEG or PNG
- **Target Size**: 224×224 pixels (ResNet/EfficientNet input size)
- **Color Space**: RGB (3 channels)
- **File Size**: Recommended <5MB per image
- **Label Convention**: Folder name = label

### Data Statistics (Current Dataset)

- **Total Images**: ~800 (example)
- **Train Set**: 70% (~560 images)
- **Validation Set**: 15% (~120 images)
- **Test Set**: 15% (~120 images)
- **Class Distribution**:
  - Non-Cancer: ~83% (majority class)
  - Cancer: ~17% (minority class)

### Preprocessing Pipeline

#### Step 1: Data Version Control

```bash
# Initialize DVC (if not done)
dvc init

# Add raw data to DVC tracking
dvc add data/raw

# Commit DVC file
git add data/raw.dvc .gitignore
git commit -m "Add raw data to DVC"

# Push data to remote storage
dvc push
```

#### Step 2: Image Preprocessing

The preprocessing script (`ml/scripts/preprocess.py`) performs:

1. **Image Loading**: Read images from raw directory
2. **Resizing**: Resize to 224×224 using high-quality resampling
3. **Format Conversion**: Standardize to PNG/JPEG
4. **Data Splitting**: Create train/val/test splits
5. **Directory Organization**: Save to structured folders

**Run preprocessing**:

```bash
# Using Makefile
make preprocess

# Or directly
python -m ml.scripts.preprocess \
  --input-dir data/raw \
  --output-dir data/processed \
  --train-split 0.7 \
  --val-split 0.15 \
  --test-split 0.15 \
  --target-size 224
```

**What happens**:
- Images are resized to 224×224
- Random stratified split (maintains class distribution)
- Saved to `data/processed/` with train/val/test subdirectories

#### Step 3: Track Processed Data

```bash
# Add processed data to DVC
dvc add data/processed

# Commit
git add data/processed.dvc
git commit -m "Add processed data to DVC"

# Push to remote
dvc push
```

### Data Augmentation

Data augmentation is applied **on-the-fly** during training (not in preprocessing):

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

augmentation = ImageDataGenerator(
    rotation_range=20,           # Random rotation ±20°
    width_shift_range=0.2,       # Horizontal shift ±20%
    height_shift_range=0.2,      # Vertical shift ±20%
    horizontal_flip=True,        # Random horizontal flip
    vertical_flip=True,          # Random vertical flip
    zoom_range=0.2,              # Random zoom ±20%
    fill_mode='nearest'          # Fill pixels after transforms
)
```

**Why on-the-fly augmentation?**
- Reduces storage (no duplicate images)
- Infinite variations during training
- Only applied to training set (not val/test)

## Model Architectures

### 1. Baseline CNN

A custom CNN built from scratch for benchmarking.

**Architecture** (`ml/src/model.py:create_baseline_cnn()`):

```python
Model: baseline_cnn
_________________________________________________________________
Layer (type)                 Output Shape              Params
=================================================================
conv2d_1 (Conv2D)           (None, 222, 222, 32)      896
max_pooling2d_1             (None, 111, 111, 32)      0
conv2d_2 (Conv2D)           (None, 109, 109, 64)      18,496
max_pooling2d_2             (None, 54, 54, 64)        0
flatten                     (None, 186,624)           0
dense_1 (Dense)             (None, 128)               23,888,000
dense_2 (Dense)             (None, 1)                 129
=================================================================
Total params: 23,907,521
Trainable params: 23,907,521
```

**Characteristics**:
- Simple architecture, fast training
- Good baseline performance
- Prone to overfitting on small datasets
- **Champion Model**: baseline_model_v2.keras

**Performance** (from `reports/champion_model_metrics.json`):
- **Accuracy**: 87.5%
- **Recall (Cancer)**: 94.7% (high sensitivity)
- **Precision (Cancer)**: 58.1% (more false positives)

### 2. Transfer Learning (EfficientNetB0)

Uses pre-trained EfficientNetB0 as feature extractor.

**Architecture** (`ml/src/model.py:create_transfer_model()`):

```python
Model: transfer_efficientnet
_________________________________________________________________
Layer (type)                 Output Shape              Params
=================================================================
efficientnetb0 (Functional) (None, 7, 7, 1280)        4,049,571 (frozen)
global_avg_pooling2d        (None, 1280)              0
dense_1 (Dense)             (None, 128)               163,968
dense_2 (Dense)             (None, 1)                 129
=================================================================
Total params: 4,213,668
Trainable params: 164,097
Non-trainable params: 4,049,571
```

**Characteristics**:
- Leverages ImageNet pre-trained weights
- Fewer trainable parameters
- Better generalization on small datasets
- Slower inference than baseline CNN

**Training Strategy**:
1. **Phase 1**: Freeze base model, train head (5-10 epochs)
2. **Phase 2**: Unfreeze top layers, fine-tune (10-20 epochs)

### 3. Regularized Transfer Learning

Transfer learning with regularization to prevent overfitting.

**Architecture** (`ml/src/model.py:create_regularized_transfer_model()`):

```python
Model: regularized_transfer
_________________________________________________________________
Layer (type)                 Output Shape              Params
=================================================================
efficientnetb0 (Functional) (None, 7, 7, 1280)        4,049,571
global_avg_pooling2d        (None, 1280)              0
dropout (Dropout)           (None, 1280)              0
dense_1 (Dense)             (None, 128)               163,968
  + L2 Regularization (0.01)
dropout_2 (Dropout)         (None, 128)               0
dense_2 (Dense)             (None, 1)                 129
=================================================================
```

**Regularization Techniques**:
- **Dropout**: 50% dropout after pooling and dense layers
- **L2 Regularization**: Weight decay of 0.01
- **Early Stopping**: Stop when validation loss plateaus

### Model Selection

| Model | Pros | Cons | Use Case |
|-------|------|------|----------|
| **Baseline CNN** | Fast, simple, good baseline | Overfits easily | Quick iterations, benchmarking |
| **Transfer Learning** | Better accuracy, less data needed | Slower, larger | Production use |
| **Regularized Transfer** | Best generalization | Slower training | Small datasets, high variance |

## Training Pipeline

### Quick Start

**Train baseline model**:

```bash
make train MODEL=baseline EPOCHS=20
```

**Train transfer learning model**:

```bash
make train MODEL=transfer EPOCHS=30
```

### Detailed Training

#### 1. Configure Training

The training script (`ml/scripts/train.py`) accepts these arguments:

```bash
python -m ml.scripts.train \
  --model baseline \              # Model type: baseline, transfer, regularized_transfer
  --epochs 20 \                   # Number of epochs
  --batch-size 32 \               # Batch size (adjust based on GPU memory)
  --learning-rate 0.001 \         # Initial learning rate
  --train-dir data/processed/train \
  --val-dir data/processed/val \
  --checkpoint-dir models/checkpoints \
  --experiment-name baseline_training
```

#### 2. Training Process

**What happens during training**:

```python
# 1. Load data
train_generator = ImageDataGenerator(...).flow_from_directory(
    'data/processed/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# 2. Build model
model = create_baseline_cnn()  # or other model type

# 3. Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall', 'auc']
)

# 4. Configure callbacks
callbacks = [
    ModelCheckpoint(
        'models/checkpoints/baseline_model_v2.keras',
        monitor='val_loss',
        save_best_only=True
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3
    )
]

# 5. Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks
)

# 6. Save final model
model.save('models/checkpoints/baseline_model_v2.keras')
```

#### 3. Monitor Training

**Using MLflow UI**:

```bash
# Start MLflow UI
mlflow ui --port 5000

# Access at http://localhost:5000
```

**Using TensorBoard** (if integrated):

```bash
tensorboard --logdir logs/
```

**Training Output**:

```
Epoch 1/20
18/18 [==============================] - 12s 650ms/step - loss: 0.6234 - accuracy: 0.6500 - val_loss: 0.5891 - val_accuracy: 0.7083
Epoch 2/20
18/18 [==============================] - 10s 560ms/step - loss: 0.4567 - accuracy: 0.7850 - val_loss: 0.4123 - val_accuracy: 0.8250
...
Epoch 15/20
18/18 [==============================] - 10s 565ms/step - loss: 0.1234 - accuracy: 0.9500 - val_loss: 0.2456 - val_accuracy: 0.8750

Early stopping triggered. Restoring best weights from epoch 12.
```

### Training Best Practices

#### 1. Label Flipping

The training script flips labels so **Cancer = 1** (positive class):

```python
# In train.py
# Original: Cancer folder = class 1, Non-Cancer folder = class 0
# Flipped: Cancer = 1 (positive), Non-Cancer = 0 (negative)
y_train = 1 - y_train  # Flip labels
```

**Why?** Medical convention treats Cancer as positive case.

#### 2. Class Imbalance

Handle imbalanced classes (83% Non-Cancer, 17% Cancer):

**Option A: Class Weights**

```python
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

model.fit(
    train_generator,
    class_weight={0: class_weights[0], 1: class_weights[1]}
)
```

**Option B: Oversampling Minority Class**

```python
from imblearn.over_sampling import SMOTE

# Oversample cancer cases
smote = SMOTE(sampling_strategy=0.5)  # Make Cancer 50% of Non-Cancer
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**Current Approach**: Class weights with threshold optimization.

#### 3. Monitoring Overfitting

Watch for these signs:

- **Training accuracy** increases while **validation accuracy** decreases
- Large gap between train and validation loss
- Validation loss increases after initial decrease

**Solutions**:
- Add dropout layers
- Use L2 regularization
- Increase data augmentation
- Reduce model complexity
- Use early stopping

## Hyperparameter Tuning

### Key Hyperparameters

| Hyperparameter | Default | Range | Impact |
|----------------|---------|-------|--------|
| Learning Rate | 0.001 | [0.0001, 0.01] | Training speed, convergence |
| Batch Size | 32 | [16, 64] | Memory usage, gradient stability |
| Epochs | 20 | [10, 50] | Training time, overfitting risk |
| Dropout Rate | 0.5 | [0.3, 0.7] | Regularization strength |
| L2 Lambda | 0.01 | [0.001, 0.1] | Weight penalty |
| Augmentation | Medium | [Low, High] | Data variation |

### Tuning Methods

#### 1. Grid Search

```python
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

def create_model(learning_rate=0.001, dropout_rate=0.5):
    model = create_baseline_cnn()
    model.add(Dropout(dropout_rate))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32)

param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01],
    'dropout_rate': [0.3, 0.5, 0.7],
    'batch_size': [16, 32, 64]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)
```

#### 2. Random Search (Faster)

```python
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'learning_rate': uniform(0.0001, 0.01),
    'dropout_rate': uniform(0.3, 0.7),
    'batch_size': [16, 32, 64]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=20,
    cv=3,
    random_state=42
)
```

#### 3. Manual Tuning with MLflow

```python
import mlflow

learning_rates = [0.0001, 0.001, 0.01]
batch_sizes = [16, 32, 64]

for lr in learning_rates:
    for bs in batch_sizes:
        with mlflow.start_run():
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size", bs)

            model = train_model(learning_rate=lr, batch_size=bs)
            val_acc = evaluate_model(model)

            mlflow.log_metric("val_accuracy", val_acc)
```

## Model Evaluation

### Evaluation Pipeline

```bash
# Evaluate trained model
make evaluate

# Or directly
python -m ml.scripts.evaluate \
  --model-path models/checkpoints/baseline_model_v2.keras \
  --test-dir data/processed/test \
  --output-report reports/baseline_metrics.json
```

### Evaluation Metrics

#### 1. Confusion Matrix

```
                Predicted
              Non-Cancer  Cancer
Actual
Non-Cancer       80        13       (TN=80, FP=13)
Cancer            1        18       (FN=1, TP=18)

Metrics:
- True Negatives (TN): 80 (correctly identified non-cancer)
- False Positives (FP): 13 (incorrectly flagged as cancer)
- False Negatives (FN): 1 (missed cancer)
- True Positives (TP): 18 (correctly identified cancer)
```

#### 2. Classification Metrics

**Accuracy**: Overall correctness
```
Accuracy = (TP + TN) / Total = (18 + 80) / 112 = 87.5%
```

**Precision** (Cancer class): When model predicts cancer, how often is it correct?
```
Precision = TP / (TP + FP) = 18 / (18 + 13) = 58.1%
```

**Recall (Sensitivity)**: Of all actual cancers, how many did we catch?
```
Recall = TP / (TP + FN) = 18 / (18 + 1) = 94.7%
```

**F1-Score**: Harmonic mean of precision and recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall) = 72.0%
```

**Specificity**: Of all non-cancers, how many did we correctly identify?
```
Specificity = TN / (TN + FP) = 80 / (80 + 13) = 86.0%
```

#### 3. ROC-AUC

Area Under the ROC Curve measures model's ability to discriminate between classes:

- **AUC = 1.0**: Perfect classifier
- **AUC = 0.9-1.0**: Excellent
- **AUC = 0.8-0.9**: Good (typical for medical imaging)
- **AUC = 0.7-0.8**: Fair
- **AUC < 0.7**: Poor

**Current Model**: AUC ≈ 0.90 (good discrimination)

### Threshold Optimization

The default threshold is 0.5, but we optimize for medical use case:

```python
from sklearn.metrics import precision_recall_curve

# Get probabilities
y_proba = model.predict(X_test)

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# Optimize for high recall (catch cancer cases)
# Target: Recall > 0.95, maximize precision
optimal_idx = np.argmax(recalls >= 0.95)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold}")  # 0.110593
```

**Decision Logic**:
```python
if probability > 0.110593:
    prediction = "Cancer"
else:
    prediction = "Non-Cancer"
```

**Trade-off**:
- Lower threshold → Higher recall (catch more cancers) but more false positives
- Higher threshold → Higher precision (fewer false alarms) but miss more cancers

**Medical Rationale**: False negative (missed cancer) is worse than false positive (unnecessary follow-up).

### Metrics Report

The evaluation script generates `reports/champion_model_metrics.json`:

```json
{
  "model_name": "baseline_cnn_v2",
  "test_accuracy": 0.875,
  "test_loss": 0.3234,
  "threshold": 0.110593,
  "classification_report": {
    "0": {
      "precision": 0.9877,
      "recall": 0.8602,
      "f1-score": 0.9195,
      "support": 93
    },
    "1": {
      "precision": 0.5806,
      "recall": 0.9474,
      "f1-score": 0.7200,
      "support": 19
    }
  },
  "confusion_matrix": [[80, 13], [1, 18]],
  "roc_auc": 0.9038
}
```

## Model Export

### Export to ONNX

ONNX (Open Neural Network Exchange) enables cross-platform inference.

**Why ONNX?**
- Run model in Go backend (no Python dependency)
- Optimized inference graph
- Cross-framework compatibility
- Smaller deployment footprint

**Export Command**:

```bash
python -m ml.scripts.export \
  --keras-model models/checkpoints/baseline_model_v2.keras \
  --output-path models/saved_models/champion_model.onnx \
  --opset 13
```

**What happens**:

```python
import tf2onnx
import onnx

# 1. Load Keras model
model = tf.keras.models.load_model('baseline_model_v2.keras')

# 2. Remove augmentation layer (if present)
# Create inference-only model
input_layer = Input(shape=(224, 224, 3))
x = model.layers[1](input_layer)  # Skip augmentation
for layer in model.layers[2:]:
    x = layer(x)
inference_model = Model(input_layer, x)

# 3. Convert to ONNX
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(inference_model, input_signature=spec, opset=13)

# 4. Save ONNX file
onnx.save(onnx_model, 'champion_model.onnx')
```

**Validation**:

```python
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession('champion_model.onnx')

# Test inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Compare with Keras
keras_output = model.predict(test_image)
onnx_output = session.run([output_name], {input_name: test_image})

assert np.allclose(keras_output, onnx_output, rtol=1e-03)
```

### Upload to Cloud Storage

```bash
# Authenticate
gcloud auth login

# Upload model
gsutil cp models/saved_models/champion_model.onnx gs://mammoscan-ai-models/

# Verify
gsutil ls -lh gs://mammoscan-ai-models/
```

## MLflow Integration

### MLflow Setup

```bash
# MLflow is installed with ml/requirements.txt

# Start MLflow UI
mlflow ui --port 5000

# Access at http://localhost:5000
```

### Experiment Tracking

MLflow automatically logs:
- **Parameters**: learning_rate, batch_size, epochs, model_type
- **Metrics**: loss, accuracy, val_loss, val_accuracy (per epoch)
- **Artifacts**: Model checkpoints, training plots

**Manual Logging**:

```python
import mlflow
import mlflow.tensorflow

mlflow.set_experiment("baseline_training")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "baseline_cnn")
    mlflow.log_param("epochs", 20)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("learning_rate", 0.001)

    # Train model
    history = model.fit(...)

    # Log metrics
    for epoch, metrics in enumerate(history.history):
        mlflow.log_metrics({
            "train_loss": metrics['loss'],
            "train_accuracy": metrics['accuracy'],
            "val_loss": metrics['val_loss'],
            "val_accuracy": metrics['val_accuracy']
        }, step=epoch)

    # Log artifacts
    mlflow.log_artifact("models/checkpoints/baseline_model_v2.keras")
    mlflow.log_artifact("reports/training_plot.png")

    # Log model
    mlflow.tensorflow.log_model(model, "model")
```

### Model Registry

Promote models to production:

```python
# Register model
mlflow.register_model(
    model_uri="runs:/<run_id>/model",
    name="mammoscan-classifier"
)

# Transition to production
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="mammoscan-classifier",
    version=2,
    stage="Production"
)
```

## Best Practices

### 1. Reproducibility

```python
# Set random seeds
import numpy as np
import tensorflow as tf
import random

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
```

### 2. Checkpoint Strategy

- Save best model based on validation loss
- Keep top-3 checkpoints
- Save final model even if not best

### 3. Monitoring

- Track metrics every epoch
- Log confusion matrix on validation set
- Visualize training curves

### 4. Version Control

- DVC for data versioning
- Git for code versioning
- MLflow for experiment versioning

### 5. Medical ML Considerations

- **Prioritize Recall**: Missing cancer is worse than false alarm
- **Interpretability**: Consider Grad-CAM for visualization
- **Validation**: Use external test set from different institution
- **Uncertainty**: Add prediction confidence intervals

## Troubleshooting

### Issue: Out of Memory

```python
# Reduce batch size
batch_size = 16  # or 8

# Use mixed precision
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Clear session between runs
from tensorflow.keras import backend as K
K.clear_session()
```

### Issue: Overfitting

- Add dropout layers (0.5)
- Increase L2 regularization
- Use more data augmentation
- Reduce model complexity
- Use early stopping

### Issue: Underfitting

- Increase model capacity (more layers/units)
- Train longer
- Reduce regularization
- Check data quality
- Verify labels are correct

### Issue: Class Imbalance

- Use class weights
- Oversample minority class
- Use focal loss
- Adjust decision threshold

### Issue: Slow Training

- Use GPU (if available)
- Increase batch size
- Use mixed precision
- Enable XLA compilation
- Use data prefetching

## Next Steps

After training a model:

1. **Evaluate** on test set
2. **Export** to ONNX format
3. **Validate** ONNX inference
4. **Upload** to GCS
5. **Update** backend configuration
6. **Deploy** to Cloud Run
7. **Monitor** production performance

## Additional Resources

- [TensorFlow Guides](https://www.tensorflow.org/guide)
- [Keras Documentation](https://keras.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [ONNX Documentation](https://onnx.ai/)
- [Medical ML Best Practices](https://arxiv.org/abs/1811.12808)

## Support

For questions or issues:
- GitHub Issues: https://github.com/josephed37/mammoscan-AI/issues
- Project Wiki: https://github.com/josephed37/mammoscan-AI/wiki
