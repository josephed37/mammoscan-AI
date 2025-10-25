# Contributing to Mammoscan AI

Thank you for your interest in contributing to Mammoscan AI! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of age, body size, disability, ethnicity, gender identity, level of experience, nationality, personal appearance, race, religion, or sexual identity.

### Our Standards

**Positive behavior includes**:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes**:
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by opening an issue or contacting the project maintainers. All complaints will be reviewed and investigated promptly and fairly.

## Getting Started

### Prerequisites

Before contributing, ensure you have:

1. **GitHub Account**: Create one at https://github.com/signup
2. **Git Installed**: Version 2.30 or later
3. **Development Environment**: Set up per [Local Development Guide](docs/LOCAL_DEVELOPMENT.md)
4. **Familiarity with Project**: Read the [README](README.md) and [Architecture docs](docs/ARCHITECTURE.md)

### First-Time Contributors

If you're new to open source:

1. **Find a good first issue**: Look for issues labeled `good-first-issue` or `help-wanted`
2. **Read the docs**: Familiarize yourself with the codebase structure
3. **Set up your environment**: Follow the [Local Development Guide](docs/LOCAL_DEVELOPMENT.md)
4. **Ask questions**: Use GitHub Discussions or comment on issues

### Repository Structure

```
mammoscan-AI/
├── backend/          # Go API backend
├── web/             # Streamlit frontend
├── ml/              # ML training pipeline
├── data/            # Dataset (DVC tracked)
├── models/          # Trained models
├── deployments/     # Infrastructure configs
├── docs/            # Documentation
└── tests/           # Test files
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

#### 1. Code Contributions

- **Bug Fixes**: Fix reported bugs or issues
- **New Features**: Implement new functionality
- **Performance Improvements**: Optimize existing code
- **Refactoring**: Improve code quality and structure

#### 2. Documentation

- **Improve Existing Docs**: Fix typos, clarify instructions
- **Add New Documentation**: Create tutorials, examples
- **API Documentation**: Document functions and endpoints
- **Translations**: Translate docs to other languages

#### 3. Testing

- **Write Tests**: Add unit tests, integration tests
- **Test Coverage**: Improve test coverage
- **Bug Reports**: Report bugs with detailed reproduction steps

#### 4. Design

- **UI/UX Improvements**: Enhance frontend design
- **Architecture Design**: Propose system improvements
- **Visualizations**: Create diagrams, charts

#### 5. Research

- **Model Improvements**: Experiment with new architectures
- **Dataset Curation**: Help improve training data
- **Benchmarking**: Compare models and approaches

## Development Workflow

### 1. Fork the Repository

```bash
# Fork via GitHub UI, then clone your fork
git clone https://github.com/YOUR_USERNAME/mammoscan-AI.git
cd mammoscan-AI

# Add upstream remote
git remote add upstream https://github.com/josephed37/mammoscan-AI.git
```

### 2. Create a Branch

```bash
# Sync with upstream
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

**Branch Naming Convention**:
- `feature/` - New features (e.g., `feature/add-user-auth`)
- `fix/` - Bug fixes (e.g., `fix/image-upload-error`)
- `docs/` - Documentation (e.g., `docs/update-readme`)
- `refactor/` - Code refactoring (e.g., `refactor/handlers-package`)
- `test/` - Adding tests (e.g., `test/add-api-tests`)

### 3. Make Changes

```bash
# Make your changes
# Run tests frequently
# Commit early and often

# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add user authentication endpoint"
```

### 4. Keep Your Branch Updated

```bash
# Fetch upstream changes
git fetch upstream

# Rebase on upstream/main
git rebase upstream/main

# If conflicts, resolve them and continue
git rebase --continue
```

### 5. Push Changes

```bash
# Push to your fork
git push origin feature/your-feature-name

# If you rebased, force push (only to your fork!)
git push --force-with-lease origin feature/your-feature-name
```

### 6. Create Pull Request

1. Go to https://github.com/josephed37/mammoscan-AI
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill out the PR template (see below)
5. Submit the pull request

## Coding Standards

### General Principles

- **Readability**: Write code that is easy to read and understand
- **Simplicity**: Prefer simple solutions over complex ones
- **DRY**: Don't Repeat Yourself - extract common functionality
- **SOLID**: Follow SOLID principles for object-oriented code
- **Documentation**: Document complex logic and public APIs

### Go Style Guide

Follow [Effective Go](https://go.dev/doc/effective_go) and these conventions:

```go
// Package comment describing the package
package handlers

import (
    "context"
    "fmt"

    "github.com/gin-gonic/gin"
)

// Public functions should have comments
// Predict handles image prediction requests
func (h *Handler) Predict(c *gin.Context) {
    // Use early returns for error handling
    if c.Request.Method != "POST" {
        c.JSON(400, gin.H{"error": "method not allowed"})
        return
    }

    // Use descriptive variable names
    imageFile, err := c.FormFile("image")
    if err != nil {
        c.JSON(400, gin.H{"error": "image file required"})
        return
    }

    // Keep functions focused and small
    result, err := h.processImage(imageFile)
    if err != nil {
        c.JSON(500, gin.H{"error": err.Error()})
        return
    }

    c.JSON(200, result)
}
```

**Go Formatting**:
```bash
# Format code
go fmt ./...

# Run linter
golangci-lint run

# Run vet
go vet ./...
```

### Python Style Guide

Follow [PEP 8](https://pep8.org/) and [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html):

```python
"""Module for training models.

This module provides functions for training mammogram classification models
using TensorFlow/Keras.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional


def train_model(
    model_type: str,
    epochs: int,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Train a mammogram classification model.

    Args:
        model_type: Type of model to train ('baseline' or 'transfer')
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate

    Returns:
        Trained Keras model

    Raises:
        ValueError: If model_type is not recognized
    """
    # Use type hints
    # Descriptive variable names
    # Early returns for validation
    if model_type not in ['baseline', 'transfer']:
        raise ValueError(f"Unknown model type: {model_type}")

    # Use constants for magic numbers
    IMAGE_SIZE = 224
    NUM_CLASSES = 1  # Binary classification

    # Build model
    model = _build_model(model_type, IMAGE_SIZE, NUM_CLASSES)

    # Compile with appropriate loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    return model


def _build_model(model_type: str, image_size: int, num_classes: int) -> tf.keras.Model:
    """Build model architecture (private function)."""
    # Implementation...
    pass
```

**Python Formatting**:
```bash
# Format with black
black ml/ web/

# Sort imports
isort ml/ web/

# Lint with flake8
flake8 ml/ web/ --max-line-length=88

# Type check with mypy
mypy ml/src/
```

### Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code change that neither fixes bug nor adds feature
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Updating build tasks, package manager configs, etc.

**Examples**:

```
feat(backend): add user authentication endpoint

Implement JWT-based authentication for API endpoints.
Includes middleware for token validation.

Closes #123
```

```
fix(frontend): resolve image upload timeout issue

Increase request timeout from 10s to 30s to handle
large image uploads.

Fixes #456
```

```
docs: update model training guide

Add section on hyperparameter tuning and cross-validation.
```

## Testing Guidelines

### Test Coverage Requirements

- **Minimum Coverage**: 80% for new code
- **Critical Paths**: 100% coverage for core functionality
- **Edge Cases**: Test boundary conditions and error handling

### Writing Tests

#### Go Tests

```go
// handlers_test.go
package handlers

import (
    "bytes"
    "encoding/json"
    "net/http"
    "net/http/httptest"
    "testing"

    "github.com/gin-gonic/gin"
    "github.com/stretchr/testify/assert"
)

func TestHealthCheck(t *testing.T) {
    // Setup
    gin.SetMode(gin.TestMode)
    router := gin.Default()
    router.GET("/healthy", HealthCheck)

    // Execute
    req := httptest.NewRequest("GET", "/healthy", nil)
    w := httptest.NewRecorder()
    router.ServeHTTP(w, req)

    // Assert
    assert.Equal(t, http.StatusOK, w.Code)

    var response map[string]string
    err := json.Unmarshal(w.Body.Bytes(), &response)
    assert.NoError(t, err)
    assert.Equal(t, "OK", response["status"])
}

func TestPredict_InvalidImage(t *testing.T) {
    // Test error handling
    router := gin.Default()
    handler := NewHandler(mockEngine)
    router.POST("/predict", handler.Predict)

    req := httptest.NewRequest("POST", "/predict", bytes.NewBuffer([]byte("invalid")))
    w := httptest.NewRecorder()
    router.ServeHTTP(w, req)

    assert.Equal(t, http.StatusBadRequest, w.Code)
}
```

#### Python Tests

```python
# test_model.py
import pytest
import numpy as np
import tensorflow as tf
from ml.src.model import create_baseline_cnn


def test_baseline_cnn_output_shape():
    """Test that baseline CNN produces correct output shape."""
    model = create_baseline_cnn()

    # Test input
    batch_size = 4
    image_size = 224
    channels = 3
    test_input = np.random.randn(batch_size, image_size, image_size, channels)

    # Predict
    output = model.predict(test_input)

    # Assert output shape
    assert output.shape == (batch_size, 1)

    # Assert output range (sigmoid)
    assert np.all(output >= 0) and np.all(output <= 1)


def test_model_training():
    """Test that model can train without errors."""
    model = create_baseline_cnn()
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Dummy data
    X_train = np.random.randn(10, 224, 224, 3)
    y_train = np.random.randint(0, 2, size=(10, 1))

    # Train for 1 epoch
    history = model.fit(X_train, y_train, epochs=1, verbose=0)

    # Assert training completed
    assert 'loss' in history.history


@pytest.fixture
def sample_image():
    """Fixture providing a sample test image."""
    return np.random.randn(1, 224, 224, 3).astype(np.float32)


def test_prediction_with_fixture(sample_image):
    """Test prediction using fixture."""
    model = create_baseline_cnn()
    output = model.predict(sample_image)

    assert output.shape == (1, 1)
```

### Running Tests

```bash
# Go tests
cd backend
go test ./...                      # Run all tests
go test -v ./internal/handlers     # Verbose output
go test -cover ./...               # With coverage
go test -race ./...                # Race detection

# Python tests
cd ml
pytest tests/                      # Run all tests
pytest -v tests/                   # Verbose
pytest --cov=src tests/            # With coverage
pytest -k test_model tests/        # Run specific test
```

### Integration Tests

```bash
# Start services
make docker-up

# Run integration tests
python tests/integration/test_api.py

# Clean up
make docker-down
```

## Documentation

### Documentation Requirements

- **Public APIs**: All public functions must have docstrings
- **Complex Logic**: Add inline comments explaining "why", not "what"
- **README**: Update if adding new features or changing setup
- **API Changes**: Update API documentation
- **Breaking Changes**: Clearly document in PR and CHANGELOG

### Writing Documentation

#### Go Documentation

```go
// Package handlers provides HTTP request handlers for the Mammoscan API.
package handlers

// Handler manages API request handling with an inference engine.
type Handler struct {
    engine *inference.Engine
}

// NewHandler creates a new Handler with the given inference engine.
//
// Example:
//   engine := inference.NewEngine("model.onnx")
//   handler := NewHandler(engine)
func NewHandler(engine *inference.Engine) *Handler {
    return &Handler{engine: engine}
}

// Predict handles POST /api/v1/predict requests.
//
// Expects multipart form data with an "image" field containing
// a JPEG or PNG mammogram image.
//
// Returns JSON response:
//   {
//     "prediction": "Cancer" | "Non-Cancer",
//     "confidence_score": 0.0-1.0,
//     "model_name": "baseline_cnn_v2",
//     "model_threshold": 0.110593
//   }
//
// Returns 400 Bad Request if image is missing or invalid.
// Returns 500 Internal Server Error if inference fails.
func (h *Handler) Predict(c *gin.Context) {
    // Implementation...
}
```

#### Python Documentation

```python
def train_model(
    model_type: str,
    epochs: int,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    data_dir: str = "data/processed",
) -> Tuple[tf.keras.Model, dict]:
    """Train a mammogram classification model.

    This function trains either a baseline CNN or transfer learning model
    on mammogram images. Training progress is logged to MLflow.

    Args:
        model_type: Type of model architecture. One of:
            - 'baseline': Simple CNN trained from scratch
            - 'transfer': EfficientNetB0 with transfer learning
        epochs: Number of training epochs. Recommended: 20-30.
        batch_size: Number of samples per gradient update.
            Default 32. Reduce if out of memory.
        learning_rate: Initial learning rate for Adam optimizer.
            Default 0.001.
        data_dir: Path to directory containing train/val subdirectories.
            Default 'data/processed'.

    Returns:
        A tuple containing:
            - model: Trained Keras model
            - history: Dictionary with training metrics per epoch

    Raises:
        ValueError: If model_type is not 'baseline' or 'transfer'
        FileNotFoundError: If data_dir does not exist
        RuntimeError: If training fails

    Example:
        >>> model, history = train_model('baseline', epochs=20)
        >>> print(f"Final accuracy: {history['val_accuracy'][-1]:.3f}")
        Final accuracy: 0.875

    Note:
        This function requires MLflow to be running for experiment tracking.
        Start MLflow UI with: mlflow ui --port 5000

    See Also:
        - evaluate_model: For model evaluation
        - export_to_onnx: For model export
    """
    # Implementation...
    pass
```

### Documentation Files

When adding documentation:

1. **API docs**: Update `docs/API.md` for endpoint changes
2. **Architecture**: Update `docs/ARCHITECTURE.md` for structural changes
3. **README**: Update if changing setup or adding features
4. **CHANGELOG**: Add entry for significant changes

## Pull Request Process

### Before Submitting

1. **Run Tests**: Ensure all tests pass
   ```bash
   make test
   ```

2. **Check Code Quality**: Run linters
   ```bash
   # Go
   cd backend && golangci-lint run

   # Python
   black ml/ web/
   flake8 ml/ web/
   ```

3. **Update Documentation**: Add/update docs if needed

4. **Rebase on Main**: Ensure your branch is up to date
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

5. **Test Locally**: Run the application and verify your changes

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement

## Related Issue
Closes #(issue number)

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
Describe tests you ran and how to reproduce:
1. Step 1
2. Step 2
3. Expected result

## Screenshots (if applicable)
Add screenshots for UI changes

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have updated the documentation accordingly
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or my feature works
- [ ] New and existing tests pass locally
- [ ] Any dependent changes have been merged and published
```

### Review Process

1. **Automatic Checks**: CI/CD pipeline runs tests
2. **Code Review**: Maintainers review your code
3. **Feedback**: Address review comments
4. **Approval**: Once approved, your PR will be merged
5. **Deployment**: Changes are deployed automatically (if on main)

### After Merge

- Delete your feature branch (local and remote)
- Update your fork's main branch
- Close related issues if applicable

## Reporting Issues

### Before Reporting

1. **Search Existing Issues**: Check if issue already exists
2. **Reproduce**: Ensure you can consistently reproduce the issue
3. **Gather Information**: Collect relevant logs, screenshots

### Issue Template

```markdown
## Bug Report / Feature Request

**Type**: Bug | Feature Request | Question

**Description**
Clear and concise description

**Steps to Reproduce** (for bugs)
1. Step 1
2. Step 2
3. See error

**Expected Behavior**
What you expected to happen

**Actual Behavior**
What actually happened

**Screenshots**
Add screenshots if applicable

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Docker version: [e.g., 20.10.17]
- Python version: [e.g., 3.11]
- Go version: [e.g., 1.24.4]

**Additional Context**
Any other relevant information

**Logs**
```
Paste relevant logs here
```
```

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good-first-issue`: Good for newcomers
- `help-wanted`: Extra attention needed
- `question`: Further information requested
- `wontfix`: This will not be worked on
- `duplicate`: This issue already exists

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Requests**: Code reviews, discussions

### Getting Help

- Read the [documentation](docs/)
- Search [existing issues](https://github.com/josephed37/mammoscan-AI/issues)
- Ask in [GitHub Discussions](https://github.com/josephed37/mammoscan-AI/discussions)
- Check the [Wiki](https://github.com/josephed37/mammoscan-AI/wiki)

### Recognition

Contributors are recognized in:
- GitHub Contributors page
- CHANGELOG.md (for significant contributions)
- Project README (for major contributions)

## License

By contributing to Mammoscan AI, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE) file).

## Questions?

If you have questions about contributing, feel free to:
- Open a GitHub Discussion
- Comment on relevant issues
- Reach out to maintainers

Thank you for contributing to Mammoscan AI! Your efforts help make this project better for everyone.
