// backend/internal/handlers/handlers.go
package handlers

import (
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/josephed37/mammoscan-AI/backend/internal/inference"
	"github.com/josephed37/mammoscan-AI/backend/internal/models"

	// Import our new preprocessing package
	"github.com/josephed37/mammoscan-AI/backend/internal/preprocess"
)

// Handler holds the dependencies for our API handlers.
type Handler struct {
	InferenceEngine *inference.ONNXInference
}

// NewHandler creates a new Handler with its dependencies.
func NewHandler(inferenceEngine *inference.ONNXInference) *Handler {
	return &Handler{
		InferenceEngine: inferenceEngine,
	}
}

// HealthCheck provides a simple health status.
func (h *Handler) HealthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "OK"})
}

// Predict handles the image upload and model prediction.
func (h *Handler) Predict(c *gin.Context) {
	// --- 1. Get the uploaded image file ---
	fileHeader, err := c.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, models.ErrorResponse{Error: "image file is required"})
		return
	}

	// Open the file to get an io.Reader
	file, err := fileHeader.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.ErrorResponse{Error: "failed to open uploaded file"})
		return
	}
	defer file.Close()

	// --- 2. Preprocess the Image ---
	// Call our new function to convert the file into a tensor.
	inputTensor, err := preprocess.PreprocessImage(file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.ErrorResponse{Error: fmt.Sprintf("failed to preprocess image: %v", err)})
		return
	}

	// --- 3. Run Inference ---
	// Pass the tensor to our ONNX model.
	prediction, err := h.InferenceEngine.Predict(inputTensor)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.ErrorResponse{Error: fmt.Sprintf("prediction failed: %v", err)})
		return
	}

	// The model returns a slice, but we only need the first value.
	confidenceScore := float64(prediction[0])

	// --- 4. Apply Threshold and Return Response ---
	// This is where we apply the optimal threshold we found.
	const modelThreshold = 0.110593
	var finalPrediction string
	if confidenceScore > modelThreshold {
		finalPrediction = "Cancer"
	} else {
		finalPrediction = "Non-Cancer"
	}

	response := models.PredictionResponse{
		Prediction:      finalPrediction,
		ConfidenceScore: confidenceScore,
		ModelName:       "baseline_cnn_v2",
		ModelThreshold:  modelThreshold,
	}

	c.JSON(http.StatusOK, response)
}
