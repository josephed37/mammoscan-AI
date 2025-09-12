// backend/internal/handlers/handlers.go
package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/josephed37/mammoscan-AI/backend/internal/inference"
	"github.com/josephed37/mammoscan-AI/backend/internal/models"
)

// Handler holds the dependencies for our API handlers, like the inference engine.
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
	// --- 1. Get the image file from the request ---
	_, err := c.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, models.ErrorResponse{Error: "image file is required"})
		return
	}

	// (Future Step: We will add the image preprocessing logic here)
	// For now, we'll just confirm the file was received.

	// --- 2. Run Inference (Dummy for now) ---
	// prediction, err := h.InferenceEngine.Predict(preprocessedImage)
	// (Error handling for prediction...)

	// --- 3. Return a successful response ---
	response := models.PredictionResponse{
		Prediction:      "Cancer", // Dummy prediction
		ConfidenceScore: 0.989,    // Dummy score
		ModelName:       "champion_model_v2",
		ModelThreshold:  0.110593,
	}

	c.JSON(http.StatusOK, response)
}
