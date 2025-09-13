// backend/internal/handlers/handlers.go
/*
 * This file defines the HTTP handlers for our API endpoints.
 *
 * Each handler is a function that receives an HTTP request, processes it,
 * and writes an HTTP response. This is where the core business logic
 * of our API resides.
 *
 * Author: Joseph Edjeani
 * Date:   September 13, 2025
 * Version: 1.0.0
 */

package handlers

import (
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/josephed37/mammoscan-AI/backend/internal/inference"
	"github.com/josephed37/mammoscan-AI/backend/internal/models"
	"github.com/josephed37/mammoscan-AI/backend/internal/preprocess"
)

// Handler is a struct that holds dependencies for our API handlers,
// such as the inference engine. This is a form of dependency injection,
// which makes our code modular and easier to test.
type Handler struct {
	InferenceEngine *inference.ONNXInference
}

// NewHandler is a constructor function that creates a new Handler
// with its required dependencies.
func NewHandler(inferenceEngine *inference.ONNXInference) *Handler {
	return &Handler{
		InferenceEngine: inferenceEngine,
	}
}

// HealthCheck is a simple handler that returns a 200 OK status.
// It's used by monitoring systems to verify that the service is alive and running.
func (h *Handler) HealthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "OK"})
}

// Predict is the core handler for our application. It orchestrates the
// entire process of receiving an image, preprocessing it, running inference,
// and returning a structured JSON response.
func (h *Handler) Predict(c *gin.Context) {
	// --- 1. Receive and Validate the Image Upload ---
	// c.FormFile retrieves the uploaded file from the "image" field of the multipart form.
	fileHeader, err := c.FormFile("image")
	if err != nil {
		// If no file is found, return a 400 Bad Request error.
		c.JSON(http.StatusBadRequest, models.ErrorResponse{Error: "image file is required"})
		return
	}

	// Open the file to get an io.Reader, which allows us to process the file's contents.
	file, err := fileHeader.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.ErrorResponse{Error: "failed to open uploaded file"})
		return
	}
	// We use defer to ensure the file is closed when the function exits.
	defer file.Close()

	// --- 2. Preprocess the Image ---
	// We pass the file to our preprocessing pipeline, which decodes, resizes,
	// and converts the image into the tensor format our model expects.
	inputTensor, err := preprocess.PreprocessImage(file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.ErrorResponse{Error: fmt.Sprintf("failed to preprocess image: %v", err)})
		return
	}

	// --- 3. Run Inference ---
	// The preprocessed tensor is passed to our ONNX model's predict method.
	prediction, err := h.InferenceEngine.Predict(inputTensor)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.ErrorResponse{Error: fmt.Sprintf("prediction failed: %v", err)})
		return
	}

	// The model returns a slice of probabilities, but since we have one output,
	// we only need the first value.
	confidenceScore := float64(prediction[0])

	// --- 4. Apply Threshold and Format the Response ---
	// This is where we apply the optimal decision threshold we found during our analysis.
	const modelThreshold = 0.110593
	var finalPrediction string

	if confidenceScore > modelThreshold {
		finalPrediction = "Cancer"
	} else {
		finalPrediction = "Non-Cancer"
	}

	// We populate our response struct with the final results.
	response := models.PredictionResponse{
		Prediction:      finalPrediction,
		ConfidenceScore: confidenceScore,
		ModelName:       "baseline_cnn_v2",
		ModelThreshold:  modelThreshold,
	}

	// Finally, we send the structured JSON response back to the client with a 200 OK status.
	c.JSON(http.StatusOK, response)
}
