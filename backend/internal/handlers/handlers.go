// backend/internal/handlers/handlers.go

package handlers

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/josephed37/mammoscan-AI/backend/internal/models"
)

// PredictionHandler handles the incoming requests for model prediction.
func PredictionHandler(w http.ResponseWriter, r *http.Request) {
	// We only want to allow POST requests to this endpoint.
	if r.Method != http.MethodPost {
		handleError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// For now, we will just return a dummy successful response.
	// In the future, this is where we'll process the image and run the model.
	response := models.PredictionResponse{
		Prediction:      "Non-Cancer", // Dummy prediction
		ConfidenceScore: 0.95,
		ModelName:       "dummy_model_v1",
		ModelThreshold:  0.5,
	}

	// Set the content type header to application/json.
	w.Header().Set("Content-Type", "application/json")
	// Set the status code to 200 OK.
	w.WriteHeader(http.StatusOK)
	// Encode our response struct into JSON and write it to the response body.
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Error encoding response: %v", err)
	}
}

// handleError is a helper function to send a consistent JSON error message.
func handleError(w http.ResponseWriter, message string, statusCode int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	response := models.ErrorResponse{Error: message}
	json.NewEncoder(w).Encode(response)
}
