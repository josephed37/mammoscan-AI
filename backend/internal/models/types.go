// backend/internal/models/types.go
package models

// We no longer need a PredictionRequest struct for file uploads.

// PredictionResponse defines the structure for a successful JSON response.
type PredictionResponse struct {
	Prediction      string  `json:"prediction"`
	ConfidenceScore float64 `json:"confidence_score"`
	ModelName       string  `json:"model_name"`
	ModelThreshold  float64 `json:"model_threshold"`
}

// ErrorResponse defines the structure for a JSON error response.
type ErrorResponse struct {
	Error string `json:"error"`
}
