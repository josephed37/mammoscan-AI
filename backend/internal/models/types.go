package models

// PredictionRequest defines the structure for incoming prediction requests.
// For now, it's empty as we will handle a direct image upload, but this
// structure is here for future expansion (e.g., if we needed to pass metadata).
type PredictionRequest struct {
	// Example field: UserID string `json:"user_id"`
}

// PredictionResponse defines the structure for our API's JSON response.
type PredictionResponse struct {
	Prediction      string  `json:"prediction"`
	ConfidenceScore float64 `json:"confidence_score"`
	ModelName       string  `json:"model_name"`
	ModelThreshold  float64 `json:"model_threshold"`
}

// ErrorResponse defines the structure for our API's JSON error response.
type ErrorResponse struct {
	Error string `json:"error"`
}
