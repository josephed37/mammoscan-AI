// backend/internal/models/types.go
/*
 * This file defines the data structures (structs) for our API.
 *
 * These structs act as the "data contract" for our application, defining the
 * expected format for JSON responses. Using structs ensures our API is
 * consistent, predictable, and easy to work with.
 *
 * Author: Joseph Edjeani
 * Date:   September 13, 2025
 * Version: 1.0.0
 */

package models

// PredictionResponse defines the structure for a successful JSON response
// when a prediction is made.
type PredictionResponse struct {
	// The final classification label (e.g., "Cancer" or "Non-Cancer").
	// The `json:"..."` tag defines how this field will be named in the JSON output.
	Prediction string `json:"prediction"`

	// The raw probability score (0.0 to 1.0) produced by the model.
	ConfidenceScore float64 `json:"confidence_score"`

	// The name of the model that produced the prediction.
	ModelName string `json:"model_name"`

	// The specific classification threshold used to make the final prediction.
	ModelThreshold float64 `json:"model_threshold"`
}

// ErrorResponse defines a standard structure for all error messages
// returned by the API. This ensures errors are consistent and easy for clients to parse.
type ErrorResponse struct {
	Error string `json:"error"`
}
