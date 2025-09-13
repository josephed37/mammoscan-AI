// backend/cmd/api/main.go
/*
 * Main entry point for the MammoScan AI backend API.
 *
 * This application initializes the ONNX model, sets up the Gin web server,
 * and defines the API routes. It serves as the central orchestrator for the
 * backend service.
 *
 * Author: Joseph Edjeani
 * Date:   September 13, 2025
 * Version: 1.0.0
 */

package main

import (
	"log"
	"os" // Used to read environment variables for configuration

	"github.com/gin-gonic/gin"
	"github.com/josephed37/mammoscan-AI/backend/internal/handlers"
	"github.com/josephed37/mammoscan-AI/backend/internal/inference"
)

func main() {
	// --- Configuration ---
	// Read the model path from an environment variable. This is a best practice
	// for containerized applications, as it allows us to configure the application
	// without changing the code.
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		// If the environment variable is not set, we provide a fallback path.
		// This makes local development easier, as we can run `go run` without
		// needing to set environment variables first.
		modelPath = "models/saved_models/champion_model.onnx"
	}

	// --- Model Loading ---
	// Initialize the ONNX inference engine at startup.
	// Loading the model into memory once ensures low latency for prediction requests,
	// as we don't have to load the 16MB file from disk every time.
	log.Println("Loading ONNX model from path:", modelPath)
	inferenceEngine, err := inference.NewONNXInference(modelPath)
	if err != nil {
		// If the model fails to load, the application cannot function.
		// We use log.Fatalf to print the error and exit immediately.
		log.Fatalf("Failed to load ONNX model: %v", err)
	}
	log.Println("✅ ONNX model loaded successfully!")

	// --- Dependency Injection ---
	// We create our handler and inject the loaded inference engine.
	// This makes our handlers testable and separates concerns.
	handler := handlers.NewHandler(inferenceEngine)

	// --- Router Setup ---
	// Initialize the Gin web framework with default middleware (logger, recovery).
	router := gin.Default()

	// Define the API endpoints and map them to their corresponding handler functions.
	router.GET("/healthz", handler.HealthCheck)
	router.POST("/api/v1/predict", handler.Predict)

	// --- Server Start ---
	// Start the HTTP server and listen for incoming requests on port 8080.
	log.Println("Starting server on :8080")
	if err := router.Run(":8080"); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
