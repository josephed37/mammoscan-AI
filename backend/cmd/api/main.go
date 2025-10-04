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
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
	"github.com/josephed37/mammoscan-AI/backend/internal/handlers"
	"github.com/josephed37/mammoscan-AI/backend/internal/inference"
)

func main() {
	// Get model path from environment or use default
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		modelPath = "/app/models/saved_models/champion_model.onnx"
	}

	log.Printf("Loading ONNX model from path: %s", modelPath)

	// Create ONNX inference engine
	inferenceEngine, err := inference.NewONNXInference(modelPath)
	if err != nil {
		log.Fatalf("Failed to load ONNX model: %v", err)
	}

	log.Println("✅ ONNX model loaded successfully!")

	// Initialize handler with inference engine
	handler := handlers.NewHandler(inferenceEngine)

	// Setup router
	router := gin.Default()
	
	// Health check endpoint
	router.GET("/healthy", handler.HealthCheck)
	
	// Prediction endpoint
	router.POST("/api/v1/predict", handler.Predict)

	// Get port from environment (Cloud Run sets this)
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Starting server on :%s", port)
	
	// Start server
	if err := http.ListenAndServe(":"+port, router); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}