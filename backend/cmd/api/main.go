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
	"path/filepath"
	"runtime" // Import the runtime package

	"github.com/gin-gonic/gin"
	"github.com/josephed37/mammoscan-AI/backend/internal/handlers"
	"github.com/josephed37/mammoscan-AI/backend/internal/inference"
)

func main() {
	// --- THIS IS THE FIX: Find the source file's path ---
	// This is the most reliable way to find our project's location.
	_, b, _, _ := runtime.Caller(0)
	// 'b' is the path to the current file (main.go)
	basepath := filepath.Dir(b)

	// Construct an absolute path by navigating up from our current file's location
	// to the project root.
	projectRoot := filepath.Join(basepath, "..", "..", "..")
	modelPath := filepath.Join(projectRoot, "models", "saved_models", "champion_model.onnx")

	// --- Load the ONNX Model ---
	log.Println("Loading ONNX model from path:", modelPath)
	inferenceEngine, err := inference.NewONNXInference(modelPath)
	if err != nil {
		log.Fatalf("Failed to load ONNX model: %v", err)
	}
	log.Println("✅ ONNX model loaded successfully!")

	// --- Dependency Injection & Router Setup ---
	handler := handlers.NewHandler(inferenceEngine)
	router := gin.Default()
	router.GET("/healthy", handler.HealthCheck)
	router.POST("/api/v1/predict", handler.Predict)

	// --- Start the Server ---
	log.Println("Starting server on :8080")
	if err := router.Run(":8080"); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
