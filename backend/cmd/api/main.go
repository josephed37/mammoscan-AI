// backend/cmd/api/main.go
package main

import (
	"log"
	"path/filepath"
	"runtime"

	"github.com/gin-gonic/gin"
	"github.com/josephed37/mammoscan-AI/backend/internal/handlers"
	"github.com/josephed37/mammoscan-AI/backend/internal/inference"
)

func main() {
	// --- Path Setup ---
	// Get the project root directory reliably.
	_, b, _, _ := runtime.Caller(0)
	basepath := filepath.Dir(b)
	projectRoot := filepath.Join(basepath, "..", "..", "..")
	modelPath := filepath.Join(projectRoot, "models", "saved_models", "champion_model.onnx")

	// --- Load the ONNX Model ---
	log.Println("Loading ONNX model...")
	inferenceEngine, err := inference.NewONNXInference(modelPath)
	if err != nil {
		log.Fatalf("Failed to load ONNX model: %v", err)
	}
	log.Println("✅ ONNX model loaded successfully!")

	// --- Dependency Injection ---
	// Create our handler and pass the loaded model to it.
	handler := handlers.NewHandler(inferenceEngine)

	// --- Setup Gin Router ---
	router := gin.Default()

	// --- Define API Routes ---
	router.GET("/healthy", handler.HealthCheck)
	// We will create a POST endpoint for prediction.
	// It will now handle multipart/form-data for image uploads.
	router.POST("/api/v1/predict", handler.Predict)

	// --- Start the Server ---
	log.Println("Starting server on :8080")
	if err := router.Run(":8080"); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
