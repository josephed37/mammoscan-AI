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
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"cloud.google.com/go/storage"
	"github.com/gin-gonic/gin"
	"github.com/josephed37/mammoscan-AI/backend/internal/handlers"
	"github.com/josephed37/mammoscan-AI/backend/internal/inference"
)

func downloadFromGCS(ctx context.Context, bucket, object, dest string) error {
	client, err := storage.NewClient(ctx)
	if err != nil {
		return fmt.Errorf("storage client: %w", err)
	}
	defer client.Close()

	os.MkdirAll(filepath.Dir(dest), 0755)
	
	rc, err := client.Bucket(bucket).Object(object).NewReader(ctx)
	if err != nil {
		return fmt.Errorf("object reader: %w", err)
	}
	defer rc.Close()

	f, err := os.Create(dest)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()

	if _, err := io.Copy(f, rc); err != nil {
		return fmt.Errorf("copy: %w", err)
	}

	log.Printf("Downloaded gs://%s/%s to %s", bucket, object, dest)
	return nil
}

func main() {
	ctx := context.Background()
	
	bucket := getEnv("MODEL_GCS_BUCKET", "mammoscan-ai-models")
	object := getEnv("MODEL_GCS_OBJECT", "champion_model.onnx")
	modelPath := getEnv("MODEL_PATH", "/tmp/champion_model.onnx")

	log.Printf("Downloading model from gs://%s/%s", bucket, object)
	if err := downloadFromGCS(ctx, bucket, object, modelPath); err != nil {
		log.Fatalf("Download failed: %v", err)
	}

	inferenceEngine, err := inference.NewONNXInference(modelPath)
	if err != nil {
		log.Fatalf("Load model failed: %v", err)
	}

	log.Println("✅ Model loaded successfully")

	handler := handlers.NewHandler(inferenceEngine)
	router := gin.Default()
	router.GET("/healthy", handler.HealthCheck)
	router.POST("/api/v1/predict", handler.Predict)

	port := getEnv("PORT", "8080")
	log.Printf("Server starting on :%s", port)
	http.ListenAndServe(":"+port, router)
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}