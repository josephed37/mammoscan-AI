// backend/cmd/api/main.go

package main

import (
	"log"
	"net/http"

	// Import our new handlers package
	"github.com/josephed37/mammoscan-AI/backend/internal/handlers"
)

const port = ":8080"

func main() {
	mux := http.NewServeMux()

	// Keep the health check endpoint.
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("OK"))
	})

	// Register our new prediction handler for the "/api/v1/predict" path.
	mux.HandleFunc("POST /api/v1/predict", handlers.PredictionHandler)

	log.Printf("Starting server on port %s", port)
	if err := http.ListenAndServe(port, mux); err != nil {
		log.Fatal(err)
	}
}
