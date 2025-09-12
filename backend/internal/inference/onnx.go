// backend/internal/inference/onnx.go

package inference

import (
	"fmt"
	"log"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

// Session holds the loaded ONNX model session.
var Session *ort.AdvancedSession

// LoadONNXModel loads the ONNX model from the specified path and initializes the session.
func LoadONNXModel(modelPath string) error {
	// --- FIX 1: Correctly set the shared library path ---
	// This function doesn't return an error, so we call it directly.
	// We'll point it to the library inside our project's backend directory.
	ort.SetSharedLibraryPath("backend/onnxruntime/lib/libonnxruntime.so.1.18.0")
	ort.InitializeEnvironment()

	// Check if the model file exists.
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return fmt.Errorf("model file not found at %s", modelPath)
	}

	// Define the input and output names for our model.
	inputNames := []string{"input"}
	outputNames := []string{"output_0"} // ONNX often names the output layer this way.

	// --- FIX 2: Call NewAdvancedSession with the correct number of arguments ---
	// We provide 'nil' for the last three arguments as we don't need them for simple loading.
	session, err := ort.NewAdvancedSession(modelPath,
		inputNames,
		outputNames,
		nil, // input values (for initialization, not needed)
		nil, // output values (for initialization, not needed)
		nil, // session options (for advanced config, not needed)
	)
	if err != nil {
		return fmt.Errorf("failed to create onnx session: %w", err)
	}

	// Store the created session in our global variable.
	Session = session
	log.Println("✅ ONNX model loaded and session created successfully.")
	return nil
}

// CloseSession cleans up and destroys the ONNX session.
func CloseSession() {
	if Session != nil {
		Session.Destroy()
	}
	ort.DestroyEnvironment()
}
