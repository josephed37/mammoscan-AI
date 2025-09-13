// backend/internal/inference/onnx.go
/*
 * This file contains the core logic for interacting with the ONNX model.
 *
 * It is responsible for loading the trained model from a file and providing
 * a simple interface to run predictions on preprocessed image data.
 * This abstracts away the complexities of the ONNX backend.
 *
 * Author: Joseph Edjeani
 * Date:   September 13, 2025
 * Version: 1.0.0
 */

package inference

import (
	"fmt"
	"os"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
)

// ONNXInference is a struct that holds the loaded model and its backend.
// This allows us to maintain the model's state in memory throughout the
// application's lifecycle, avoiding the need to reload it for every request.
type ONNXInference struct {
	model   *onnx.Model
	backend onnx.Backend
}

// NewONNXInference is a constructor function that loads an ONNX model
// from the specified file path and initializes the inference engine.
func NewONNXInference(modelPath string) (*ONNXInference, error) {
	// --- Step 1: Read the Model File ---
	// We read the entire .onnx model file into a byte slice.
	modelData, err := os.ReadFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read model file: %w", err)
	}

	// --- Step 2: Initialize the Backend and Model ---
	// Create a new Gorgonia backend, which is the computation engine that will
	// execute the model's operations.
	backend := gorgonnx.NewGraph()
	// Create a new model object that will use our backend.
	model := onnx.NewModel(backend)

	// --- Step 3: Decode the Model ---
	// UnmarshalBinary parses the raw byte data and decodes it into the
	// structured model object, building the computation graph.
	err = model.UnmarshalBinary(modelData)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal ONNX model: %w", err)
	}

	// Return the ready-to-use inference engine.
	return &ONNXInference{
		model:   model,
		backend: backend,
	}, nil
}

// Predict runs inference on a preprocessed input tensor.
func (o *ONNXInference) Predict(inputTensor tensor.Tensor) ([]float32, error) {
	// --- Step 1: Set the Input ---
	// We set the input tensor for the model. The '0' indicates that this is
	// the first (and in our case, only) input to the model.
	err := o.model.SetInput(0, inputTensor)
	if err != nil {
		return nil, fmt.Errorf("failed to set input: %w", err)
	}

	// --- Step 2: Run Inference ---
	// We run the backend's computation graph. This is where the actual
	// deep learning inference happens.
	g, ok := o.backend.(*gorgonnx.Graph)
	if !ok {
		return nil, fmt.Errorf("backend is not a *gorgonnx.Graph")
	}
	err = g.Run()
	if err != nil {
		return nil, fmt.Errorf("failed to run model: %w", err)
	}

	// --- Step 3: Get the Output ---
	// We retrieve the output tensors from the model after the run is complete.
	outputs, err := o.model.GetOutputTensors()
	if err != nil {
		return nil, fmt.Errorf("failed to get output: %w", err)
	}
	if len(outputs) == 0 {
		return nil, fmt.Errorf("no output tensors found")
	}

	// --- Step 4: Extract and Return the Result ---
	// We convert the output tensor's data into a simple slice of float32,
	// which is the raw probability score our application needs.
	outputData, ok := outputs[0].Data().([]float32)
	if !ok {
		return nil, fmt.Errorf("failed to convert output to float32 slice")
	}

	return outputData, nil
}
