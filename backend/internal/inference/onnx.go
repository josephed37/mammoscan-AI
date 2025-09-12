// backend/internal/inference/onnx.go
package inference

import (
	"fmt"
	"os"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
)

// ONNXInference holds the loaded model and backend.
type ONNXInference struct {
	model   *onnx.Model
	backend onnx.Backend
}

// NewONNXInference loads an ONNX model from the specified file path.
func NewONNXInference(modelPath string) (*ONNXInference, error) {
	// Read the model data from the file.
	modelData, err := os.ReadFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read model file: %w", err)
	}

	// Create a new Gorgonia backend (the engine that runs the model).
	backend := gorgonnx.NewGraph()
	// Create a new model object.
	model := onnx.NewModel(backend)

	// Decode the model data into the model object.
	err = model.UnmarshalBinary(modelData)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal ONNX model: %w", err)
	}

	return &ONNXInference{
		model:   model,
		backend: backend,
	}, nil
}

// Predict runs inference on the input data.
// (We will complete this in a future step when we add preprocessing).
func (o *ONNXInference) Predict(inputTensor tensor.Tensor) ([]float32, error) {
	// Set the input tensor to the model.
	err := o.model.SetInput(0, inputTensor)
	if err != nil {
		return nil, fmt.Errorf("failed to set input: %w", err)
	}

	// Run the model's computation graph.
	g, ok := o.backend.(*gorgonnx.Graph)
	if !ok {
		return nil, fmt.Errorf("backend is not a *gorgonnx.Graph")
	}
	err = g.Run()
	if err != nil {
		return nil, fmt.Errorf("failed to run model: %w", err)
	}

	// Get the prediction result from the model.
	outputs, err := o.model.GetOutputTensors()
	if err != nil {
		return nil, fmt.Errorf("failed to get output: %w", err)
	}

	if len(outputs) == 0 {
		return nil, fmt.Errorf("no output tensors found")
	}

	// Convert the output tensor to a float32 slice.
	outputData, ok := outputs[0].Data().([]float32)
	if !ok {
		return nil, fmt.Errorf("failed to convert output to float32 slice")
	}

	return outputData, nil
}