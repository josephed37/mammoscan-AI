// backend/internal/preprocess/image.go
/*
 * This file contains the image preprocessing pipeline for the Go backend.
 *
 * Its primary responsibility is to take a raw image file uploaded by a user
 * and transform it into a standardized tensor that can be fed directly into
 * our ONNX model for inference. This ensures that the data format for
 * prediction perfectly matches the format used during model training.
 *
 * Author: Joseph Edjeani
 * Date:   September 13, 2025
 * Version: 1.0.0
 */

package preprocess

import (
	"fmt"
	"image"

	// We must perform a blank import of the image formats we want to support.
	// This registers the decoders with the `image` package, allowing it to
	// automatically detect and decode JPEG and PNG files.
	_ "image/jpeg"
	_ "image/png"
	"io"

	"github.com/nfnt/resize"
	"gorgonia.org/tensor"
)

// PreprocessImage orchestrates the entire image transformation pipeline.
// It takes an io.Reader (like an uploaded file), decodes it into an image object,
// resizes it to the model's required input dimensions, and finally converts it
// into a multi-dimensional tensor.
func PreprocessImage(file io.Reader) (tensor.Tensor, error) {
	// --- Step 1: Decode the Image ---
	// The `image.Decode` function reads the raw bytes from the file reader and,
	// thanks to our blank imports, automatically determines the correct format
	// (e.g., JPEG, PNG) and decodes it into a generic `image.Image` object.
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}

	// --- Step 2: Resize the Image ---
	// Our neural network expects a fixed input size of 224x224 pixels.
	// We use the `resize.Resize` function to downscale or upscale the image.
	// `resize.Lanczos3` is a high-quality interpolation algorithm that produces
	// a clear image with minimal artifacts.
	resizedImg := resize.Resize(224, 224, img, resize.Lanczos3)

	// --- Step 3: Convert Image to Tensor ---
	// The ONNX model requires the input data to be in a specific tensor format:
	// a 4D tensor with shape [batch_size, height, width, channels] and float32 values.
	height := resizedImg.Bounds().Dy()
	width := resizedImg.Bounds().Dx()
	// We create a flat slice to hold all the pixel data.
	tensorData := make([]float32, 1*height*width*3) // batch_size=1, channels=3 (R,G,B)

	// This loop iterates through each pixel of the resized image.
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// The `At(x, y).RGBA()` method returns the color of a pixel.
			r, g, b, _ := resizedImg.At(x, y).RGBA()

			// The returned RGBA values are 16-bit (0-65535). Our model was trained
			// on 8-bit values (0-255). The `>> 8` bit-shift operation is an
			// efficient way to convert from 16-bit to 8-bit.
			// The baseIndex calculation ensures we place the R, G, B values
			// sequentially in the flat slice, matching the "channels-last" format (HWC).
			baseIndex := (y*width + x) * 3
			tensorData[baseIndex+0] = float32(r >> 8) // Red channel
			tensorData[baseIndex+1] = float32(g >> 8) // Green channel
			tensorData[baseIndex+2] = float32(b >> 8) // Blue channel
		}
	}

	// Finally, we create a Gorgonia tensor object, wrapping our flat slice
	// of pixel data and applying the correct 4D shape that our model requires.
	inputTensor := tensor.New(
		tensor.WithShape(1, height, width, 3),
		tensor.WithBacking(tensorData),
	)

	return inputTensor, nil
}
