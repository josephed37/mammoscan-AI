// backend/internal/preprocess/image.go
package preprocess

import (
	"fmt"
	"image"
	// We need to import specific image formats for the decoder to recognize them.
	_ "image/jpeg"
	_ "image/png"
	"io"

	"github.com/nfnt/resize"
	"gorgonia.org/tensor"
)

// PreprocessImage takes an image file, decodes, resizes, and converts it into a tensor.
func PreprocessImage(file io.Reader) (tensor.Tensor, error) {
	// --- 1. Decode the Image ---
	// The `image.Decode` function automatically detects the format (JPEG, PNG, etc.).
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}

	// --- 2. Resize the Image ---
	// We resize it to 224x224 pixels, which is the required input size for our model.
	// `resize.Lanczos3` is a high-quality resampling filter.
	resizedImg := resize.Resize(224, 224, img, resize.Lanczos3)

	// --- 3. Convert Image to Tensor ---
	// Our model expects a tensor of shape [1, 224, 224, 3] with float32 values.
	// We iterate through the pixels and convert them into a flat float32 slice.
	height := resizedImg.Bounds().Dy()
	width := resizedImg.Bounds().Dx()
	tensorData := make([]float32, 1*height*width*3) // batch_size * h * w * channels

	// This loop extracts the R, G, B values for each pixel.
	// TensorFlow's default is channels-last (H, W, C).
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// Get the color of the pixel at (x, y).
			r, g, b, _ := resizedImg.At(x, y).RGBA()
			
			// The RGBA() method returns values from 0-65535. We need to scale them
			// down to the 0-255 range that our model was trained on.
			// The index calculation ensures we lay out the data correctly in the flat slice.
			baseIndex := (y*width + x) * 3
			tensorData[baseIndex+0] = float32(r >> 8) // Red channel
			tensorData[baseIndex+1] = float32(g >> 8) // Green channel
			tensorData[baseIndex+2] = float32(b >> 8) // Blue channel
		}
	}

	// Create the final tensor with the correct shape.
	inputTensor := tensor.New(
		tensor.WithShape(1, height, width, 3),
		tensor.WithBacking(tensorData),
	)

	return inputTensor, nil
}