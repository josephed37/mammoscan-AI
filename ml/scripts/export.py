"""
This script converts a trained TensorFlow/Keras model (.keras format)
into the ONNX (Open Neural Network Exchange) format.

ONNX is a standardized format that allows models to be used across
different frameworks and platforms, making it perfect for deploying
our Python-trained model in a Go application.
"""

import os
import sys
import argparse
import tensorflow as tf
import tf2onnx

# --- Path Setup ---
# This ensures the script can find our custom modules in ml/src.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Custom Modules ---
# Import our model-building functions to reconstruct the architecture.
from ml.src.model import build_full_model, create_baseline_cnn

def export_to_onnx(keras_model_path, onnx_model_path):
    """Loads a Keras model's weights and saves its core architecture in ONNX format."""
    print(f"--- 🚀 Starting ONNX Export ---")
    
    # --- Step 1: Load the Full Training Model to Access Weights ---
    print("Loading full training model to access weights...")
    # We build the full training architecture (including augmentation layers)
    # so that we can correctly load the weights from the saved file.
    full_training_model = build_full_model(input_shape=(224, 224, 3))
    full_training_model.load_weights(keras_model_path)
    print("✅ Keras model weights loaded successfully.")

    # Extract the core CNN part, which contains the learned weights we care about.
    trained_cnn = full_training_model.layers[1]

    # --- Step 2: Create a Clean Inference-Only Model (The "Blueprint Transfer") ---
    # To avoid compatibility issues, we create a fresh, clean model that only
    # contains the layers needed for inference (no augmentation).
    print("Building a clean, functional-API model for conversion...")
    # We use the Keras Functional API because it's more explicit and robust for conversion.
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input")
    # We then take our extracted, trained CNN and "call" it on the new inputs.
    outputs = trained_cnn(inputs)
    # This creates a new model with a clean graph, perfect for the ONNX converter.
    inference_model = tf.keras.Model(inputs, outputs)
    print("✅ Functional model created successfully.")

    # --- Step 3: Convert the Clean Model to ONNX ---
    print(f"Converting model to ONNX format...")
    # We define the input signature, telling ONNX to expect a batch of images of our chosen size.
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    
    # The main conversion call. Opset 13 is a stable and widely supported version.
    tf2onnx.convert.from_keras(inference_model, input_signature=spec, opset=13, output_path=onnx_model_path)

    # --- Step 4: Verify the output ---
    if os.path.exists(onnx_model_path):
        print(f"✅ Model successfully converted and saved to: {onnx_model_path}")
    else:
        print(f"🔥 Error: ONNX model conversion failed.")


if __name__ == '__main__':
    # --- Command-Line Argument Parser ---
    # This makes our script a flexible tool that can be used in automated pipelines.
    parser = argparse.ArgumentParser(description="Convert a Keras model to ONNX.")
    
    # Define default paths for our champion model for convenience.
    default_keras_path = os.path.join(project_root, 'models', 'checkpoints', 'baseline_model_v2.keras')
    default_onnx_path = os.path.join(project_root, 'models', 'saved_models', 'champion_model.onnx')

    parser.add_argument("--keras-path", type=str, default=default_keras_path)
    parser.add_argument("--onnx-path", type=str, default=default_onnx_path)

    args = parser.parse_args()
    
    # Ensure the output directory exists before saving.
    os.makedirs(os.path.dirname(args.onnx_path), exist_ok=True)
    
    export_to_onnx(args.keras_path, args.onnx_path)