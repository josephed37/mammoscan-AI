# ml/scripts/export.py
# FINAL CORRECTED VERSION

import os
import sys
import argparse
import tensorflow as tf
import tf2onnx

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Custom Modules ---
from ml.src.model import build_full_model

def export_to_onnx(keras_model_path, onnx_model_path):
    """Loads a Keras model and saves it in ONNX format."""
    print(f"--- 🚀 Starting ONNX Export ---")
    
    # --- Step 1: Load the Full Training Model ---
    print("Loading full training model to access weights...")
    full_training_model = build_full_model(input_shape=(224, 224, 3))
    full_training_model.load_weights(keras_model_path)
    print("✅ Keras model weights loaded successfully.")

    # Extract the core CNN part that has the learned weights
    trained_cnn = full_training_model.layers[1]

    # --- Step 2: Create a Clean Inference Model using the FUNCTIONAL API ---
    print("Building a clean, functional-API model for conversion...")
    # Define the explicit input layer
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input")
    # Pass the inputs through our trained CNN model
    outputs = trained_cnn(inputs)
    # Create the final, functional model
    inference_model = tf.keras.Model(inputs, outputs)
    print("✅ Functional model created successfully.")

    # --- Step 3: Convert the Functional Model to ONNX ---
    print(f"Converting model to ONNX format...")
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(inference_model, input_signature=spec, opset=13, output_path=onnx_model_path)

    # --- Step 4: Verify ---
    if os.path.exists(onnx_model_path):
        print(f"✅ Model successfully converted and saved to: {onnx_model_path}")
    else:
        print(f"🔥 Error: ONNX model conversion failed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a Keras model to ONNX.")
    default_keras_path = os.path.join(project_root, 'models', 'checkpoints', 'baseline_model_v2.keras')
    default_onnx_path = os.path.join(project_root, 'models', 'saved_models', 'champion_model.onnx')
    parser.add_argument("--keras-path", type=str, default=default_keras_path)
    parser.add_argument("--onnx-path", type=str, default=default_onnx_path)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.onnx_path), exist_ok=True)
    export_to_onnx(args.keras_path, args.onnx_path)