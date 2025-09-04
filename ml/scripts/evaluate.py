# ml/scripts/evaluate.py
"""
This script provides the final, unbiased evaluation of a trained model.

It uses a "specialist function" approach, where each model type has its own
dedicated evaluation logic to ensure clarity and robustness.
"""

# ml/scripts/evaluate.py
# FINAL CORRECTED VERSION

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Custom Modules ---
# Import our trusted model-building functions
from ml.src.model import build_full_model, create_regularized_transfer_model


# --- Constants ---
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
CLASS_NAMES = ['Non-Cancer', 'Cancer'] # Class 0, Class 1

def evaluate_model(model_name, model_path, report_path, threshold):
    """
    Loads a trained model, evaluates it on the test set, and saves metrics.
    """
    print("--- 🕵️‍ Starting Final Model Evaluation ---")
    
    # --- 1. Load Data ---
    print("Loading test dataset...")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'test'),
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=False
    )
    # Remap labels to match training (Cancer=1)
    test_dataset = test_dataset.map(lambda x, y: (x, 1 - y))

    # Apply preprocessing if required by the model
    if model_name == 'transfer':
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        test_dataset = test_dataset.map(lambda x, y: (preprocess_input(x), y))

    # --- 2. Build, Compile, and Load Weights (The Robust Method) ---
    print(f"Re-creating '{model_name}' model architecture...")
    if model_name == 'baseline':
        # The baseline model was trained with augmentation layers, so we build that structure
        model = build_full_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    elif model_name == 'transfer':
        # The transfer model was trained with dropout, so we build that structure
        model = create_regularized_transfer_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    print("Compiling model for evaluation...")
    model.compile(metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])
    
    print(f"Loading weights from {model_path}...")
    model.load_weights(model_path)
    print("✅ Model loaded successfully.")

    # --- 3. Get Predictions ---
    true_labels = np.concatenate([y for x, y in test_dataset], axis=0)
    raw_predictions = model.predict(test_dataset)
    predicted_labels = (raw_predictions > threshold).astype(int)

    # --- 4. Generate and Save Report ---
    save_report(model_path, threshold, true_labels, predicted_labels, report_path)

def save_report(model_path, threshold, true_labels, predicted_labels, report_path):
    """Calculates metrics and saves them to a JSON file."""
    print("\n--- Generating Final Report ---")
    report_dict = classification_report(true_labels, predicted_labels, target_names=CLASS_NAMES, output_dict=True)
    
    final_metrics = {
        'model_path': model_path,
        'threshold': threshold,
        'classification_report': report_dict
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)

    print(f"--- ✅ Evaluation Complete ---")
    print(f"Final Recall for 'Cancer' at threshold {threshold}: {report_dict['Cancer']['recall']:.4f}")
    print(f"Metrics saved to: {report_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--model-name", type=str, required=True, choices=['baseline', 'transfer'])
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--report-path", type=str, required=True)
    parser.add_argument(
    "--threshold", 
    type=float, 
    default=0.5, # Set the default value here
    help="Classification threshold to use. Defaults to 0.5."
)

    
    args = parser.parse_args()
    evaluate_model(args.model_name, args.model_path, args.report_path, args.threshold)