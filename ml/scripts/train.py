"""
This is the main, flexible training script for all models in the project.

It uses a model registry to select an architecture and can be configured
with command-line arguments for automated MLOps pipelines.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Custom Modules ---
from ml.src.model import build_full_model, create_regularized_transfer_model

# --- Model Registry ---
MODEL_REGISTRY = {
    "baseline": build_full_model,
    "transfer": create_regularized_transfer_model,
}

# --- Constants ---
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')
MODEL_SAVE_DIR = os.path.join(project_root, 'models', 'checkpoints')
IMG_HEIGHT = 224
IMG_WIDTH = 224

def train_model(model_name, model_save_path, epochs, batch_size, learning_rate):
    """Main function to run the model training pipeline."""
    print(f"🚀 Starting training pipeline for '{model_name}' model...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # --- Data Loading and Preparation ---
    print("Loading datasets...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'train'),
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        label_mode='binary'
    )
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'val'),
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        label_mode='binary'
    )

    # Flip labels so that Cancer = 1
    train_dataset = train_dataset.map(lambda x, y: (x, 1 - y))
    val_dataset = val_dataset.map(lambda x, y: (x, 1 - y))

    # Apply model-specific preprocessing
    if model_name != 'baseline':
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y))
        val_dataset = val_dataset.map(lambda x, y: (preprocess_input(x), y))
    print("✅ Datasets configured.")

    # --- Class Weights ---
    remapped_labels = np.concatenate([y for x, y in train_dataset], axis=0)
    non_cancer_count, cancer_count = np.bincount(remapped_labels.astype(int).flatten())
    total = non_cancer_count + cancer_count
    weight_for_0 = (1 / non_cancer_count) * (total / 2.0)
    weight_for_1 = (1 / cancer_count) * (total / 2.0)
    class_weights = {0: weight_for_0, 1: weight_for_1}
    print(f"Calculated class weights: {class_weights}")

    # --- Model Creation ---
    model_creation_function = MODEL_REGISTRY.get(model_name)
    model = model_creation_function(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )
    
    # --- Training ---
    print("\n--- Starting model training ---")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )
    print("--- Model training finished ---\n")

    # --- Save Model ---
    model.save(model_save_path)
    print(f"✅ Model saved successfully to: {model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--model-name", type=str, required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--model-save-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    
    args = parser.parse_args()
    train_model(args.model_name, args.model_save_path, args.epochs, args.batch_size, args.learning_rate)