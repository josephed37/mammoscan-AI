"""
This script is the main training pipeline for the MammoScan AI model.
It now uses a model with on-the-fly data augmentation to prevent overfitting.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Custom Modules ---
# We now import our new, more advanced model-building function.
from ml.src.model import build_full_model

# --- Constants ---
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')
MODEL_SAVE_DIR = os.path.join(project_root, 'models', 'checkpoints')
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'augmented_model.keras') # New model name

# Training Hyperparameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20 # We can train for a few more epochs now

def main():
    """Main function to run the model training pipeline."""
    print("🚀 Starting model training pipeline with on-the-fly augmentation...")
    
    # IMPORTANT NOTE: For this training run, ensure your `data/processed` directory
    # was generated WITHOUT the pre-augmented data to avoid data leakage.
    # This was the experiment we ran in the previous step.

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    print("Loading datasets...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'train'),
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'val'),
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    
    class_names = train_dataset.class_names
    print(f"Classes found: {class_names}")

    print("Calculating class weights for imbalanced data...")
    labels = np.concatenate([y for x, y in train_dataset], axis=0)
    neg, pos = np.bincount(labels.astype(int).flatten())
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weights = {0: weight_for_0, 1: weight_for_1}
    print(f"Calculated class weights: {class_weights}")

    print("Creating and compiling the model with data augmentation layers...")
    # Here, we call our new function to get the complete, robust model.
    model = build_full_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )
    model.summary()

    print("\n--- Starting model training ---")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        class_weight=class_weights
    )
    print("--- Model training finished ---\n")

    print(f"Saving augmented model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("✅ Model saved successfully!")

if __name__ == '__main__':
    main()