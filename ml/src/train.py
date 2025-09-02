"""
This script is the main training pipeline for the MammoScan AI model.

It performs the following steps:
1.  Sets up paths and constants.
2.  Loads the preprocessed train and validation datasets.
3.  Calculates class weights to handle data imbalance.
4.  Creates and compiles the baseline CNN model.
5.  Trains the model using the datasets and class weights.
6.  Saves the final trained model to a file.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Path Setup ---
# This ensures the script can find our custom modules in ml/src
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Custom Modules ---
from ml.src.model import create_baseline_cnn

# --- Constants ---
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')
MODEL_SAVE_DIR = os.path.join(project_root, 'models', 'checkpoints')
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'baseline_model.keras') # Using the modern .keras format

# Training Hyperparameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10 # Start with a small number for the first run

def main():
    """Main function to run the model training pipeline."""
    print("🚀 Starting model training pipeline...")

    # Step 1: Create the directories if they don't exist
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Step 2: Load the prepped datasets using TensorFlow's utility
    print("Loading datasets...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'train'),
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='binary' # For binary classification (Cancer/Non-Cancer)
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'val'),
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    
    # Extract class names (e.g., ['Cancer', 'Non-Cancer'])
    class_names = train_dataset.class_names
    print(f"Classes found: {class_names}")

    # Step 3: Calculate Class Weights to handle imbalance
    # This is how we solve the problem we found in our EDA
    labels = np.concatenate([y for x, y in train_dataset], axis=0)
    neg = np.sum(labels == 0) # Assuming 0 is the negative class
    pos = np.sum(labels == 1) # Assuming 1 is the positive class
    total = neg + pos

    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    class_weights = {0: weight_for_0, 1: weight_for_1}
    print(f"Calculated class weights: {class_weights}")

    # Step 4: Create and compile the model
    print("Creating and compiling the model...")
    model = create_baseline_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )
    model.summary()

    # Step 5: Train the model
    print("\n--- Starting model training ---")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        class_weight=class_weights # Pass the weights here
    )
    print("--- Model training finished ---\n")

    # Step 6: Save the trained model
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("✅ Model saved successfully!")

if __name__ == '__main__':
    main()