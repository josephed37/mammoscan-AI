"""
This script defines the architecture for our machine learning models.
Each function in this file will build and return a Keras model,
allowing us to keep our model definitions separate, clean, and reusable.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def create_baseline_cnn(input_shape=(224, 224, 3)):
    """
    Builds and returns a simple, baseline Convolutional Neural Network (CNN).

    This model serves as our starting point. It's a standard architecture
    designed to extract features from images and make a binary classification.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).

    Returns:
        keras.Model: The compiled Keras model.
    """
    model = models.Sequential(name="baseline_cnn")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def build_full_model(input_shape=(224, 224, 3)):
    """
    Builds the complete model by attaching on-the-fly data augmentation
    layers to the front of our baseline CNN. This is the professional
    solution to combat overfitting when working with smaller datasets.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).

    Returns:
        keras.Model: The final, complete Keras model with augmentation.
    """
    # --- The "Kaleidoscope": On-the-fly Data Augmentation ---
    # These layers will randomly alter the training images as they are fed to the model.
    # This creates a virtually infinite training set and forces the model
    # to learn the true underlying patterns of the data, not just memorize the samples.
    data_augmentation = models.Sequential(
        [
            # Randomly flip the image horizontally.
            layers.RandomFlip("horizontal", input_shape=input_shape),
            # Randomly rotate the image by a small amount (up to 10% of a full circle).
            layers.RandomRotation(0.1),
            # Randomly zoom in or out on the image by up to 10%.
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    # Get our baseline CNN "engine".
    baseline_model = create_baseline_cnn(input_shape)

    # --- Chain Everything Together ---
    # We build the final model by connecting the pieces in order:
    # Input -> Augmentation Layers -> Baseline CNN -> Output
    full_model = models.Sequential([
        data_augmentation,
        baseline_model
    ])
    
    return full_model

if __name__ == '__main__':
    print("--- Creating and summarizing the full model with augmentation ---")
    final_model = build_full_model()
    final_model.summary()