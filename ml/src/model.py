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
    # Initialize a sequential model, which lets us build the network layer by layer.
    model = models.Sequential()

    # Define the input layer. This tells the model what shape of image to expect.
    model.add(layers.Input(shape=input_shape))

    # --- First Feature Extraction Block (First Detective Team) ---
    # A convolutional layer scans the image with 32 filters to find basic patterns (edges, corners).
    # 'relu' activation introduces non-linearity, allowing the model to learn complex patterns.
    # 'padding="same"' ensures we don't lose information at the edges of the image.
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    # A max pooling layer summarizes the findings from the convolutional layer,
    # keeping the most important information and reducing the image's size.
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Second Feature Extraction Block (Second Detective Team) ---
    # This block is more experienced (64 filters) and looks for more complex patterns
    # by analyzing the features found by the first block.
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Flattening Layer ---
    # This layer takes the 2D feature maps from the extraction blocks and "flattens"
    # them into a single, long vector. This prepares the data for the final decision-making layers.
    model.add(layers.Flatten())

    # --- Decision-Making Block (The Chief Detective) ---
    # A fully connected 'Dense' layer analyzes the flattened list of features to find high-level patterns.
    model.add(layers.Dense(128, activation='relu'))
    # The final output layer has a single neuron that makes the final binary decision.
    # The 'sigmoid' activation function squashes the output to be a number between 0 and 1,
    # representing the probability of the image belonging to the positive class (Cancer).
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

if __name__ == '__main__':
    # This block of code runs only when you execute this script directly
    # (e.g., `python ml/src/model.py`). It's a great way to do a quick
    # self-test of the model architecture.
    print("--- Creating and summarizing the baseline CNN model ---")
    baseline_model = create_baseline_cnn()
    # .summary() provides a clean, table-like overview of the model's layers and parameters.
    baseline_model.summary()