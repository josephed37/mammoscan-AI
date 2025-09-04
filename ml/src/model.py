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


def create_transfer_model(input_shape=(224, 224, 3), num_classes=1):
    """
    Builds a transfer learning model using the pre-trained EfficientNetB0 base.

    This function implements the "fine-tuning" strategy:
    1.  Loads the powerful, pre-trained EfficientNetB0 model.
    2.  Freezes the weights of the base model to retain its knowledge.
    3.  Adds a new, custom classification head on top.
    4.  Returns the complete model, ready for training only the new head.

    Args:
        input_shape (tuple): The shape of the input images.
        num_classes (int): The number of output classes (1 for binary classification).

    Returns:
        keras.Model: The complete transfer learning model.
    """
    # --- Step 1: Load the Pre-Trained "Brain" ---
    # We load EfficientNetB0, pre-trained on the massive ImageNet dataset.
    # `include_top=False` means we chop off the original head.
    # `weights='imagenet'` specifies which pre-trained weights to use.
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    # --- Step 2: Freeze the "Brain" ---
    # We lock the pre-trained weights so they don't change during our
    # initial training. We want to leverage their existing knowledge.
    base_model.trainable = False

    # --- Step 3: Add Our Custom "Head" ---
    # We build our own set of decision-making layers.
    # We start with a Global Average Pooling layer to aggregate the features
    # from the base model in a smart way.
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    
    # We add a final Dense layer with a sigmoid activation for our binary decision.
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='sigmoid')

    # --- Step 4: Chain Everything Together ---
    # We use the Keras Functional API to build the final model.
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False) # `training=False` is important for frozen layers
    x = global_average_layer(x)
    outputs = prediction_layer(x)
    
    model = tf.keras.Model(inputs, outputs)

    return model

def create_regularized_transfer_model(input_shape=(224, 224, 3), num_classes=1):
    """
    Builds a transfer learning model with a Dropout layer for regularization.
    """
    # Load a fresh pre-trained EfficientNetB0 base model.
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    # The base is trainable, allowing for fine-tuning.
    base_model.trainable = True

    # Freeze the early layers and only fine-tune the top portion.
    fine_tune_at = len(base_model.layers) // 3 * 2
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # --- Create our new custom "head" with regularization ---
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    
    # Add a Dropout layer to randomly ignore 50% of the neurons during training.
    # This makes the model more robust and less likely to overfit.
    dropout_layer = tf.keras.layers.Dropout(0.5) 
    
    # Final decision-making layer.
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='sigmoid')

    # Chain all the components together using the Keras Functional API.
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = global_average_layer(x)
    x = dropout_layer(x) # Apply dropout before the final prediction.
    outputs = prediction_layer(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model



if __name__ == '__main__':
    print("--- Creating and summarizing the baseline CNN model ---")
    baseline_model = create_baseline_cnn()
    baseline_model.summary()

    print("\n\n--- Creating and summarizing the transfer learning model ---")
    transfer_model = create_transfer_model()
    transfer_model.summary()
    
    print("\n\n--- Creating and summarizing the regularized transfer learning model ---")
    regularized_model = create_regularized_transfer_model()
    regularized_model.summary()