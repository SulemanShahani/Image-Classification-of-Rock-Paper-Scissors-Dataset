# model.py

import tensorflow as tf
from tensorflow.keras import layers

def create_model(input_shape, num_classes):
    """
    Create an image classification model.
    Args:
        input_shape: Shape of input images (e.g., (height, width, channels)).
        num_classes: Number of classes for classification.
    Returns:
        model: Compiled Keras model.
    """
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
