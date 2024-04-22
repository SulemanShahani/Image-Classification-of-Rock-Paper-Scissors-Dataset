# data_loader.py

import tensorflow_datasets as tfds
import numpy as np

def load_data():
    """
    Load Rock Paper Scissors dataset from TensorFlow Datasets.
    Returns:
        dataset: TensorFlow dataset containing images and labels.
    """
    dataset, info = tfds.load('rock_paper_scissors', with_info=True, as_supervised=True)
    return dataset

def preprocess_data(dataset):
    """
    Preprocess images and labels.
    Args:
        dataset: TensorFlow dataset containing images and labels.
    Returns:
        processed_images: Preprocessed images as NumPy array.
        processed_labels: Preprocessed labels as NumPy array.
    """
    images = []
    labels = []
    for image, label in dataset:
        # Resize images to a fixed size (e.g., 100x100)
        image = tf.image.resize(image, (100, 100))
        # Normalize pixel values
        image = image / 255.0
        images.append(image.numpy())
        labels.append(label.numpy())
    
    processed_images = np.array(images)
    processed_labels = np.array(labels)
    
    return processed_images, processed_labels

