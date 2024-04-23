# data_loader.py

import os
import numpy as np
import cv2

def load_data(data_dir):
    """
    Load images from the specified directory.
    Args:
        data_dir: Directory containing images.
    Returns:
        images: List of loaded images.
        labels: List of corresponding labels.
    """
    images = []
    labels = []
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(label)
    
    return images, labels

def preprocess_data(images, labels):
    """
    Preprocess images and labels.
    Args:
        images: List of images.
        labels: List of corresponding labels.
    Returns:
        processed_images: Preprocessed images as NumPy array.
        processed_labels: Preprocessed labels as NumPy array.
    """
    # Convert images to NumPy array and normalize pixel values
    processed_images = np.array(images, dtype=np.float32) / 255.0
    
    # Convert labels to NumPy array
    processed_labels = np.array(labels)
    
    return processed_images, processed_labels
