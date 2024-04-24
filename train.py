# train.py

import tensorflow_datasets as tfds
from data_loader import preprocess_data
from model import create_model

# Load dataset
builder = tfds.builder('rock_paper_scissors')
builder.download_and_prepare()
dataset = builder.as_dataset()

# Convert dataset to NumPy arrays
images = []
labels = []
for example in dataset['train']:
    images.append(example['image'])
    labels.append(example['label'])

# Preprocess data
processed_images, processed_labels = preprocess_data(images, labels)

# Define input shape and number of classes
input_shape = processed_images.shape[1:]
num_classes = len(set(processed_labels))

# Create model
model = create_model(input_shape, num_classes)

# Train model
model.fit(processed_images, processed_labels, epochs=3, batch_size=32, validation_split=0.2)

# Save trained model
model.save('image_classification_model.h5')
