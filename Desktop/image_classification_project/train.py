# train.py

from data_loader import load_data, preprocess_data
from model import create_model

# Define dataset directory
data_dir = 'data/'

# Load and preprocess dataset
images, labels = load_data(data_dir)
processed_images, processed_labels = preprocess_data(images, labels)

# Define input shape and number of classes
input_shape = processed_images.shape[1:]
num_classes = len(set(processed_labels))

# Create model
model = create_model(input_shape, num_classes)

# Train model
model.fit(processed_images, processed_labels, epochs=10, batch_size=32, validation_split=0.2)

# Save trained model
model.save('image_classification_model.h5')
