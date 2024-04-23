from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

# Define image preprocessing function
# Define image preprocessing function
def preprocess_image(image):
    # Resize image to (300, 300)
    image = image.resize((300, 300))
    # Convert image to numpy array
    image = np.array(image)
    # Normalize pixel values to range [0, 1]
    image = image / 255.0
    # Check if the image is grayscale
    if len(image.shape) == 2:  # Grayscale image
        # Add channel dimension
        image = np.expand_dims(image, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 1:  # Grayscale image with channel dimension
        pass  # No need to change shape
    elif image.shape[2] == 4:  # RGBA image
        # Convert RGBA to RGB
        image = image[:, :, :3]
    # Ensure image has 3 channels (for RGB images)
    if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale image
        # Repeat grayscale channel to create three channels
        image = np.repeat(image, 3, axis=-1)
    # Print shape after preprocessing
    print("Image shape after preprocessing:", image.shape)
    return image

    # Test the function





# Initialize Flask application
app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model('image_classification_model.h5')

# Define class labels
class_labels = ['rock', 'paper', 'scissors']

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for image upload and classification
@app.route('/upload', methods=['POST'])
def upload():
    print(request.files)  # Print request files to see if 'image' key is present
    # Get image file from request
    file = request.files['image']

    # Open image using PIL
    img = Image.open(file)
   
    # Preprocess image
    img = preprocess_image(img)

    # Make prediction
    prediction = model.predict(np.expand_dims(img, axis=0))

    # Get predicted class label
    predicted_class = class_labels[np.argmax(prediction)]

    # Prepare response
    response = {
        'prediction': predicted_class
    }
    return jsonify(response)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
