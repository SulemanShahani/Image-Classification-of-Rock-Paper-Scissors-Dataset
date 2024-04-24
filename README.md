# Image Classification Project

This project is a simple image classification application built using Flask and TensorFlow. It allows users to upload images of rock, paper, or scissors, and the application predicts the class of the uploaded image using a pre-trained deep learning model.

## Features

- Upload images of rock, paper, or scissors.
- Classify uploaded images into one of the three classes.
- Display the predicted class label.

## Requirements

- Python 3.x
- Flask
- TensorFlow
- PIL (Python Imaging Library)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SulemanShahani/Image-Classification-of-Rock-Paper-Scissors-Dataset

## Install dependencies

cd image-classification-project
pip install -r requirements.txt

## Install TensorFlow

Since TensorFlow files are large and not included in this repository, you will need to install TensorFlow separately. Please refer to the official TensorFlow installation guide for instructions on how to install it for your system: TensorFlow Installation Guide

## Download Trained Model

You can download the pre-trained model file from Google Drive link [https://drive.google.com/file/d/1peTHWuyQSe-Rr2rEQBlm8B9p2DHy2f1E/view?usp=drive_link]. Once downloaded, place the model file in the root directory of the project.

## Usage

Start the Flask application:

python app.py
Open a web browser and go to <http://localhost:5000>.
Upload an image using the provided form.
Click the "Upload" button to classify the uploaded image.

## Acknowledgments

The pre-trained image classification model used in this project is based on TensorFlow's official tutorials.
Inspiration for this project came from [https://github.com/KeithGalli].
