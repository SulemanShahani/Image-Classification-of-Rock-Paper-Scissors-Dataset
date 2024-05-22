# Image Classification Project

## Introduction

This project uses the Rock, Paper, Scissors dataset to train a CNN model to classify images into one of three classes: rock, paper, or scissors. The dataset is loaded from TensorFlow Datasets and preprocessed to extract only the red channel from RGB images for simplicity.
This project is a simple image classification application built using Flask and TensorFlow


## Dataset

The dataset used in this project is the Rock, Paper, Scissors dataset provided by TensorFlow Datasets. It contains images of hands in rock, paper, and scissors poses.

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


Create and activate a virtual environment:


source venv/bin/activate  # On Windows use `venv\Scripts\activate`

## Install dependencies

cd image-classification-project
pip install -r requirements.txt





## Usage

Start the Flask application:

python app.py
Open a web browser and go to <http://localhost:5000>.
Upload an image using the provided form.
Click the "Upload" button to classify the uploaded image.

## Acknowledgments


Inspiration for this project came from [https://github.com/KeithGalli].
