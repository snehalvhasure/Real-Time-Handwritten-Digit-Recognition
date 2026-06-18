# Real-Time-Handwritten-Digit-Recognition
A real-time handwritten digit recognition system using a live camera feed. The model was trained on MNIST and integrated with OpenCV for real-time inference.
# MNIST Digit Recognition using Artificial Neural Network (ANN)

## Project Overview

This project implements a Handwritten Digit Recognition System using an Artificial Neural Network (ANN) built with TensorFlow and Keras.

The model is trained on the MNIST dataset and can recognize digits (0–9) from a live webcam feed. The webcam image is processed in real-time, converted to the MNIST format, and passed through the trained neural network for prediction.
## Features

* Train an ANN model using the MNIST dataset
* Multi-layer neural network architecture
* Real-time digit recognition using webcam
* Confidence score display
* Training and validation accuracy visualization
* Model evaluation on test dataset
* Live prediction overlay on camera feed

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
  
## Dataset

The project uses the MNIST Handwritten Digit Dataset.
Dataset Details:

* 60,000 training images
* 10,000 testing images
* Grayscale images
* Image size: 28 × 28 pixels
* Classes: Digits 0–9

## Neural Network Architecture

Input Layer:
* 28 × 28 image

Hidden Layers:

* Dense(128, ReLU)
* Dense(256, ReLU)
* Dense(512, ReLU)
* Dense(256, ReLU)
* Dense(128, ReLU)

Output Layer:
* Dense(10, Softmax)

## Workflow

1. Load MNIST dataset
2. Normalize image values
3. Build ANN model
4. Train model
5. Evaluate performance
6. Plot training accuracy
7. Open webcam feed
8. Capture frames
9. Preprocess image
10. Predict digit
11. Display prediction and confidence score

## Installation
Clone the repository:
git clone https://github.com/yourusername/mnist-digit-recognition.git
Navigate to project directory:
cd mnist-digit-recognition
Install dependencies:
pip install tensorflow opencv-python numpy matplotlib

## Run the Project
DL_project.py

## Sample Output
Test Accuracy: 98%+
Predicted Digit: 5
Confidence: 0.99

## Learning Outcomes

Through this project I learned:

* Artificial Neural Networks (ANN)
* Deep Learning fundamentals
* TensorFlow and Keras
* Image preprocessing
* Computer Vision using OpenCV
* Model evaluation techniques
* Real-time prediction systems

## Limitations Identified

During implementation several limitations were observed:

- ANN loses spatial information after flattening the image.
- Pixel relationships are not preserved.
- Real-world webcam images differ significantly from MNIST images.
- Performance decreases when background noise is present.
- ANN is not ideal for advanced computer vision applications.
These observations motivated the development of a CNN-based version of the project.

## Author
Snehal Vhasure | 
Full Stack Developer | AI Enthusiast | GenAI and AgenticAI Learner
