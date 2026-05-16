# Brain Tumor Detection using CNN

This is a Deep Learning project that classifies brain MRI images into different tumor categories using a Convolutional Neural Network.

## Project Overview

The goal of this project is to build a CNN-based image classification model for brain tumor detection from MRI images.

The model classifies MRI images into four categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

This project is created for learning purposes and is not intended for real medical diagnosis.

## Dataset

The project uses the Brain Tumor MRI Dataset available on Kaggle.

Dataset classes:

- glioma
- meningioma
- notumor
- pituitary

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn

## Model Architecture

The CNN model contains:

- Convolutional layers
- MaxPooling layers
- Flatten layer
- Dense layers
- Dropout layer
- Softmax output layer

## Workflow

1. Load MRI images from dataset folders
2. Resize images
3. Normalize pixel values
4. Split data into training and testing sets
5. Train CNN model
6. Evaluate model accuracy
7. Predict tumor class for sample images

## Results

The model is trained to classify MRI images into four classes. Accuracy may vary depending on dataset size, preprocessing, and training settings.

## How to Run

1. Clone the repository

```bash
git clone https://github.com/your-username/Brain-Tumor-Detection-Using-CNN.git
