# Feature Extraction for Image Datasets

This repository provides a feature extraction pipeline for multiple popular image datasets, including MNIST, Fashion MNIST, MedMNIST, CIFAR-10, and the Street View House Number (SVHN) dataset. The pipeline extracts curvature and shape features from the images and adds Gaussian noise to the features. The L2 distance between the original and noisy features is also calculated.

## Supported Datasets
- **MNIST**: Handwritten digits (28x28 grayscale)
- **Fashion MNIST**: Fashion items (28x28 grayscale)
- **MedMNIST**: Various medical image datasets (RGB or grayscale)
- **CIFAR-10**: 32x32 RGB images of 10 different classes
- **SVHN**: Street View House Number dataset (RGB)

## Features Extracted
- **Curvatures**: Curvature features are extracted from the contours of the images.
- **Shapes**: Shape features such as area, perimeter, and aspect ratio are extracted from the contours.

USAGE:
python fe_dataset.py <dataset_name>
python noise.py <dataset_name>

Replace <dataset_name> with one of the supported datasets:

mnist
fashion_mnist
pathmnist (or any other MedMNIST dataset)
cifar10
svhn


File Output:
The extracted curvature and shape features will be saved as .npy files:
<dataset_name>_curvatures.npy
<dataset_name>_shapes.npy


Gaussian Noise and L2 Distance:
Gaussian noise is added to the features, and the L2 distance between the original and noisy features is calculated and printed.