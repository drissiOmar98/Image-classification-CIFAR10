# CIFAR-10 Image Classification with Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.12+](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/Numpy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white)](https://matplotlib.org/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/cifar10-classification/blob/main/CIFAR10_Classification.ipynb)

A deep learning project comparing ANN and CNN architectures for classifying 32×32 color images across 10 categories in the CIFAR-10 dataset.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Installation](#installation)


## Overview
This project implements and compares two approaches for image classification:

- **Artificial Neural Network (ANN)**: Baseline fully-connected model
- **Convolutional Neural Network (CNN)**: Advanced architecture featuring:
  - Convolutional layers with ReLU activation
  - Batch normalization
  - Dropout regularization (30-50%)
  - Data augmentation pipeline
  - L2 weight regularization

The CIFAR-10 dataset contains 50,000 training and 10,000 test images across 10 object categories.

## Key Features
✔️ Comprehensive ANN vs CNN performance comparison  
✔️ Advanced data augmentation pipeline  
✔️ Real-time training visualization  
✔️ Detailed classification metrics (precision/recall/F1)  
✔️ Confusion matrix analysis  
✔️ Regularization suite (Dropout, BatchNorm, L2)  
✔️ Early stopping with model checkpointing  

## Technologies Used

### Core Frameworks
| Technology | Purpose | Version |
|------------|---------|---------|
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white) | Deep learning framework | 2.12+ |
| ![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white) | High-level API | Built-in |
| ![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white) | Model evaluation | 1.2+ |

### Data Processing
| Technology | Purpose |
|------------|---------|
| ![NumPy](https://img.shields.io/badge/Numpy-013243?logo=numpy&logoColor=white) | Array operations |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) | Data analysis |

### Visualization
| Technology | Purpose |
|------------|---------|
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white) | Training plots |
| ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?logo=python&logoColor=white) | Statistical visuals |

### Model Optimization
| Technology | Purpose |
|------------|---------|
| ![ImageDataGenerator](https://img.shields.io/badge/Keras_Augmentation-5C3EE8) | Real-time augmentation |
| ![EarlyStopping](https://img.shields.io/badge/Keras_Callbacks-D00000) | Prevent overfitting |

## Results

### Performance Comparison
| Model | Train Accuracy | Test Accuracy | Parameters|
|-------|----------------|---------------|------------|
| ANN   | 49.4%          | 46.2%         | 820,874 |
| CNN   | **73.2%**      | **72.4%**     |  1,253,674 |



## Installation
```bash
# Clone repository
git clone https://github.com/drissiOmar98/Image-classification-CIFAR10.git
cd Image-classification-CIFAR10

