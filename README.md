# Chest-X-ray-Classification-using-CNN-and-VGG16-Detecting-COVID-19-Pneumonia-and-Tuberculosis
This repository contains a deep learning project that classifies chest X-ray images into four categories: Normal, Pneumonia, COVID-19, and Tuberculosis.

**Project Overview**

This project implements convolutional neural networks (CNNs) to analyze chest X-ray images and detect respiratory diseases. The model is trained to classify images into four categories, helping to automate the preliminary screening process.

****Dataset****

The project uses the "chest-xray-pneumoniacovid19tuberculosis" dataset from Kaggle, which contains X-ray images divided into:


- COVID-19
- Normal
- Pneumonia
- Tuberculosis

**Implementation Details**

The project implements two main approaches:

**1.Custom VGG16-like Architecture:** A CNN built from scratch following the VGG16 architecture pattern with:

- Multiple convolutional blocks with increasing filter sizes
- Max pooling layers
- Dense layers with dropout for classification


**2.Transfer Learning with Pre-trained VGG16:**

- Using ImageNet pre-trained weights
- Two variants:

    - Feature extraction (frozen base model)
    - Fine-tuning (unfreezing later layers for additional training)

**Key Features**

- Data augmentation to improve model generalization
- Image preprocessing and normalization
- Model training with learning rate scheduling
- Comprehensive evaluation metrics
- Confusion matrix visualization for analyzing model performance
- Performance comparison between different approaches

**Technical Components**

- **Framework:** TensorFlow/Keras
- **Pre-trained Model:** VGG16
- **Image Size:** 224 × 224 pixels
- **Batch Size:** 32
- **Evaluation Metrics:** Accuracy, confusion matrix

**Results**
The models are evaluated based on their accuracy on a separate test dataset, with confusion matrices showing the classification performance for each disease category.

**Requirements**

- TensorFlow 
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Kaggle API access (for dataset download)
