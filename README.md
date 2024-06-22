# Venue Classification using Decision Trees

Venue Classification using Decision Trees and CNN
This project explores the effectiveness of supervised and semi-supervised learning approaches for venue classification using decision tree models and a convolutional neural network (CNN). The models are evaluated on the "Indoor Scene Recognition" dataset from MIT for venue classification. Hyperparameter tuning is performed using Random Search, and the models are assessed using cross-validation.


## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [CNN Model Architecture](#cnn-model-architecture)
  
## Overview
Automatic venue classification from visual data has emerged as a critical research area with extensive applications. Intelligent systems capable of identifying locations depicted in images can enhance user experience across diverse domains, including navigation and social media.

This project implements and evaluates supervised and semi-supervised learning approaches for venue classification, utilizing decision tree models and a CNN. Key strategies include:

- Hyperparameter optimization using Random Search
- Cross-validation for robust evaluation
- Model persistence for reproducibility

## Dataset
The dataset used is the MIT Indoor Scenes dataset, which contains images of 67 different indoor categories. For this project, a subset of 5 categories (Bedroom, Grocery, Office, Restaurant, and Station) is used.

## Project Structure
The project includes the following files:

- main.py: Main script to run the project.
- Config.py: Configuration file for hyperparameters.
- Preprocessing.py: Contains functions for data preprocessing.
- SupervisedImageClassifier.py: Implements the supervised learning model.
- SemiSupervisedImageClassifier.py: Implements the semi-supervised learning model.
- Spinner.py: Utility class for displaying progress.
- Utils.py: Additional utility functions.
- CNN.py: Implementation of the CNN model for image classification.
- main_cnn.py: Script for running the CNN model.
  
## Setup Instructions

Clone the Repository:
git clone [[Link](https://github.com/HaithamHany/DeepLearningVenueClassifier.git)]

Download dataset:
Download the MIT Indoor Scenes dataset and place it in the dataset directory. [[Link](https://web.mit.edu/torralba/www/indoor.html)]

## Hyperparameter Tuning
Hyperparameters are defined in the Config.py file. Modify the values as needed to optimize the model performance.

## CNN Model Architecture
The CNN model architecture is defined in CNN.py using PyTorch. It consists of convolutional layers followed by fully connected layers for classification. Hyperparameters such as learning rate, batch size, and number of epochs can be adjusted for optimal performance.


By integrating the CNN model alongside decision trees, this project provides a comprehensive approach to venue classification using both traditional machine learning and deep learning techniques.
