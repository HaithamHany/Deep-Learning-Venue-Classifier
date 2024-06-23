# Venue Classification Using Decision Trees and CNN

This project explores the effectiveness of supervised and semi-supervised learning approaches for venue classification using decision tree models and a convolutional neural network (CNN). The models are evaluated on the "Indoor Scene Recognition" dataset from MIT for venue classification. Hyperparameter tuning is performed using Random Search, and the models are assessed using cross-validation.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup Requirements](#setup-requirements)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [CNN Model Architecture](#cnn-model-architecture)
- [Training and Validation](#training-and-validation)
- [Running the Pre-trained Model](#running-the-pre-trained-model)
- [Source Code](#source-code)

## Overview
Automatic venue classification from visual data has emerged as a critical research area with extensive applications. Intelligent systems capable of identifying locations depicted in images can enhance user experience across diverse domains, including navigation and social media.

This project implements and evaluates supervised and semi-supervised learning approaches for venue classification, utilizing decision tree models and a CNN. Key strategies include:
- Hyperparameter optimization using Random Search
- Cross-validation for robust evaluation
- Model persistence for reproducibility

## Dataset
The dataset used is the MIT Indoor Scenes dataset, which contains images of 67 different indoor categories (link: [Original Dataset](https://web.mit.edu/torralba/www/indoor.html)). For this project, a subset of 5 categories (Bedroom, Grocery, Office, Restaurant, and Station) is used.

The subset dataset can be found in the GitHub Repository: [Dataset](dataset/)

A smaller test dataset with 5 images, one from each class can also be found in the GitHub Repository: [Test Dataset](dataset/Test)

## Project Structure
The project includes the following files:
- `main.py`: Main script to run the project.
- `Config.py`: Configuration file for hyperparameters.
- `Preprocessing.py`: Contains functions for data preprocessing.
- `SupervisedImageClassifier.py`: Implements the supervised learning model.
- `SemiSupervisedImageClassifier.py`: Implements the semi-supervised learning model.
- `Spinner.py`: Utility class for displaying progress.
- `Utils.py`: Additional utility functions.
- `CNN.py`: Implementation of the CNN model for image classification.
- `main_cnn.py`: Script for running the CNN model.
- `main_decision_tree.py`: Script for running the decision tree models.
- `requirements.txt`: Contains all the required libraries for running the program

## Setup Requirements

### Clone the Repository:

git clone https://github.com/HaithamHany/DeepLearningVenueClassifier.git

Download dataset:
Download the MIT Indoor Scenes dataset and place it in the dataset directory (Only if not already downloaded through the clone of the repository). [Dataset](dataset/)

Install Dependencies:
Ensure you have Python 3.x installed. Install the required libraries using:

pip install -r requirements.txt

## Hyperparameter Tuning
Hyperparameters are defined in the Config.py file. Modify the values as needed to optimize the model performance.

## CNN Model Architecture
The CNN model architecture is defined in CNN.py using PyTorch. It consists of convolutional layers followed by fully connected layers for classification. Hyperparameters such as learning rate, batch size, and number of epochs can be adjusted for optimal performance.

## Training and Validation
Training the Decision Tree Model/CNN Model:

run: python Main.py
You will be prompted to select the model (CNN or Decision Tree) and the mode (supervised or semi-supervised). Follow the on-screen instructions to proceed with training or loading a pre-trained model.

The script will handle loading the dataset, splitting it into training, validation, and testing sets, and training the Decision Tree/CNN model.

## Running the Pre-trained Model
If you have a pre-trained model and want to evaluate it on the test dataset, ensure the model file is in the correct directory and run:

python Main.py
Select the option to load a previously trained model when prompted.

The script will detect the pre-trained model and prompt you to load it for evaluation on a test dataset.

## Source Code
The source code is provided in the repository with the following key files:

- main.py: Main entry point for the program.
- main_decision_tree.py: Main entry point for the Decision Tree Models.
- main_cnn.py: Main entry point for the CNN model.
- CNN.py: Implementation of the CNN model.
- Config.py: Configuration file for hyperparameters.
- Preprocessing.py: Data preprocessing functions.
- SupervisedImageClassifier.py: Supervised learning implementation.
- SemiSupervisedImageClassifier.py: Semi-supervised learning implementation.
- Spinner.py: Utility class for progress display.
- Utils.py: Additional utility functions.



By integrating the CNN model alongside decision trees, this project provides a comprehensive approach to venue classification using both traditional machine learning and deep learning techniques.
