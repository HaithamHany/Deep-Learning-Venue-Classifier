# Venue Classification using Decision Trees

This project explores the effectiveness of supervised and semi-supervised learning approaches for venue classification using decision tree models. The models are evaluated on the "Indoor Scene Recognition" dataset from MIT for venue classification. Hyperparameter tuning is performed using Random Search, and the robustness of the models is assessed using 5-fold cross-validation.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [References](#references)

## Overview

Automatic venue classification from visual data has emerged as a critical research area with extensive applications. Intelligent systems capable of identifying locations depicted in images can enhance user experience across diverse domains, including navigation and social media.

This project implements and evaluates supervised and semi-supervised learning approaches for venue classification, focusing on the following strategies:
- Hyperparameter optimization using Random Search
- Cross-validation for robust evaluation
- Model persistence for reproducibility

## Dataset

The dataset used is the MIT Indoor Scenes dataset, which contains images of 67 different indoor categories. For this project, a subset of 5 categories (Bedroom, Grocery, Office, Restaurant, and Station) is used.

## Project Structure

The project includes the following files:

- `main.py`: Main script to run the project.
- `Config.py`: Configuration file for hyperparameters.
- `Preprocessing.py`: Contains functions for data preprocessing.
- `SupervisedImageClassifier.py`: Implements the supervised learning model.
- `SemiSupervisedImageClassifier.py`: Implements the semi-supervised learning model.
- `Spinner.py`: Utility class for displaying progress.
- `Utils.py`: Additional utility functions.

## Setup Instructions

1. **Clone the Repository**:
   https://github.com/HaithamHany/DeepLearningVenueClassifier.git

2. **Download dataset**:
Download the MIT Indoor Scenes dataset and place it in the dataset directory. [Link]

## Hyperparameter Tuning
Hyperparameters are defined in the Config.py file. Modify the values as needed to optimize the model performance.
