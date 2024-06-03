import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from Preprocessing import Preprocessing
from SupervisedImageClassifier import SupervisedImageClassifier
from SemiSupervisedImageClassifier import SemiSupervisedImageClassifier
from Spinner import Spinner
from Config import config

# Configuration
data_dir = 'dataset'
img_height, img_width = 128, 128


def get_user_choice():
    while True:
        print()  # Add an empty print statement for a blank line
        print("Select the mode for classification:")
        print("1. Supervised learning")
        print("2. Semi-supervised learning")
        choice = input("Enter 1 or 2: ")
        if choice == '1' or choice == '2':
            return choice
        else:
            print("Invalid choice. Please enter 1 or 2.")


def get_model_choice():
    while True:
        print()  # Add an empty print statement for a blank line
        print("Select the action:")
        print("1. Train a new model")
        print("2. Load a previously trained model")
        choice = input("Enter 1 or 2: ")
        if choice == '1' or choice == '2':
            return choice
        else:
            print("Invalid choice. Please enter 1 or 2.")


# Main execution
if __name__ == "__main__":

    spinner = Spinner()
    spinner.set_msg("Loading Images")
    spinner.start()

    # Preprocessing
    preprocessing = Preprocessing(data_dir, img_height, img_width)
    images, labels = preprocessing.load_images_parallel()
    spinner.stop()

    spinner.set_msg("Encoding Labels")
    spinner.start()
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    spinner.stop()

    spinner.set_msg("Display distribution and sample images")
    spinner.start()
    preprocessing.display_distribution(labels)
    preprocessing.display_sample_images(images, labels)
    spinner.stop()

    spinner.set_msg("Normalizing images")
    spinner.start()
    images = images / 255.0
    spinner.stop()

    spinner.set_msg("Splitting data into training and testing sets")
    spinner.start()
    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
    spinner.stop()

    # User choice for classifier mode
    print()
    mode = get_user_choice()

    # Classifier selection and training
    if mode == '1':
        print("Selected mode: Supervised learning")
        classifier = SupervisedImageClassifier(preprocessing, label_encoder)
    else:
        print("Selected mode: Semi-supervised learning")
        classifier = SemiSupervisedImageClassifier(preprocessing, label_encoder)


    # User choice for model training or loading
    print()
    model_choice = get_model_choice()

    if model_choice == '1':
        print("Selected action: Train a new model")
    else:
        print("Selected action: Load a previously trained model")
        model_filename = input("Enter the filename of the trained model to load: ")
        classifier.load_model(model_filename)

    spinner.set_msg("Training and evaluating classifier")
    spinner.start()
    classifier.train_and_evaluate_classifier(X_train, y_train, X_test, y_test, config)
    spinner.stop()
