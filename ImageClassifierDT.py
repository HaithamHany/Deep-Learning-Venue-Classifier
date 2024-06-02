import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelSpreading
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import tensorflow as tf
from Spinner import Spinner
from Preprocessing import Preprocessing


class ImageClassifierDT:
    def __init__(self, mode='supervised', data_dir='dataset'):
        self.mode = mode
        self.data_dir = data_dir
        self.spinner = Spinner()
        self.preprocessing = Preprocessing(data_dir)
        self.label_encoder = LabelEncoder()

    def load_and_preprocess_data(self):
        self.spinner.set_msg("Loading Images")
        self.spinner.start()
        images, labels = self.preprocessing.load_images_parallel()
        self.spinner.stop()

        self.spinner.set_msg("Encoding Labels")
        self.spinner.start()
        labels_encoded = self.label_encoder.fit_transform(labels)
        self.spinner.stop()

        self.spinner.set_msg("Display distribution and sample images")
        self.spinner.start()
        self.preprocessing.display_distribution(labels)
        self.preprocessing.display_sample_images(images, labels)
        self.spinner.stop()

        self.spinner.set_msg("Normalizing images")
        self.spinner.start()
        images = images / 255.0
        self.spinner.stop()

        self.spinner.set_msg("Splitting data into training and testing sets")
        self.spinner.start()
        X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
        self.spinner.stop()

        if self.mode == 'semi_supervised':
            self.spinner.set_msg("Augmenting the training data")
            self.spinner.start()
            X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
            y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.int64)
            augmented_dataset = self.preprocessing.augment_dataset(X_train_tensor, y_train_tensor)

            X_train_augmented = []
            y_train_augmented = []
            for batch in augmented_dataset:
                augmented_images, augmented_labels = batch
                X_train_augmented.append(augmented_images.numpy())
                y_train_augmented.append(augmented_labels.numpy())

            X_train_augmented = np.concatenate(X_train_augmented)
            y_train_augmented = np.concatenate(y_train_augmented)
            self.spinner.stop()

            self.spinner.set_msg("Flattening images")
            self.spinner.start()
            X_train_augmented_flat = self.preprocessing.flatten_images(X_train_augmented)
            X_test_flat = self.preprocessing.flatten_images(X_test)
            self.spinner.stop()

            return X_train_augmented_flat, X_test_flat, y_train_augmented, y_test
        else:
            return X_train, X_test, y_train, y_test

    def train_and_evaluate_classifier(self, X_train, y_train, X_test, y_test):
        if self.mode == 'semi_supervised':
            # Split the augmented training data into labeled and unlabeled sets
            n_labeled = int(0.8 * len(y_train))
            indices = np.arange(len(y_train))
            np.random.shuffle(indices)
            labeled_indices = indices[:n_labeled]
            unlabeled_indices = indices[n_labeled:]

            X_train_labeled = X_train[labeled_indices]
            y_train_labeled = y_train[labeled_indices]

            X_train_unlabeled = X_train[unlabeled_indices]
            y_train_unlabeled = -1 * np.ones(len(unlabeled_indices), dtype=int)  # Use -1 for unlabeled data

            # Combine labeled and unlabeled data
            X_train_combined = np.concatenate((X_train_labeled, X_train_unlabeled), axis=0)
            y_train_combined = np.concatenate((y_train_labeled, y_train_unlabeled), axis=0)

            # Apply LabelSpreading for semi-supervised learning
            label_spreading = LabelSpreading(kernel='knn', alpha=0.8)
            label_spreading.fit(X_train_combined, y_train_combined)

            # Predict the labels for the unlabeled data
            y_train_unlabeled_pred = label_spreading.transduction_[len(labeled_indices):]

            # Combine the newly labeled data with the original labeled data
            X_train_final = np.concatenate((X_train_labeled, X_train_unlabeled), axis=0)
            y_train_final = np.concatenate((y_train_labeled, y_train_unlabeled_pred), axis=0)
        else:
            # For supervised learning, use the data as is
            X_train_final, y_train_final = X_train, y_train

        # Train a Decision Tree Classifier
        clf = DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=5, criterion='entropy', splitter='best', random_state=42)
        clf.fit(X_train_final, y_train_final)
        model_filename = 'venue_classifier_decision_tree.joblib'
        joblib.dump(clf, model_filename)
        print(f"Model saved to {model_filename}")

        # Make predictions and evaluate
        y_pred = clf.predict(X_test)

        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)

        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        print("Confusion Matrix:")
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()


# Main execution
if __name__ == "__main__":
    mode = 'semi_supervised'  # or 'supervised'
    classifier = ImageClassifierDT(mode=mode)

    X_train, X_test, y_train, y_test = classifier.load_and_preprocess_data()
    classifier.train_and_evaluate_classifier(X_train, y_train, X_test, y_test)
