import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from Config import config


class SupervisedImageClassifier:
    def __init__(self, preprocessing, label_encoder):
        """
        Initialize the SupervisedImageClassifier class.
        """
        self.preprocessing = preprocessing
        self.label_encoder = label_encoder
        self.trained_model = None

    def load_model(self, model_filename):
        try:
            self.trained_model = joblib.load(model_filename)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading the model: {e}")

    def train_classifier(self, X_train, y_train, config):
        """
        Train a Decision Tree Classifier with specified hyperparameters.
        Perform a randomized search over the provided hyperparameters.
        """
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

        # Flatten images
        X_train_augmented_flat = self.preprocessing.flatten_images(X_train_augmented)

        # Train the classifier
        param_distributions = {
            'max_depth': config.get("max_depth"),
            'min_samples_split': config.get("min_samples_split"),
            'min_samples_leaf': config.get("min_samples_leaf"),
        }

        clf = DecisionTreeClassifier(random_state=config.get("random_state"))

        randomized_search = RandomizedSearchCV(
            estimator=clf,
            param_distributions=param_distributions,
            n_iter=config.get("n_iter", 5),
            scoring='accuracy',
            n_jobs=config.get("n_jobs", -1),
            cv=config.get("cv", 3),
            verbose=1,
            random_state=config.get("random_state")
        )

        randomized_search.fit(X_train_augmented_flat, y_train_augmented)

        # Save the best trained model
        self.trained_model = randomized_search.best_estimator_
        joblib.dump(self.trained_model, 'supervised_model.pkl')

        best_params = randomized_search.best_params_
        best_score = randomized_search.best_score_

        print(f"Best Hyperparameters: {best_params}")
        print(f"Best Cross-Validation Accuracy: {best_score}")

    def evaluate_classifier(self, X_test, y_test):
        """
        Evaluate the trained classifier on the test set.
        """
        if self.trained_model is None:
            raise ValueError("No trained model found. Please train the model before evaluation.")

        X_test_flat = self.preprocessing.flatten_images(X_test)

        y_pred = self.trained_model.predict(X_test_flat)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_, zero_division=0))

        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
