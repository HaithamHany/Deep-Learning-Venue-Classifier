import joblib
import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class SemiSupervisedImageClassifier:
    def __init__(self, preprocessing, label_encoder):
        """
        Initialize the SemiSupervisedImageClassifier class.
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
        Train a semi-supervised classifier using SelfTrainingClassifier with randomized search for hyperparameter optimization.
        """
        # Split data into labeled and unlabeled
        X_train_semi, y_train_semi = self.preprocessing.split_labeled_unlabeled(X_train, y_train, labeled_ratio=0.8)

        # Flatten images
        X_train_flat = self.preprocessing.flatten_images(X_train_semi)

        # Define parameter distributions for RandomizedSearchCV
        param_distributions = {
            'base_estimator__max_depth': config.get("max_depth"),
            'base_estimator__min_samples_split': config.get("min_samples_split"),
            'base_estimator__min_samples_leaf': config.get("min_samples_leaf"),
        }

        # Base classifier
        base_clf = DecisionTreeClassifier(random_state=config.get("random_state"))

        # Self-training classifier
        self_training_clf = SelfTrainingClassifier(base_clf)

        # Perform randomized search with parallelization
        randomized_search = RandomizedSearchCV(
            estimator=self_training_clf,
            param_distributions=param_distributions,
            n_iter=config.get("n_iter", 5),
            scoring='accuracy',
            n_jobs=config.get("n_jobs", -1),
            cv=config.get("cv", 3),
            verbose=1,
            random_state=config.get("random_state")
        )

        randomized_search.fit(X_train_flat, y_train_semi)

        # Save the best trained model
        self.trained_model = randomized_search.best_estimator_
        joblib.dump(self.trained_model, 'semi_supervised_model.pkl')

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
