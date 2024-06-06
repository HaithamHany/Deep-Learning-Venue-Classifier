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
import Utils
from Config import config

class SemiSupervisedImageClassifier:
    def __init__(self, preprocessing, label_encoder):
        """
        Initialize the SemiSupervisedImageClassifier class.
        """
        self.preprocessing = preprocessing
        self.label_encoder = label_encoder
        self.trained_model = None

    def load_model(self, model_filename):
        self.trained_model = Utils.load_model(model_filename, config)

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
        Utils.evaluate_classifier(self.trained_model, X_test, y_test, self.preprocessing, self.label_encoder)