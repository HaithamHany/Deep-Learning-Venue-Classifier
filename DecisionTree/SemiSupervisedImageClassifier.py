import joblib
import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from . import Utils
import time
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

    def pseudo_label(self, model, X_unlabeled, confidence_threshold):
        """
        Pseudo-label the unlabeled data based on model predictions and confidence threshold.
        """
        predicts = model.predict_proba(X_unlabeled)
        max_conf = np.max(predicts, axis=1)
        pseudo_labels = np.argmax(predicts, axis=1)
        confident_indices = np.where(max_conf >= confidence_threshold)[0]
        return X_unlabeled[confident_indices], pseudo_labels[confident_indices]

    def train_classifier(self, X_train, y_train, X_val, y_val):
        """
        Train a semi-supervised classifier using SelfTrainingClassifier with randomized search for hyperparameter optimization.
        """
        # Split data into labeled and unlabeled
        X_train_semi, y_train_semi, X_unlabeled, _ = self.preprocessing.split_labeled_unlabeled(X_train, y_train,
                                                                                                labeled_ratio=0.8)

        # Create an array for y_train with unlabeled samples marked as -1
        y_train_semi_with_unlabeled = np.concatenate((y_train_semi, -1 * np.ones(len(X_unlabeled), dtype=int)))

        # Combine labeled and unlabeled data for fitting
        X_combined_train = np.concatenate((X_train_semi, X_unlabeled))

        # Flatten images
        X_combined_flat = self.preprocessing.flatten_images(X_combined_train)
        X_val_flat = self.preprocessing.flatten_images(X_val)

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

        randomized_search.fit(X_combined_flat, y_train_semi_with_unlabeled)

        best_params = randomized_search.best_params_
        best_score = randomized_search.best_score_

        print(f"Best Hyperparameters: {best_params}")
        print(f"Best Cross-Validation Accuracy: {best_score}")

        # Validate on validation set
        val_score = randomized_search.score(X_val_flat, y_val)
        print(f"Validation Accuracy: {val_score}")

        # Pseudo-label the unlabeled data
        X_unlabeled_flat = self.preprocessing.flatten_images(X_unlabeled)
        X_pseudo, y_pseudo = self.pseudo_label(randomized_search.best_estimator_, X_unlabeled_flat, 0.97)

        # Ensure X_pseudo has the same number of dimensions as X_train_semi
        if X_pseudo.ndim == 2:  # Flattened images need to be reshaped to match
            X_pseudo = X_pseudo.reshape((-1,) + X_train_semi.shape[1:])

        # Combine labeled and pseudo-labeled data
        X_combined_final = np.concatenate((X_train_semi, X_pseudo))
        y_combined_final = np.concatenate((y_train_semi, y_pseudo))

        # Flatten the final combined data
        X_combined_final_flat = self.preprocessing.flatten_images(X_combined_final)

        # Train final model on combined data
        self_training_clf.fit(X_combined_final_flat, y_combined_final)

        # Save the trained model
        self.trained_model = self_training_clf
        joblib.dump(self.trained_model, 'semi_supervised_model.pkl')
        print("Model saved successfully.")

    def evaluate_classifier(self, X_test, y_test):
        """
        Evaluate the trained classifier on the test set.
        """
        Utils.evaluate_classifier(self.trained_model, X_test, y_test, self.preprocessing, self.label_encoder)