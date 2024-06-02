import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
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
from sklearn.semi_supervised import LabelSpreading
import joblib
from Spinner import Spinner
from Config import config
from Preprocessing import Preprocessing

class ImageClassifier:
    def __init__(self, mode='supervised', data_dir='dataset'):
        self.mode = mode
        self.spinner = Spinner()
        self.preprocessing = Preprocessing(data_dir)
        self.label_encoder = None

    def load_and_preprocess_data(self):
        self.spinner.set_msg("Loading Images")
        self.spinner.start()
        images, labels = self.preprocessing.load_images_parallel()
        self.spinner.stop()

        self.spinner.set_msg("Encoding Labels")
        self.spinner.start()
        labels_encoded, self.label_encoder = self.preprocessing.encode_labels(labels)
        self.spinner.stop()

        self.spinner.set_msg("Display distribution and sample images")
        self.spinner.start()
        self.preprocessing.display_distribution(labels)
        self.preprocessing.display_sample_images(images, labels)
        self.spinner.stop()

        self.spinner.set_msg("Normalizing images")
        self.spinner.start()
        images = self.preprocessing.normalize_images(images)
        self.spinner.stop()

        self.spinner.set_msg("Splitting data into training and testing sets")
        self.spinner.start()
        X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
        self.spinner.stop()

        return X_train, X_test, y_train, y_test

    def train_and_evaluate_classifier(self, X_train, y_train, X_test, y_test):
        if self.mode == 'semi_supervised':
            n_labeled = int(0.8 * len(y_train))
            indices = np.arange(len(y_train))
            np.random.shuffle(indices)
            labeled_indices = indices[:n_labeled]
            unlabeled_indices = indices[n_labeled:]

            X_train_labeled = X_train[labeled_indices]
            y_train_labeled = y_train[labeled_indices]

            X_train_unlabeled = X_train[unlabeled_indices]
            y_train_unlabeled = -1 * np.ones(len(unlabeled_indices), dtype=int)

            # Flatten images
            self.spinner.set_msg("Flattening images")
            self.spinner.start()
            X_train_labeled_flat = self.preprocessing.flatten_images(X_train_labeled)
            X_train_unlabeled_flat = self.preprocessing.flatten_images(X_train_unlabeled)
            self.spinner.stop()

            # Combine labeled and unlabeled data
            X_train_combined = np.concatenate((X_train_labeled_flat, X_train_unlabeled_flat), axis=0)
            y_train_combined = np.concatenate((y_train_labeled, y_train_unlabeled), axis=0)

            label_spreading = LabelSpreading(kernel='knn', alpha=0.8)
            label_spreading.fit(X_train_combined, y_train_combined)

            y_train_unlabeled_pred = label_spreading.transduction_[len(labeled_indices):]

            X_train_final = np.concatenate((X_train_labeled_flat, X_train_unlabeled_flat), axis=0)
            y_train_final = np.concatenate((y_train_labeled, y_train_unlabeled_pred), axis=0)

        else:
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

            # Flatten images
            self.spinner.set_msg("Flattening images")
            self.spinner.start()
            X_train_final = self.preprocessing.flatten_images(X_train_augmented)
            X_test = self.preprocessing.flatten_images(X_test)
            self.spinner.stop()

            y_train_final = y_train_augmented

        X_test = self.preprocessing.flatten_images(X_test)

        param_distributions = {
            'max_depth': config.get("max_depth"),
            'min_samples_split': config.get("min_samples_split"),
            'min_samples_leaf': config.get("min_samples_leaf"),
        }

        self.spinner.set_msg("Training and evaluating classifier")
        self.spinner.start()
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

        randomized_search.fit(X_train_final, y_train_final)

        best_clf = randomized_search.best_estimator_
        best_params = randomized_search.best_params_
        best_score = randomized_search.best_score_

        y_pred = best_clf.predict(X_test)
        self.spinner.stop()

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print("Best Hyperparameters:", best_params)
        print("Best Cross-Validation Accuracy:", best_score)
        print("Test Accuracy:", accuracy)
        print("Test Precision:", precision)
        print("Test Recall:", recall)
        print("Test F1-score:", f1)
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

        best_model = randomized_search.best_estimator_
        self.save_model(best_model, 'best_model.pkl')

    def save_model(self, model, filename):
        joblib.dump(model, filename)

    def load_model(self, filename):
        return joblib.load(filename)

    def run(self):
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
        self.train_and_evaluate_classifier(X_train, y_train, X_test, y_test)


# Main execution
if __name__ == "__main__":
    classifier = ImageClassifier(mode='semi_supervised')  # or 'supervised'
    classifier.run()
