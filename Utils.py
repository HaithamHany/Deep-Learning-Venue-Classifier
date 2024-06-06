import joblib
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


def load_model(model_filename, config):
    try:
        # Load the model
        trained_model = joblib.load(model_filename)
        print("Model loaded successfully!")

        # Extract and print the parameters of the best estimator
        config_params = ["max_depth", "min_samples_split", "min_samples_leaf", "random_state"]
        search_params = {"n_iter": config.get("n_iter"), "cv": config.get("cv"), "n_jobs": config.get("n_jobs")}

        best_params = trained_model.get_params()

        print("Best parameters found:")
        for param in config_params:
            if param in best_params:
                print(f"  {param}: {best_params[param]}")
        for param, value in search_params.items():
            if value is not None:
                print(f"  {param}: {value}")

        # best_params = self.trained_model.get_params()
        # print("Best parameters found:")
        # for param, value in best_params.items():
        # print(f"  {param}: {value}")

        return trained_model

    except Exception as e:
        print(f"Error loading the model: {e}")
        return None


def evaluate_classifier(trained_model, X_test, y_test, preprocessing, label_encoder):
    if trained_model is None:
        raise ValueError("No trained model found. Please train the model before evaluation.")

    X_test_flat = preprocessing.flatten_images(X_test)

    y_pred = trained_model.predict(X_test_flat)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")