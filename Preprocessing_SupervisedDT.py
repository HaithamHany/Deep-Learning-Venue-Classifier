import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
from multiprocessing import Pool
from Spinner import Spinner
from Config import config
from sklearn.model_selection import RandomizedSearchCV

# Set up directories
data_dir = 'dataset'
categories = os.listdir(data_dir)

# Parameters for image processing
img_height, img_width = 128, 128  # Reduced size for faster processing

def process_image(params):
    """
    Load and process an image: convert to RGB, resize, and return as a numpy array.
    """
    category, img_name = params
    img_path = os.path.join(data_dir, category, img_name)
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((img_width, img_height))
        img_array = np.array(img)
        return img_array, category
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None, None

def load_images_parallel():
    """
    Load images in parallel using multiprocessing Pool.
    """
    images = []
    labels = []
    params = [(category, img_name) for category in categories for img_name in os.listdir(os.path.join(data_dir, category))]
    with Pool() as pool:
        results = pool.map(process_image, params)
    for img_array, category in results:
        if img_array is not None:
            images.append(img_array)
            labels.append(category)
    return np.array(images), np.array(labels)

def display_distribution(labels):
    """
    Display the distribution of images per category.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=labels)
    plt.title('Distribution of Images per Category')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

def display_sample_images(images, labels):
    """
    Display a few sample images from each category.
    """
    fig, axes = plt.subplots(len(categories), 5, figsize=(15, 15))
    for i, category in enumerate(categories):
        category_indices = np.where(labels == category)[0]
        sample_indices = np.random.choice(category_indices, 5, replace=False)
        for j, index in enumerate(sample_indices):
            axes[i, j].imshow(images[index])
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(category)
    plt.show()

def augment_image(image):
    """
    Apply random augmentations to an image.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_with_crop_or_pad(image, img_height + 10, img_width + 10)
    image = tf.image.random_crop(image, size=[img_height, img_width, 3])
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image

def augment_dataset(X_train, y_train):
    """
    Create an augmented dataset using TensorFlow's tf.data API.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(lambda x, y: (augment_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def flatten_images(images):
    """
    Flatten images for input into the decision tree classifier.
    """
    return images.reshape((images.shape[0], -1))


def train_and_evaluate_classifier(X_train_flat, y_train, X_test_flat, y_test, config):
    """
    Train and evaluate a Decision Tree Classifier with specified hyperparameters.
    Perform a randomized search over the provided hyperparameters.
    """
    param_distributions = {
        'max_depth': config.get("max_depth"),
        'min_samples_split': config.get("min_samples_split"),
        'min_samples_leaf': config.get("min_samples_leaf"),
    }

    clf = DecisionTreeClassifier(criterion='entropy', random_state=config.get("random_state"))

    # Perform randomized search with parallelization
    randomized_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_distributions,
        n_iter=config.get("n_iter", 5),  # Default to 5 if not specified
        scoring='accuracy',
        n_jobs=config.get("n_jobs", -1),  # Default to using all cores if not specified
        cv=config.get("cv", 3),  # Default to 3-fold cross-validation if not specified
        verbose=1,  # Default to verbosity level 1 if not specified
        random_state=config.get("random_state")
    )

    randomized_search.fit(X_train_flat, y_train)

    best_clf = randomized_search.best_estimator_
    best_params = randomized_search.best_params_
    best_score = randomized_search.best_score_

    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Cross-Validation Accuracy: {best_score}")

    # Evaluate the best model on the test set
    y_pred = best_clf.predict(X_test_flat)

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
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Main execution
if __name__ == "__main__":

    spinner = Spinner()
    spinner.set_msg("Loading Images")
    spinner.start()
    # Load images in parallel
    images, labels = load_images_parallel()

    spinner.stop()
    spinner.set_msg("Encoding Labels")
    spinner.start()


    # Encode the labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    spinner.stop()
    spinner.set_msg("Display distribution and sample images")
    spinner.start()

    # Display distribution and sample images
    display_distribution(labels)
    display_sample_images(images, labels)

    spinner.stop()
    spinner.set_msg("Normalizing images")
    spinner.start()

    # Normalize images
    images = images / 255.0

    spinner.stop()
    spinner.set_msg("Splitting data into training and testing sets")
    spinner.start()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

    spinner.stop()
    spinner.set_msg("Augmenting the training data")
    spinner.start()

    # Augment the training data
    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.int64)
    augmented_dataset = augment_dataset(X_train_tensor, y_train_tensor)

    X_train_augmented = []
    y_train_augmented = []
    for batch in augmented_dataset:
        augmented_images, augmented_labels = batch
        X_train_augmented.append(augmented_images.numpy())
        y_train_augmented.append(augmented_labels.numpy())

    X_train_augmented = np.concatenate(X_train_augmented)
    y_train_augmented = np.concatenate(y_train_augmented)

    spinner.stop()
    spinner.set_msg("Flattening images")
    spinner.start()

    # Flatten images
    X_train_augmented_flat = flatten_images(X_train_augmented)
    X_test_flat = flatten_images(X_test)

    spinner.stop()
    spinner.set_msg("Training and evaluating classifier")
    spinner.start()

    # Train and evaluate classifier
    train_and_evaluate_classifier(X_train_augmented_flat, y_train_augmented, X_test_flat, y_test, config)

    spinner.stop()
    spinner.set_msg(" ")
