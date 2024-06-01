import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf

# Set up directories
data_dir = 'dataset'
categories = os.listdir(data_dir)

# Parameters for image processing
img_height, img_width = 256, 256

# Initialize lists for images and labels
images = []
labels = []

# Load the images
for category in categories:
    category_path = os.path.join(data_dir, category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((img_width, img_height))
            img_array = np.array(img)
            images.append(img_array)
            labels.append(category)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Display the distribution of images per category
plt.figure(figsize=(10, 6))
sns.countplot(x=labels)
plt.title('Distribution of Images per Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Display a few sample images from each category
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

# Normalize the images
images = images / 255.0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Data Augmentation using TensorFlow
def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, size=[img_height, img_width, 3])
    image = tf.image.resize(image, [img_height, img_width])
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image

# Convert numpy arrays to TensorFlow tensors
X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)

# Augment the training data
X_train_augmented = []
y_train_augmented = []

for img, label in zip(X_train_tensor, y_train):
    X_train_augmented.append(img.numpy())
    y_train_augmented.append(label)
    for _ in range(5):  # create 5 augmented images per original image
        aug_img = augment_image(img)
        X_train_augmented.append(aug_img.numpy())
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

# Flatten the images for the decision tree
X_train_augmented_flat = X_train_augmented.reshape((X_train_augmented.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_augmented_flat, y_train_augmented)

# Make predictions and evaluate
y_pred = clf.predict(X_test_flat)

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
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
