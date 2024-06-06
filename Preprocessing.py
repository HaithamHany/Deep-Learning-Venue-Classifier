import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from multiprocessing import Pool
from sklearn.preprocessing import LabelEncoder

class Preprocessing:
    def __init__(self, data_dir='dataset', img_height=128, img_width=128):
        """
        Initialize the Preprocessing class.
        """
        self.data_dir = data_dir
        self.categories = os.listdir(data_dir)
        self.img_height = img_height
        self.img_width = img_width

    def process_image(self, params):
        """
        Load and process an image: convert to RGB, resize, and return as a numpy array.
        """
        category, img_name = params
        img_path = os.path.join(self.data_dir, category, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.img_width, self.img_height))
            img_array = np.array(img)
            return img_array, category
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

    def load_images_parallel(self):
        """
        Load images in parallel using multiprocessing Pool.
        """
        images = []
        labels = []
        params = [(category, img_name) for category in self.categories for img_name in os.listdir(os.path.join(self.data_dir, category))]
        with Pool() as pool:
            results = pool.map(self.process_image, params)
        for img_array, category in results:
            if img_array is not None:
                images.append(img_array)
                labels.append(category)
        return np.array(images), np.array(labels)

    def display_distribution(self, labels):
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

    def display_sample_images(self, images, labels):
        """
        Display a few sample images from each category.
        """
        fig, axes = plt.subplots(len(self.categories), 5, figsize=(15, 15))
        for i, category in enumerate(self.categories):
            category_indices = np.where(labels == category)[0]
            sample_indices = np.random.choice(category_indices, 5, replace=False)
            for j, index in enumerate(sample_indices):
                axes[i, j].imshow(images[index])
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(category)
        plt.show()

    def augment_image(self, image):
        """
        Apply random augmentations to an image.
        """
        image = tf.image.random_flip_left_right(image)
        image = tf.image.resize_with_crop_or_pad(image, self.img_height + 10, self.img_width + 10)
        image = tf.image.random_crop(image, size=[self.img_height, self.img_width, 3])
        image = tf.image.random_brightness(image, max_delta=0.3)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        return image

    def augment_dataset(self, X_train, y_train):
        """
        Create an augmented dataset using TensorFlow's tf.data API.
        """
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.map(lambda x, y: (self.augment_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(32)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def flatten_images(self, images):
        """
        Flatten images for input into the decision tree classifier.
        """
        return images.reshape((images.shape[0], -1))

    def normalize_images(self, images):
        """
        Normalize images to a range of 0 to 1.
        """
        return images / 255.0

    def encode_labels(self, labels):
        """
        Encode the labels using LabelEncoder.
        """
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        return labels_encoded, label_encoder

    def split_labeled_unlabeled(self, X_train, y_train, labeled_ratio=0.8):
        """
        Split the training data into labeled and unlabeled sets.
        """
        np.random.seed(42)
        num_samples = len(y_train)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        num_labeled = int(num_samples * labeled_ratio)
        labeled_indices = indices[:num_labeled]
        unlabeled_indices = indices[num_labeled:]

        X_labeled = X_train[labeled_indices]
        y_labeled = y_train[labeled_indices]
        X_unlabeled = X_train[unlabeled_indices]
        y_unlabeled = y_train[unlabeled_indices]

        return X_labeled, y_labeled, X_unlabeled, y_unlabeled


