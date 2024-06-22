import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import random
import numpy as np
from CNN.CNN import CNN
from PIL import Image
from Config import config_cnn_architecture, config_cnn  # Import your CNN config
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")  # Changed: Directly assign device
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    device = torch.device("cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 4
num_classes = 5
learning_rate = 0.001


# Setting a fixed seed for all random number generators to ensure reproducibility
# and consistent behavior across different runs of the code.

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(0)


def load_data():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    # preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128 if they are not already
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = ImageFolder(root=data_dir, transform=transform)

    # Split dataset into training, validation, and testing sets
    total_count = len(dataset)
    train_count = int(0.7 * total_count)
    val_count = int(0.15 * total_count)
    test_count = total_count - train_count - val_count

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_count, val_count, test_count])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # Get class names from the directory structure
    classes = sorted([d.name for d in os.scandir(data_dir) if d.is_dir()])

    return train_loader, val_loader, test_loader, classes, transform


def train_and_evaluate_cnn(train_loader, test_loader, val_loader, classes, transform, learning_rate, batch_size,
                           num_epochs):
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Update data loaders with new batch size
    train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_loader.dataset, batch_size=1000, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_loader.dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Total steps per epoch
    total_step = len(train_loader)
    loss_list = []
    acc_list = []

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy:.2f}%')

    # Save the model and best parameters after training (overwrite existing file)
    torch.save({
        'model_state_dict': model.state_dict(),
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'architecture': config_cnn_architecture
    }, 'cnn_model.pth')

    # Evaluate on test set
    accuracy, precision, recall, f1, conf_matrix, per_class_metrics = evaluate_model(model, test_loader, classes)

    return precision, recall, f1, conf_matrix


def evaluate_model(model, test_loader, classes):
    model.eval()

    # Variables to gather full outputs
    true_labels = []
    predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())  # Save predictions
            true_labels.extend(labels.cpu().numpy())  # Save true labels

    # Calculate metrics using sklearn
    accuracy = accuracy_score(true_labels, predictions)
    overall_precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    overall_recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    overall_f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(true_labels, predictions)

    # Print overall metrics
    print(f"Overall Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Overall Precision: {overall_precision:.2f}")
    print(f"Overall Recall: {overall_recall:.2f}")
    print(f"Overall F1 Score: {overall_f1:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot confusion matrix as an image
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Calculate and print per-class metrics
    per_class_metrics = {}
    for i, class_label in enumerate(classes):
        class_true = [1 if x == i else 0 for x in true_labels]
        class_pred = [1 if x == i else 0 for x in predictions]

        class_accuracy = accuracy_score(class_true, class_pred)
        class_precision = precision_score(class_true, class_pred, zero_division=0)
        class_recall = recall_score(class_true, class_pred, zero_division=0)
        class_f1 = f1_score(class_true, class_pred, zero_division=0)

        per_class_metrics[class_label] = {
            'Accuracy': class_accuracy,
            'Precision': class_precision,
            'Recall': class_recall,
            'F1 Score': class_f1
        }

        print(f"Metrics for class {class_label}:")
        print(f"  Accuracy: {class_accuracy * 100:.2f}%")
        print(f"  Precision: {class_precision:.2f}")
        print(f"  Recall: {class_recall:.2f}")
        print(f"  F1 Score: {class_f1:.2f}")

    return accuracy, overall_precision, overall_recall, overall_f1, conf_matrix, per_class_metrics


def predict_image(model, classes, transform, image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Apply same transformation as training and add batch dimension
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return classes[predicted.item()]


def single_image_prediction_prompt(classes, transform):
    test_images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'test'))
    test_images = os.listdir(test_images_dir)
    print("Available images:")
    for idx, img_name in enumerate(test_images):
        print(f"{idx}: {img_name}")
    try:
        image_index = int(input("Enter the index of the image: ").strip())
        if image_index < 0 or image_index >= len(test_images):
            raise ValueError("Index out of range")
        image_name = test_images[image_index]
        image_path = os.path.join(test_images_dir, image_name)
        model = CNN().to(device)
        checkpoint = torch.load('cnn_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        prediction = predict_image(model, classes, transform, image_path)
        print(f'Predicted class for the image: {prediction}')
    except ValueError as e:
        print(f"Invalid input: {e}")



def hyperparameters_tuning(train_loader, val_loader, test_loader, classes, transform):
    # Hyperparameter tuning
    best_accuracy = 0
    best_params = {}
    for lr in config_cnn['learning_rate']:
        for bs in config_cnn['batch_size']:
            for epochs in config_cnn['num_epochs']:
                print(f"Training with learning_rate={lr}, batch_size={bs}, num_epochs={epochs}")
                precision, recall, f1, _ = train_and_evaluate_cnn(train_loader, test_loader, val_loader, classes,
                                                                  transform, lr, bs, epochs)
                if precision > best_accuracy:  # Assuming you want to use precision or choose another metric
                    best_accuracy = precision
                    best_params = {'learning_rate': lr, 'batch_size': bs, 'num_epochs': epochs}

    return best_params, best_accuracy




def run_cnn():
    train_loader, val_loader, test_loader, classes, transform = load_data()
    model = CNN().to(device)

    def load_existing_model():
        checkpoint = torch.load('cnn_model.pth')
        config_cnn_architecture.update(checkpoint['architecture'])  # Update the configuration with loaded architecture
        model.load_state_dict(checkpoint['model_state_dict'])
        learning_rate = checkpoint['learning_rate']
        batch_size = checkpoint['batch_size']
        num_epochs = checkpoint['num_epochs']
        print(f"Loaded model with learning_rate={learning_rate}, batch_size={batch_size}, num_epochs={num_epochs}")

    def train_new_model():
        best_params, best_accuracy = hyperparameters_tuning(train_loader, val_loader, test_loader, classes, transform)
        print(f"Best Accuracy: {best_accuracy}% with params: {best_params}")
        model.load_state_dict(
            torch.load('cnn_model.pth')['model_state_dict'])  # Load the best model after hyperparameter tuning

    def prompt_for_loading_model():
        load_model = input("Model found. Do you want to load the existing model? (yes/no): ").strip().lower()
        if load_model == 'yes':
            load_existing_model()
        else:
            train_new_model()

    def prompt_user_between_single_and_testData():
        choice = input(
            "Do you want to evaluate the entire test dataset or classify a single image? (test/single): ").strip().lower()
        if choice == 'test':
            evaluate_model(model, test_loader, classes)  # Evaluate the test dataset
        elif choice == 'single':
            single_image_prediction_prompt(classes, transform)  # Predict a single image
        else:
            print("Invalid choice. Please enter 'test' or 'single'.")

    if os.path.exists('cnn_model.pth'):
        prompt_for_loading_model()
    else:
        train_new_model()

    # Prompt user to choose between evaluating test dataset or single image
    prompt_user_between_single_and_testData()