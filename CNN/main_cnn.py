import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import random
import numpy as np
from CNN import CNN
from PIL import Image
from Config import config_cnn_architecture, config_cnn  # Import your CNN config
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    # Check the number of GPUs and their names:
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available!")
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 4
num_classes = 10
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

    # Define transformation with data augmentation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Get class names from the directory structure
    classes = sorted([d.name for d in os.scandir(data_dir) if d.is_dir()])

    return train_loader, val_loader, test_loader, classes, transform

def train_and_evaluate_cnn(train_loader, test_loader, val_loader, classes, transform, learning_rate, batch_size, num_epochs):
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Reconfigure data loaders with the correct batch size
    train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_loader.dataset, batch_size=1000, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_loader.dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
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
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.2f}%')

    # Save the model after all epochs
    torch.save({
        'model_state_dict': model.state_dict(),
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'architecture': config_cnn_architecture
    }, 'cnn_model.pth')

    # Evaluate on test set
    return evaluate_model(model, test_loader)

def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode

    # Variables to gather full outputs
    true_labels = []
    predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())  # Save predictions
            true_labels.extend(labels.cpu().numpy())  # Save true labels

    # Calculate metrics using sklearn
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(true_labels, predictions)

    # Print metrics
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    return accuracy, precision, recall, f1, conf_matrix

def predict_image(model, classes, transform, image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Apply same transformation as training and add batch dimension
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return classes[predicted.item()]


def single_image_prediction_prompt(classes, transform):
    predict_img = input("Do you want to predict an individual image? (yes/no): ").strip().lower()
    if predict_img == 'yes':
        test_images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'test'))
        print("Available images:")
        for img_name in os.listdir(test_images_dir):
            print(img_name)
        image_name = input("Enter the name of the image (e.g., image.jpg): ").strip()
        image_path = os.path.join(test_images_dir, image_name)
        model = CNN().to(device)
        model.load_state_dict(torch.load('cnn_model.pth'))
        model.eval()
        prediction = predict_image(model, classes, transform, image_path)
        print(f'Predicted class for the image: {prediction}')


def hyperparameters_tuning(train_loader, val_loader, test_loader, classes, transform):
    best_accuracy = 0  # Initialize to 0
    best_params = {}

    for lr in config_cnn['learning_rate']:
        for bs in config_cnn['batch_size']:
            for epochs in config_cnn['num_epochs']:
                print(f"Training with learning_rate={lr}, batch_size={bs}, num_epochs={epochs}")
                _, precision, recall, f1, _ = train_and_evaluate_cnn(train_loader, test_loader, val_loader, classes, transform, lr, bs, epochs)
                if precision > best_accuracy:  # Assuming you want to use precision or choose another metric
                    best_accuracy = precision
                    best_params = {'learning_rate': lr, 'batch_size': bs, 'num_epochs': epochs}

    return best_params, best_accuracy

if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes, transform = load_data()

    if os.path.exists('cnn_model.pth'):
        load_model = input("Model found. Do you want to load the existing model? (yes/no): ").strip().lower()
        if load_model == 'yes':
            checkpoint = torch.load('cnn_model.pth')
            config_cnn_architecture.update(checkpoint['architecture'])  # Update the configuration with loaded architecture
            model = CNN().to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model with learning_rate={checkpoint['learning_rate']}, batch_size={checkpoint['batch_size']}, num_epochs={checkpoint['num_epochs']}")
            evaluate_model(model, test_loader)
        else:
            # Ensure to include 'classes' and 'transform' in the function call
            best_params, best_accuracy = hyperparameters_tuning(train_loader, val_loader, test_loader, classes, transform)
            print(f"Best Accuracy: {best_accuracy}% with params: {best_params}")
    else:
        # Also here, include 'classes' and 'transform'
        best_params, best_accuracy = hyperparameters_tuning(train_loader, val_loader, test_loader, classes, transform)
        print(f"Best Accuracy: {best_accuracy}% with params: {best_params}")

    # Additional functionality such as predicting an individual image
    single_image_prediction_prompt(classes, transform)



