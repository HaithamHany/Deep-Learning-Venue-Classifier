import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import random
import numpy as np
from CNN import CNN
from PIL import Image
from Config import config_cnn_architecture, config_cnn  # Import your CNN config

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
    # preprocessing
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize images to 32x32 if they are not already
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
    ])

    # Load dataset
    dataset = ImageFolder(root=data_dir, transform=transform)

    # Split dataset into training and testing sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size  # 20% for testing
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

    # Get class names from the directory structure
    classes = sorted([d.name for d in os.scandir(data_dir) if d.is_dir()])

    return train_loader, test_loader, classes, transform

def train_and_evaluate_cnn(train_loader, test_loader, classes, transform, learning_rate, batch_size, num_epochs):
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Update data loaders with new batch size
    train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_loader.dataset, batch_size=1000, shuffle=False, num_workers=2)

    # Total steps per epoch
    total_step = len(train_loader)
    loss_list = []
    acc_list = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  # Iterate batches
            # Forward pass
            outputs = model(images)  # Compute predictions
            loss = criterion(outputs, labels)  # Calculate loss
            loss_list.append(loss.item())  # Record loss

            # Backprop and optimisation
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            # Train accuracy
            total = labels.size(0)  # Total labels
            _, predicted = torch.max(outputs.data, 1)  # Predicted labels
            correct = (predicted == labels).sum().item()  # Correct predictions
            acc_list.append(correct / total)  # Record accuracy

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))

        # Calculate and print average loss and accuracy for the epoch
        avg_loss_epoch = sum(loss_list) / len(loss_list)
        avg_acc_epoch = sum(acc_list) / len(acc_list) * 100
        print('Epoch [{}/{}] Complete, Average Loss: {:.4f}, Average Accuracy: {:.2f}%'
              .format(epoch + 1, num_epochs, avg_loss_epoch, avg_acc_epoch))

    # Save the model and best parameters after training (overwrite existing file)
    torch.save({
        'model_state_dict': model.state_dict(),
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'architecture': config_cnn_architecture
    }, 'cnn_model.pth')
    print('Model and parameters saved to cnn_model.pth')

    # Set model to evaluation mode
    model.eval()
    # Disable gradient computation
    with torch.no_grad():
        correct = 0
        total = 0
        # Iterate test batches
        for images, labels in test_loader:
            # Compute predictions
            outputs = model(images)
            # Get predicted labels
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100
    print('Test Accuracy of the model: {} %'.format(accuracy))

    return accuracy



def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = (correct / total) * 100
    print('Test Accuracy of the loaded model: {} %'.format(accuracy))
    return accuracy


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
        model = CNN()
        model.load_state_dict(torch.load('cnn_model.pth'))
        model.eval()
        prediction = predict_image(model, classes, transform, image_path)
        print(f'Predicted class for the image: {prediction}')


def hyperparameters_tuning(train_loader, test_loader, classes, transform):
    # Hyperparameter tuning
    best_accuracy = 0
    best_params = {}
    for lr in config_cnn['learning_rate']:
        for bs in config_cnn['batch_size']:
            for epochs in config_cnn['num_epochs']:
                print(f"Training with learning_rate={lr}, batch_size={bs}, num_epochs={epochs}")
                accuracy = train_and_evaluate_cnn(train_loader, test_loader, classes, transform, lr, bs, epochs)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'learning_rate': lr, 'batch_size': bs, 'num_epochs': epochs}

    return best_params, best_accuracy


if __name__ == "__main__":
    train_loader, test_loader, classes, transform = load_data()

    if os.path.exists('cnn_model.pth'):
        load_model = input("Model found. Do you want to load the existing model? (yes/no): ").strip().lower()
        if load_model == 'yes':
            checkpoint = torch.load('cnn_model.pth')
            config_cnn_architecture.update(
                checkpoint['architecture'])  # Update the configuration with loaded architecture
            model = CNN()
            model.load_state_dict(checkpoint['model_state_dict'])
            learning_rate = checkpoint['learning_rate']
            batch_size = checkpoint['batch_size']
            num_epochs = checkpoint['num_epochs']
            print(f"Loaded model with learning_rate={learning_rate}, batch_size={batch_size}, num_epochs={num_epochs}")
            evaluate_model(model, test_loader)
        else:
            best_params, best_accuracy = hyperparameters_tuning(train_loader, test_loader, classes, transform)
            print(f"Best Accuracy: {best_accuracy}% with params: {best_params}")
    else:
        best_params, best_accuracy = hyperparameters_tuning(train_loader, test_loader, classes, transform)
        print(f"Best Accuracy: {best_accuracy}% with params: {best_params}")

    # Option to predict an individual image
    single_image_prediction_prompt(classes, transform)

