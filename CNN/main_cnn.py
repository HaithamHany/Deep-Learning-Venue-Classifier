import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from CNN import CNN

num_epochs = 4
num_classes = 10
learning_rate = 0.001

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

    return train_loader, test_loader, classes

def cnn():
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader, test_loader, classes = load_data()

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

    print('Test Accuracy of the model on the 10000 test images: {} %'
          .format((correct / total) * 100))

if __name__ == "__main__":
    cnn()
