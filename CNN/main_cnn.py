import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from CNN import CNN
from PIL import Image

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

    return train_loader, test_loader, classes, transform

def cnn(train_loader, test_loader, classes, transform):
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

    # Save the model after training
    torch.save(model.state_dict(), 'cnn_model.pth')

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

    print('Test Accuracy of the model on the test images: {} %'
          .format((correct / total) * 100))

    return model, classes, transform

def predict_image(model, classes, transform, image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Apply same transformation as training and add batch dimension
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return classes[predicted.item()]

if __name__ == "__main__":

    train_loader, test_loader, classes, transform = load_data()

    # Option to train or load the model
    if os.path.exists('cnn_model.pth'):
        load_model = input("Model found. Do you want to load the existing model? (yes/no): ").strip().lower()
        if load_model == 'yes':
            model = CNN()
            model.load_state_dict(torch.load('cnn_model.pth'))
            model.eval()
            print("Loaded the model from 'cnn_model.pth'")
        else:
            model, classes, transform = cnn(train_loader, test_loader, classes, transform)
    else:
        model, classes, transform = cnn(train_loader, test_loader, classes, transform)

    # Evaluate on the test dataset
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the loaded model: {} %'.format((correct / total) * 100))

    # Option to predict an individual image
    predict_img = input("Do you want to predict an individual image? (yes/no): ").strip().lower()
    if predict_img == 'yes':
        test_images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'test'))
        print("Available images:")
        for img_name in os.listdir(test_images_dir):
            print(img_name)
        image_name = input("Enter the name of the image (e.g., image.jpg): ").strip()
        image_path = os.path.join(test_images_dir, image_name)
        prediction = predict_image(model, classes, transform, image_path)
        print(f'Predicted class for the image: {prediction}')


