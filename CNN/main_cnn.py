import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

data_dir = '../dataset'
def load_data(data_dir):
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
