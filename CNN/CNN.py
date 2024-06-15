import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import config_cnn_architecture

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        layers = []

        in_channels = 3  # Assuming RGB images
        for i in range(config_cnn_architecture["num_layers"]):
            out_channels = config_cnn_architecture["num_filters"][i]
            kernel_size = config_cnn_architecture["filter_sizes"][i]
            stride = config_cnn_architecture["strides"][i]
            padding = config_cnn_architecture["paddings"][i]

            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(inplace=True))

            in_channels = out_channels

            # Add max pooling after every two convolutional layers
            if (i + 1) % 2 == 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_layer = nn.Sequential(*layers)
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(3 * 3 * 512, 1000),  # Adjusted input size
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)  # Apply convolutional layers
        # print("Output shape before flattening:", x.shape)  # Print the shape here
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc_layer(x)  # Apply fully connected layers
        return x
