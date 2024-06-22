import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import config_cnn_architecture

class CNN(nn.Module):
    def __init__(self, input_size=128):
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

        conv_output_size = calculate_conv_output_size(input_size, config_cnn_architecture)
        fc_input_size = conv_output_size * conv_output_size * config_cnn_architecture["num_filters"][-1]

        self.conv_layer = nn.Sequential(*layers)
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(fc_input_size, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # conv layers.  Passes the input x through the convolutional layers.
        x = self.conv_layer(x)
        # flatten. Flattens the output of the convolutional layers to prepare it for the fully connected layers.
        x = x.view(x.size(0), -1)
        # fc layer. Passes the flattened data through the fully connected layers.
        x = self.fc_layer(x)
        return x




def calculate_conv_output_size(input_size, config):
    size = input_size
    for i in range(config["num_layers"]):
        size = (size - config["filter_sizes"][i] + 2 * config["paddings"][i]) // config["strides"][i] + 1
        if (i + 1) % 2 == 0:  # Apply max pooling after every two layers
            size = size // 2
    return size