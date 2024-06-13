import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Defines a sequence of convolutional layers, batch normalization layers, LeakyReLU activations, and max pooling layers.
        self.conv_layer = nn.Sequential(
            # Applies a 2D convolution over an input signal
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            # Applies Batch Normalization over a 4D input.
            nn.BatchNorm2d(32),
            # Applies the LeakyReLU activation function.
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            # Applies a 2D max pooling over an input signal
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Defines a sequence of fully connected layers, dropout layers, and ReLU activations.
        self.fc_layer = nn.Sequential(
            # Randomly zeroes some of the elements of the input tensor with probability
            nn.Dropout(p=0.1),
            # Applies a linear transformation to the incoming data.
            nn.Linear(8 * 8 * 64, 1000),
            # Applies the ReLU activation function.
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
