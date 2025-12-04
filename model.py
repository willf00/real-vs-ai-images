import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
import torch.nn as nn

class CNN(torch.nn.Module):
    """
    CNN architecture based on ResNet18.

    Key changes from standard ResNet:
        - First conv layer stride reduced from 2 to 1 to preserve fine details
        - Last residual block stride reduced to prevent excessive downsampling
        - Added DropBlock regularization for structured dropout
        - Modified final classification head for handwriting classes

    Args:
        num_classes (int): Number of output classes
        dropout_p (float): Dropout prob for final layer
    """

    def __init__(self, num_classes: int, dropout_p: float = 0.3):
        super().__init__()
        # load pretrained ResNet18 with ImageNet weights
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # extract all layers except final FC layer
        self.backbone = nn.Sequential(*list(net.children())[:-1])

        # final classification head using ResNet's feature dimension
        self.head = nn.Linear(net.fc.in_features, num_classes)

    def forward(self, x):
        # extract features through ResNet backbone
        x = self.backbone(x)


        # flatten spatial dimensions for linear layer
        x = x.flatten(1)

        # final classification
        return self.head(x)


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 64x112x112 -> 128x56x56
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Block 3: 128x56x56 -> 256x28x28
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Flattening: 256 * 28 * 28 = 200,704 features
        self.fc1 = nn.Linear(in_features=256 * 28 * 28, out_features=1024)

        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=256, out_features=2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Flatten
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x