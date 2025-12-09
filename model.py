"""
model.py: the SimpleCNN and ResNet-based CNN architectures
Authors: Will Fete & Jason Albanus
Date: 12/7/2025
Notice: SimpleCNN is based around HW5. CNN uses ResNet with custom classification head
"""
import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
import torch.nn as nn

class CNN(torch.nn.Module):
    """
    CNN architecture using a pretrained ResNet18 backbone

    What it does:
      - Loads torchvision resnet18 with ImageNet weights.
      - Removes the original final FC layer and adds a custom Linear head for `num_classes`.
      - Forwards inputs through the frozen structure of resnet18 (up to avgpool) and then the new head.

    Args:
        num_classes (int): number of output classes for the custom head.
    """


    def __init__(self, num_classes: int):
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
    """
    SimpleCNN: three conv blocks with batch norm and maxpool, then two
    fully connected layers with dropout, ending in a 2-class output.

    What it does: extracts features with conv/pool, flattens, then classifies
    with dense layers and dropout for regularization.

    Args:
        (none) - output classes are fixed to 2 in this head.
    """

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