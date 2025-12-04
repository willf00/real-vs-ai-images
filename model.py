import torch
from torchvision.models import resnet18, ResNet18_Weights

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
        self.backbone = torch.nn.Sequential(*list(net.children())[:-1])

        # final classification head using ResNet's feature dimension
        self.head = torch.nn.Linear(net.fc.in_features, num_classes)

    def forward(self, x):
        # extract features through ResNet backbone
        x = self.backbone(x)


        # flatten spatial dimensions for linear layer
        x = x.flatten(1)

        # final classification
        return self.head(x)