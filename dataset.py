from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
"""
dataset.py: Handles the raw dataset and turns it into test and training dataloaders for PyTorch
Authors: Will Fete
Date: 12/7/2025
"""


def get_dataTransforms():
    """
        Args:
        None

        Output: A dictionary of PyTorch transforms
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)), # Resize to standard size for ResNet
            transforms.RandomHorizontalFlip(), # Data augmentation for training
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def get_data(path, data_transforms):
    """
        Args:
        path: String of the path of the dataset in the current directory
        data_transforms: A dictionary of PyTorch transforms

        Output: A dictionary containing the train and test datasets
    """
    img_datasets = {
        'train': datasets.ImageFolder(os.path.join(path, 'train'), data_transforms['train']),
        'test': datasets.ImageFolder(os.path.join(path, 'test'), data_transforms['test'])
    }
    return img_datasets

def get_dataloaders(img_datasets):
    """
        Args:
        img_datasets: A dictionary containing the train and test datasets

        Output: PyTorch Dataloaders for test and train
    """
    trainloader =  DataLoader(img_datasets['train'], batch_size=64, shuffle=True, num_workers=4)
    testloader = DataLoader(img_datasets['test'], batch_size=64, shuffle=False, num_workers=4)

    return trainloader, testloader

if __name__ == '__main__':
    """
        This just shows the dataloader and how it works
    """
    transform = get_dataTransforms()
    dataset = get_data('dataset', transform)
    trainloader, testloader = get_dataloaders(dataset)

    # Get dataset sizes and class names
    dataset_sizes = {x: len(dataset[x]) for x in ['train', 'test']}
    class_names = dataset['train'].classes

    print(f"Classes found: {class_names}")

    # Get a batch of training data
    inputs, classes = next(iter(trainloader))

    print(f"Batch shape: {inputs.shape}")
    print(f"Labels: {classes}")