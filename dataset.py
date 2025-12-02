import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

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

path = "dataset"

image_datasets = {
    'train': datasets.ImageFolder(os.path.join(path, 'train'), data_transforms['train']),
    'test': datasets.ImageFolder(os.path.join(path, 'test'), data_transforms['test'])
}

if __name__ == '__main__':
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=4)
    }

    # Get dataset sizes and class names
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    print(f"Classes found: {class_names}")

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    print(f"Batch shape: {inputs.shape}")
    print(f"Labels: {classes}")