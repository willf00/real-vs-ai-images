import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_dataTransforms():
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
    img_datasets = {
        'train': datasets.ImageFolder(os.path.join(path, 'train'), data_transforms['train']),
        'test': datasets.ImageFolder(os.path.join(path, 'test'), data_transforms['test'])
    }
    return img_datasets

def get_dataloaders(img_datasets):
    trainloader =  DataLoader(img_datasets['train'], batch_size=64, shuffle=True, num_workers=4)
    testloader = DataLoader(img_datasets['test'], batch_size=64, shuffle=False, num_workers=4)

    return trainloader, testloader

if __name__ == '__main__':
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