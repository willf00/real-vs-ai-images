"""
run.py: trains and evaluates a model (SimpleCNN or ResNet) on the dataset
Authors: Will Fete & Jason Albanus
Date: 12/7/2025
Notice: Uses utils.train / utils.evaluate and saves the trained weights
"""
import model
import dataset as d
import utils
import torch
from torch import nn
from torch import optim

if __name__ == '__main__':
    # simple CNN
    model = model.SimpleCNN()
    transform = d.get_dataTransforms()
    dataset = d.get_data('dataset', transform)
    trainloader, testloader = d.get_dataloaders(dataset)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model.to(device)

    utils.train(model, trainloader, criterion, optimizer, device)
    utils.evaluate(model, testloader, device)

    CNN_MODEL_PATH = './cnn_model.pth'
    torch.save(model.state_dict(), CNN_MODEL_PATH)
    
    # resnet
    # transforms = d.get_dataTransforms()
    # datasets = d.get_data('dataset', transforms)
    # trainloader, testloader = d.get_dataloaders(datasets)
    # num_classes = len(datasets['train'].classes)

    # net = model.CNN(num_classes=num_classes)  # uses resnet18 backbone
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Device: {device}")
    # net.to(device)

    # utils.train(net, trainloader, criterion, optimizer, device, epochs=5)
    # utils.evaluate(net, testloader, device)