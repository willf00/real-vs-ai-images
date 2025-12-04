import model
import dataset as d
import utils
import torch
from torch import nn
from torch import optim

if __name__ == '__main__':
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