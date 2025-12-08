import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm  # For progress bar


def imshow(img, title=None):
    """Used to display image"""
    img = img.numpy().transpose((1, 2, 0))

    # Undoes normalization done in transforms
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = std * img + mean
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# This function is the train function from HW5 of ECE4554
def train(model, loader, criterion, optimizer,device, epochs=5):
  '''Train a model from training data.

  Args:
    - model: Neural network to train
    - epochs: Number of epochs to train the model
    - loader: Dataloader to train the model with
    - device: PyTorch device (CPU vs GPU)
  '''
  print('Start Training')

  for epoch in range(epochs):  # loop over the dataset multiple times

      running_loss = 0.0
      total = 0
      correct = 0
      for i, data in enumerate(tqdm(loader)):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data
          inputs, labels = inputs.to(device), labels.to(device) # from piazza was broken
          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # the class with the highest energy is what we choose as prediction
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

          # print statistics
          running_loss += loss.item()
          if i % 500 == 499:    # print every 500 mini-batches
              tqdm.write(f'[epoch {epoch + 1}, batch {i + 1:5d}] loss: {running_loss / 500:.3f}')
              running_loss = 0.0

              print(f'\nAccuracy of the network on last {500} training images: {100 * correct // total:.8f} %')
              total = 0
              correct = 0

  print('\nFinished Training')


# This function is the evaluation function from HW5 of ECE4554 with slight changes
def evaluate(model, loader, device):
  '''Evaluate a model and output its accuracy on a test dataset.

  Args:
    - model: Neural network to evaluate
    - loader: Dataloader containing test dataset
    - device: PyTorch device (CPU vs GPU)
  '''
  model.eval()
  # Evaluate accuracy on validation / test set
  correct = 0
  total = 0
  TP = TN = FP = FN = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
  with torch.no_grad():
    for data in tqdm(loader):
      images, labels = data
      images, labels = images.to(device), labels.to(device) # from piazza was broken
      # calculate outputs by running images through the network
      outputs = model(images)
      # the class with the highest energy is what we choose as prediction
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      # TP: Predicted 1, Actual 1
      TP += ((predicted == 1) & (labels == 1)).sum().item()
      
      # TN: Predicted 0, Actual 0
      TN += ((predicted == 0) & (labels == 0)).sum().item()
      
      # FP: Predicted 1, Actual 0
      FP += ((predicted == 1) & (labels == 0)).sum().item()
      
      # FN: Predicted 0, Actual 1
      FN += ((predicted == 0) & (labels == 1)).sum().item()

  # Calculate final metrics
  precision = TP / (TP + FP) if (TP + FP) > 0 else 0
  recall = TP / (TP + FN) if (TP + FN) > 0 else 0
  f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

  print(f"\nAccuracy of the network on the {total} test images: {correct / total:.8f}")
  print(f"\nAccuracy of the network on the {total} test images: {100 * correct / total:.8f} %")
  print(f'\nPrecision of the network on the {total} test images: {precision}')
  print(f'\nRecall of the network on the {total} test images: {recall}')
  print(f'\nF1 Score of the network on the {total} test images: {f1_score}')
  print(f'\nTP = {TP}, FP = {FP}, FN = {FN}, TN = {TN}')