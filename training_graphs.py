"""
training_graphs.py: Training/validation data collection and graph generation for SimpleCNN and ResNet
Authors: Will Fete & Jason Albanus
Date: 12/7/2025
Notice: Trained the same way as the train funciton
"""
import torch
import utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


def train_with_metrics(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                       criterion, optimizer, device: str, epochs: int = 5, log_interval: int = 500,) -> Dict[str, Any]:
    """
    Train a model and track training and validation metrics
    What: runs the full train/val loop and stores loss/accuracy for later plotting
    Args:
        model: the PyTorch model to train
        train_loader: dataloader for training data
        val_loader: dataloader for validation data
        criterion: loss function
        optimizer: optimizer (e.g., SGD)
        device: "cuda" or "cpu"
        epochs: how many passes over the training set
        log_interval: how many batches before logging a train point
    """
    history = {
        "train_steps": [], "train_loss": [], "train_acc": [], "epochs": [], "train_loss_epoch": [],
        "train_acc_epoch": [], "val_epochs": [], "val_loss": [], "val_acc": [],
    }

    global_step = 0

    print("start training with graphs")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        # for epoch-level training stats
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            global_step += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # accumulate for running averages
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            running_total += batch_size

            # accumulate for full-epoch averages
            epoch_loss += loss.item() * batch_size
            epoch_correct += (preds == labels).sum().item()
            epoch_total += batch_size

            # log every mini-batches/epoch
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = running_loss / max(1, running_total)
                avg_acc = running_correct / max(1, running_total)

                history["train_steps"].append(global_step)
                history["train_loss"].append(avg_loss)
                history["train_acc"].append(avg_acc)

                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Train loss: {avg_loss:.8f} | "
                    f"Train acc: {avg_acc * 100:.8f}%"
                )

                # reset running stats for next logging window
                running_loss = 0.0
                running_correct = 0
                running_total = 0

        # Training metrics collected per epoch
        epoch_avg_loss = epoch_loss / max(1, epoch_total)
        epoch_avg_acc = epoch_correct / max(1, epoch_total)
        history["epochs"].append(epoch + 1)
        history["train_loss_epoch"].append(epoch_avg_loss)
        history["train_acc_epoch"].append(epoch_avg_acc)

        # Validation at the end of each epoch
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += batch_size

        avg_val_loss = val_loss / max(1, val_total)
        avg_val_acc = val_correct / max(1, val_total)

        history["val_epochs"].append(epoch + 1)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)

        print(
            f"Validation after epoch {epoch + 1}: "
            f"Val loss: {avg_val_loss:.4f} | "
            f"Val acc: {avg_val_acc * 100:.8f}%"
        )

    print("Finished training with graphs")
    return history

def plot_train_loss_vs_accuracy(history: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    plots training loss vs training accuracy.
    What: shows how loss and accuracy move together across each epoch
    Args:
        history: metrics dict returned by train_with_metrics
        save_path: optional filepath to save instead of showing
    """
    epochs = history.get("epochs", [])
    train_loss_epoch = history.get("train_loss_epoch", [])
    train_acc_epoch = history.get("train_acc_epoch", [])

    if not epochs or not train_loss_epoch or not train_acc_epoch:
        print("no data found")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Loss on left y-axis
    ax1.plot(epochs, train_loss_epoch, "b-o", label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # Accuracy on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, [a * 100 for a in train_acc_epoch], "r-s", label="Train accuracy")
    ax2.set_ylabel("Accuracy (%)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Training loss vs training accuracy")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def _plot_train_loss_vs_accuracy_on_ax(ax, history: Dict[str, Any], title: str) -> None:
    """
    Plot training loss vs training accuracy on a given axes
    What: same as above but draws on a provided subplot
    Args:
        ax: matplotlib axes to draw on
        history: metrics dict from train_with_metrics
        title: title for this subplot
    """
    epochs = history.get("epochs", [])
    train_loss_epoch = history.get("train_loss_epoch", [])
    train_acc_epoch = history.get("train_acc_epoch", [])

    if not epochs or not train_loss_epoch or not train_acc_epoch:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_title(title)
        return

    ax_loss = ax
    ax_acc = ax.twinx()

    ax_loss.plot(epochs, train_loss_epoch, "b-o", label="Train loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss", color="b")
    ax_loss.tick_params(axis="y", labelcolor="b")

    ax_acc.plot(epochs, [a * 100 for a in train_acc_epoch], "r-s", label="Train acc")
    ax_acc.set_ylabel("Accuracy (%)", color="r")
    ax_acc.tick_params(axis="y", labelcolor="r")

    lines1, labels1 = ax_loss.get_legend_handles_labels()
    lines2, labels2 = ax_acc.get_legend_handles_labels()
    ax_loss.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax_loss.set_title(title)
    ax_loss.grid(True, alpha=0.3)


def _plot_train_vs_val_accuracy_on_ax(ax, history: Dict[str, Any], title: str) -> None:
    """
    Plot training vs validation accuracy on a given axes
    What: compares train and val accuracy per epoch on a provided subplot.
    Args:
        ax: matplotlib axes to draw on
        history: metrics dict from train_with_metrics
        title: title for this subplot
    """
    epochs = history.get("epochs", [])
    train_acc_epoch = history.get("train_acc_epoch", [])
    val_epochs = history.get("val_epochs", [])
    val_acc = history.get("val_acc", [])

    if not epochs or not train_acc_epoch or not val_epochs or not val_acc:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_title(title)
        return

    ax.plot(epochs, [a * 100 for a in train_acc_epoch], "b-o", label="Train acc")
    ax.plot(val_epochs, [a * 100 for a in val_acc], "r-s", label="Val acc")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_train_vs_val_accuracy(history: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plot training vs validation accuracy
    """
    epochs = history.get("epochs", [])
    train_acc_epoch = history.get("train_acc_epoch", [])
    val_epochs = history.get("val_epochs", [])
    val_acc = history.get("val_acc", [])

    if not epochs or not train_acc_epoch or not val_epochs or not val_acc:
        print("no data found")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [a * 100 for a in train_acc_epoch], "b-o", label="Train acc")
    plt.plot(val_epochs, [a * 100 for a in val_acc], "r-s", label="Val acc")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs validation accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    """
    To run both SimpleCNN and ResNet and view graphs
    Takes a lot longer than simply running it on run.py
    """
    from torch import nn, optim
    import dataset as d
    import model as model_mod

    # build data once
    transforms = d.get_dataTransforms()
    datasets = d.get_data("dataset", transforms)
    train_loader, test_loader = d.get_dataloaders(datasets)
    num_classes = len(datasets["train"].classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    criterion = nn.CrossEntropyLoss()

    # simple cnn
    simple_net = model_mod.SimpleCNN().to(device)
    simple_opt = optim.SGD(simple_net.parameters(), lr=0.001, momentum=0.9)
    simple_history = train_with_metrics(
        simple_net,
        train_loader,
        test_loader,
        criterion,
        simple_opt,
        device,
        epochs=5,
        log_interval=500,
    )

    # resnet
    resnet_net = model_mod.CNN(num_classes=num_classes).to(device)
    resnet_opt = optim.SGD(resnet_net.parameters(), lr=0.001, momentum=0.9)
    resnet_history = train_with_metrics(
        resnet_net,
        train_loader,
        test_loader,
        criterion,
        resnet_opt,
        device,
        epochs=5,
        log_interval=500,
    )

    # evaluate both models on test set with confusion/metrics
    print("\n== SimpleCNN evaluation ==")
    utils.evaluate(simple_net, test_loader, device)
    print("\n== ResNet evaluation ==")
    utils.evaluate(resnet_net, test_loader, device)

    # Combined 2x2 figure: rows = models, cols = plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    _plot_train_loss_vs_accuracy_on_ax(
        axes[0, 0], simple_history, "SimpleCNN: Train loss vs acc"
    )
    _plot_train_vs_val_accuracy_on_ax(
        axes[0, 1], simple_history, "SimpleCNN: Train vs val acc"
    )
    _plot_train_loss_vs_accuracy_on_ax(
        axes[1, 0], resnet_history, "ResNet: Train loss vs acc"
    )
    _plot_train_vs_val_accuracy_on_ax(
        axes[1, 1], resnet_history, "ResNet: Train vs val acc"
    )

    plt.tight_layout()
    plt.show()
