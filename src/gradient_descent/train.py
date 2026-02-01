import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import argparse

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import data_manager as data
from .model import Model
from .test import test_model
from . import config

def evaluate_accuracy(model: nn.Module, dataloader: DataLoader) -> float:
    """Compute and return the classification accuracy (0..1) on the dataset.

    Handles both integer class labels (1-D) and one-hot encoded labels (2-D).

    For small TensorDataset-backed dataloaders, prefer evaluating the model on the full
    dataset at once (to avoid batch-indexing quirks in synthetic tests). Falls back to
    batch-wise evaluation for general dataloaders.
    """
    # fast-path: if dataloader.dataset exposes tensors (e.g., TensorDataset), evaluate all at once
    dataset = getattr(dataloader, "dataset", None)
    try:
        if dataset is not None and hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
            images = dataset.tensors[0]
            labels = dataset.tensors[1]
            with torch.no_grad():
                outputs = model(images)
                _, predicted_classes = torch.max(outputs, 1)

                if labels.ndim == 1:
                    true_classes = labels
                elif labels.ndim == 2:
                    true_classes = labels.argmax(dim=1)
                else:
                    raise ValueError("Unsupported label shape for accuracy evaluation")

                total = true_classes.size(0)
                correct = (predicted_classes == true_classes).sum().item()

            return correct / total if total > 0 else 0.0
    except Exception:
        # fall back to batch-wise evaluation below
        pass

    correct, total = 0, 0

    # disable gradient computation for faster inference
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted_classes = torch.max(outputs, 1)

            # support both index labels (shape: [N]) and one-hot (shape: [N, C])
            if labels.ndim == 1:
                true_classes = labels
            elif labels.ndim == 2:
                true_classes = labels.argmax(dim=1)
            else:
                raise ValueError("Unsupported label shape for accuracy evaluation")

            total += true_classes.size(0)
            correct += (predicted_classes == true_classes).sum().item()

    return correct / total if total > 0 else 0.0

def average_epoch_and_loss_data(epoch_data: np.ndarray, loss_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average per-batch epoch and loss arrays into per-epoch values.

    Returns two 1-D arrays of length `config.NUM_EPOCHS`. If no batch data exists for an epoch,
    that epoch's loss is set to `np.nan` so plotting and downstream analysis can handle missing
    epochs gracefully.
    """
    # map fractional epoch indicators (e.g., 1.23) to integer epoch indices
    epoch_numbers = np.rint(epoch_data).astype(int)

    # if epochs are zero-based (0,1,...), shift to 1-based indexing
    if epoch_numbers.min() == 0:
        epoch_numbers = epoch_numbers + 1

    num_epochs = config.NUM_EPOCHS
    loss_data_average = np.full(num_epochs, np.nan, dtype=float)

    for e in range(1, num_epochs + 1):
        mask = epoch_numbers == e
        if np.any(mask):
            loss_data_average[e - 1] = loss_data[mask].mean()

    epoch_indices = np.arange(1, num_epochs + 1)

    return epoch_indices, loss_data_average

def plot_data(data_x: np.ndarray, data_y: np.ndarray, data_y2: np.ndarray | None = None) -> None:
    # primary axis: loss
    fig, ax1 = plt.subplots()
    ax1.plot(data_x, data_y, 'o--', color='blue', label='average cross entropy loss per epoch')
    ax1.set_xlabel('Epoch Number')
    ax1.set_ylabel('Cross Entropy (averaged per epoch)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # secondary axis: accuracy (optional)
    if data_y2 is not None:
        ax2 = ax1.twinx()
        ax2.plot(data_x, data_y2, 's-', color='green', label='accuracy per epoch')
        ax2.set_ylabel('Accuracy (0..1)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        # combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2)
    else:
        ax1.legend()

    # show can fail in headless environments; don't crash the program
    try:
        plt.show()
    except Exception as e:
        print(f"warning: plotting failed: {e}")

def train_model(data_loader: DataLoader, model: type[nn.Module], num_epochs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train `model` with SGD and record per-batch epoch progress and loss.

    Returns three numpy arrays: (epochs_per_batch, losses_per_batch, accuracies_per_epoch).
    """
    optimizer = SGD(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()

    losses = []
    epochs = []
    accuracies: list[float] = []

    # prefer to fail fast if no data is provided
    if hasattr(data_loader, "__len__") and len(data_loader) == 0:
        raise ValueError("data_loader is empty; cannot train without data")

    loader_len = len(data_loader) if hasattr(data_loader, "__len__") else None

    for epoch in range(num_epochs):
        print(f'training: epoch {epoch + 1}/{num_epochs}')
        for i, (images, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            
            loss_value = loss(model(images), labels)
            loss_value.backward()
            optimizer.step()

            # record approximate fractional epoch progress and loss value
            fraction = 1 / loader_len if loader_len else 0.0
            epochs.append(epoch + fraction)
            losses.append(loss_value.item())

        # evaluate accuracy at the end of the epoch
        acc = evaluate_accuracy(model, data_loader)
        accuracies.append(acc)

    return np.array(epochs), np.array(losses), np.array(accuracies)

def run_training_mode(model_path: str, dataloader: DataLoader) -> None:
    model = Model()

    try:
        print("training model...")
        epoch_data, loss_data, accuracy_data = train_model(dataloader, model, config.NUM_EPOCHS)
        epoch_indices, loss_data_average = average_epoch_and_loss_data(epoch_data, loss_data)

        print("plotting data...")
        plot_data(epoch_indices, loss_data_average, accuracy_data)
        
        print("testing model...")
        test_model(model, dataloader)
        
        print("saving model...")
        data.save_model(model_path, model)
    except Exception as e:
        print(f"error during training run: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main() -> None:
    dataloader = DataLoader(data.TRAIN_DATASET, batch_size=config.BATCH_SIZE)

    parser = argparse.ArgumentParser(description="Train a new MNIST model with gradient descent")
    parser.add_argument(
        "--name", "--n",
        type=str,
        help="Model name (without .pt). Saved to models directory"
    )

    args = parser.parse_args()

    if args.name is None:
        model_path = data.GENETIC_ALGORITHM_MODEL_PATH
    else:
        model_path = f"{data.MODEL_WEIGHTS_DIR}/{args.name}.pt"

    try:
        run_training_mode(model_path, dataloader)
    except Exception as e:
        print(f"fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()