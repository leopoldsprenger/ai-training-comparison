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
from model import Model
from test import test_model
import config

def evaluate_accuracy(model: nn.Module, dataloader: DataLoader) -> float:
    """Compute and return the classification accuracy (0..1) on the dataset."""
    correct, total = 0, 0

    # disable gradient computation for faster inference
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted_classes = torch.max(outputs, 1)
            true_classes = labels.argmax(dim=1)
            
            total += labels.size(0)
            correct += (predicted_classes == true_classes).sum().item()

    return correct / total

def average_epoch_and_loss_data(epoch_data: np.ndarray, loss_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average per-batch epoch and loss arrays into per-epoch values.

    Returns two 1-D arrays of length `config.NUM_EPOCHS`.
    """
    epoch_data_average = epoch_data.reshape(config.NUM_EPOCHS, -1)
    loss_data_average = loss_data.reshape(config.NUM_EPOCHS, -1)
    
    epoch_data_average = epoch_data_average.mean(axis=1)
    loss_data_average = loss_data_average.mean(axis=1)
    
    return epoch_data_average, loss_data_average

def plot_data(data_x: np.ndarray, data_y: np.ndarray) -> None:
    plt.figure(1)

    plt.plot(data_x, data_y, 'o--', color='blue', label='average cross entropy loss per epoch')
    
    plt.xlabel('Epoch Number')
    plt.ylabel('Cross Entropy (averaged per epoch)')
    plt.title('Cross Entropy (averaged per epoch)')
    
    plt.legend()
    # show can fail in headless environments; don't crash the program
    try:
        plt.show()
    except Exception as e:
        print(f"warning: plotting failed: {e}")

def train_model(data_loader: DataLoader, model: type[nn.Module], num_epochs: int) -> tuple[np.ndarray, np.ndarray]:
    """Train `model` with SGD and record per-batch epoch progress and loss.

    Returns two numpy arrays: (epochs, losses) collected per batch.
    """
    optimizer = SGD(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()

    losses = []
    epochs = []

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

    return np.array(epochs), np.array(losses)

def run_training_mode(model_path: str, dataloader: DataLoader) -> None:
    model = Model()

    try:
        print("training model...")
        epoch_data, loss_data = train_model(dataloader, model, config.NUM_EPOCHS)
        epoch_data_average, loss_data_average = average_epoch_and_loss_data(epoch_data, loss_data)

        print("plotting data...")
        plot_data(epoch_data_average, loss_data_average)
        
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
        "--name",
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