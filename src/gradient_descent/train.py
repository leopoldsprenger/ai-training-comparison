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
import config

from src.gradient_descent.model import Model
from test import test_model

def average_epoch_and_loss_data(epoch_data: np.ndarray, loss_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    plt.show()

def train_model(data_loader: DataLoader, model: type[nn.Module], num_epochs: int) -> tuple[np.ndarray, np.ndarray]:
    optimizer = SGD(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()

    losses = []
    epochs = []

    for epoch in range(num_epochs):
        print(f'Training: Epoch {epoch + 1}/{num_epochs}')
        for i, (images, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            
            loss_value = loss(model(images), labels)
            loss_value.backward()
            optimizer.step()

            epochs.append(epoch + 1 / len(data_loader))
            losses.append(loss_value.item())

    return np.array(epochs), np.array(losses)

def run_training_mode(model_path: str, dataloader: DataLoader) -> None:
    model = Model()

    print("Training model...")
    epoch_data, loss_data = train_model(dataloader, model, config.NUM_EPOCHS)
    epoch_data_average, loss_data_average = average_epoch_and_loss_data(epoch_data, loss_data)

    print("Plotting data...")
    plot_data(epoch_data_average, loss_data_average)
    
    print("Testing model...")
    test_model(model)
    
    print("Saving model...")
    data.save_model(model_path, model)

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

    run_training_mode(model_path, dataloader)

if __name__ == "__main__":
    main()