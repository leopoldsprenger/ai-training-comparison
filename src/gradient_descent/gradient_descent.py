# import all the libraries needed
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import config
import data_manager as data

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.Matrix1 = nn.Linear(data.TENSOR, 100)
        self.Matrix2 = nn.Linear(100, 50)
        self.Matrix3 = nn.Linear(50, data.NUM_CLASSES)

        self.rectifier = nn.ReLU()

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        input_images = input_images.view(-1, data.TENSOR_SIZE)

        input_images = self.rectifier(self.Matrix1(input_images))
        input_images = self.rectifier(self.Matrix2(input_images))
        input_images = self.Matrix3(input_images)

        return input_images.squeeze()

def train_model(data_loader: DataLoader, neural_network: type[nn.Module], num_epochs: int) -> tuple[np.ndarray, np.ndarray]:
    optimizer = SGD(neural_network.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()

    losses = []
    epochs = []

    for epoch in range(num_epochs):
        print(f'Training: Epoch {epoch + 1}/{num_epochs}')
        for i, (images, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            
            loss_value = loss(neural_network(images), labels)
            loss_value.backward()
            optimizer.step()

            epochs.append(epoch + 1 / len(data_loader))
            losses.append(loss_value.item())

    return np.array(epochs), np.array(losses)

def test_model(neural_network: nn.Module) -> None:
    test_images, test_labels = data.TEST_DATASET[0:40]
    test_label_predictions = neural_network(test_images).argmax(axis=1)

    figure, axis = plt.subplots(4, 10, figsize=(22.5, 15))

    for i in range(40):
        plt.subplot(4, 10, i + 1)
        plt.imshow(test_images[i])
        plt.title(f'Predicted Digit: {test_label_predictions[i]}')
    
    figure.tight_layout()
    plt.show()

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

def run_training_mode(train_dataloader: DataLoader) -> None:
    neural_network = NeuralNetwork()

    print("Training model...")
    epoch_data, loss_data = train_model(train_dataloader, neural_network, config.NUM_EPOCHS)
    epoch_data_average, loss_data_average = average_epoch_and_loss_data(epoch_data, loss_data)

    print("Plotting data...")
    plot_data(epoch_data_average, loss_data_average)
    
    print("Testing model...")
    test_model(neural_network)
    
    print("Saving model...")
    data.save_model(neural_network, data.GRADIENT_DESCENT_MODEL_PATH)

def run_load_mode() -> None:
    print("Loading model...")
    neural_network = data.load_model(
        data.GRADIENT_DESCENT_MODEL_PATH,
        NeuralNetwork()
    )

    print("Testing model...")
    test_model(neural_network)

def main() -> None:
    train_dataloader = DataLoader(data.TRAIN_DATASET, batch_size=config.BATCH_SIZE)

    while True:
        mode = input(
            'Train and test model with gradient descent: 0\n'
            'Load and test existing model: 1\n'
            'Which mode would you like to do: '
        )

        match mode:
            case '0':
                run_training_mode(train_dataloader)
            case '1':
                run_load_mode()
            case _:
                print("Input wasn't accepted. Please try again.")

if __name__ == '__main__':
    main()
