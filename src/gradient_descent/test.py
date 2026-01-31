import torch.nn as nn
import matplotlib.pyplot as plt
import argparse

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import data_manager as data
from neural_network import NeuralNetwork

def test_model(neural_network: nn.Module) -> None:
    test_images, _ = data.TEST_DATASET[0:40]
    test_label_predictions = neural_network(test_images).argmax(axis=1)

    figure, _ = plt.subplots(4, 10, figsize=(22.5, 15))

    for i in range(40):
        plt.subplot(4, 10, i + 1)

        img = test_images[i].view(data.IMAGE_SIZE, data.IMAGE_SIZE).detach().cpu().numpy()
        plt.imshow(img, cmap="gray")
        
        plt.title(f'Predicted Digit: {test_label_predictions[i]}')
    
    figure.tight_layout()
    plt.show()

def run_testing_mode(model_path: str) -> None:
    print("Loading model...")
    neural_network = data.load_model(
        model_path,
        NeuralNetwork()
    )

    print("Testing model...")
    test_model(neural_network)

def main() -> None:
    parser = argparse.ArgumentParser(description="Test a trained MNIST model with gradient descent")
    parser.add_argument(
        "--name",
        type=str,
        help="Model name (without .pt). Loaded from models directory"
    )

    args = parser.parse_args()

    if args.name is None:
        model_path = data.GENETIC_ALGORITHM_MODEL_PATH
    else:
        model_path = f"{data.MODEL_WEIGHTS_DIR}/{args.name}.pt"

    run_testing_mode(model_path)

if __name__ == "__main__":
    main()