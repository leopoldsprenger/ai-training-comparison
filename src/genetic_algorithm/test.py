import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import argparse

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import data_manager as data
from .model import Model
from . import config

def test_model(model: nn.Module, dataloader: DataLoader) -> None:
    """Print accuracy and display the first 40 test images with predicted labels."""
    from .train import evaluate_accuracy
    
    # print testing accuracy
    accuracy = evaluate_accuracy(model, dataloader)
    print(f"accuracy: {round(accuracy*100, 2)} %")

    # graph testing plot
    test_images, _ = data.TEST_DATASET[0:40]
    predictions = model(test_images).argmax(axis=1)
    
    figure, _ = plt.subplots(4, 10, figsize=(22.5, 15))

    for i in range(40):
        plt.subplot(4, 10, i + 1)
        
        img = test_images[i].view(data.IMAGE_SIZE, data.IMAGE_SIZE).detach().cpu().numpy()
        plt.imshow(img, cmap="gray")

        plt.title(f'Predicted Digit: {predictions[i]}')
    
    figure.tight_layout()
    plt.show()

def run_testing_mode(model_path: str, dataloader: DataLoader) -> None:
    print("loading model...")
    try:
        model = data.load_model(model_path, Model())
    except Exception as e:
        print(f"error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        print("testing model...")
        test_model(model, dataloader)
    except Exception as e:
        print(f"error while testing model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main() -> None:
    test_dataloader = DataLoader(data.TEST_DATASET, batch_size=config.BATCH_SIZE)

    parser = argparse.ArgumentParser(description="Test a trained MNIST model with the genetic algorithm")
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
        run_testing_mode(model_path, test_dataloader)
    except Exception as e:
        print(f"fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()