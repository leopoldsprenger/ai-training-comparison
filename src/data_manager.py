import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, filepath):
        """Load and validate a saved (images, labels) dataset.

        Raises FileNotFoundError, IOError, or ValueError on invalid input.
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        try:
            data = torch.load(filepath)
        except Exception as e:
            raise IOError(f"Failed to load dataset from {filepath}: {e}")

        if not (isinstance(data, (list, tuple)) and len(data) == 2):
            raise ValueError("Expected dataset file to contain (images, labels) tuple")

        images, labels = data

        if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("Loaded dataset must contain torch.Tensors for images and labels")

        # reshape to (N, TENSOR_SIZE) and normalize to 0..1
        self.images = images.view(-1, TENSOR_SIZE).float() / 255.0

        # ensure label dtype is integer before one-hot encoding
        if labels.dtype != torch.long:
            labels = labels.long()
        self.labels = F.one_hot(labels, num_classes=NUM_CLASSES).to(torch.float)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, image_index):
        return self.images[image_index], self.labels[image_index]

def save_model(save_path: str, model: nn.Module) -> None:
    """Save model state_dict to `save_path`, creating parent directories as needed."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    try:
        torch.save(model.state_dict(), save_path)
        print(f"saved model to path: {save_path}")
    except Exception as e:
        raise IOError(f"Failed to save model to {save_path}: {e}")

def load_model(load_path: str, model: nn.Module) -> nn.Module:
    """Load model weights from `load_path` into `model` and return the model.

    Raises FileNotFoundError or IOError on failure.
    """
    if not Path(load_path).exists():
        raise FileNotFoundError(f"Model weights not found: {load_path}")

    try:
        state = torch.load(load_path)
    except Exception as e:
        raise IOError(f"Failed to load model from {load_path}: {e}")

    if not isinstance(state, dict):
        raise ValueError("Loaded object is not a state_dict (expected dict)")

    model.load_state_dict(state)
    model.eval()
    return model

IMAGE_SIZE = 28
TENSOR_SIZE = IMAGE_SIZE**2
NUM_CLASSES = 10

MODEL_WEIGHTS_DIR = 'models'
GENETIC_ALGORITHM_MODEL_PATH = f"{MODEL_WEIGHTS_DIR}/genetic_algorithm.pt"
GRADIENT_DESCENT_MODEL_PATH = f"{MODEL_WEIGHTS_DIR}/gradient_descent.pt"

# ensure model dir exists (harmless if it already does)
os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)

# Data downloaded from www.di.ens.fr/~lelarge/MNIST.tar.gz
# Construct datasets defensively so import-time failures don't crash the module
try:
    TRAIN_DATASET = MNISTDataset('data/processed/training.pt')
except Exception as e:
    warnings.warn(f"Failed to load training dataset: {e}")
    TRAIN_DATASET = []

try:
    TEST_DATASET = MNISTDataset('data/processed/test.pt')
except Exception as e:
    warnings.warn(f"Failed to load test dataset: {e}")
    TEST_DATASET = []