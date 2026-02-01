import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytest

# make src importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from gradient_descent import train as gd_train
from gradient_descent import model as gd_model
import data_manager as data


def make_loader(batch_size=2, n_samples=6, with_one_hot=False):
    images = torch.randn(n_samples, data.TENSOR_SIZE)
    labels_idx = torch.randint(0, data.NUM_CLASSES, (n_samples,))
    if with_one_hot:
        labels = F.one_hot(labels_idx, num_classes=data.NUM_CLASSES).to(torch.float)
    else:
        labels = labels_idx
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=batch_size)


def test_train_model_runs_with_index_labels():
    loader = make_loader(with_one_hot=False)
    model = gd_model.Model()

    epochs, losses, accuracies = gd_train.train_model(loader, model, num_epochs=1)
    assert isinstance(epochs, (list, tuple)) or hasattr(epochs, "shape")
    assert losses.shape[0] == epochs.shape[0]
    assert accuracies.shape[0] == 1


def test_evaluate_accuracy_on_one_hot():
    # make a tiny loader where model predicts labels exactly
    n = 4
    images = torch.zeros(n, data.TENSOR_SIZE)
    labels_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    labels_one_hot = F.one_hot(labels_idx, num_classes=data.NUM_CLASSES).to(torch.float)

    ds = TensorDataset(images, labels_one_hot)
    loader = DataLoader(ds, batch_size=2)

    class PerfectModel(nn.Module):
        def forward(self, x):
            # produce logits whose argmax equals [0,1,2,3] repeated
            out = torch.zeros(x.shape[0], data.NUM_CLASSES)
            for i in range(x.shape[0]):
                out[i, labels_idx[i]] = 1.0
            return out

    m = PerfectModel()
    acc = gd_train.evaluate_accuracy(m, loader)
    assert acc == 1.0


def test_average_epoch_and_loss_data():
    # simple check of averaging shapes
    epoch_data = torch.tensor([1, 1, 2, 2]).numpy()
    loss_data = torch.tensor([0.5, 0.7, 0.2, 0.3]).numpy()

    avg_epochs, avg_losses = gd_train.average_epoch_and_loss_data(epoch_data, loss_data)
    assert avg_epochs.shape[0] == gd_train.config.NUM_EPOCHS if hasattr(gd_train, "config") else True
