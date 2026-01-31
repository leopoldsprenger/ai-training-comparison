import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import pytest

# make src importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from genetic_algorithm import train as ga_train
from genetic_algorithm import model as ga_model
import data_manager as data


def make_loader_one_hot(n_samples=6, batch_size=2):
    images = torch.randn(n_samples, data.TENSOR_SIZE)
    labels_idx = torch.randint(0, data.NUM_CLASSES, (n_samples,))
    labels = F.one_hot(labels_idx, num_classes=data.NUM_CLASSES).to(torch.float)
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=batch_size), labels_idx


def test_initialize_crossover_and_mutate():
    pop = ga_train.initialize_population(4, ga_model.Model)
    assert len(pop) == 4
    offspring = ga_train.crossover(pop[:2], 2)
    assert len(offspring) == 2

    # original params for comparison
    orig_params = [p.clone() for p in offspring[0].parameters()]
    mutated = ga_train.mutate([offspring[0]], mutation_rate=1.0, mutation_strength=0.1)
    # at least one parameter tensor should have changed
    changed = any(not torch.allclose(a, b) for a, b in zip(orig_params, mutated[0].parameters()))
    assert changed


def test_evaluate_population_with_custom_loss():
    loader, _ = make_loader_one_hot()
    pop = ga_train.initialize_population(3, ga_model.Model)

    # use a custom MSE loss that works with one-hot labels
    def mse_loss(outputs, labels):
        return ((outputs - labels) ** 2).mean()

    fitness, train_accs, test_accs = ga_train.evaluate_population(pop, loader, loader, mse_loss)
    assert len(fitness) == 3
    assert all(0 <= a <= float('inf') for a in fitness)
    assert len(train_accs) == 3


def test_genetic_algorithm_run_small(monkeypatch):
    # create loaders with index labels since GA uses CrossEntropyLoss internally
    n = 8
    images = torch.randn(n, data.TENSOR_SIZE)
    labels_idx = torch.randint(0, data.NUM_CLASSES, (n,))
    ds = TensorDataset(images, labels_idx)
    loader = DataLoader(ds, batch_size=2)

    # monkeypatch evaluate_accuracy to accept index labels
    def eval_acc_idx(model, dataloader):
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labs in dataloader:
                preds = model(imgs).argmax(dim=1)
                total += labs.size(0)
                correct += (preds == labs).sum().item()
        return correct / total

    monkeypatch.setattr(ga_train, 'evaluate_accuracy', eval_acc_idx)

    best, best_accs, avg_test_accs = ga_train.genetic_algorithm(
        loader, loader, ga_model.Model, num_generations=2, population_size=4, num_parents=2, mutation_rate=0.1, mutation_strength=0.01
    )

    assert isinstance(best, nn.Module)
    assert len(best_accs) == 2
