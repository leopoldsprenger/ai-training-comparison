from pathlib import Path
import sys
import argparse
from typing import Optional

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ensure top-level `src/` is importable
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import data_manager as data
from gradient_descent import train as gd_train
from genetic_algorithm import train as ga_train
import gradient_descent.config as gd_config
import genetic_algorithm.config as ga_config


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GA vs GD training runs and save models")
    parser.add_argument("--name", "--n", type=str, required=True, help="Base name for saved models (no extension)")
    parser.add_argument(
        "--generations", "--g",
        type=int,
        default=ga_config.NUM_GENERATIONS,
        help="Number of generations (GA) or epochs (GD) to run"
    )

    return parser.parse_args(argv)


def make_dataloaders(batch_size: int):
    if not hasattr(data, "TRAIN_DATASET") or len(data.TRAIN_DATASET) == 0:
        raise RuntimeError("TRAIN_DATASET not available or empty")

    if not hasattr(data, "TEST_DATASET") or len(data.TEST_DATASET) == 0:
        raise RuntimeError("TEST_DATASET not available or empty")

    train_loader = DataLoader(data.TRAIN_DATASET, batch_size=batch_size)
    test_loader = DataLoader(data.TEST_DATASET, batch_size=batch_size)

    return train_loader, test_loader


def run_compare(name: str, generations: int) -> None:
    print(f"running comparison: name={name}, generations={generations}")

    # dataloaders (use gradient descent batch size for GD and GA for consistency)
    train_loader, test_loader = make_dataloaders(batch_size=gd_config.BATCH_SIZE)

    # --- Genetic Algorithm ---
    print("starting genetic algorithm training...")
    best_model, best_accuracies, average_accuracies = ga_train.genetic_algorithm(
        train_loader,
        test_loader,
        ga_train.Model,
        num_generations=generations,
        population_size=ga_config.POPULATION_SIZE,
        num_parents=ga_config.NUM_PARENTS,
        mutation_rate=ga_config.MURATION_RATE,
        mutation_strength=ga_config.MUTATION_STRENGTH,
    )

    # --- Gradient Descent ---
    print("starting gradient descent training...")
    gd_model = gd_train.Model()
    _, _, gd_accuracies = gd_train.train_model(train_loader, gd_model, num_epochs=generations)

    # --- Save models ---
    ga_path = f"{data.MODEL_WEIGHTS_DIR}/{name}_ga.pt"
    gd_path = f"{data.MODEL_WEIGHTS_DIR}/{name}_gd.pt"

    print(f"saving GA model to {ga_path}")
    data.save_model(ga_path, best_model)

    print(f"saving GD model to {gd_path}")
    data.save_model(gd_path, gd_model)

    # --- Plot comparison ---
    # x-axis for GA (generations) and GD (epochs)
    ga_x = np.arange(1, len(best_accuracies) + 1)
    gd_x = np.arange(1, len(gd_accuracies) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(ga_x, best_accuracies, 'o--', color='blue', label='GA: Best Accuracy')
    plt.plot(ga_x, average_accuracies, 's--', color='red', label='GA: Average Accuracy')
    plt.plot(gd_x, gd_accuracies, 'd-', color='green', label='GD: Accuracy per Epoch')

    plt.xlabel('Generation / Epoch')
    plt.ylabel('Accuracy (0..1)')
    plt.title(f'GA vs GD comparison for "{name}"')
    plt.legend()

    try:
        plt.show()
    except Exception as e:
        print(f"warning: plotting failed: {e}")


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    run_compare(args.name, args.generations)


if __name__ == "__main__":
    main()
