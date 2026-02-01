# OCR AI Methods Comparison

**German translation:** see `docs/README_de.md` 📘

This repository implements and evaluates two lightweight feed-forward classifiers on the MNIST dataset using two distinct training paradigms: Genetic Algorithm (GA) and Gradient Descent (GD). 

It emphasizes reproducibility, configurable experiments, and clear evaluation metrics, making it suitable for research, benchmarking, and educational purposes.

## Table of Contents
- [Overview](#overview)
- [Why This Project 📝](#why-this-project-)
- [Quickstart & Installation 🔧](#quickstart--installation-)
- [Running: train and test individual models 🧪](#running-train-and-test-individual-models-)
- [Comparing GA vs GD programmatically ⚖️](#comparing-ga-vs-gd-programmatically-)
- [Visual results 📊](#visual-results-)
- [Data Pipeline](#data-pipeline)
- [License](#license)

## Overview

This repository contains two main approaches to train small feed-forward classifiers on MNIST:

- **Genetic Algorithm (GA)**: A population-based search that evolves model parameters over generations.
- **Gradient Descent (GD)**: A conventional neural network trained with stochastic gradient descent.

## Why This Project 📝

I chose this project as my 9th-grade thesis to investigate a fundamental question in AI: whether different training paradigms lead to equivalent outcomes, and if not, how their efficiency and performance differ. This project allowed me to implement two distinct methods, Genetic Algorithms and Gradient Descent, in a reproducible, modular framework and to compare their behavior and results on the MNIST dataset.

## Quickstart & Installation 🔧

Clone the repository and install dependencies:

```bash
git clone https://github.com/leopoldsprenger/ai-training-comparison.git
cd ai-training-comparison
pip install -r requirements.txt
```

Note: this README avoids pinning a specific Python version. Ensure that you have a working Python environment with PyTorch installed (see `requirements.txt`).

Run unit tests with:

```bash
pytest -q
```

## Running: train and test individual models 🧪

The training and testing routines expose simple command-line interfaces (argparse). There are two ways to invoke them: as module scripts (recommended) or by running the file directly.

Examples (module form):

- Train the Genetic Algorithm model:
  ```bash
  python3 src/genetic_algorithm/train.py --name model_name
  ```

- Test a saved Genetic Algorithm model:
  ```bash
  python3 src/genetic_algorithm/test.py --name model_name
  ```

- Train with Gradient Descent:
  ```bash
  python3 src/gradient_descent/train.py --name model_name
  ```

- Test a saved Gradient Descent model:
  ```bash
  python3 -m src/gradient_descent/test.py --name model_name
  ```

Argparse flags:
- `--name` / `-n`  : model base name (saved/loaded from `models/` directory; omit to use default paths)
- `--generations` / `-g` (used by the comparison script) : number of generations / epochs to run

## Comparing GA vs GD programmatically ⚖️

A utility script compares a GA run to a GD run, saves both models, and plots accuracies:

```bash
python -m src.utils.compare_trainings --name experiment1 --generations 50
```

Flags:
- `--name` / `-n` (required): base name used for saved model files (no extension). The script writes to `models/{name}_ga.pt` and `models/{name}_gd.pt`.
- `--generations` / `-g` (optional): number of generations (GA) or epochs (GD) to execute.

## Visual results 📊

After training, inspect the `imgs/` directory for saved figures. Below are a few representative results with short captions to explain what each figure illustrates.

| GA - after hyperparameter adjustments | GA - 50 generations |
|--------------------------------------|--------------------|
| ![GA after hyperparameter](imgs/genetic_algorithm/test_3_after_adjusting_hyper_parameters.png) | ![GA 50 generations](imgs/genetic_algorithm/test_4_test_with_50_gens.png) |
| Demonstrates early generation behavior after tuning selection and mutation parameters. | Best and average accuracy progression across 50 generations. |

| GD - Cross-entropy per epoch | GD - Test dataset predictions |
|------------------------------|------------------------------|
| ![Cross entropy per epoch](imgs/gradient_descent/cross_entropy_per_epochs_graph.png) | ![GD predictions](imgs/gradient_descent/gradient_descent_test_dataset_predictions.png) |
| Useful to assess convergence behaviour and stability. | Example predictions on the test dataset (first 40 images) with predicted labels. |

![Comparison](imgs/accuracy_comparison_gradient_descent_and_genetic_algorithm.png)  
*Comparison — GA (best), GA (average), and GD (per-epoch) accuracies plotted together for direct comparison.*

## Data Pipeline

The `src/data_manager.py` module provides a small, defensive dataset abstraction used throughout the project. The key elements are:

- `MNISTDataset(filepath)` : expects a saved `(images, labels)` tuple in a torch `.pt` file. The constructor validates file existence and types and raises informative errors on invalid input.
- `images` : tensors are reshaped to `(N, TENSOR_SIZE)` and normalized to the range [0, 1].
- `labels` : converted to integer dtype and one-hot encoded as float tensors (`NUM_CLASSES = 10`).
- `TRAIN_DATASET` and `TEST_DATASET` : module-level objects constructed from `data/processed/training.pt` and `data/processed/test.pt` (the module falls back to empty lists with warnings if loading fails).

Constants exposed: `IMAGE_SIZE`, `TENSOR_SIZE`, `NUM_CLASSES`, and model-weight path helpers such as `MODEL_WEIGHTS_DIR`.

To use the datasets in training or evaluation, create a DataLoader as shown in the codebase:

```python
from torch.utils.data import DataLoader
import data_manager as data
train_loader = DataLoader(data.TRAIN_DATASET, batch_size=32)
```

Note on model configuration: the two training pipelines use different model classes and algorithmic settings that were chosen after empirical exploration. For reproducible and systematic experiments, algorithm hyperparameters are declared in `src/genetic_algorithm/config.py` and `src/gradient_descent/config.py`; researchers are encouraged to explore these configuration values to investigate sensitivity and convergence phenomena ("play around with the config" to perform controlled studies).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.