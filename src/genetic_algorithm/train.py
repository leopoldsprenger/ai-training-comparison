import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import random
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
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted_classes = torch.max(outputs, 1)
            true_classes = labels.argmax(dim=1)
            
            total += labels.size(0)
            correct += (predicted_classes == true_classes).sum().item()

    return correct / total

def initialize_population(population_size: int, model_class: type[nn.Module]) -> list[nn.Module]:
    """Create a population of models with random normal-initialized parameters."""
    population = []
    for _ in range(population_size):
        individual = model_class()
        
        # initialize each parameter with standard normal noise
        for parameter in individual.parameters():
            parameter.data = torch.randn_like(parameter)
        
        population.append(individual)
    
    return population

def evaluate_population(population: list[nn.Module], train_loader: DataLoader, test_loader: DataLoader, loss_function: nn.Module) -> tuple[list[float], list[float], list[float]]:
    """Evaluate each individual and return (fitness_scores, train_accs, test_accs).

    Fitness is computed as average training loss (lower is better).
    """
    fitness_scores = []
    train_accuracies = []
    test_accuracies = []

    for individual in population:
        losses = [
            loss_function(individual(images), labels).item()
            for images, labels in train_loader
        ]
        average_loss = sum(losses) / len(losses)

        fitness_scores.append(average_loss)
        train_accuracies.append(evaluate_accuracy(individual, train_loader))
        test_accuracies.append(evaluate_accuracy(individual, test_loader))

    return fitness_scores, train_accuracies, test_accuracies

def select_parents(population: list[nn.Module], fitness_scores: list[float], num_parents: int) -> list[nn.Module]:
    def get_fitness(pair):
        return pair[0]

    scored_population = list(zip(fitness_scores, population))
    scored_population.sort(key=get_fitness)

    return [individual for _, individual in scored_population[:num_parents]]

def crossover(parents: list[nn.Module], num_offspring: int) -> list[nn.Module]:
    """Create `num_offspring` children using one-point crossover of flattened parameters."""
    offspring = []

    for _ in range(num_offspring):
        parent_1, parent_2 = random.choices(parents, k=2)
        child = Model()

        for param1, param2, child_param in zip(parent_1.parameters(), parent_2.parameters(), child.parameters()):
            flat_param1 = param1.detach().view(-1)
            flat_param2 = param2.detach().view(-1)

            # one-point crossover along flattened parameter vector
            crossover_point = random.randint(0, flat_param1.numel())
            flat_child = torch.cat([flat_param1[:crossover_point], flat_param2[crossover_point:]])

            child_param.data = flat_child.view_as(param1)

        offspring.append(child)

    return offspring

def mutate(individuals: list[nn.Module], mutation_rate: float, mutation_strength: float) -> list[nn.Module]:
    """Apply Gaussian perturbations to parameters with probability `mutation_rate`.

    Perturbations are scaled by `mutation_strength`.
    """
    for individual in individuals:
        for parameter in individual.parameters():
            if random.random() < mutation_rate:
                parameter.data += torch.randn_like(parameter) * mutation_strength
    
    return individuals

def genetic_algorithm(
        train_loader: DataLoader, 
        test_loader: DataLoader, 
        model: type[nn.Module], 
        num_generations: int, 
        population_size: int, 
        num_parents: int, 
        mutation_rate: float, 
        mutation_strength: float
    ) -> tuple[nn.Module, list[float], list[float]]:
    """Run the genetic algorithm and return the best model and accuracy histories.

    Returns (best_individual, best_accuracies, average_test_accuracies).
    """
    loss_function = nn.CrossEntropyLoss()

    population = initialize_population(population_size, model)
    best_individual = None

    best_accuracies = []
    average_train_accuracies = []
    average_test_accuracies = []
 
    for generation in range(num_generations):
        print(f"training: generation {generation + 1}/{num_generations}")

        fitness_scores, train_accuracies, test_accuracies = evaluate_population(population, train_loader, test_loader, loss_function)

        best_index = test_accuracies.index(max(test_accuracies))
        current_best_individual = population[best_index]

        # update global best if current generation produced a better test accuracy
        if best_individual is None or test_accuracies[best_index] > max(best_accuracies, default=0):
            best_individual = current_best_individual

        best_accuracies.append(test_accuracies[best_index])
        average_train_accuracies.append(np.mean(train_accuracies))
        average_test_accuracies.append(np.mean(test_accuracies))

        parents = [best_individual] + select_parents(population, fitness_scores, num_parents - 1)

        offspring = crossover(parents, population_size - num_parents)        
        offspring = mutate(offspring, mutation_rate, mutation_strength)

        population = [best_individual] + offspring

    return best_individual, best_accuracies, average_test_accuracies

def plot_accuracies_data(best_accuracies: list[float], average_accuracies: list[float]) -> None:
    """Plot best and average accuracies across generations (blocking)."""
    plt.figure(figsize=(10, 6))

    plt.plot(range(len(best_accuracies)), best_accuracies, 'o--', color='blue', label='Best Accuracy')
    plt.plot(range(len(average_accuracies)), average_accuracies, 'o--', color='red', label='Average Accuracy')

    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.title('Best and Average Accuracy per Generation')
    
    plt.legend()
    plt.show()

def run_training_mode(model_path: str, train_dataloader: DataLoader, test_dataloader: DataLoader) -> None:
    """Execute the genetic algorithm workflow and persist the best model to `model_path`."""
    print("training model...")
    best_model, best_accuracies, average_accuracies = genetic_algorithm(
        train_dataloader, test_dataloader,
        Model, 
        config.NUM_GENERATIONS,
        config.POPULATION_SIZE,
        config.NUM_PARENTS,
        config.MURATION_RATE,
        config.MUTATION_STRENGTH
    )

    print("plotting best and average accuracies...")
    plot_accuracies_data(best_accuracies, average_accuracies)

    print("testing model...")
    test_model(best_model, test_dataloader)

    print("saving model...")
    data.save_model(model_path, best_model)

def main() -> None:
    train_dataloader = DataLoader(data.TRAIN_DATASET, batch_size=config.BATCH_SIZE)
    test_dataloader = DataLoader(data.TEST_DATASET, batch_size=config.BATCH_SIZE)

    parser = argparse.ArgumentParser(description="Test a trained MNIST model")
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

    run_training_mode(model_path, train_dataloader, test_dataloader)

if __name__ == "__main__":
    main()