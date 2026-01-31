import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random

import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import config
import data_manager as data

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Linear(data.TENSOR_SIZE, 100)
        self.layer2 = nn.Linear(100, 50)
        self.layer3 = nn.Linear(50, data.NUM_CLASSES)
        
        self.relu = nn.ReLU()

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        input_images = input_images.view(-1, data.TENSOR_SIZE)

        input_images = self.relu(self.layer1(input_images))
        input_images = self.relu(self.layer2(input_images))

        return self.layer3(input_images).squeeze()

def evaluate_accuracy(model: nn.Module, data_loader: DataLoader) -> float:
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted_classes = torch.max(outputs, 1)
            true_classes = labels.argmax(dim=1)
            
            total += labels.size(0)
            correct += (predicted_classes == true_classes).sum().item()

    return correct / total

def initialize_population(population_size: int, model_class: type[nn.Module]) -> list[nn.Module]:
    population = []
    for _ in range(population_size):
        individual = model_class()
        
        for parameter in individual.parameters():
            parameter.data = torch.randn_like(parameter)
        
        population.append(individual)
    
    return population

def evaluate_population(population: list[nn.Module], train_loader: DataLoader, test_loader: DataLoader, loss_function: nn.Module) -> tuple[list[float], list[float], list[float]]:
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
    offspring = []

    for _ in range(num_offspring):
        parent_1, parent_2 = random.choices(parents, k=2)
        child = NeuralNetwork()

        for param1, param2, child_param in zip(parent_1.parameters(), parent_2.parameters(), child.parameters()):
            flat_param1 = param1.detach().view(-1)
            flat_param2 = param2.detach().view(-1)

            crossover_point = random.randint(0, flat_param1.numel())
            flat_child = torch.cat([flat_param1[:crossover_point], flat_param2[crossover_point:]])

            child_param.data = flat_child.view_as(param1)

        offspring.append(child)

    return offspring

def mutate(individuals: list[nn.Module], mutation_rate: float, mutation_strength: float) -> list[nn.Module]:
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
    loss_function = nn.CrossEntropyLoss()

    population = initialize_population(population_size, model)
    best_individual = None

    best_accuracies = []
    average_train_accuracies = []
    average_test_accuracies = []
 
    for generation in range(num_generations):
        print(f"Training: Generation {generation + 1}/{num_generations}")

        fitness_scores, train_accuracies, test_accuracies = evaluate_population(population, train_loader, test_loader, loss_function)

        best_index = test_accuracies.index(max(test_accuracies))
        current_best_individual = population[best_index]

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
    plt.figure(figsize=(10, 6))

    plt.plot(range(len(best_accuracies)), best_accuracies, 'o--', color='blue', label='Best Accuracy')
    plt.plot(range(len(average_accuracies)), average_accuracies, 'o--', color='red', label='Average Accuracy')

    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.title('Best and Average Accuracy per Generation')
    
    plt.legend()
    plt.show()

def test_model(neural_network: nn.Module) -> None:
    test_images, _ = data.TEST_DATASET[0:40]
    predictions = neural_network(test_images).argmax(axis=1)
    
    figure, _ = plt.subplots(4, 10, figsize=(22.5, 15))

    for i in range(40):
        plt.subplot(4, 10, i + 1)
        plt.imshow(test_images[i])
        plt.title(f'Predicted Digit: {predictions[i]}')
    
    figure.tight_layout()
    plt.show()

def run_training_mode(train_dataloader: DataLoader, test_dataloader: DataLoader) -> None:
    while True:
        try:
            num_generations = int(input('How many generations should the model train for: '))
            break
        except ValueError:
            print('Generation number was not valid. Please try again...')

    print("Training model...")
    best_model, best_accuracies, average_accuracies = genetic_algorithm(
        train_dataloader, test_dataloader,
        NeuralNetwork, num_generations,
        config.POPULATION_SIZE,
        config.NUM_PARENTS,
        config.MURATION_RATE,
        config.MUTATION_STRENGTH
    )

    print("Plotting best and average accuracies...")
    plot_accuracies_data(best_accuracies, average_accuracies)

    print("Testing model...")
    test_model(best_model)

    print("Saving model...")
    data.save_model(
        data.GENETIC_ALGORITHM_MODEL_PATH, 
        best_model
    )

def run_load_mode() -> None:
    print("Loading model...")
    neural_network = data.load_model(
        data.GENETIC_ALGORITHM_MODEL_PATH,
        NeuralNetwork()
    )

    print("Testing model...")
    test_model(neural_network)

def main() -> None:
    train_dataloader = DataLoader(data.TRAIN_DATASET, batch_size=config.BATCH_SIZE)
    test_dataloader = DataLoader(data.TEST_DATASET, batch_size=config.BATCH_SIZE)

    while True:
        mode = input(
            'Train and test model with genetic algorithm: 0\n'
            'Load and test existing model: 1\n'
            'Which mode would you like to do: '
        )

        match mode:
            case '0':
                run_training_mode(train_dataloader, test_dataloader)
                break
            case '1':
                run_load_mode()
                break
            case _:
                print("Input wasn't accepted. Please try again.")

if __name__ == '__main__':
    main()
