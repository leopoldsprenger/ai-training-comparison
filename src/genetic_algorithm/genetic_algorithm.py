import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import data_manager

class NeuralNetwork(nn.Module):
    def __init__(self):
        # inherite everything from nn.Module
        super().__init__()
        # define the neural net
        # first layer has 784 neurons and passes the values onto 100 neurons in the second layer
        self.layer1 = nn.Linear(28**2, 100)
        # second layer has 100 neurons
        # it processes data from the input layer and then sends the values to 50 new neurons
        self.layer2 = nn.Linear(100, 50)
        # same as second layer but it has 50 neurons, passes it to 10 output neurons
        # there are 10 output neurons because we have 10 different digits it can be
        self.layer3 = nn.Linear(50, 10)
        # define the rectified linear unit (ReLU)
        self.relu = nn.ReLU()

    def forward(self, input_images):
        # convert the tensor into an array as mentioned previously
        input_images = input_images.view(-1, 28**2)
        # pass the images through the layers
        # use ReLU to make sure that only meaningful output gets passed to next neuron
        input_images = self.relu(self.layer1(input_images))
        input_images = self.relu(self.layer2(input_images))
        # return the predictions of the Neural Network
        # use squeeze to remove unnecessary dimensions from the tensor
        return self.layer3(input_images).squeeze()

def evaluate_accuracy(model, data_loader):
    # define the total number of images and the number of images the model has correctly predicted
    correct, total = 0, 0
    # the the model not to calculate or store the gradient
    # this is because we only want to evaluate the model, not train it
    # this still has to be defined, as pytorch is built more towards gradient-based optimization than genetic algorithms
    with torch.no_grad():
        # loop through all images and labels (which are pairs) in the dataloader
        for images, labels in data_loader:
            # get the tensor output prediction of the model (i.e.: [0.2, 0.4, 0.7, 0.2, ...])
            outputs = model(images)
            # get the highest value for the output
            # _ is the value of the highest output, which we don't care about
            # predicted is a list of indices for what class has the highest value
            _, predicted = torch.max(outputs, 1)
            # add the amount of labels or images to the total by taking the size of the first dimension
            total += labels.size(0)
            # add the number of correct predictions to the correct counter
            # to do this: find a list of the highest values for labels (1 wherever the ground truth is)
            # compare the predicted list with the list of ground truths
            # sum that comparison to get one number and run item() to turn it from a tensor to an int
            correct += (predicted == labels.argmax(dim=1)).sum().item()
    # return a fraction of the correct / total values (i.e. 0.7 is 70 % accuracy)
    return correct / total

def initialize_population(population_size, model_class=NeuralNetwork):
    # define a list of individuals as the population
    population = []
    # for every number in the population_size
    for _ in range(population_size):
        # create a new neural network
        individual = model_class()
        # Randomize the weights using a normal distribution
        for parameter in individual.parameters():
            parameter.data = torch.randn_like(parameter)
        # add the neural_network to the population list
        population.append(individual)
    # return the list of the population
    return population

def evaluate_population(population, train_loader, test_loader, loss_function):
    # define a list of fitness scortes, train and test accuracies
    fitness_scores = []
    train_accuracies = []
    test_accuracies = []
    # loop through every individual in the population
    for individual in population:
        # return the sum of the loss function differences between the models prediction and the actual labels
        total_loss = sum(loss_function(individual(images), labels).item() for images, labels in train_loader)
        # calculate the average loss
        average_loss = total_loss / len(train_loader)
        # add the average loss to the fitness scores
        fitness_scores.append(average_loss)
        # get a list of all of the training accuracies of all of the models
        train_accuracies.append(evaluate_accuracy(individual, train_loader))
        # get a list of all of the test accuracies of all of the models
        # we evaluate on test data as well, since we want to test generalization
        # note that the test data is not used for training the model
        test_accuracies.append(evaluate_accuracy(individual, test_loader))
    # return the lists, which are now filled with data
    return fitness_scores, train_accuracies, test_accuracies

def select_parents(population, fitness_scores, num_parents):
    # creates a dictionary (list of pairs) for the fitness scores and their corresponding individual
    # sorts that population based on argument 0 (the fitness scores)
    sorted_population = sorted(zip(fitness_scores, population), key=lambda x: x[0])
    # takes in the first few (and also best) individuals from the list based on the number of parents
    # return only the neural network associated with the best scores, the scores aren't needed
    return [individual for _, individual in sorted_population[:num_parents]]

def crossover(parents, num_offspring):
    # define a list of all of the offspring or children
    offspring = []
    # loop through the following for how many times we wan't offsprings'
    for _ in range(num_offspring):
        # select to random parents from the parents list
        parent_1, parent_2 = random.choices(parents, k=2)
        # define a new neural network as the child
        child = NeuralNetwork()
        # perform crossover for the parameters of each layer separately
        for parameter_1, parameter_2, child_parameter in zip(parent_1.parameters(), parent_2.parameters(), child.parameters()):
            # detach parameters from the computational graph to not track the gradients during operations
            # flatten the tensor to perform crossover in a sequence
            flat_parameter_1 = parameter_1.detach().view(-1)
            flat_parameter_2 = parameter_2.detach().view(-1)
            # make a clone of the first parameter to modify
            flat_child_parameter = flat_parameter_1.clone()
            # calculate a random crossover point based on the size of the parameter tensor
            # numel() gives the total number of elements in the flat_parameter_1
            crossover_point = random.randint(0, flat_parameter_1.numel())
            # perform crossover by slicing the flat tensors
            # everything before the crossover point is carried over from the first parameter
            flat_child_parameter[:crossover_point] = flat_parameter_1[:crossover_point]
            # everything after the crossover point is carried over from the second parameter
            flat_child_parameter[crossover_point:] = flat_parameter_2[crossover_point:]
            # reshape the child parameter back to the original shape
            child_parameter.data = flat_child_parameter.view_as(parameter_1)
        # append the newly created offpsring to the list
        offspring.append(child)
    # return the list of offspring
    return offspring

def mutate(individuals, mutation_rate, mutation_strength):
    # loop through allindividuals
    for individual in individuals:
        # loop through all parameters for each individual
        for parameter in individual.parameters():
            # if the randomly generated number is less than the mutation rate, mutate
            if random.random() < mutation_rate:
                # add a random value to the parameter using a standard normal distribution
                parameter.data += torch.randn_like(parameter) * mutation_strength
    # return the mutated list of individuals
    return individuals

def genetic_algorithm(train_loader, test_loader, model, num_generations, population_size, num_parents, mutation_rate, mutation_strength):
    # define the loss function as cross entropy loss
    loss_function = nn.CrossEntropyLoss()
    # get the population by initializing it with the defined method
    population = initialize_population(population_size, model)
    # define variable to store the best individual
    best_individual = None
    # define lists to store the best accuracies and average train and test accuracies
    best_accuracies = []
    average_train_accuracies = []
    average_test_accuracies = []
    # loop through all generations
    for generation in range(num_generations):
        print(f"Training: Generation {generation + 1}/{num_generations}") # +1 to not start at the normal index of 0
        # evaluate fitness (loss) of each individual and their accuracies
        fitness_scores, train_accuracies, test_accuracies = evaluate_population(population, train_loader, test_loader, loss_function)
        # find the best individual in this generation
        # do this by getting the index with the best value
        best_index = test_accuracies.index(max(test_accuracies))
        # and then setting the current best indivudal based on the neural network with that index
        current_best_individual = population[best_index]
        # update the tracked best individual if current one is better
        # either if it's gen one and there is no best indiviidual yet
        # or if the testing accuracy is better than any seen so far
        if best_individual is None or test_accuracies[best_index] > max(best_accuracies, default=0):
            best_individual = current_best_individual
        # track the best and average accuracies
        best_accuracies.append(test_accuracies[best_index])
        average_train_accuracies.append(np.mean(train_accuracies))
        average_test_accuracies.append(np.mean(test_accuracies))
        # select parents for crossover, including the best individual
        # -1 here because we already have the best individual as one parent, so we only select enough to fill thee total
        parents = [best_individual] + select_parents(population, fitness_scores, num_parents - 1) # -1 here because we already have the best individual as one parent
        # crossover to create offspring
        offspring = crossover(parents, population_size - num_parents)
        # mutate only the offspring
        offspring = mutate(offspring, mutation_rate, mutation_strength)
        # form the new population with elitism
        population = [best_individual] + offspring

    return best_individual, best_accuracies, average_test_accuracies

def plot_accuracies_data(best_accuracies, average_accuracies):
    # define a new plot with size 10x6 inches
    plt.figure(figsize=(10, 6))
    # generate a line graph based on the number of generations or the length of the list and the actual values
    # set the style to be o--, the color and the label
    # len() gets the length (i.e.) 50 and range turns it into a list (i.e. [0, 1, 2, 3, ..., 49])
    plt.plot(range(len(best_accuracies)), best_accuracies, 'o--', color='blue', label='Best Accuracy')
    plt.plot(range(len(average_accuracies)), average_accuracies, 'o--', color='red', label='Average Accuracy')

    # define the labels and title for the plot
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.title('Best and Average Accuracy per Generation')
    # toggle the legend
    plt.legend()
    # show the plot
    plt.show()

def test_model(neural_network):
    print("Testing model...")
    # splice of the first 2000 entries in the test dataset
    test_images, test_labels = data_manager.test_dataset[0:2000]
    test_label_predictions = neural_network(test_images).argmax(axis=1)
    # define the figure and 2D array of subplots
    figure, axis = plt.subplots(4, 10, figsize=(22.5, 15))

    for i in range(40):
        # navigate to a specific subplot
        plt.subplot(4, 10, i+1)
        # show the image for that subplot
        plt.imshow(test_images[i])
        # set the title of the cell as the predicted image
        plt.title(f'Predicted Digit: {test_label_predictions[i]}')
    # use the inbuilt tight_layout() function to properly space all cells and text
    figure.tight_layout()
    # show the plot in a new window
    plt.show()

def main():
    # load the test and train dataloader from the train and test datasets stored in data_manager.py
    train_dataloader = data_manager.DataLoader(data_manager.train_dataset, batch_size=batch_size)
    test_dataloader = data_manager.DataLoader(data_manager.test_dataset, batch_size=batch_size)
    # let the user choose wether to train a model from scratch or showcase an existing one
    mode = input('Train and test model with genetic algorithm: 0\nLoad and test existing model: 1\nWhich mode would you like to do: ')
    # if the user picked to train a model from scratch
    if mode == '0':
        try:
            num_generations = int(input('How many generations should the model train for: '))
            print("Training model...")
            # execute the genetic_algorithm only if num_generations is valid
            best_model, best_accuracies, average_accuracies = genetic_algorithm(
                train_dataloader, test_dataloader,
                NeuralNetwork, num_generations, population_size, num_parents, mutation_rate, mutation_strength
            )
            # plot the best and average accuracies for visualization
            print("Plotting best and average accuracies...")
            plot_accuracies_data(best_accuracies, average_accuracies)
            # test the best model from the last generation of training
            test_model(best_model)
            # save the best model
            data_manager.save_model(best_model, f"{data_manager.model_weights_path}/genetic_algorithm.pt")
        except ValueError:
            print('Generation number was not valid. Please try again...')
            main()
    # if the user picked to showcase an existing model
    elif mode == '1':
        # load a neural network using the load_model function from the data_manager
        neural_network = data_manager.load_model(f"{data_manager.model_weights_path}/genetic_algorithm.pt", NeuralNetwork())
        # run the test_model function with the neural_network as input
        test_model(neural_network)
    else:
        # have a fail safe in case the input wasn't '0' or '1'
        print("Input wasn't accepted. Please try again.")
        main()

# define the number of images per batch for the dataloader
# the dataloader will take a batch of x images from the dataset every time to train the model
batch_size = 64
# define for how many generations the model will be trained for
num_generations = 1
# define how many individuals or neural networks will be in any one generation
population_size = 50
num_parents = 25
# define at what rate the parameters of any individual will mutate per generation
# in this case, 5 % of the parameters will mutate
mutation_rate = 0.05
# define how much the parameters will mutate per generation
mutation_strength = 0.1

if __name__ == '__main__':
    main()
