# import all the libraries needed
import torch.nn as nn
from torch.optim import SGD
import numpy as np
import matplotlib.pyplot as plt

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
        self.Matrix1 = nn.Linear(28**2, 100)
        # second layer has 100 neurons
        # it processes data from the input layer and then sends the values to 50 new neurons
        self.Matrix2 = nn.Linear(100, 50)
        # same as second layer but it has 50 neurons, passes it to 10 output neurons
        # there are 10 output neurons because we have 10 different digits it can be
        self.Matrix3 = nn.Linear(50, 10)
        # define the rectified linear unit (ReLU)
        self.rectifier = nn.ReLU()

    def forward(self, input_images):
        # convert the tensor into an array as mentioned previously
        input_images = input_images.view(-1, 28**2)
        # pass the images through the layers
        # use ReLU to make sure that only meaningful output gets passed to next neuron
        input_images = self.rectifier(self.Matrix1(input_images))
        input_images = self.rectifier(self.Matrix2(input_images))
        input_images = self.Matrix3(input_images)
        # return the predictions of the Neural Network
        # use squeeze to remove unnecessary dimensions from the tensor
        return input_images.squeeze()

def train_model(data_loader, neural_network, num_epochs):
    print("Training model...")
    # optimize the model using SGD
    optimizer = SGD(neural_network.parameters(), lr=0.01)
    # define the loss function
    loss = nn.CrossEntropyLoss()

    # train the model
    # define the lists to store the data in
    losses = []
    epochs = []
    # iterate through each epoch
    for epoch in range(num_epochs):
        print(f'Training: Epoch {epoch+1}/{num_epochs}') # +1 to not start at the normal index of 0
        for i, (images, labels) in enumerate(train_dataloader):
            # update the weights of the neural network
            # reset the stored gradients of the model to avoid errors
            optimizer.zero_grad()
            # calculate the loss
            loss_value = loss(neural_network(images), labels)
            # use backpropagation to find the gradients of the weights
            loss_value.backward()
            # adjust the weights in the direction of least loss
            optimizer.step()
            # store the training data
            epochs.append(epoch+1/len(train_dataloader))
            losses.append(loss_value.item())
    # return the epochs and losses, converted to numpy arrays to better work with them
    return np.array(epochs), np.array(losses)

def test_model(neural_network):
    print("Testing model...")
    # splice of the first 2000 entries in the test dataset
    test_images, test_labels = data_manager.test_dataset[0:2000]
    # get the predicted class of the neural network
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

def average_epoch_and_loss_data(epoch_data, loss_data):
    # calculate the average loss for each epoch so we can better visualize the Dataset
    # split the loss and epoch data into num_epochs different batches
    epoch_data_average = epoch_data.reshape(num_epochs, -1)
    loss_data_average = loss_data.reshape(num_epochs, -1)
    # calculate the mean loss of each epoch where axis 1 corresponds to the epochs
    epoch_data_average = epoch_data_average.mean(axis=1)
    loss_data_average = loss_data_average.mean(axis=1)
    # return the averaged epoch and loss data
    return epoch_data_average, loss_data_average

def plot_data(data_x, data_y):
    print(f"Loss Data: {data_y}")
    plt.figure(1)
    # plot the data as a graph of markers connected by dashed lines
    plt.plot(data_x, data_y, 'o--', color='blue', label='average cross entropy loss per epoch')
    # set the labels and title of the graph
    plt.xlabel('Epoch Number')
    plt.ylabel('Cross Entropy (averaged per epoch)')
    plt.title('Cross Entropy (averaged per epoch)')
    plt.legend()

def train_model_from_scratch():
    # train the model and return the epoch data and loss data
    epoch_data, loss_data = train_model(train_dataloader, neural_network, num_epochs)
    # calculate the average of the epoch and loss for better visualization
    epoch_data_average, loss_data_average = average_epoch_and_loss_data(epoch_data, loss_data)
    # plot the data for visualization
    plot_data(epoch_data_average, loss_data_average)
    # evaluate the model on images the model hasn't seen using the test dataset
    test_model(neural_network)
    # save the currently trained model's weights
    # note that this will overwrite any existing file saved under the same path
    data_manager.save_model(neural_network, f"{data_manager.model_weights_path}/gradient_descent.pt")

def main():
    # give the user a choice if he wants to train a new model or test / showcase an existing one
    mode = input('Train and test model from scratch: 0\nLoad and test existing model: 1\nWhich mode would you like to do: ')
    if mode == '0':
        train_model_from_scratch()
    elif mode == '1':
        # load an existing model's weights'
        neural_network = data_manager.load_model(f"{data_manager.model_weights_path}/gradient_descent.pt", NeuralNetwork())
        # test the existing model
        test_model(neural_network)
    else:
        # have a fail safe for when the user types something else
        print("Input wasn't accepted. Please try again.")
        main()

# number of images per batch
batch_size = 5
# load the dataset into the dataloader with a batch size of 5
# the dataloader will give 1200 batches of 5 images every time (6000 images / 5 images per batch)
# this is important for training, as you can easily iterate through the dataloader
train_dataloader = data_manager.DataLoader(data_manager.train_dataset, batch_size=batch_size)
# initialize the neural network
neural_network = NeuralNetwork()
# set the number of epochs that the model will be trained for
num_epochs = 20

if __name__ == '__main__':
    main()
