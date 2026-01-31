# import libraries needed for managing data
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, filepath):
        # Data downloaded from www.di.ens.fr/~lelarge/MNIST.tar.gz
        # Load the images and labels from the preprocessed data
        self.images, self.labels = torch.load(filepath, weights_only=True)
        # turn the data from a torch tensor to an array of 784 values that can be worked with more easily
        # -1 means to keep the first property: 6000 images if you look at images.shape
        # and since the images are 28 x 28 in length, we have a list of 28**2 values (one per pixel)
        self.images.view(-1, 28**2)
        # divide each value of the tensor by 255 so we get a value 0-1
        self.images = self.images / 255
        # use the one hot encoding method to encode each label into a floating point vector
        self.labels = F.one_hot(self.labels, num_classes=10).to(float)

    # return the amount of images by returning the first value of x where x is ([6000, 784])
    # the function names have to be with two underscore before and after
    # this is so we can call len(dataset) and dataset[index] and not have to call the function of the class
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, image_index):
        return self.images[image_index], self.labels[image_index]

def save_model(model, save_path):
    # save the models weights to the save_path
    torch.save(model.state_dict(), save_path)
    print(f"saved model to path: {save_path}")

def load_model(load_path, neural_network):
    print("Loading model...")
    # create a new instance of the neural network
    model = neural_network
    # load the saved state dictionary into the model
    model.load_state_dict(torch.load(load_path, weights_only=True))
    # Set the model to evaluation mode
    model.eval()
    # return the model
    return model

# define a path to save and load the model to and from
model_weights_path = 'models'
# load the dataset that the model will use to train with
train_dataset = MNISTDataset('data/processed/training.pt')
# load the dataset that the model will be tested on
# note that this is data the model hasn't seen during training, to test generalization
test_dataset = MNISTDataset('data/processed/test.pt')
