# import libraries needed for managing data
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, filepath):
        self.images, self.labels = torch.load(filepath, weights_only=True)

        self.images.view(-1, TENSOR_SIZE)
        # convert to values of 0-1
        self.images = self.images / 255
        
        self.labels = F.one_hot(self.labels, num_classes=NUM_CLASSES).to(float)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, image_index):
        return self.images[image_index], self.labels[image_index]

def save_model(save_path, model):
    torch.save(model.state_dict(), save_path)
    print(f"saved model to path: {save_path}")

def load_model(load_path, model):
    model.load_state_dict(torch.load(load_path, weights_only=True))
    model.eval()
    return model

IMAGE_SIZE = 28
TENSOR_SIZE = IMAGE_SIZE**2
NUM_CLASSES = 10

MODEL_WEIGHTS_DIR = 'models'
GENETIC_ALGORITHM_MODEL_PATH = f"{MODEL_WEIGHTS_DIR}/genetic_algorithm.pt"
GRADIENT_DESCENT_MODEL_PATH = f"{MODEL_WEIGHTS_DIR}/gradient_descent.pt"

# Data downloaded from www.di.ens.fr/~lelarge/MNIST.tar.gz
TRAIN_DATASET = MNISTDataset('data/processed/training.pt')
TEST_DATASET = MNISTDataset('data/processed/test.pt')