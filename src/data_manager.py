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
        
        self.labels = F.one_hot(self.labels, num_classes=10).to(float)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, image_index):
        return self.images[image_index], self.labels[image_index]

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"saved model to path: {save_path}")

def load_model(load_path, model):
    print("Loading model...")   
    model.load_state_dict(torch.load(load_path, weights_only=True))
    model.eval()
    return model

IMAGE_SIZE = 28
TENSOR_SIZE = IMAGE_SIZE**2

MODEL_WEIGHTS_PATH = 'models'
# Data downloaded from www.di.ens.fr/~lelarge/MNIST.tar.gz
TRAIN_DATASET = MNISTDataset('data/processed/training.pt')
TEST_DATASET = MNISTDataset('data/processed/test.pt')