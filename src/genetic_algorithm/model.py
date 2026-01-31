import torch
import torch.nn as nn

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

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