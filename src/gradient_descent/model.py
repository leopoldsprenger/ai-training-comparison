import torch
import torch.nn as nn

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import data_manager as data

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.Matrix1 = nn.Linear(data.TENSOR_SIZE, 100)
        self.Matrix2 = nn.Linear(100, 50)
        self.Matrix3 = nn.Linear(50, data.NUM_CLASSES)

        self.rectifier = nn.ReLU()

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        input_images = input_images.view(-1, data.TENSOR_SIZE)

        input_images = self.rectifier(self.Matrix1(input_images))
        input_images = self.rectifier(self.Matrix2(input_images))
        input_images = self.Matrix3(input_images)

        return input_images.squeeze()