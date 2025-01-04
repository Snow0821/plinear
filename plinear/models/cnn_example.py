import torch
import torch.nn as nn
import torch.nn.functional as F

from plinear.core import RMSNorm
from plinear.btnn import Conv2d
from plinear.btnn import Linear

class CNN_Example(nn.Module):
    def __init__(self):
        super(CNN_Example, self).__init__()
        # self.features = nn.Sequential(
        #     Conv2d(1, 16, kernel_size=3, padding=1),
        #     Conv2d(16, 32, kernel_size=3, padding=1),
        #     nn.MaxPool2d(2, 2)
        # )
        self.fc1 = Linear(28*28, 128)
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        # x = self.features(x)
        # x = x.flatten(1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
