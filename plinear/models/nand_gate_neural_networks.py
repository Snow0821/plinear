from torch import nn
from plinear import layers

class ngnn_mnist(nn.Module):
    def __init__(self):
        super(ngnn_mnist, self).__init__()
        self.fc1 = layers.SparseBtnn_Nand(28*28, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
