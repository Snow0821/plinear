from plinear import layers
from torch import nn

class mnist_classifier_conv2d(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.ModuleList()
        self.conv.append(layers.Conv2d_Nand(1, 16, 2, stride=2))
        self.conv.append(layers.Conv2d_Nand(16, 16, 2, stride=2))
        self.conv.append(layers.Conv2d_Nand(16, 16, 2))
        self.conv.append(layers.Conv2d_Nand(16, 16, 2, stride=2))
        self.conv.append(layers.Conv2d_Nand(16, 16, 2, stride=2))
        self.conv.append(layers.Conv2d_Nand(16, 10, 1))
        
        

    def forward(self, x):
        x = ((x > 0).float()) * 2 - 1
        for ly in self.conv:
            x = ly(x)
        x = x.flatten(1)
        return x

class mnist_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        linear = layers.Compacted_Nand
        # linear = nn.Linear
        self.layers = nn.ModuleList()
        self.layers.append(linear(28 * 28, 256))
        for _ in range(2):
            self.layers.append(linear(256, 256))
        self.layers.append(linear(256, 10))
    
    def forward(self, x):
        x = x.flatten(1)
        for ly in self.layers:
            x = ly(x)
        return x

class mnist_classifier_Compacted_Nand(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(layers.Compacted_Nand(28 * 28, 128))
        for _ in range(5):
            self.layers.append(layers.Compacted_Nand(512, 512))
        self.layers.append(layers.Compacted_Nand(512, 128))
        self.layers.append(layers.Compacted_Nand(128, 10))
    
    def forward(self, x):
        x = x.flatten(1)
        for ly in self.layers:
            x = ly(x)
        return x

class mnist_classifier_multihead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(layers.SparseBtnn_Nand_Multihead(28 * 28, 16, 16))
        for _ in range(10):
            self.layers.append(layers.SparseBtnn_Nand_Multihead(256, 16, 16))
        self.layers.append(layers.SparseBtnn_Nand(256, 10))
    
    def forward(self, x):
        x = x.flatten(1)
        for ly in self.layers:
            x = ly(x)
        return x