import pytest
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from plinear.plinear import PLinear

from plinear.tester import tester

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = PLinear(28*28, 512)
        self.fc2 = PLinear(512, 512)
        self.fc3 = PLinear(512, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

@pytest.fixture
def mnist_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader

def test_simple_nn(mnist_data):
    model = SimpleNN()
    domains = ['Real']
    num_epochs = 10
    path = 'tests/results/mnist_plinear/'
    train_loader, test_loader = mnist_data
    tester(model, domains, num_epochs, path, train_loader, test_loader)