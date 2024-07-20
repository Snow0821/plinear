import pytest
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from plinear.plinear import PLinear_Complex as PL

from plinear.tester import tester

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.complex = torch.zeros(1, 28*28)
        self.fc1 = PL(28*28, 256)
        self.fc2 = PL(256, 256)
        self.fc3 = PL(256, 10)

    def forward(self, x):
        real = torch.flatten(x, 1)
        complex = self.complex
        real, complex = self.fc1(real, complex)
        real, complex = self.fc2(real, complex)
        real, complex = self.fc3(real, complex)
        return real

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
    domains = ['Real', 'Complex']
    num_epochs = 10
    path = 'tests/results/mnist_plinear_complex/'
    train_loader, test_loader = mnist_data
    tester(model, domains, num_epochs, path, train_loader, test_loader)
