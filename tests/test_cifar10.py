import pytest
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from plinear.plinear import PLinear, PLinear_Complex

from plinear.tester import ExampleTester

class RealModel(nn.Module):
    def __init__(self):
        super(RealModel, self).__init__()
        self.fc1 = PLinear(32*32*3, 2048)
        self.fc2 = PLinear(2048, 2048)
        self.fc3 = PLinear(2048, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.complex = torch.zeros(32*32*3)
        self.fc1 = PLinear_Complex(32*32*3, 1024)
        self.fc2 = PLinear_Complex(1024, 1024)
        self.fc3 = PLinear_Complex(1024, 10)

    def forward(self, x):
        real = torch.flatten(x, 1)
        complex = self.complex
        real, complex = self.fc1(real, complex)
        real, complex = self.fc2(real, complex)
        real, complex = self.fc3(real, complex)
        return real

@pytest.fixture
def cifar10_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader

def test_real(cifar10_data):
    model = RealModel()
    domains = ['Real']
    num_epochs = 10
    path = 'tests/results/cifar10_real/'
    train_loader, test_loader = cifar10_data
    ExampleTester(model, domains, num_epochs, path, train_loader, test_loader)

def test_complex(cifar10_data):
    model = ComplexModel()
    domains = ['Real', 'Complex']
    num_epochs = 10
    path = 'tests/results/cifar10_complex/'
    train_loader, test_loader = cifar10_data
    ExampleTester(model, domains, num_epochs, path, train_loader, test_loader)