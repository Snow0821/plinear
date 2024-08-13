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
        self.fc3 = PLinear(2048, 2048)
        self.fc4 = PLinear(2048, 2048)
        self.fc5 = PLinear(2048, 2048)
        self.fc6 = PLinear(2048, 2048)
        self.fc7 = PLinear(2048, 2048)
        self.fc8 = PLinear(2048, 2048)
        self.fc9 = PLinear(2048, 2048)
        self.fcA = PLinear(2048, 100)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        x = self.fcA(x)

        return x

class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.complex = torch.zeros(32*32*3)
        self.fc1 = PLinear_Complex(32*32*3, 1024)
        self.fc2 = PLinear_Complex(1024, 1024)
        self.fc3 = PLinear_Complex(1024, 1024)
        self.fc4 = PLinear_Complex(1024, 1024)
        self.fc5 = PLinear_Complex(1024, 1024)
        self.fc6 = PLinear_Complex(1024, 1024)
        self.fc7 = PLinear_Complex(1024, 1024)
        self.fc8 = PLinear_Complex(1024, 1024)
        self.fc9 = PLinear_Complex(1024, 1024)
        self.fcA = PLinear_Complex(1024, 100)

    def forward(self, x):
        real = torch.flatten(x, 1)
        complex = self.complex
        real, complex = self.fc1(real, complex)
        real, complex = self.fc2(real, complex)
        real, complex = self.fc3(real, complex)
        real, complex = self.fc4(real, complex)
        real, complex = self.fc5(real, complex)
        real, complex = self.fc6(real, complex)
        real, complex = self.fc7(real, complex)
        real, complex = self.fc8(real, complex)
        real, complex = self.fc9(real, complex)
        real, complex = self.fcA(real, complex)
        return real

@pytest.fixture
def cifar100_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader

def test_real(cifar100_data):
    model = RealModel()
    domains = ['Real']
    num_epochs = 100
    path = 'tests/results/cifar100_real/'
    train_loader, test_loader = cifar100_data
    ExampleTester(model, domains, num_epochs, path, train_loader, test_loader)

def test_complex(cifar100_data):
    model = ComplexModel()
    domains = ['Real', 'Complex']
    num_epochs = 100
    path = 'tests/results/cifar100_complex/'
    train_loader, test_loader = cifar100_data
    ExampleTester(model, domains, num_epochs, path, train_loader, test_loader)