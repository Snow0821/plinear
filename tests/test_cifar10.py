import pytest
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from plinear import PLinear

@pytest.fixture(scope="module")
def cifar10_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    return train_loader

def test_plinear_cifar10(cifar10_data):
    model = PLinear(32*32*3, 10)

    for images, labels in cifar10_data:
        images = images.view(images.size(0), -1)
        output = model(images)
        assert output.shape[-1] == 2, "Each output value should be a tuple of length 2"
        break
