import pytest
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from plinear import PLinear

@pytest.fixture(scope="module")
def cifar100_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    return train_loader

def test_plinear_cifar100(cifar100_data):
    model = PLinear(32*32*3, 100)

    for images, labels in cifar100_data:
        images = images.view(images.size(0), -1)
        output = model(images)
        assert output.shape[-1] == 2, "Each output value should be a tuple of length 2"
        break
