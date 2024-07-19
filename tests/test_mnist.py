import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from plinear.plinear import PLinear
from plinear.vis import save_weight_images, create_animation_from_images, plot_metrics, save_confusion_matrix
import os

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

@pytest.fixture(scope="module")
def mnist_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader

def test_mnist(mnist_data):
    def train(model, train_loader):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        return

    def evaluation(model, test_loader, path, epoch):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                preds = output.argmax(dim=1, keepdim=True)
                all_preds.extend(preds.view(-1).cpu().tolist())
                all_labels.extend(target.view(-1).cpu().tolist())

        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')

        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Accuracy: {accuracy}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")

        save_confusion_matrix(confusion_matrix(all_labels, all_preds), path, epoch)
        save_weight_images(model, path, epoch)
        
        return
    
    def save_animations(model, num_epochs, accuracy_list, recall_list, precision_list, path):
        for i, layer in enumerate(model.children()):
            if isinstance(layer, model):
                create_animation_from_images(path, f'layer_{i + 1}', num_epochs, 'Real')
        create_animation_from_images(path,'cm', num_epochs)

        # Plot accuracy, recall, precision over epochs and save to file
        epochs = range(1, num_epochs + 1)
        plot_metrics(epochs, accuracy_list, recall_list, precision_list, path)
        return


    path = 'tests/results/mnist_plinear/'
    os.makedirs(path, exist_ok=True)
    train_loader, test_loader = mnist_data

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1)  # Basic SGD optimizer

    num_epochs = 3  # EPOCHS
    accuracy_list = []
    recall_list = []
    precision_list = []

    # Training and evaluation loop
    for epoch in range(num_epochs):
        train(model, train_loader)
        evaluation(model, test_loader, path, epoch)
    
    save_animations(model, num_epochs, accuracy_list, recall_list, precision_list, path)

    # Assert final accuracy
    assert accuracy_list[-1] > 0.9, "Final accuracy should be higher than 90%"