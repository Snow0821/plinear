import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from plinear.plinear import PLinear_Complex as PL
from plinear.visualization import save_weight_image_complex, create_animation_from_images, create_confusion_matrix_animation, plot_metrics
import os

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

@pytest.fixture(scope="module")
def mnist_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader

def test_mnist(mnist_data):
    result_path = 'tests/results/mnist_plinear_complex/'
    os.makedirs(result_path, exist_ok=True)
    train_loader, test_loader = mnist_data

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1)  # Basic SGD optimizer

    num_epochs = 10  # epochs
    accuracy_list = []
    recall_list = []
    precision_list = []
    confusion_matrices = []

    # Training and evaluation loop
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Evaluation
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

        cm = confusion_matrix(all_labels, all_preds)
        confusion_matrices.append(cm)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Accuracy: {accuracy}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")

        # saving images
        for i, layer in enumerate(model.children()):
            if isinstance(layer, PL):
                save_weight_image_complex(layer, result_path, i, epoch)

    # create animation
    for i, layer in enumerate(model.children()):
        if isinstance(layer, PL):
            create_animation_from_images(result_path, i, num_epochs, postfix='_real')
            create_animation_from_images(result_path, i, num_epochs, postfix='_complex')

    # Plot accuracy, recall, precision over epochs and save to file
    epochs = range(1, num_epochs + 1)
    plot_metrics(epochs, accuracy_list, recall_list, precision_list, result_path)

    # Animate confusion matrices and save to file
    create_confusion_matrix_animation(confusion_matrices, result_path, num_epochs)

    # Assert final accuracy
    assert accuracy_list[-1] > 0.9, "Final accuracy should be higher than 90%"
