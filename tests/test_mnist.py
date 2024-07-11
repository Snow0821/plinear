import pytest
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from plinear import PLinear

@pytest.fixture(scope="module")
def mnist_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    return test_loader

def test_plinear_mnist(mnist_data):
    model = PLinear(28*28, 10)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in mnist_data:
            images = images.view(images.size(0), -1)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()  # Convert to Python list
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())  # Convert to Python list

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')

    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    assert accuracy > 0.9, "Accuracy should be higher than 90%"
