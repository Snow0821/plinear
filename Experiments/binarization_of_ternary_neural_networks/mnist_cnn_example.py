import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchinfo import summary

from plinear import btnn
from plinear.core import RMSNorm

# 간단한 CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = btnn.Conv2d(1, 16, kernel_size=3, padding=1)  # 입력 채널: 1, 출력 채널: 16
        self.conv2 = btnn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 Max Pooling
        self.fc1 = btnn.Linear(32 * 7 * 7, 128)  # Fully Connected Layer
        self.fc2 = btnn.Linear(128, 10)  # 출력 노드: 10 (MNIST 클래스)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = RMSNorm(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = RMSNorm(x)
        x = x.flatten(1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = RMSNorm(x)
        x = self.fc2(x)
        return x

# 데이터셋 및 DataLoader 준비
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 모델, 손실 함수 및 옵티마이저 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 요약 출력
print(summary(model, input_size=(64, 1, 28, 28)))

# 학습 루프
num_epochs = 10
train_loss_history = []
train_accuracy_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        progress_bar.set_postfix(loss=loss.item(), accuracy=correct / total)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    train_loss_history.append(epoch_loss)
    train_accuracy_history.append(epoch_accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# 테스트 루프
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

# 테스트 실행
test_accuracy = test_model(model, test_loader)

# 학습 손실 및 정확도 시각화
plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_loss_history, marker='o', label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid()
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracy_history, marker='o', label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Over Epochs")
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig("mnist_simplecnn_training_metrics.png")
plt.show()
