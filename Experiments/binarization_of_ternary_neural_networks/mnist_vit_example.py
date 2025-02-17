import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from plinear.models import vit_example as vit
from plinear import btnn

from tqdm import tqdm

# MNIST 데이터셋 로드
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 모델 초기화
config = {'img_size' : 32, 
          'patch_size' : 4,
          'in_channels' : 1, 
          'embed_dim' : 512, 
          'num_heads' : 32, 
          'depth' : 6, 
          'mlp_dim' : 1024, 
          'num_classes' : 10}

model = vit.VisionTransformer(**config)

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_loss_history = []
train_accuracy_history = []

# 학습 함수
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct = 0, 0

    # tqdm Progress Bar 추가
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Training", leave=True)

    for batch_idx, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 손실 및 정확도 계산
        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()

        # Progress Bar에 실시간 업데이트
        progress_bar.set_postfix({
            "Batch Loss": loss.item(),
            "Avg Loss": total_loss / (batch_idx + 1),
            "Accuracy": total_correct / ((batch_idx + 1) * loader.batch_size)
        })

    # 최종 평균 손실 및 정확도 반환
    avg_loss = total_loss / len(loader)
    avg_accuracy = total_correct / len(loader.dataset)

    train_loss_history.append(avg_loss)
    train_accuracy_history.append(avg_accuracy)

    return avg_loss, avg_accuracy

# 테스트 함수
def test(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), total_correct / len(loader.dataset)

# GPU 또는 CPU 설정
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model = model.to(device)

print(f"Using device: {device}")

# 학습 및 평가 루프
epochs = 10
for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test(model, test_loader, criterion, device)

    print(f"Epoch [{epoch + 1}/{epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    model.push_to_hub(f'snowian/mnist_btvit_{epoch + 1}')

from torchinfo import summary

summary(model, input_size=(1, 1, 32, 32))

import matplotlib.pyplot as plt

# 학습 손실 및 정확도 시각화
plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_loss_history, marker='o', label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid()
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracy_history, marker='o', label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Over Epochs")
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig("mnist_simplecnn_training_metrics.png")
plt.show()

