from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# 스트리밍 모드로 ImageNet 데이터셋 로드
try:
    dataset = load_dataset("imagenet-1k", split="train", streaming=True)
except Exception as e:
    print("Error loading dataset. Ensure you have access to 'imagenet-1k' and are authenticated.")
    raise e


total_samples = 1281167  # ImageNet train 데이터 샘플 수
batch_size = 64

# 데이터 변환 정의
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda x: x.convert("RGB")),  # 강제로 RGB로 변환
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 배치 데이터 로드 함수
def collate_fn(batch):
    images, labels = [], []
    for item in batch:
        try:
            image = transform(item["image"])
            images.append(image)
            labels.append(item["label"])
        except Exception as e:
            print(f"Error processing image: {e}")
            continue
    return torch.stack(images), torch.tensor(labels)

# DataLoader 설정
train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


model = models.resnet18(pretrained=False, num_classes=1000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 결과 저장을 위한 리스트
loss_history = []
accuracy_history = []
iteration_loss = []
iteration_accuracy = []

# 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    running_samples = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", total=total_samples // batch_size)
    for i, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        running_corrects += torch.sum(preds == labels).item()
        running_samples += labels.size(0)
        running_loss += loss.item()

        # Iteration-level metrics
        iteration_loss.append(loss.item())
        iteration_accuracy.append(running_corrects / running_samples)

        progress_bar.set_postfix(loss=loss.item(), accuracy=running_corrects / total_samples)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = running_corrects / total_samples
    loss_history.append(epoch_loss)
    accuracy_history.append(epoch_accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# 학습 완료 후 모델 저장
torch.save(model.state_dict(), "resnet18_imagenet.pth")

# 결과 시각화
plt.figure(figsize=(10, 5))

# Iteration-level metrics in a single plot
plt.plot(range(1, len(iteration_loss) + 1), iteration_loss, label="Iteration Loss", alpha=0.7)
plt.plot(range(1, len(iteration_accuracy) + 1), iteration_accuracy, label="Iteration Accuracy", alpha=0.7)
plt.xlabel("Iteration")
plt.ylabel("Metric")
plt.title("Loss and Accuracy Per Iteration")
plt.legend()
plt.grid()
plt.savefig("training_iteration_metrics_combined.png")
plt.show()

# Epoch-level metrics
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), loss_history, marker='o', label="Epoch Loss")
plt.plot(range(1, num_epochs + 1), accuracy_history, marker='o', label="Epoch Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Loss and Accuracy Over Epochs")
plt.legend()
plt.grid()
plt.savefig("training_epoch_metrics.png")
plt.show()

# Save metrics to CSV
metrics_data = {
    "Iteration": list(range(1, len(iteration_loss) + 1)),
    "Iteration Loss": iteration_loss,
    "Iteration Accuracy": iteration_accuracy,
    "Epoch": list(range(1, num_epochs + 1)),
    "Epoch Loss": loss_history,
    "Epoch Accuracy": accuracy_history,
}

# Create DataFrame and save to CSV
iteration_df = pd.DataFrame({"Iteration": metrics_data["Iteration"],
                              "Loss": metrics_data["Iteration Loss"],
                              "Accuracy": metrics_data["Iteration Accuracy"]})
epoch_df = pd.DataFrame({"Epoch": metrics_data["Epoch"],
                          "Loss": metrics_data["Epoch Loss"],
                          "Accuracy": metrics_data["Epoch Accuracy"]})

iteration_df.to_csv("iteration_metrics.csv", index=False)
epoch_df.to_csv("epoch_metrics.csv", index=False)

print("Metrics saved to CSV files.")
