from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

try:
    dataset = load_dataset("benjamin-paine/imagenet-1k-32x32", split="train", streaming=False)
except Exception as e:
    print("Error loading dataset. Ensure you have access to 'benjamin-paine/imagenet-1k-32x32' and are authenticated.")
    raise e


total_samples = 1281167
batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor()
])

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

train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

from plinear.models import vit_example as vit

config = {'img_size' : 256, 
          'patch_size' : 16,
          'in_channels' : 3, 
          'embed_dim' : 512, 
          'num_heads' : 16, 
          'depth' : 4, 
          'mlp_dim' : 1024, 
          'num_classes' : 1000}

model = vit.VisionTransformer(**config)

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

import time

training_start_time = time.time()
# 학습 루프
num_epochs = 1
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
        
    model.push_to_hub(f'snowian/imageNet_btvit_{epoch + 1}')
    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_corrects / total_samples
    loss_history.append(epoch_loss)
    accuracy_history.append(epoch_accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

training_end_time = time.time()  # 학습 종료 시간
training_duration = training_end_time - training_start_time  # 전체 학습 소요 시간
print(f"Total Training Time: {training_duration:.2f} seconds")

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
# plt.show()

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
# plt.show()

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

import pandas as pd

def evaluate_model(model, dataloader, criterion, device, save_path=None):
    model.eval()
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            running_corrects += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

    accuracy = running_corrects / total_samples
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # 검증 결과 저장
    if save_path:
        results = {"Accuracy": [accuracy]}
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path, index=False)
        print(f"Test results saved to {save_path}")

    return accuracy

# 검증 데이터로 평가
validation_dataset = load_dataset("benjamin-paine/imagenet-1k-32x32", split="validation", streaming=False)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=collate_fn)

val_start_time = time.time()

test_accuracy = evaluate_model(
    model, validation_loader, criterion, device, save_path="validation_results.csv"
)

val_end_time = time.time()  # 테스트 종료 시간
val_duration = val_end_time - val_start_time  # 테스트 소요 시간
print(f"Total Validation Time: {val_duration:.2f} seconds")



test_dataset = load_dataset("benjamin-paine/imagenet-1k-32x32", split="test", streaming=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

test_start_time = time.time()

test_accuracy = evaluate_model(
    model, test_loader, criterion, device, save_path="test_results.csv"
)

test_end_time = time.time()  # 테스트 종료 시간
test_duration = test_end_time - test_start_time  # 테스트 소요 시간
print(f"Total Test Time: {test_duration:.2f} seconds")

from torchinfo import summary

summary(model, input_size=(1, 3, 32, 32))