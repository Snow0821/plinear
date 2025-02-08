import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl

from plinear import layers

# LightningModule 정의
class LitModel(pl.LightningModule):
    def __init__(self, linear = nn.Linear, conv2d = nn.Conv2d):
        super().__init__()

        self.model = rcvit_xs(num_classes=100, linear = linear, conv2d = conv2d)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Batch 단위 결과 로깅
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return {"loss": loss, "accuracy": acc}

    def train_epoch_end(self, outputs):
        # 에포크 단위 결과 요약
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        print(f"\nEpoch End - Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_acc:.4f}")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Batch 단위 결과 로깅
        self.log("val_acc", acc, prog_bar=True)
        return {"val_accuracy": acc}

    def validation_step_epoch_end(self, outputs):
        # 에포크 단위 결과 요약
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        print(f"Validation Accuracy: {avg_acc:.4f}")

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_accuracy", acc)
        return {"test_accuracy": acc}

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

"""# Data Module"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # 데이터 전처리 변환 정의
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        # 데이터셋 다운로드
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # 데이터셋 정의
        if stage == "fit" or stage is None:
            self.cifar_train = datasets.CIFAR100(self.data_dir, train=True, transform=self.transform_train)
            self.cifar_val = datasets.CIFAR100(self.data_dir, train=True, transform=self.transform_test)

            # 훈련/검증 나누기
            val_size = 5000
            train_size = len(self.cifar_train) - val_size
            self.cifar_train, self.cifar_val = torch.utils.data.random_split(self.cifar_train, [train_size, val_size])

        if stage == "test" or stage is None:
            self.cifar_test = datasets.CIFAR100(self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=4)

"""# Train Test"""

# 학습 및 테스트 실행
if __name__ == "__main__":
    model = LitModel(linear = btnnLinear, conv2d = btnnConv2d)
    data_module = CIFAR100DataModule(batch_size=64)

    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(device)
    # 학습
    logger = pl.loggers.CSVLogger("logs", name = "CIFAR100_BT_CASViT")
    trainer = pl.Trainer(max_epochs=5, devices = 1, accelerator=device, logger = logger)
    trainer.fit(model, data_module)

    # 테스트
    test_results = trainer.test(datamodule=data_module)
    print("Test Results:", test_results)

# 학습 및 테스트 실행
if __name__ == "__main__":
    model = LitModel()
    data_module = CIFAR100DataModule(batch_size=64)

    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(device)
    # 학습
    logger = pl.loggers.CSVLogger("logs", name = "CIFAR100_CASViT")
    trainer = pl.Trainer(max_epochs=5, devices = 1, accelerator=device, logger = logger)
    trainer.fit(model, data_module)

    # 테스트
    test_results = trainer.test(datamodule=data_module)
    print("Test Results:", test_results)

import pandas as pd
import matplotlib.pyplot as plt

# 두 모델의 metrics.csv 파일 경로
paths = {
    "BT_CASViT": "/content/logs/CIFAR100_BT_CASViT/version_0/metrics.csv",
    "CASViT": "/content/logs/CIFAR100_CASViT/version_0/metrics.csv"
}

# 데이터를 로드하고 처리
data = {}
for model_name, path in paths.items():
    df = pd.read_csv(path)

    # train 데이터와 validation 데이터 필터링
    df_train = df[df["train_acc"].notna()]  # train 데이터 (step 단위)
    df_val = df[df["val_acc"].notna()]  # validation 데이터 (epoch 단위)

    data[model_name] = {
        "train_step": df_train["step"].values,
        "train_acc": df_train["train_acc"].values,
        "train_loss": df_train["train_loss"].values,
        "val_epoch": range(1, len(df_val) + 1),  # epoch 번호
        "val_acc": df_val["val_acc"].values
    }

save_dir = "/content/logs/"

# Training Accuracy 그래프
plt.figure(figsize=(12, 6))
for model_name in data:
    plt.plot(data[model_name]["train_step"], data[model_name]["train_acc"], label=f"{model_name} Train Accuracy")
plt.title("Training Accuracy")
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(f"{save_dir}training_accuracy.png")
plt.close()

# Training Loss 그래프
plt.figure(figsize=(12, 6))
for model_name in data:
    plt.plot(data[model_name]["train_step"], data[model_name]["train_loss"], label=f"{model_name} Train Loss")
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"{save_dir}training_loss.png")
plt.close()

# Validation Accuracy 그래프
plt.figure(figsize=(12, 6))
for model_name in data:
    plt.plot(data[model_name]["val_epoch"], data[model_name]["val_acc"], marker='o', linestyle='--', label=f"{model_name} Validation Accuracy")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(f"{save_dir}validation_accuracy.png")
plt.close()