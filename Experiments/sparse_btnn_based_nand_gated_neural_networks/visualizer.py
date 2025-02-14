import pandas as pd
import matplotlib.pyplot as plt

# 데이터를 로드하고 처리
path = ""
df = pd.read_csv(path)

# train 데이터와 validation 데이터 필터링
df_train = df[df["train_acc"].notna()]  # train 데이터 (step 단위)
df_val = df[df["val_acc"].notna()]  # validation 데이터 (epoch 단위)

data = {
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