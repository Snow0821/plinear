import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ResNet-18 모델 정의
def ResNet18(num_classes=1000):
    return models.resnet18(pretrained=False, num_classes=num_classes)

# 데이터 전처리 및 로드
def get_dataloaders(data_dir, batch_size, num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = datasets.ImageFolder(root=f"{data_dir}", transform=transform_train)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}", transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

# 학습 함수
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10):
    train_loader, val_loader = dataloaders
    
    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0
        
        # Training phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (outputs.argmax(1) == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        
        # Validation phase
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += (outputs.argmax(1) == labels).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")

# 실행 코드
if __name__ == "__main__":
    # 설정
    data_dir = "./data"  # ImageNet 데이터셋 경로
    batch_size = 64
    num_epochs = 10
    num_classes = 1000

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    
    # 데이터 로더 준비
    train_loader, val_loader = get_dataloaders(data_dir, batch_size)
    
    # 모델 준비
    model = ResNet18(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 학습
    train_model(model, (train_loader, val_loader), criterion, optimizer, device, num_epochs)