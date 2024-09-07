import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def mnist_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader

def train(model, train_loader, crit, opt, device):
    model.train()
    losses = 0
    samples = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        outputs = model(x)
        loss = crit(outputs, y)
        losses += loss
        samples += x.size(0)
        loss.backward()
        opt.step()

    epoch_loss = losses / samples
    print(f"Loss : {epoch_loss} ")
    return epoch_loss


def test(model, test_loader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            pred = outputs.argmax(dim = 1)
            preds.extend(pred.cpu().tolist)
            labels.extend(y.cpu().tolist)

    return preds, labels

from torch import nn

class Model(nn.Module):
    def __init__(self, linear, width, depth):
        super(Model, self).__init__()
        self.layers = [linear(width) for _ in range(depth)]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
from sklearn.metrics import accuracy_score
import torch.optim as optim

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(nn.Linear, 32, 3)
    epochs = 3
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters())


    model = model.to(device)
    train_loader, test_loader = mnist_data

    for epoch in range(epochs):
        train(model, train_loader, crit, opt, device)
        preds, labels = test(model, test_loader)
        print(f"Epoch {epoch + 1} / {epochs} Acc = {accuracy_score(labels, preds)}")


if __name__ == "__main__":
    main()