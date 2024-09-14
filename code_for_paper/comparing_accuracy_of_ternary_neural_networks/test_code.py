import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def mnist_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
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
            preds.extend(pred.cpu().tolist())
            labels.extend(y.cpu().tolist())

    return preds, labels

from torch import nn
from code_for_paper.comparing_accuracy_of_ternary_neural_networks.bitLinear import RMSNorm

class Model(nn.Module):
    def __init__(self, linear, width, depth, i, o, RMSNorm = False):
        super(Model, self).__init__()
        self.reader = linear(i, width)
        self.layers = [linear(width, width) for _ in range(depth)]
        self.writer = linear(width, o)
        self.RMSNorm = RMSNorm
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.reader(x)
        for layer in self.layers:
            if self.RMSNorm:
                x = RMSNorm(x)
            x = layer(x)
        x = self.writer(x)
        return x
        
from sklearn.metrics import accuracy_score
import torch.optim as optim

def case(linear, width, depth, epochs, features, target, device, data, rms):
    train_loader, test_loader = data()
    model = Model(linear, width, depth, features, target, rms)
    opt = optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss()
    model = model.to(device)
    

    for epoch in range(epochs):
        train(model, train_loader, crit, opt, device)
        preds, labels = test(model, test_loader)
        print(f"Epoch {epoch + 1} / {epochs} Acc = {accuracy_score(labels, preds)}")

from code_for_paper.comparing_accuracy_of_ternary_neural_networks import bitLinear, ternaries

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 3
    linears = [
        nn.Linear, 
        bitLinear.BitLinear,
        ternaries.naive_tern_det, 
        ternaries.naive_tern_stoc,
        ternaries.pow2_tern,
        ternaries.exp_tern_det,
        ternaries.exp_tern_stoc
        ]
    names = [
        "linear",
        "bitlinear",
        "naive_det",
        "naive_stoc",
        "pow2",
        "exp_det",
        "exp_stoc"
    ]
    feature_size = 28 * 28
    target_size = 10
    width = 32
    depth = 3
    data = mnist_data
    for i in range(2, len(linears)):
        linear = linears[i]
        print(names[i])
        case(linear, width, depth, epochs, feature_size, target_size, device, data, False)
if __name__ == "__main__":
    main()