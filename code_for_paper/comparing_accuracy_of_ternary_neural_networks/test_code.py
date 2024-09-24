#https://ar5iv.labs.arxiv.org/html/1906.00425
#https://ar5iv.labs.arxiv.org/html/1905.11946

import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from code_for_paper.comparing_accuracy_of_ternary_neural_networks import bitLinear, ternaries

def mnist_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader, "mnist"

def cifar100_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader, "cifar100"

def zero_ratio_after_tern(model):
    zero_count = 0
    total_count = 0
    for name, module in model.named_modules():
        if isinstance(module, ternaries.ternary):
            tern_w = module.tern(module.weight)
            zero_count += (tern_w == 0).sum().item()
            total_count += tern_w.numel()
    if total_count:
        zero_ratio = zero_count / total_count
        return zero_ratio
    return

def train(model, train_loader, crit, opt, device):
    model.train()
    losses = 0
    samples = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        outputs = model(x)
        loss = crit(outputs, y)
        losses += loss.item()
        samples += x.size(0)
        loss.backward()
        opt.step()

    epoch_loss = losses / samples
    print(f"Loss : {epoch_loss} ")
    
    return epoch_loss

def test(model, test_loader):
    # model.eval()
    preds = []
    labels = []

    # with torch.no_grad():
    with torch.inference_mode():
        for x, y in test_loader:
            outputs = model(x)
            pred = outputs.argmax(dim = 1)
            preds.extend(pred.cpu().tolist())
            labels.extend(y.cpu().tolist())

    return preds, labels

from torch import nn
from code_for_paper.comparing_accuracy_of_ternary_neural_networks.bitLinear import RMSNorm

class Model(nn.Module):
    def __init__(self, linear, width, depth, i, o):
        super(Model, self).__init__()
        self.reader = linear(i, width)
        self.layers = nn.ModuleList([linear(width, width) for _ in range(depth)])
        # self.layer_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.writer = linear(width, o)
        self.ReLU = nn.LeakyReLU()
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.reader(x)
        # for layer, ln in zip(self.layers, self.layer_norms):
        for layer in self.layers:
            x = RMSNorm(x)
            x = layer(x)
            # x = ln(x)
            x = self.ReLU(x)
        x = self.writer(x)
        return x
        
from sklearn.metrics import accuracy_score
import torch.optim as optim
import code_for_paper.comparing_accuracy_of_ternary_neural_networks.vis as vis
from plinear.plinear import PLinear

def case(linear, name, width, depth, epochs, features, target, device, data):
    train_loader, test_loader, dname = data()
    model = Model(linear, width, depth, features, target)
    opt = optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss().to(device)
    model = model.to(device)
    test_accuracies = []
    zero_ratios = []

    for epoch in range(epochs):
        train(model, train_loader, crit, opt, device)
        preds, labels = test(model, test_loader)
        accuracy = accuracy_score(labels, preds)
        test_accuracies.append(accuracy)
        with torch.no_grad():
            zero_ratios.append(zero_ratio_after_tern(model))
        print(f"Epoch {epoch + 1} / {epochs} Acc = {accuracy}")
    with torch.no_grad():
        vis.save_parameters(model, dname, width, depth, name)
    vis.save_results_to_csv(dname, name, width, depth, test_accuracies, "accuracy")
    vis.save_results_to_csv(dname, name, width, depth, zero_ratios, "zero_ratio")

import gc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    linears = [
        nn.Linear, 
        bitLinear.BitLinear,
        ternaries.naive_tern_det, 
        ternaries.naive_tern_stoc,
        ternaries.pow2_tern,
        ternaries.exp_tern_det,
        ternaries.exp_tern_stoc,
        PLinear
        ]
    names = [
        "linear",
        "bitlinear_tern",
        "naive_tern_det",
        "naive_tern_stoc",
        "pow2_tern",
        "exp_tern_det",
        "exp_tern_stoc",
        "plinear"
    ]

    zipped = list(zip(names, linears))
    epochs = 100

    data = mnist_data
    dname = 'mnist'
    feature_size = 28 * 28
    target_size = 10
    
    for d in range(4):
        depth = 2**d
        for w in range(1, 5):
            width = 4 * 4**w
            for name, linear in zipped:
                print(name)
                case(linear, name, width, depth, epochs, feature_size, target_size, device, data)
                gc.collect()
            vis.plot_from_csv(dname, width, depth, "accuracy")
            vis.plot_from_csv(dname, width, depth, "zero_ratio")

    data = cifar100_data
    dname = 'cifar100'
    feature_size = 32 * 32 * 3
    target_size = 100

    for d in range(4):
        depth = 2**d
        for w in range(1, 5):
            width = 4 * 4**w
            for name, linear in zipped:
                print(name)
                case(linear, name, width, depth, epochs, feature_size, target_size, device, data)
                gc.collect()
            vis.plot_from_csv(dname, width, depth, "accuracy")
            vis.plot_from_csv(dname, width, depth, "zero_ratio")
            


def stressTest():
    data = mnist_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(100):
        case(ternaries.exp_tern_stoc, 'exp_tern_stoc', 512, 3, 3, 28*28, 10, device, data)
    return

if __name__ == "__main__":
    # stressTest()
    main()