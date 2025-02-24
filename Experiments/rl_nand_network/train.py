import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0).float())  # Binarize
])

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

def check_binary_tensor(tensor, tensor_name="tensor"):
    assert torch.all((tensor == 0) | (tensor == 1)), f"{tensor_name} contains values other than -1 or 1! as {tensor}"

def my_hard_mask(logits):
    softmax_probs = torch.softmax(logits, dim=1)
    hard_mask = torch.zeros_like(softmax_probs).scatter_(1, softmax_probs.argmax(dim=1, keepdim=True), 1.0)
    return hard_mask - softmax_probs.detach() + softmax_probs  # STE

def my_action_mask(logits, action):
    softmax_probs = torch.softmax(logits, dim=1)
    hard_mask = torch.zeros_like(softmax_probs).scatter_(1, action, 1.0)
    return hard_mask - softmax_probs.detach() + softmax_probs  # STE

class SparseBtnn_Selector(nn.Module):
    def __init__(self, x, y):
        super(SparseBtnn_Selector, self).__init__()
        self.selector = nn.Linear(x, y, bias=False)
        torch.nn.init.xavier_uniform_(self.selector.weight)
    
    def forward(self, x, reward=None, train_mode=False):
        selector_logits = self.selector.weight
        action_probs = torch.softmax(selector_logits, dim=1)
        
        if train_mode and reward is not None:
            action = torch.multinomial(action_probs, num_samples=1)
            policy_loss = -10 * torch.log(action_probs.gather(1, action)) * reward
            policy_loss.mean().backward(retain_graph=True)  # retain_graph
            mask = my_action_mask(selector_logits, action)
        else:
            mask = my_hard_mask(selector_logits)  # had mask
        out = F.linear(x, mask)
        check_binary_tensor(out, "selector")
        return out

class Compacted_Nand(nn.Module):
    def __init__(self, x, y):
        super().__init__()
        self.a = SparseBtnn_Selector(x, y)
        self.b = SparseBtnn_Selector(x, y)
    
    def forward(self, x, reward=None, train_mode=False):
        out_a = self.a(x, reward=reward, train_mode=train_mode)
        out_b = self.b(x, reward=reward, train_mode=train_mode)
        out = 1 - (out_a * out_b)
        check_binary_tensor(out, "nand")
        return out

def get_reward(prediction, target):
    return torch.where(prediction == target, torch.tensor(1.0, device=prediction.device), torch.tensor(-1.0, device=prediction.device))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Compacted_Nand(28*28, 10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).view(data.size(0), -1), target.to(device)
        
        logits = model(data, train_mode=True)
        reward = get_reward(logits.argmax(dim=1), target)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == target).sum().item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {correct / len(train_loader.dataset):.4f}")
