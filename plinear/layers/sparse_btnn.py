import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseBtnn_Selector(nn.Module):
    def __init__(self, *dim):
        super(SparseBtnn_Selector, self).__init__()
        self.selector = nn.Linear(*dim)

        torch.nn.init.uniform_(self.selector.weight, -1, 1)
    
    def forward(self, x):
        selector = self.selector.weight

        mask = torch.zeros_like(self.selector.weight)
        mask.scatter_(1, self.selector.weight.argmax(dim=1, keepdim=True), 1.0)

        masked = mask - selector.detach() + selector

        return F.linear(masked, x)

class SparseBtnn_And(nn.Module):
    def __init__(self, *dim):
        super(SparseBtnn_And, self).__init__()
        self.a = SparseBtnn_Selector(*dim)
        self.b = SparseBtnn_Selector(*dim)
    
    def forward(self, x):
        a = self.a(x)
        b = self.b(x)

        return a * b

class SparseBtnn_Not(nn.Module):
    def __init__(self, *dim):
        super(SparseBtnn_Not, self).__init__()
        self.a = nn.parameter.Parameter(torch.randn(*dim),)

    def forward(self, x):
        a = self.a.expand(x.shape[-1], -1)
        qa = (a > 0).float() - a.detach() + a
        x = x.permute(1, 0)

        return qa + x - 2 * qa * x
    
class SparseBtnn_Nand(nn.Module):
    def __init__(self, x, y):
        super(SparseBtnn_Nand, self).__init__()
        self.a = SparseBtnn_And(x, y)
        self.n = SparseBtnn_Not(y)
    
    def forward(self, x):
        out = self.a(x)
        out = self.n(out)
        return out
    
class SparseBtnn_Nand_Multihead(nn.Module):
    def __init__(self, x, y, n_heads):
        super(SparseBtnn_Nand_Multihead, self).__init__()
        self.attns = nn.ModuleList([
            SparseBtnn_Nand(x, y) for _ in range(n_heads)
        ])
    
    def forward(self, x):
        return 