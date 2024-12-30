import torch
import torch.nn as nn
import torch.nn.functional as F

from core.posNet import posNet

class Linear(nn.Module):
    def __init__(self, x, y):
        super(Linear, self).__init__()
        self.real_pos = nn.Linear(x, y)
        self.real_neg = nn.Linear(x, y)

        torch.nn.init.uniform_(self.real_pos.weight, -1, 1)
        torch.nn.init.uniform_(self.real_neg.weight, -1, 1)

    def forward(self, x):
        w_pos = self.real_pos.weight
        w_neg = self.real_neg.weight
        tern_pos = posNet(w_pos)
        tern_neg = posNet(w_neg)

        # Apply quantization using posNet with detach
        tern_pos = tern_pos - w_pos.detach() + w_pos
        tern_neg = tern_neg - w_neg.detach() + w_neg

        # Compute linear transformations
        y_pos = F.linear(x, tern_pos)
        y_neg = F.linear(x, tern_neg)

        # Combine positive and negative parts
        y = y_pos - y_neg

        return y