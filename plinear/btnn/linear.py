import torch
import torch.nn as nn
import torch.nn.functional as F

from plinear.core import posNet

class Linear(nn.Module):
    def __init__(self, x, y):
        super(Linear, self).__init__()
        self.pr = nn.Linear(x, y)
        self.nr = nn.Linear(x, y)

        torch.nn.init.uniform_(self.pr.weight, -1, 1)
        torch.nn.init.uniform_(self.nr.weight, -1, 1)

    def forward(self, x):
        pr = self.pr.weight
        nr = self.nr.weight
        qpr = posNet(pr)
        qnr = posNet(nr)

        # Apply quantization using posNet with detach
        qpr = qpr - pr.detach() + pr
        qnr = qnr - nr.detach() + nr

        # Compute linear transformations
        yr = F.linear(x, qpr) - F.linear(x, qnr)

        return yr