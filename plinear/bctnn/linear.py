import torch
import torch.nn as nn
import torch.nn.functional as F

from plinear.core import posNet

class Linear(nn.Module):
    def __init__(self, x, y):
        super(Linear, self).__init__()
        self.pr = nn.Linear(x, y)
        self.nr = nn.Linear(x, y)
        self.pi = nn.Linear(x, y)
        self.ni = nn.Linear(x, y)

        torch.nn.init.uniform_(self.pr.weight, -1, 1)
        torch.nn.init.uniform_(self.nr.weight, -1, 1)
        torch.nn.init.uniform_(self.pi.weight, -1, 1)
        torch.nn.init.uniform_(self.ni.weight, -1, 1)
    
    def forward(self, xr, xi):
        pr = self.pr.weight
        nr = self.nr.weight
        pi = self.pi.weight
        ni = self.ni.weight

        qpr = posNet(pr) - pr.detach() + pr
        qnr = posNet(nr) - nr.detach() + nr
        qpi = posNet(pi) - pi.detach() + pi
        qni = posNet(ni) - ni.detach() + ni

        yr = F.linear(xr, qpr) - F.linear(xr, qnr) - F.linear(xi, qpi) + F.linear(xi, qni)
        yi = F.linear(xr, qpi) - F.linear(xr, qni) + F.linear(xi, qpr) - F.linear(xi, qnr)

        return yr, yi