from torch import nn
from plinear import bctnn
from plinear import btnn

class bctAttn(nn.Module):
    def __init__(self, dim):
        super(bctAttn, self).__init__()

        self.q = btnn.Linear(dim, dim)
        self.k = btnn.Linear(dim, dim)
        self.mixer = bctnn.Linear(dim, dim)

    def forward(self, x):
        q, k = self.q(x), self.k(x)
        q, k = self.mixer(q, k)
        attn = q * k

        return attn