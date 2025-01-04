import torch.nn as nn
import plinear.core as core

class RMSNorm(nn.Module):
    def __init__(self):
        super(RMSNorm, self).__init__()
    
    def forward(self, x):
        return core.RMSNorm(x)
