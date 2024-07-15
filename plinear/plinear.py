import torch.nn as nn

class PLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)
