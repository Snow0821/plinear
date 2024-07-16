import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PF:
    @staticmethod
    def posNet(w: torch.Tensor):
        return (w > 0).float()

class PLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_bias: bool = False):
        super(PLinear, self).__init__()
        self.use_bias = use_bias
        self.linear_pos = nn.Linear(in_features, out_features, bias=False)
        self.linear_neg = nn.Linear(in_features, out_features, bias=False)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clip weights to be within -1 and 1
        # self.linear_pos.weight.data.clamp_(-1, 1)
        # self.linear_neg.weight.data.clamp_(-1, 1)
        # Soft clipping using tanh
        w_pos = torch.tanh(self.linear_pos.weight)
        w_neg = torch.tanh(self.linear_neg.weight)

        # w_pos = self.linear_pos.weight
        # w_neg = self.linear_neg.weight

        # Apply quantization using posNet with detach
        w_pos_quant = w_pos + (PF.posNet(w_pos) - w_pos).detach()
        w_neg_quant = w_neg + (PF.posNet(w_neg) - w_neg).detach()

        # Compute linear transformations
        y_pos = F.linear(x, w_pos_quant, self.bias if self.use_bias else None)
        y_neg = F.linear(x, w_neg_quant, self.bias if self.use_bias else None)

        # Combine positive and negative parts
        y = y_pos - y_neg

        return y