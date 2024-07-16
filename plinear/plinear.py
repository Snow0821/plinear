import torch
import torch.nn as nn
import torch.nn.functional as F

class PF:
    def posNet(w: torch.Tensor):
        return (w > 0).float()

    def negNet(w: torch.Tensor):
        return -(w <= 0).float()

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
        w_pos = self.linear_pos.weight
        w_neg = self.linear_neg.weight

        w_pos_quant = w_pos + (PF.posNet(w_pos) - w_pos).detach()
        w_neg_quant = w_neg + (PF.negNet(w_neg) - w_neg).detach()

        y_pos = F.linear(x, w_pos_quant, self.bias if self.use_bias else None)
        y_neg = F.linear(x, w_neg_quant, self.bias if self.use_bias else None)

        y = y_pos + y_neg
        return y


