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
        # Tanh makes network learn smoothly somehow. especially with lr of 1 and no momentum.
        w_pos = torch.tanh(self.linear_pos.weight)
        w_neg = torch.tanh(self.linear_neg.weight)

        # Apply quantization using posNet with detach
        w_pos_quant = w_pos + (PF.posNet(w_pos) - w_pos).detach()
        w_neg_quant = w_neg + (PF.posNet(w_neg) - w_neg).detach()

        # Compute linear transformations
        y_pos = F.linear(x, w_pos_quant, self.bias if self.use_bias else None)
        y_neg = F.linear(x, w_neg_quant, self.bias if self.use_bias else None)

        # Combine positive and negative parts
        y = y_pos - y_neg

        return y

class PLinear_Complex(nn.Module):
    def __init__(self, ins, outs):
        super(PLinear_Complex, self).__init__()
        self.real_pos = nn.Linear(ins, outs)
        self.real_neg = nn.Linear(ins, outs)
        self.complex_pos = nn.Linear(ins, outs)
        self.complex_neg = nn.Linear(ins, outs)
    
    def forward(self, x_real: torch.Tensor, x_complex: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real_pos = torch.tanh(self.real_pos.weight)
        real_neg = torch.tanh(self.real_neg.weight)
        complex_pos = torch.tanh(self.complex_pos.weight)
        complex_neg = torch.tanh(self.complex_neg.weight)

        real_pos_q = real_pos + (PF.posNet(real_pos) - real_pos).detach()
        real_neg_q = real_neg + (PF.posNet(real_neg) - real_neg).detach()
        complex_pos_q = complex_pos + (PF.posNet(complex_pos) - complex_pos).detach()
        complex_neg_q = complex_neg + (PF.posNet(complex_neg) - complex_neg).detach()

        y_real = F.linear(x_real, real_pos_q) - F.linear(x_real, real_neg_q) + F.linear(x_complex, complex_pos_q) - F.linear(x_complex, complex_neg_q)
        y_complex = F.linear(x_real, complex_pos_q) - F.linear(x_real, complex_neg_q) + F.linear(x_complex, real_pos_q) - F.linear(x_complex, real_neg_q)

        return y_real, y_complex