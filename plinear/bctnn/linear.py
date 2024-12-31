import torch
import torch.nn as nn
import torch.nn.functional as F

from core.posNet import posNet

class PLinear_Complex(nn.Module):
    def __init__(self, ins, outs):
        super(PLinear_Complex, self).__init__()
        self.real_pos = nn.Linear(ins, outs)
        self.real_neg = nn.Linear(ins, outs)
        self.complex_pos = nn.Linear(ins, outs)
        self.complex_neg = nn.Linear(ins, outs)

        torch.nn.init.uniform_(self.real_pos.weight, -1, 1)
        torch.nn.init.uniform_(self.real_neg.weight, -1, 1)
        torch.nn.init.uniform_(self.complex_pos.weight, -1, 1)
        torch.nn.init.uniform_(self.complex_neg.weight, -1, 1)
    
    def forward(self, x_real: torch.Tensor, x_complex: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real_pos = self.real_pos.weight
        real_neg = self.real_neg.weight
        complex_pos = self.complex_pos.weight
        complex_neg = self.complex_neg.weight

        real_pos_q = posNet(real_pos)
        real_neg_q = posNet(real_neg)
        complex_pos_q = posNet(complex_pos)
        complex_neg_q = posNet(complex_neg)

        real_pos_q = real_pos_q - real_pos.detach() + real_pos
        real_neg_q = real_neg_q - real_neg_q.detach() + real_neg_q
        complex_pos_q = complex_pos_q - complex_pos_q.detach() + complex_pos_q
        complex_neg_q = complex_neg_q - complex_neg_q.detach() + complex_neg_q

        y_real = F.linear(x_real, real_pos_q) - F.linear(x_real, real_neg_q) - F.linear(x_complex, complex_pos_q) + F.linear(x_complex, complex_neg_q)
        y_complex = F.linear(x_real, complex_pos_q) - F.linear(x_real, complex_neg_q) + F.linear(x_complex, real_pos_q) - F.linear(x_complex, real_neg_q)

        return y_real, y_complex