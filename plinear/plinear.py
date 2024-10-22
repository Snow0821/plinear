import torch
import torch.nn as nn
import torch.nn.functional as F

class PF:
    @staticmethod
    def posNet(w: torch.Tensor):
        return (w > 0).float()

class PLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(PLinear, self).__init__()
        self.real_pos = nn.Linear(in_features, out_features)
        self.real_neg = nn.Linear(in_features, out_features)

        torch.nn.init.uniform_(self.real_pos.weight, -1, 1)
        torch.nn.init.uniform_(self.real_neg.weight, -1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_pos = self.real_pos.weight
        w_neg = self.real_neg.weight
        tern_pos = PF.posNet(w_pos)
        tern_neg = PF.posNet(w_neg)

        # Apply quantization using posNet with detach
        tern_pos = tern_pos - w_pos.detach() + w_pos
        tern_neg = tern_neg - w_neg.detach() + w_neg

        # Compute linear transformations
        y_pos = F.linear(x, tern_pos)
        y_neg = F.linear(x, tern_neg)

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

        torch.nn.init.uniform_(self.real_pos.weight, -1, 1)
        torch.nn.init.uniform_(self.real_neg.weight, -1, 1)
        torch.nn.init.uniform_(self.complex_pos.weight, -1, 1)
        torch.nn.init.uniform_(self.complex_neg.weight, -1, 1)
    
    def forward(self, x_real: torch.Tensor, x_complex: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real_pos = self.real_pos.weight
        real_neg = self.real_neg.weight
        complex_pos = self.complex_pos.weight
        complex_neg = self.complex_neg.weight

        real_pos_q = PF.posNet(real_pos)
        real_neg_q = PF.posNet(real_neg)
        complex_pos_q = PF.posNet(complex_pos)
        complex_neg_q = PF.posNet(complex_neg)

        real_pos_q = real_pos_q - real_pos.detach() + real_pos
        real_neg_q = real_neg_q - real_neg_q.detach() + real_neg_q
        complex_pos_q = complex_pos_q - complex_pos_q.detach() + complex_pos_q
        complex_neg_q = complex_neg_q - complex_neg_q.detach() + complex_neg_q

        y_real = F.linear(x_real, real_pos_q) - F.linear(x_real, real_neg_q) - F.linear(x_complex, complex_pos_q) + F.linear(x_complex, complex_neg_q)
        y_complex = F.linear(x_real, complex_pos_q) - F.linear(x_real, complex_neg_q) + F.linear(x_complex, real_pos_q) - F.linear(x_complex, real_neg_q)

        return y_real, y_complex