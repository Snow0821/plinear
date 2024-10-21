import torch
import torch.nn as nn
import torch.nn.functional as F

class PLinear(nn.Module):
    def posNet(w: torch.Tensor):
        bin_w = torch.full_like(w, 0.0)  # default 0
        bin_w = torch.where(w > 0, 1.0, bin_w)
        return bin_w

    def __init__(self, in_features: int, out_features: int):
        super(PLinear, self).__init__()
        self.real_pos = nn.Linear(in_features, out_features)
        self.real_neg = nn.Linear(in_features, out_features)

        torch.nn.init.uniform_(self.real_pos.weight, -1, 1)
        torch.nn.init.uniform_(self.real_neg.weight, -1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_pos = self.real_pos.weight.to(x.device)
        w_neg = self.real_neg.weight.to(x.device)
        tern_pos = self.posNet(w_pos)
        tern_neg = self.posNet(w_neg)

        # Apply quantization using posNet with detach
        tern_pos = tern_pos - w_pos.detach() + w_pos
        tern_neg = tern_neg - w_neg.detach() + w_neg

        # Compute linear transformations
        y_pos = F.linear(x, tern_pos)
        y_neg = F.linear(x, tern_neg)

        # Combine positive and negative parts
        y = y_pos - y_neg

        return y

# class PLinear_Complex(nn.Module):
#     def __init__(self, ins, outs):
#         super(PLinear_Complex, self).__init__()
#         self.real_pos = nn.Linear(ins, outs)
#         self.real_neg = nn.Linear(ins, outs)
#         self.complex_pos = nn.Linear(ins, outs)
#         self.complex_neg = nn.Linear(ins, outs)
    
#     def forward(self, x_real: torch.Tensor, x_complex: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         real_pos = torch.tanh(self.real_pos.weight).to(x_real.device)
#         real_neg = torch.tanh(self.real_neg.weight).to(x_real.device)
#         complex_pos = torch.tanh(self.complex_pos.weight).to(x_real.device)
#         complex_neg = torch.tanh(self.complex_neg.weight).to(x_real.device)

#         real_pos_q = real_pos + (PF.posNet(real_pos) - real_pos).detach().to(x_real.device)
#         real_neg_q = real_neg + (PF.posNet(real_neg) - real_neg).detach().to(x_real.device)
#         complex_pos_q = complex_pos + (PF.posNet(complex_pos) - complex_pos).detach().to(x_real.device)
#         complex_neg_q = complex_neg + (PF.posNet(complex_neg) - complex_neg).detach().to(x_real.device)

#         y_real = F.linear(x_real, real_pos_q) - F.linear(x_real, real_neg_q) + F.linear(x_complex, complex_pos_q) - F.linear(x_complex, complex_neg_q)
#         y_complex = F.linear(x_real, complex_pos_q) - F.linear(x_real, complex_neg_q) + F.linear(x_complex, real_pos_q) - F.linear(x_complex, real_neg_q)

#         return y_real, y_complex