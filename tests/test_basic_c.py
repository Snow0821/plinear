# tests/test_module.py

import torch
from plinear.plinear import PLinear_Complex as PL, PF

def test_basic_forward():
    model = PL(10, 5)
    x = torch.randn(1, 10)
    x2 = torch.zeros(1, 10)
    real, complex = model(x, x2)

    assert real.shape == (1, 5), "real shape mismatch"
    assert complex.shape == (1, 5), "complex shape mismatch"

# def test_weight_quantization():
#     model = PL(10, 5)
#     x = torch.randn(1, 10)
#     x2 = torch.randn(1, 10)
#     real, complex = model(x, x2)
    
#     w_pos = model.linear_pos.weight
#     w_neg = model.linear_neg.weight

#     w_pos_quant = w_pos + (PF.posNet(w_pos) - w_pos).detach()
#     w_neg_quant = w_neg + (PF.posNet(w_neg) - w_neg).detach()

#     real_pos = real.
#     real_neg = torch.tanh(self.real_neg)
#     complex_pos = torch.tanh(self.complex_pos)
#     complex_neg = torch.tanh(self.complex_neg)

#     real_pos_q = real_pos + (PF.posNet(real_pos) - real_pos).detach()
#     real_neg_q = real_neg + (PF.posNet(real_neg) - real_neg).detach()
#     complex_pos_q = complex_pos + (PF.posNet(complex_pos) - complex_pos).detach()
#     complex_neg_q = complex_neg + (PF.posNet(complex_neg) - complex_neg).detach()

#     assert torch.all((w_pos_quant == 0) | (w_pos_quant == 1)), "Weight quantization for posNet failed"
#     assert torch.all((w_neg_quant == 0) | (w_neg_quant == 1)), "Weight quantization for negNet failed"
