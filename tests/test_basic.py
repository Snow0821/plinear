# tests/test_module.py

import torch
from plinear.plinear import PLinear, PF

def test_basic_forward():
    model = PLinear(10, 5, use_bias=False)
    x = torch.randn(1, 10)
    output = model(x)
    assert output.shape == (1, 5), "Output shape mismatch"

def test_basic_forward_with_bias():
    model = PLinear(10, 5, use_bias=True)
    x = torch.randn(1, 10)
    output = model(x)
    assert output.shape == (1, 5), "Output shape mismatch"

def test_weight_quantization():
    model = PLinear(10, 5, use_bias=False)
    x = torch.randn(1, 10)
    output = model(x)
    
    w_pos = model.linear_pos.weight
    w_neg = model.linear_neg.weight

    w_pos_quant = w_pos + (PF.posNet(w_pos) - w_pos).detach()
    w_neg_quant = w_neg + (PF.posNet(w_neg) - w_neg).detach()

    assert torch.all((w_pos_quant == 0) | (w_pos_quant == 1)), "Weight quantization for posNet failed"
    assert torch.all((w_neg_quant == 0) | (w_neg_quant == 1)), "Weight quantization for negNet failed"
