# tests/test_module.py

import torch
from plinear.plinear import PLinear, PF

def test_basic_forward():
    model = PLinear(10, 5)
    x = torch.randn(1, 10)
    output = model(x)
    assert output.shape == (1, 5), "Output shape mismatch"