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
