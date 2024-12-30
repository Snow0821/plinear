#RMSNorm : https://github.com/microsoft/unilm/blob/master/YOCO/yoco/models/decoder/rms_norm.py

import torch

def RMSNorm(x, eps: float = 1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)