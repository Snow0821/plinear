import torch

def posNet(w: torch.Tensor):
    return (w > 0).float()
