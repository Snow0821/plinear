import torch

class Binarizer:
    @staticmethod
    def posNet(w: torch.Tensor):
        return (w > 0).float()
