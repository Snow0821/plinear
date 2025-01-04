#Referenced from https://openreview.net/pdf?id=rAnB7JSMXL

import torch.nn as nn
import plinear.btnn as btnn
import plinear.core.RMSNorm as RMSNorm
from huggingface_hub import PyTorchModelHubMixin

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return self.fn(x) + x

# def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
#     return nn.Sequential(
#         nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
#         nn.GELU(),
#         nn.BatchNorm2d(dim),
#         *[nn.Sequential(
#             Residual(nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
#             nn.GELU(),
#             nn.BatchNorm2d(dim)
#             )),
#             nn.Conv2d(dim, dim, kernel_size=1),
#             nn.GELU(),
#             nn.BatchNorm2d(dim)
#             ) for i in range(depth)],
#         nn.AdaptiveAvgPool2d((1,1)),
#         nn.Flatten(),
#         nn.Linear(dim, n_classes)
#     )

class ConvMixer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, dim, depth, kernel_size, patch_size, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            btnn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            *[nn.Sequential(
                Residual(nn.Sequential(
                    btnn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.ReLU(),
                    nn.BatchNorm2d(dim)
                    )),
                btnn.Conv2d(dim, dim, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(dim)
            ) for _ in range(depth)],
            # btnn.Conv2d(dim, 1, kernel_size = 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            btnn.Linear(dim, n_classes)
        )
    
    def forward(self, x):
        return self.model(x)
