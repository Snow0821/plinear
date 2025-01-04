import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import PyTorchModelHubMixin

from plinear.core import RMSNorm
from plinear import btnn
from plinear import blocks

class TransformerReducer(nn.Module):
    def __init__(self, de):
        return

class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super(Encoder, self).__init__()
        self.attention = blocks.btAttn(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            btnn.Linear(embed_dim, mlp_dim),
            btnn.RMSNorm(),
            nn.ReLU(),
            btnn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attention(RMSNorm(x))
        x = x + self.mlp(RMSNorm(x))
        return x

class btViT(nn.Module, PyTorchModelHubMixin,):
    def __init__(self, embed_dim, num_heads, depth, mlp_dim, num_classes):
        super(btViT, self).__init__()
        self.transformer = nn.Sequential(
            *[Encoder(embed_dim, num_heads, mlp_dim) for _ in range(depth)]
        )
        self.mlp_head = btnn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        cls_token = x[:, 0]  # Extract the [CLS] token
        return self.mlp_head(RMSNorm(cls_token))
