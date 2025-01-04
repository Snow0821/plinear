import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import PyTorchModelHubMixin

from plinear.core import RMSNorm
from plinear import btnn
from plinear import blocks

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.projection = btnn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x).flatten(2).permute(0, 2, 1)
        return x
    
class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super(Encoder, self).__init__()
        self.attention = blocks.btAttn(embed_dim, num_heads)
        self.bn1 = nn.BatchNorm1d(num_heads)
        self.bn2 = nn.BatchNorm1d(num_heads)
        self.mlp = nn.Sequential(
            btnn.Linear(embed_dim, mlp_dim),
            btnn.RMSNorm(),
            nn.ReLU(),
            btnn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attention(self.bn1(x))
        x = x + self.mlp(self.bn2(x))
        return x

class btViT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, embed_dim, depth, mlp_dim, img_size, patch_size, channels, num_classes):
        super(btViT, self).__init__()
        num_heads = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(patch_size, channels, embed_dim)
        self.transformer = nn.Sequential(
            *[Encoder(embed_dim, num_heads, mlp_dim) for _ in range(depth)]
        )
        self.reducer = btnn.Conv1d(num_heads, 1, kernel_size=1)
        self.mlp_head = btnn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.reducer(x)
        x = x.flatten(1)
        x = self.mlp_head(RMSNorm(x))
        return x
