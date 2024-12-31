import torch
import torch.nn as nn
import torch.nn.functional as F

from plinear.core import RMSNorm

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.projection(x).flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_token, x), dim=1)  # [B, num_patches + 1, embed_dim]
        x = x + self.pos_embed
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = model(embed_dim, embed_dim * 3)
        self.proj = model(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # Scaled dot-product attention
        # attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, model):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, model)
        self.mlp = nn.Sequential(
            model(embed_dim, mlp_dim),
            nn.ReLU(),
            model(mlp_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attention(RMSNorm(x))
        x = x + self.mlp(RMSNorm(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, depth, mlp_dim, num_classes, model = nn.Linear):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, model) for _ in range(depth)]
        )
        self.mlp_head = model(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        cls_token = x[:, 0]  # Extract the [CLS] token
        return self.mlp_head(RMSNorm(cls_token))
