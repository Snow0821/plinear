from torch import nn
# from plinear import bctnn
from plinear import btnn

class btAttn(nn.Module):
    def __init__(self, dim, num_heads):
        super(btAttn, self).__init__()
        self.q = btnn.Linear(dim, dim)
        self.k = btnn.Linear(dim, dim)
        self.mixer = btnn.Conv1d(num_heads, num_heads, 1)

    def forward(self, x):
        q, k = self.q(x), self.k(x)
        attn = q * k

        return self.mixer(attn)

# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads

#         self.qkv = btnn.Linear(embed_dim, embed_dim * 3)
#         self.proj = btnn.Linear(embed_dim, embed_dim)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
#         q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]

#         # attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # Scaled dot-product attention
#         attn = (q @ k.transpose(-2, -1))  # Just dot-product attention
#         # attn = attn.softmax(dim=-1) # skipping softmax
#         out = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         return self.proj(out)