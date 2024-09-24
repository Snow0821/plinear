#https://arxiv.org/pdf/1608.06902

import torch
import torch.nn as nn
import torch.nn.functional as F

class ternary(nn.Linear):
    def __init__(self, feat, tar):
        super().__init__(feat, tar)
        torch.nn.init.uniform_(self.weight, -1, 1)

    def tern(self, w):
        return w

    def forward(self, x):
        w = self.weight
        tern_w = self.tern(w)
        tern_w = tern_w - w.detach() + w
        return F.linear(x, tern_w)

class naive_tern_det(ternary):
    # def __init__(self, feat, tar):
    #     super().__init__(feat, tar)
    #     torch.nn.init.uniform_(self.weight, -1, 1)

    def tern(self, w):
        tern_w = torch.full_like(w, 0.0)  # default 0
        tern_w = torch.where(w > 0.5, 1.0, tern_w)
        tern_w = torch.where(w <= -0.5, -1.0, tern_w)

        return tern_w

class naive_tern_stoc(ternary):
    def tern(self, w):
        prob = torch.clip(torch.abs(2 * w), 0, 1)
        sign = torch.sign(w)
        sample = torch.bernoulli(prob)
        return sign * sample

class pow2_tern(ternary):
    # def __init__(self, feat, tar):
    #     super().__init__(feat, tar)
    #     torch.nn.init.uniform_(self.weight, -1, 1)

    def tern(self, w):
        # Step 1: Clip weights to the range allowed by 2-bit precision (-2 to 2)
        w_clip = torch.clip(w, -2, 2)

        # Step 2: Scale the weights by 2^fractional_bits and round them
        w_scaled = torch.round(w_clip * 2) / 2

        # Step 3: Apply ternary quantization (-1, 0, 1)
        tern_w = torch.full_like(w_scaled, 0.0)  # default 0
        tern_w = torch.where(w_scaled >= 0.5, 1.0, tern_w)
        tern_w = torch.where(w_scaled <= -0.5, -1.0, tern_w)

        return tern_w

class exp_tern_det(ternary):
    def tern(self, w):
        eps = 1e-6 
        abs_w = torch.abs(w) + eps
        floor_w = torch.floor(torch.log2(abs_w))
        ceil_w = torch.ceil(torch.log2(abs_w))
        p = abs_w * (2 ** -floor_w) - 1
        quant_w = torch.where(p > 0.5, 1.0, 0.0)

        return quant_w * torch.sign(w)

class exp_tern_stoc(ternary):
    def tern(self, w):
        eps = 1e-6 
        abs_w = torch.abs(w) + eps
        floor_w = torch.floor(torch.log2(abs_w))
        # ceil_w = torch.ceil(torch.log2(abs_w))
        p = torch.clip(abs_w * (2 ** -floor_w) - 1, 0, 1)

        sample = torch.bernoulli(p)
        quant_w = torch.where(sample == 1, 1.0, 0.0)
        return  quant_w * torch.sign(w)
            

            