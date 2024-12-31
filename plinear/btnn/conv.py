import torch
import torch.nn as nn
import torch.nn.functional as F

from core.posNet import posNet

class Conv2d(nn.Module):
    def __init__(self, x, y, kernel_size, stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.real_pos = nn.Conv2d(x, y, kernel_size, stride=stride, padding=padding, bias=False)
        self.real_neg = nn.Conv2d(x, y, kernel_size, stride=stride, padding=padding, bias=False)

        torch.nn.init.uniform_(self.real_pos.weight, -1, 1)
        torch.nn.init.uniform_(self.real_neg.weight, -1, 1)

    def forward(self, x):
        w_pos = self.real_pos.weight
        w_neg = self.real_neg.weight
        tern_pos = posNet(w_pos)
        tern_neg = posNet(w_neg)

        # Apply quantization using posNet with detach
        tern_pos = tern_pos - w_pos.detach() + w_pos
        tern_neg = tern_neg - w_neg.detach() + w_neg

        # 양자화된 가중치로 Conv2d 수행
        y_pos = F.conv2d(
            x, 
            tern_pos, 
            bias=None,               # bias는 None으로 명시
            stride=self.conv.stride, 
            padding=self.conv.padding, 
            dilation=self.conv.dilation, 
            groups=self.conv.groups
        )

        y_neg = F.conv2d(
            x, 
            tern_neg, 
            bias=None,               # bias는 None으로 명시
            stride=self.conv.stride, 
            padding=self.conv.padding, 
            dilation=self.conv.dilation, 
            groups=self.conv.groups
        )

        return y_pos - y_neg