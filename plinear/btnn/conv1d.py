import torch
import torch.nn as nn
import torch.nn.functional as F

from plinear.core import posNet

class Conv1d(nn.Module):
    def __init__(self, x, y, kernel_size, stride=1, padding=0, groups = 1):
        super(Conv1d, self).__init__()
        self.real_pos = nn.Conv1d(x, y, kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.real_neg = nn.Conv1d(x, y, kernel_size, stride=stride, padding=padding, groups=groups, bias=False)

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
        y_pos = F.conv1d(
            x, 
            tern_pos, 
            bias=None,               
            stride=self.real_pos.stride, 
            padding=self.real_pos.padding, 
            dilation=self.real_pos.dilation, 
            groups=self.real_pos.groups
        )

        y_neg = F.conv1d(
            x, 
            tern_neg, 
            bias=None,               
            stride=self.real_neg.stride, 
            padding=self.real_neg.padding, 
            dilation=self.real_neg.dilation, 
            groups=self.real_neg.groups
        )

        return y_pos - y_neg