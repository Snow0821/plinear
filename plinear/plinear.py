import torch
import torch.nn as nn

class PLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(PLinear, self).__init__(in_features, out_features, bias)
        self.threshold1 = 0.5
        self.threshold2 = -0.5

    def quantize(self, x):
        if x > self.threshold1:
            return 1, 0
        elif x < self.threshold2:
            return 0, 1
        else:
            return 0, 0

    def forward(self, input):
        output = super(PLinear, self).forward(input)
        quantized_output = torch.tensor([self.quantize(val) for val in output.view(-1)], dtype=torch.float32)
        return quantized_output.view(output.shape[0], output.shape[1], -1)
