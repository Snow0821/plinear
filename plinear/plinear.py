import torch.nn as nn

class PLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(PLinear, self).__init__(in_features, out_features, bias)
