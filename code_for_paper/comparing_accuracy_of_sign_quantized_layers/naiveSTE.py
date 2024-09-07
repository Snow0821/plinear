#Written by chatGPT4-o

import torch
import torch.nn as nn

class STEQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Forward pass: Quantize input (example: rounding to nearest integer)
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass: Pass through the gradient without modification
        return grad_output

class STELayer(nn.Module):
    def __init__(self):
        super(STELayer, self).__init__()
        # Define any layer parameters here (if needed)
        # For example, this could be a linear layer: nn.Linear(in_features, out_features)

    def forward(self, x):
        # Use the STE quantization during forward pass
        quantized_x = STEQuantizeFunction.apply(x)
        return quantized_x