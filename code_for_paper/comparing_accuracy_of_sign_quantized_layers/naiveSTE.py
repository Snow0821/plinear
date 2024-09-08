#Written by chatGPT4-o

import torch
import torch.nn as nn
import torch.nn.functional as F

class STEQuantizeTernaryFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights):
        """
        Forward pass: Quantize weights to {-1, 0, 1}.
        Quantization rules:
        - weights > 0.5 -> 1
        - weights < -0.5 -> -1
        - otherwise -> 0
        """
        quantized_weights = torch.zeros_like(weights)
        quantized_weights[weights > 0.5] = 1
        quantized_weights[weights < -0.5] = -1
        return quantized_weights

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Straight-through estimator (STE).
        Pass the gradient as is.
        """
        return grad_output  # No modification to gradient, STE

class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(TernaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Linear layer weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        # Apply ternary quantization to the weights using STE
        quantized_weight = STEQuantizeTernaryFunction.apply(self.weight)

        # Perform the linear operation with quantized weights and input (no bias)
        return F.linear(x, quantized_weight)