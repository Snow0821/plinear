# plinear

Github for parrallel - linear layer

You can install PLinear using pip:

```sh
pip install plinear
```

### idea inspired from

https://arxiv.org/pdf/2402.17764?trk=public_post_comment-text

### code inspired from

https://github.com/kyegomez/BitNet/blob/main/bitnet/bitlinear.py

# Ideas and Road Map

## parrallel neural network (PLinear)

Binarizing ternary layers by making posNet and negNet and add them.

No activation function required.

Suggested usage :

```
import torch
import torch.nn as nn
from plinear import PLinear

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = PLinear(28*28, 128)
        self.fc2 = PLinear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Example usage
model = SimpleNN()
print(model)

```

###

## visualization (Not finised for documentation)

## Brute Force optimization of 3 x 3 CNN (Only Idea)

Since I parrallelized layers, each 3 x 3 CNN layer can be brute forced in 2^9 \* 2 weights for ternary, which is very cheap against previous models.

Even the model is same, the layer is still at least 9 times smaller even if the model seeked through every cases.

And we can reduce the model with simple searching tasks.

I believe this can be used to vectorize images in proper size of vector which can be reused for image generation or more.

I guess vectorizing concepts and dynamically allocating them with layers would be the final goal of this project.

## complex layers (Only Idea)

# Develope Note

### 15, July, 2024

Checked plinear works on colab

### 16, July, 2024

version 0.1.2.2.

# changelog

#### 0.1.1.2.

Documented readme.md

Preflight testing done for mnist both layer and visualization.
