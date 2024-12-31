# plinear

Github for parrallel - linear layer

You can install PLinear using pip:

```sh
pip install plinear
```

#### Example usage

```
import torch
import torch.nn as nn
from plinear import btnn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = btnn.Linear(28*28, 128)
        self.fc2 = btnn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
```