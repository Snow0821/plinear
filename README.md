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
```

#### Example usage

```
import torch
import torch.nn as nn
from plinear import PLinear_Complex as PL

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.complex = torch.zeros(28*28, 1)
        self.fc1 = PL(28*28, 128)
        self.fc2 = PL(128, 10)

    def forward(self, x):
        real = torch.flatten(x, 1)
        real, complex = self.fc1(real, complex)
        real, complex = self.fc2(real, complex)
        return real
```