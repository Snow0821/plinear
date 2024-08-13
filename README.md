# plinear

Github for parrallel - linear layer

You can install PLinear using pip:

```sh
pip install plinear
```

### Idea inspired from

https://arxiv.org/pdf/2402.17764?trk=public_post_comment-text

### Code inspired from

https://github.com/kyegomez/BitNet/blob/main/bitnet/bitlinear.py

# Ideas and Road Map

## Parrallel neural network (PLinear)

#### Layer composition

Binarizing ternary layers by making posNet and negNet and compute them.

Both are created with posNet, which returns 1 if the weight if over 0 and 0 else.

Result comes out with posNet - negNet to mimic ternary.

Found out that tanh(weight) makes the model to fit in and learn without normalizing entire layer.

No additional activation function used in test.

#### tanh makes learning much more stable (quite stunning if you see the curve.)

![tanh](images/readme/plinear/tanh.svg)

![posNet](images/readme/plinear/w_pos.svg)

W_neg is calculated just as W_pos

![Equation](images/readme/plinear/equation.svg)

#### Learning w' - w as bitNet does

![Learning](images/readme/plinear//learning.svg)

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

#### Test code for Mnist example (Should clone whole git)

```
pytest -k mnist -s
```

Results can be found in tests/result_mnist.

You are offered with precision, accuracy, recall per epochs.

Also confusion matrix and full visualization of weights per epochs will be offered in animation.

## Complex layers

parrellized 4 binarized nn.Linears (using same algorithms with PLinear).

2 for real and 2 for complex.

# Real and Complex Results

## Real Result

![Real Result](images/readme/complex/Rout_equation.svg)

## Complex Result

![Complex Result](images/readme/complex/Cout_equation.svg)

### Notations

![Notations](images/readme/complex/descriptions.svg)

#### P.S.

I used torch.zeros if there is no complex input to feed.

pretty lovely result comes out, I suggest you to try this.

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

#### Test code for Mnist example (Should clone whole git)

```
pytest -k mnist_c -s
```

## visualization (Not finised for documentation)

## Exhaustive search of 3 x 3 CNN (Only Idea)

Since I parrallelized layers, each 3 x 3 CNN layer can be brute forced in 2^9 \* 2 weights for ternary, which is very cheap against previous models.

Even the model is same, the layer is still at least 9 times smaller even if the model seeked through every cases.

And we can reduce the model with simple searching tasks.

I believe this can be used to vectorize images in proper size of vector which can be reused for image generation or more.

I guess vectorizing concepts and dynamically allocating them with layers would be the final goal of this project.

# Developer Note

### 15, July, 2024

Checked plinear works on colab

### 16, July, 2024

version 0.1.2.2.
version 0.1.2.3.

### 19, July, 2024

version 0.1.3.0 - PLinear_Complex

fixed some documentaion

### 14, Auguest, 2024

version 0.1.3.1 - Minor Tester change

# changelog

#### 0.1.2.2.

Documented readme.md

Preflight testing done for mnist both layer and visualization.

#### 0.1.2.3

Integrated posNet, negNet functions to posNet.

Layer now does posNet - negNet instead of posNet + negNet since negNet is not negative in real now.

Weight is now processed with tanh and shows much stable learning curve.

Removed test result from the git.

#### 0.1.3.0

Complex layer created and tested on MNIST

mnist run case fixed to show result through cmd

#### 0.1.3.1

Teseted SAM optimizer and somehow it did not work well.

##### Found out that square models way better than non-square models.

###### Hypothesis

While backpropagation if the tensor got smaller, the change should spread thinner.

Need compensation since unlike traditional linear, quantized models does not show the change until it passes threshold.
