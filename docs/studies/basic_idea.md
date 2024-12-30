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