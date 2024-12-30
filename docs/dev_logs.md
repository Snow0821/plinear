# Dev Logs
## version 0.2
### 0.2.0
#### 0.2.0.0 
##### 30th December, 2024
- Removed softmax in qkv calculation
    stablized learning curve

- Splited 'plinear' into core, btnn, ctnn
    core is for core algorithms which now only contains baisc binarization
    btnn stands for binarized ternary neural network which was basic plinear
    bctnn stands for binarized complex ternary neural network which was plinear_complex

- Reformed folder structure
    Seperating dev logs from README.md
    Now presenting binarized ternary linear as btnn and binarized complex ternary linear as ctnn
    Begin new research for binarizing ternary transformers with ViT.

## version 0.1
#### 0.1.3.3
- plinear device setting added

#### 0.1.3.2
- Minor instalation version check.
- pytorch 2.2.2 for darwin (intel Macos)

#### 0.1.3.1
- Teseted SAM optimizer and somehow it did not work well.

#### 0.1.3.0
- Complex layer created and tested on MNIST
- mnist run case fixed to show result through cmd

#### 0.1.2.3
- Integrated posNet, negNet functions to posNet.
    Layer now does posNet - negNet instead of posNet + negNet since negNet is not negative in real now.
    Weight is now processed with tanh and shows much stable learning curve.
- Removed test result from the git.

#### 0.1.2.2.
- Documented readme.md
- Preflight testing done for mnist both layer and visualization.