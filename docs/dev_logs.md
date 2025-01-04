# Dev Logs
## version 0.2
### 0.2.0

#### 0.2.0.4
##### 2nd January, 2025
- fixed major bug
    bctnn (binarized complex ternary neural network) has been calculated incorrectly

#### 0.2.0.3
##### 31th December, 2024
- fixed minor bugs
    corrected include error from btnn

#### 0.2.0.2
##### 31th December, 2024

- included 3 modules into plinear again.
    plinear is not yet that much magnificant.

#### 0.2.0.1
##### 31th December, 2024
- btnn.conv2D layer on working
    not yet tested

- added core to distribution
    plinear was not working

- fixed README.md
    to working example

#### 0.2.0.0
##### 30th December, 2024
- Removed softmax in qkv calculation
    stablized learning curve

- Splited 'plinear' into core, btnn, bctnn
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