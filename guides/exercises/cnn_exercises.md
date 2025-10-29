# CNN Exercises - Module 7

**Time:** 4-5 hours
**Difficulty:** Advanced
**Materials needed:** Paper, pencil, calculator, NumPy

Implement CNNs from scratch. Solutions are in `guides/solutions/cnn_solutions.md`

---

## Part 1: Convolution Basics (35 min)

### Exercise 1.1: 2D Convolution by Hand
Input 4×4:
```
[[1, 2, 3, 0],
 [0, 1, 2, 3],
 [3, 0, 1, 2],
 [2, 3, 0, 1]]
```
Kernel 3×3:
```
[[1, 0, -1],
 [1, 0, -1],
 [1, 0, -1]]
```

1. Compute convolution output (no padding, stride=1)
2. What is output size? Formula: (n - k + 1) × (n - k + 1)
3. Calculate first output value (top-left)
4. Calculate all 4 output values

### Exercise 1.2: Padding Effects
Same input and kernel as 1.1:

1. Add zero padding (pad=1) to input → new size?
2. Compute output with padding
3. What is output size now?
4. Why use padding? (preserve spatial dimensions)

### Exercise 1.3: Stride Effects
5×5 input, 3×3 kernel, stride=2:

1. Calculate output size: (n - k)/stride + 1
2. Perform convolution
3. Compare with stride=1
4. Why use larger strides? (downsample, reduce computation)

### Exercise 1.4: Multiple Channels
Input: 3×3×2 (2 channels), Kernel: 3×3×2 (matches input channels)

1. Convolve each input channel with corresponding kernel channel
2. Sum results across channels
3. Output is single 1×1 value
4. How many multiplications per output value?

---

## Part 2: Pooling Operations (30 min)

### Exercise 2.1: Max Pooling
Input 4×4:
```
[[1, 3, 2, 4],
 [5, 6, 7, 8],
 [9, 2, 3, 1],
 [4, 5, 6, 7]]
```

Apply 2×2 max pooling, stride=2:
1. Divide into 2×2 regions
2. Take max of each region
3. Output size?
4. Why max pooling? (translation invariance, reduce spatial size)

### Exercise 2.2: Average Pooling
Same input as 2.1:

1. Apply 2×2 average pooling
2. Compare outputs with max pooling
3. When to use average vs max?

### Exercise 2.3: Global Average Pooling
Input 4×4×3 (3 channels):

1. Compute mean of each channel → 3 values
2. Output shape: (3,)
3. Use case: replace fully-connected at end
4. Benefit: no parameters, any input size

---

## Part 3: CNN Forward Pass (30 min)

### Exercise 3.1: Single Conv Layer
Input: 5×5×1
Conv layer: 3×3 kernel, 2 filters, stride=1, no padding

1. How many kernels? (2, one per output channel)
2. Kernel dimensions? (3×3×1) each
3. Output shape?
4. Parameters count? (2×(3×3×1 + 1)) = 20

### Exercise 3.2: Conv + ReLU + Pool
Input: 8×8×1

Layer 1: Conv 3×3, 4 filters, stride=1, pad=1 → ReLU → MaxPool 2×2
Layer 2: Conv 3×3, 8 filters, stride=1, pad=1 → ReLU → MaxPool 2×2

1. After Layer 1 Conv: size?
2. After Layer 1 Pool: size?
3. After Layer 2 Conv: size?
4. After Layer 2 Pool: size?
5. Total parameters?

### Exercise 3.3: Receptive Field
3-layer CNN: Conv 3×3 → Conv 3×3 → Conv 3×3

1. First layer receptive field: 3×3
2. Second layer: what region of input affects one output neuron?
3. Third layer: calculate receptive field
4. Formula: RF = 1 + Σ(k - 1)·∏previous strides

---

## Part 4: CNN Backpropagation (45 min)

### Exercise 4.1: Convolution Backward Pass
Forward: Y = conv(X, W)
Given: dL/dY (gradient w.r.t output)

1. To get dL/dW: convolve X with dL/dY
2. To get dL/dX: full convolution of dL/dY with flipped W
3. Implement for 3×3 case
4. Verify with numerical gradients

### Exercise 4.2: Max Pooling Backward
Forward: Y = maxpool(X), saved indices of max values

Backward given dL/dY:
1. Route gradient only to positions that were max
2. Other positions get 0
3. Implement for 2×2 pooling
4. Compare with average pooling backward (distribute evenly)

### Exercise 4.3: Full CNN Backward
Network: Conv → ReLU → MaxPool → Flatten → FC → Softmax

1. Start from softmax gradient
2. Backprop through FC layer
3. Unflatten to spatial dimensions
4. Backprop through MaxPool (route to max indices)
5. Backprop through ReLU (pass if forward > 0)
6. Backprop through Conv (convolve X with gradient)

### Exercise 4.4: Gradient Flow
Deep CNN with 20 Conv layers:

1. Why doesn't this suffer from vanishing gradients like deep FC nets?
2. Role of skip connections (ResNet)
3. Batch normalization effect
4. Compare gradient magnitudes: first vs last layer

---

## Part 5: BatchNorm (30 min)

### Exercise 5.1: BatchNorm Forward
Batch of 4 samples, each 3-dimensional:
```
X = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9],
     [10, 11, 12]]
```

1. Compute mean μ per feature (column)
2. Compute variance σ² per feature
3. Normalize: X_norm = (X - μ) / √(σ² + ε)
4. Scale and shift: Y = γ·X_norm + β (learnable γ, β)

### Exercise 5.2: BatchNorm Backward
Given dL/dY, compute dL/dX, dL/dγ, dL/dβ:

1. dL/dβ = sum of dL/dY over batch
2. dL/dγ = sum of dL/dY ⊙ X_norm over batch
3. dL/dX requires chain rule through normalization
4. Why is BatchNorm backward complex?

### Exercise 5.3: Why BatchNorm Works
Network with and without BatchNorm:

1. Internal covariate shift problem
2. Smoother loss landscape → larger learning rates
3. Regularization effect
4. Different behavior train vs test (use running mean/var at test)

---

## Part 6: CNN Architectures (35 min)

### Exercise 6.1: LeNet-5
Design LeNet for 28×28 grayscale images (MNIST):

1. Conv 5×5, 6 filters → Pool 2×2
2. Conv 5×5, 16 filters → Pool 2×2
3. FC 120 → FC 84 → FC 10
4. Calculate output shapes at each layer
5. Total parameters?

### Exercise 6.2: VGG Block
VGG uses multiple 3×3 convs instead of larger kernels:

1. Stack two 3×3 convs: receptive field?
2. Compare params: two 3×3 vs one 5×5
3. Why 3×3 better? (fewer params, more nonlinearity)
4. Implement VGG block: Conv-ReLU-Conv-ReLU-Pool

### Exercise 6.3: ResNet Skip Connection
Residual block: F(x) = Conv(Conv(x)) + x

Input: 16×16×64, output: 16×16×64

1. Implement: x → Conv 3×3,64 → BN → ReLU → Conv 3×3,64 → BN
2. Add input: out = F(x) + x
3. Apply ReLU
4. Why does skip connection help training?

---

## Challenge Problems (Optional)

### Challenge 1: Implement Conv2D from Scratch
NumPy only, no libraries:

1. Forward: conv2d(X, W, stride, pad)
2. Backward: gradient w.r.t X and W
3. Handle batches and multiple channels
4. Verify with numerical gradients

### Challenge 2: Train Tiny CNN on MNIST
Implement and train:

1. Architecture: Conv-Pool-Conv-Pool-FC-Softmax
2. Load MNIST (or generate toy data)
3. Train with SGD + momentum
4. Achieve >95% accuracy

---

## NumPy Implementation

```python
import numpy as np

# Exercise 1.1 - 2D Convolution
def conv2d(X, K, stride=1, pad=0):
    if pad > 0:
        X = np.pad(X, pad, mode='constant')

    n, m = X.shape
    k_h, k_w = K.shape
    out_h = (n - k_h) // stride + 1
    out_w = (m - k_w) // stride + 1

    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = X[i*stride:i*stride+k_h, j*stride:j*stride+k_w]
            out[i, j] = np.sum(region * K)

    return out

X = np.array([[1, 2, 3, 0],
              [0, 1, 2, 3],
              [3, 0, 1, 2],
              [2, 3, 0, 1]])

K = np.array([[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]])

result = conv2d(X, K)
print("Convolution output:")
print(result)

# Exercise 2.1 - Max Pooling
def maxpool2d(X, pool_size=2, stride=2):
    n, m = X.shape
    out_h = (n - pool_size) // stride + 1
    out_w = (m - pool_size) // stride + 1

    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = X[i*stride:i*stride+pool_size,
                      j*stride:j*stride+pool_size]
            out[i, j] = np.max(region)

    return out

X = np.array([[1, 3, 2, 4],
              [5, 6, 7, 8],
              [9, 2, 3, 1],
              [4, 5, 6, 7]])

result = maxpool2d(X)
print("\nMax pooling output:")
print(result)

# Exercise 3.2 - Conv + ReLU + Pool Pipeline
class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.W = np.random.randn(out_channels, in_channels,
                                 kernel_size, kernel_size) * 0.01
        self.b = np.zeros(out_channels)

    def forward(self, X):
        # Simplified: assume batch size 1, stride 1, pad 1
        return conv2d(X, self.W) + self.b

# Exercise 5.1 - Batch Normalization
def batchnorm_forward(X, gamma, beta, eps=1e-8):
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    X_norm = (X - mu) / np.sqrt(var + eps)
    out = gamma * X_norm + beta

    cache = (X, mu, var, X_norm, gamma, eps)
    return out, cache

X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]], dtype=float)

gamma = np.ones(3)
beta = np.zeros(3)

out, cache = batchnorm_forward(X, gamma, beta)
print("\nBatchNorm output:")
print(out)
print("Mean per feature:", np.mean(out, axis=0))
print("Var per feature:", np.var(out, axis=0))

# Challenge 1 - Full Conv2D Implementation
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialization
        self.W = np.random.randn(out_channels, in_channels,
                                 kernel_size, kernel_size) * np.sqrt(2.0 / in_channels)
        self.b = np.zeros(out_channels)

    def forward(self, X):
        batch_size, C_in, H_in, W_in = X.shape
        assert C_in == self.in_channels

        # Apply padding
        if self.padding > 0:
            X = np.pad(X, ((0,0), (0,0),
                          (self.padding, self.padding),
                          (self.padding, self.padding)), mode='constant')

        _, _, H_padded, W_padded = X.shape

        # Calculate output dimensions
        H_out = (H_padded - self.kernel_size) // self.stride + 1
        W_out = (W_padded - self.kernel_size) // self.stride + 1

        # Initialize output
        out = np.zeros((batch_size, self.out_channels, H_out, W_out))

        # Convolution
        for n in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        region = X[n, :,
                                  h_start:h_start+self.kernel_size,
                                  w_start:w_start+self.kernel_size]
                        out[n, c_out, h, w] = np.sum(region * self.W[c_out]) + self.b[c_out]

        self.cache = X
        return out

    def backward(self, dout):
        X = self.cache
        batch_size, C_in, H_in, W_in = X.shape
        _, C_out, H_out, W_out = dout.shape

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(X)

        # Gradient w.r.t weights and bias
        for n in range(batch_size):
            for c_out in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        region = X[n, :,
                                  h_start:h_start+self.kernel_size,
                                  w_start:w_start+self.kernel_size]
                        dW[c_out] += dout[n, c_out, h, w] * region
                        db[c_out] += dout[n, c_out, h, w]
                        dX[n, :,
                           h_start:h_start+self.kernel_size,
                           w_start:w_start+self.kernel_size] += dout[n, c_out, h, w] * self.W[c_out]

        self.dW = dW / batch_size
        self.db = db / batch_size

        # Remove padding from dX if needed
        if self.padding > 0:
            dX = dX[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return dX

# Test convolution
conv = Conv2D(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
X_test = np.random.randn(2, 1, 5, 5)  # batch=2, channels=1, 5x5
out = conv.forward(X_test)
print(f"\nConv2D output shape: {out.shape}")

# Backward pass
dout = np.random.randn(*out.shape)
dX = conv.backward(dout)
print(f"Conv2D dX shape: {dX.shape}")
```

---

## Tips for Success

1. **Visualize convolutions** - Draw receptive fields
2. **Check shapes** - Dimensional analysis prevents bugs
3. **Stride vs pooling** - Both downsample, different purposes
4. **Padding preserves size** - pad = (k-1)/2 for same size
5. **Channels multiply params** - Be careful with width
6. **BatchNorm placement** - After conv, before activation
7. **Skip connections** - Enable very deep networks
8. **Start simple** - Single conv layer, then build up
