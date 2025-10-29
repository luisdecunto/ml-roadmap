# CNN Solutions - Module 7

Complete step-by-step solutions for all CNN exercises.

---

## Part 1: Convolution Basics

### Solution 1.1: 2D Convolution by Hand

**Given:**
- Input 4×4: `[[1,2,3,0], [0,1,2,3], [3,0,1,2], [2,3,0,1]]`
- Kernel 3×3: `[[1,0,-1], [1,0,-1], [1,0,-1]]`
- No padding, stride=1

**Step 1: Output size**
Formula: (n - k + 1) × (n - k + 1) = (4 - 3 + 1) × (4 - 3 + 1) = **2×2**

**Step 2: Compute first output value (position [0,0])**
```
Top-left 3×3 region:
[[1, 2, 3],
 [0, 1, 2],
 [3, 0, 1]]

Element-wise multiply with kernel:
[[1×1,  2×0,  3×(-1)],
 [0×1,  1×0,  2×(-1)],
 [3×1,  0×0,  1×(-1)]]

= [[1, 0, -3],
   [0, 0, -2],
   [3, 0, -1]]

Sum = 1 + 0 - 3 + 0 + 0 - 2 + 3 + 0 - 1 = -2
```

**Step 3: All 4 output values**
```
Position [0,0]: -2 (calculated above)

Position [0,1] (top-right region):
[[2, 3, 0],
 [1, 2, 3],
 [0, 1, 2]]
Sum: 2×1 + 3×0 + 0×(-1) + 1×1 + 2×0 + 3×(-1) + 0×1 + 1×0 + 2×(-1)
   = 2 + 0 + 0 + 1 + 0 - 3 + 0 + 0 - 2 = -2

Position [1,0] (bottom-left region):
[[0, 1, 2],
 [3, 0, 1],
 [2, 3, 0]]
Sum: 0 + 0 - 2 + 3 + 0 - 1 + 2 + 0 + 0 = 2

Position [1,1] (bottom-right region):
[[1, 2, 3],
 [0, 1, 2],
 [3, 0, 1]]
Sum: 1 + 0 - 3 + 0 + 0 - 2 + 3 + 0 - 1 = -2

Final output: [[-2, -2],
               [ 2, -2]]
```

---

### Solution 1.2: Padding Effects

**Step 1: Add zero padding (pad=1)**
Original 4×4 becomes **6×6** after padding:
```
[[0, 0, 0, 0, 0, 0],
 [0, 1, 2, 3, 0, 0],
 [0, 0, 1, 2, 3, 0],
 [0, 3, 0, 1, 2, 0],
 [0, 2, 3, 0, 1, 0],
 [0, 0, 0, 0, 0, 0]]
```

**Step 2: Output size with padding**
Formula: (n + 2p - k + 1) × (n + 2p - k + 1)
= (4 + 2×1 - 3 + 1) × (4 + 2×1 - 3 + 1) = **4×4**

**Step 3: Why use padding?**
- **Preserve spatial dimensions**: With pad=1 and k=3×3, output same size as input
- **Don't lose border information**: Without padding, edge pixels contribute less
- **Control output size**: Padding gives flexibility in architecture design
- **Common choice**: "same" padding where output size = input size

---

### Solution 1.3: Stride Effects

**Given:** 5×5 input, 3×3 kernel, stride=2

**Step 1: Output size**
Formula: ((n - k) / stride) + 1 = ((5 - 3) / 2) + 1 = **2×2**

**Step 2: Compare with stride=1**
- Stride=1: output would be 3×3
- Stride=2: output is 2×2 (downsampled by factor of 2)

**Step 3: Why use larger strides?**
- **Downsample**: Reduce spatial dimensions quickly
- **Reduce computation**: Fewer output positions to compute
- **Alternative to pooling**: Can replace pooling layers
- **Increase receptive field**: Each output neuron sees larger input region

**Trade-off:** Lose spatial resolution, may miss fine details

---

### Solution 1.4: Multiple Channels

**Given:** Input 3×3×2 (2 channels), Kernel 3×3×2

**Step 1: Process each channel**
```
Channel 0: Input[..., 0] * Kernel[..., 0] → intermediate result
Channel 1: Input[..., 1] * Kernel[..., 1] → intermediate result
```

**Step 2: Sum across channels**
Final output = sum of both intermediate results → **single 1×1 value**

**Step 3: Multiplications per output**
For C input channels, kernel size k×k:
- Multiplications = k × k × C = 3 × 3 × 2 = **18 multiplications**

**Key insight:** Convolution always sums across ALL input channels to produce one output channel value.

---

## Part 2: Pooling Operations

### Solution 2.1: Max Pooling

**Given:** 4×4 input, 2×2 pool, stride=2
```
[[1, 3, 2, 4],
 [5, 6, 7, 8],
 [9, 2, 3, 1],
 [4, 5, 6, 7]]
```

**Step 1: Divide into 2×2 regions**
```
Region [0,0] (top-left):    Region [0,1] (top-right):
[[1, 3],                    [[2, 4],
 [5, 6]]                     [7, 8]]
max = 6                     max = 8

Region [1,0] (bottom-left): Region [1,1] (bottom-right):
[[9, 2],                    [[3, 1],
 [4, 5]]                     [6, 7]]
max = 9                     max = 7
```

**Step 2: Output**
```
[[6, 8],
 [9, 7]]
```

**Step 3: Output size**
(n / pool_size) × (n / pool_size) = (4 / 2) × (4 / 2) = **2×2**

**Step 4: Why max pooling?**
- **Translation invariance**: Small shifts don't change max
- **Reduce spatial size**: Downsample feature maps
- **Keep strongest activations**: Preserve most important features
- **Reduce parameters**: Fewer values to process in next layer

---

### Solution 2.2: Average Pooling

**Same 4×4 input, 2×2 average pooling:**

```
Region [0,0]: (1+3+5+6)/4 = 15/4 = 3.75
Region [0,1]: (2+4+7+8)/4 = 21/4 = 5.25
Region [1,0]: (9+2+4+5)/4 = 20/4 = 5.0
Region [1,1]: (3+1+6+7)/4 = 17/4 = 4.25

Output:
[[3.75, 5.25],
 [5.0,  4.25]]
```

**Comparison with max pooling:**
- Max pooling: `[[6, 8], [9, 7]]`
- Average pooling: `[[3.75, 5.25], [5.0, 4.25]]`

**When to use each:**
- **Max pooling**: Most common, emphasizes strong features, used in classification
- **Average pooling**: Smoother, preserves background, used in some segmentation tasks
- **Global average pooling**: Often at end of network to replace FC layers

---

### Solution 2.3: Global Average Pooling

**Given:** Input 4×4×3 (3 channels)

**Step 1: Compute mean per channel**
```
Channel 0: mean of all 16 values → scalar_0
Channel 1: mean of all 16 values → scalar_1
Channel 2: mean of all 16 values → scalar_2
```

**Step 2: Output shape**
**(3,)** - one value per channel

**Step 3: Use case**
Replace final fully-connected layers:
- Traditional: Conv → Flatten → FC(1000) → FC(10)
- With GAP: Conv → GlobalAvgPool → FC(10)

**Step 4: Benefits**
- **No parameters**: Just averaging operation
- **Any input size**: Works with variable-sized images
- **Regularization**: Forces feature maps to be class-specific
- **Less overfitting**: Fewer parameters than FC layers

---

## Part 3: CNN Forward Pass

### Solution 3.1: Single Conv Layer

**Given:** Input 5×5×1, Conv 3×3, 2 filters, stride=1, no padding

**1. How many kernels?**
**2 kernels** (one per output channel/filter)

**2. Kernel dimensions?**
Each kernel: **3×3×1** (must match input channels)

**3. Output shape?**
- Spatial: (5 - 3 + 1) × (5 - 3 + 1) = 3×3
- Channels: 2 (number of filters)
- **Output: 3×3×2**

**4. Parameters count?**
- Each kernel: 3×3×1 = 9 weights + 1 bias = 10 params
- Total: 2 × 10 = **20 parameters**

---

### Solution 3.2: Conv + ReLU + Pool

**Input: 8×8×1**

**Layer 1: Conv 3×3, 4 filters, stride=1, pad=1 → ReLU → MaxPool 2×2**

1. **After Conv:**
   - Spatial: (8 + 2×1 - 3)/1 + 1 = 8×8
   - Channels: 4
   - Shape: **8×8×4**

2. **After Pool:**
   - Spatial: 8/2 = 4×4
   - Shape: **4×4×4**

**Layer 2: Conv 3×3, 8 filters, stride=1, pad=1 → ReLU → MaxPool 2×2**

3. **After Conv:**
   - Spatial: (4 + 2×1 - 3)/1 + 1 = 4×4
   - Channels: 8
   - Shape: **4×4×8**

4. **After Pool:**
   - Spatial: 4/2 = 2×2
   - Shape: **2×2×8**

**5. Total parameters:**
- Layer 1: (3×3×1×4) + 4 = 36 + 4 = 40
- Layer 2: (3×3×4×8) + 8 = 288 + 8 = 296
- **Total: 336 parameters**

---

### Solution 3.3: Receptive Field

**3-layer CNN: Conv 3×3 → Conv 3×3 → Conv 3×3 (all stride=1)**

**Layer 1:** RF = 3×3

**Layer 2:**
- Each Layer 2 neuron sees 3×3 from Layer 1
- Each Layer 1 neuron sees 3×3 from input
- Layer 2 RF = 3 + (3-1) = **5×5**

**Layer 3:**
- Each Layer 3 neuron sees 3×3 from Layer 2
- Each Layer 2 neuron sees 5×5 from input
- Layer 3 RF = 5 + (3-1) = **7×7**

**General formula:**
RF_n = RF_(n-1) + (k - 1) × ∏(previous strides)

For stride=1: RF_n = 1 + Σ(k_i - 1) for all layers

**Key insight:** Stacking small kernels (3×3) gives large receptive field efficiently.

---

## Part 4: CNN Backpropagation

### Solution 4.1: Convolution Backward Pass

**Forward:** Y = conv(X, W)
**Given:** dL/dY (gradient w.r.t output)
**Need:** dL/dW and dL/dX

**1. Gradient w.r.t weights (dL/dW):**
- dL/dW = conv(X, dL/dY)
- Slide dL/dY over X like convolution
- Each position accumulates gradient for corresponding weight

**2. Gradient w.r.t input (dL/dX):**
- dL/dX = full_conv(dL/dY, flip(W))
- "Full" convolution (padding to restore size)
- Flip W by 180° (reverse both dimensions)

**3. Example (3×3 case):**
```python
# Forward
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
W = np.array([[1, 0],
              [0, 1]])
Y = conv2d(X, W)  # shape: 2×2

# Backward
dL_dY = np.ones((2, 2))  # upstream gradient

# dL/dW: convolve X with dL/dY
dL_dW = np.zeros_like(W)
for i in range(2):
    for j in range(2):
        region = X[i:i+2, j:j+2]
        dL_dW += dL_dY[i, j] * region

# dL/dX: full convolution
W_flipped = np.flip(W)
dL_dX = full_conv2d(dL_dY, W_flipped)
```

**4. Verification:**
Use numerical gradients: (f(x+h) - f(x-h)) / (2h)
Compare with analytical gradients, should match to ~1e-7

---

### Solution 4.2: Max Pooling Backward

**Forward:** Y = maxpool(X), save indices of max values

**Backward given dL/dY:**

**1. Routing gradient:**
- Only position that was max gets gradient
- Other 3 positions get 0

**2. Example:**
```python
# Forward
X = np.array([[1, 3, 2, 4],
              [5, 6, 7, 8],
              [9, 2, 3, 1],
              [4, 5, 6, 7]])

# 2×2 pooling, stride=2
# Region [0:2, 0:2]: max=6 at position (1,1)
# Region [0:2, 2:4]: max=8 at position (1,3)
# etc.

# Backward
dL_dY = np.array([[1, 2],
                  [3, 4]])

dL_dX = np.zeros_like(X)
# Route gradient only to max positions:
dL_dX[1, 1] = dL_dY[0, 0]  # 1
dL_dX[1, 3] = dL_dY[0, 1]  # 2
dL_dX[2, 0] = dL_dY[1, 0]  # 3
dL_dX[2, 3] = dL_dY[1, 1]  # 4

# Result:
# [[0, 0, 0, 0],
#  [0, 1, 0, 2],
#  [3, 0, 0, 4],
#  [0, 0, 0, 0]]
```

**3. Compare with average pooling:**
```python
# Average pooling backward: distribute evenly
dL_dX = np.zeros_like(X)
for each 2×2 region:
    dL_dX[region] = dL_dY[i, j] / 4  # split gradient equally
```

---

### Solution 4.3: Full CNN Backward

**Network:** Conv → ReLU → MaxPool → Flatten → FC → Softmax

**Backward pass (reverse order):**

**1. Start from softmax gradient:**
```python
# Cross-entropy loss + softmax: dL/dZ = (p - y)
dL_dZ = softmax_output - y_true  # shape: (batch, num_classes)
```

**2. Backprop through FC layer:**
```python
# FC: Z = X @ W + b
dL_dW_fc = X.T @ dL_dZ
dL_db_fc = np.sum(dL_dZ, axis=0)
dL_dX_fc = dL_dZ @ W.T  # shape: (batch, flattened_size)
```

**3. Unflatten to spatial dimensions:**
```python
# Reshape back to (batch, channels, height, width)
dL_dPool_out = dL_dX_fc.reshape(original_spatial_shape)
```

**4. Backprop through MaxPool:**
```python
# Route gradient to max indices (saved during forward)
dL_dReLU_out = route_to_max_indices(dL_dPool_out, max_indices)
```

**5. Backprop through ReLU:**
```python
# ReLU gradient: pass if forward input > 0, else 0
dL_dConv_out = dL_dReLU_out * (conv_output > 0)
```

**6. Backprop through Conv:**
```python
# Compute dL/dW_conv and dL/dX
dL_dW_conv = convolve_for_weight_gradient(input, dL_dConv_out)
dL_dX = full_convolve(dL_dConv_out, flipped_weights)
```

**Complete code:**
```python
def cnn_backward(X, y, cache):
    conv_out, relu_out, pool_out, fc_out, softmax_out = cache

    # 1. Softmax
    dL_dZ = softmax_out - y

    # 2. FC
    dL_dW_fc = pool_out.T @ dL_dZ
    dL_dPool = dL_dZ @ W_fc.T

    # 3. Unflatten
    dL_dPool = dL_dPool.reshape(pool_out_shape)

    # 4. MaxPool
    dL_dReLU = maxpool_backward(dL_dPool, max_indices)

    # 5. ReLU
    dL_dConv = dL_dReLU * (conv_out > 0)

    # 6. Conv
    dL_dW_conv = conv_backward_weights(X, dL_dConv)
    dL_dX = conv_backward_input(dL_dConv, W_conv)

    return dL_dW_conv, dL_dW_fc
```

---

### Solution 4.4: Gradient Flow

**Deep CNN with 20 Conv layers:**

**1. Why doesn't it suffer from vanishing gradients like deep FC nets?**
- **Skip connections** (ResNet): Gradient flows directly through identity path
- **Fewer multiplications**: Each layer multiplies by kernel (9 values), not full weight matrix
- **Batch normalization**: Normalizes activations, prevents gradient explosion/vanishing
- **ReLU**: Doesn't saturate like sigmoid/tanh (gradient is 1 or 0, not tiny)

**2. Role of skip connections (ResNet):**
```python
# Without skip: F(x)
# With skip: F(x) + x

# Gradient: dL/dF + dL/dx (identity path)
# Always has "1" component, ensures gradient flow
```

**3. Batch normalization effect:**
- Normalizes layer inputs to zero mean, unit variance
- Smoother loss landscape
- Can use larger learning rates
- Reduces internal covariate shift

**4. Gradient magnitudes comparison:**
```python
# Typical observation:
# Layer 1 (early): gradient ≈ 1e-3
# Layer 20 (late): gradient ≈ 1e-2 (with ResNet + BN)

# Without ResNet/BN:
# Layer 1: gradient ≈ 1e-10 (vanished!)
# Layer 20: gradient ≈ 1e-2
```

**Key insight:** Modern techniques (skip connections, BatchNorm, ReLU) enable training very deep CNNs without vanishing gradients.

---

## Part 5: BatchNorm

### Solution 5.1: BatchNorm Forward

**Given batch:**
```
X = [[1,  2,  3],
     [4,  5,  6],
     [7,  8,  9],
     [10, 11, 12]]
```

**Step 1: Compute mean μ per feature (column)**
```
μ₀ = (1 + 4 + 7 + 10) / 4 = 22/4 = 5.5
μ₁ = (2 + 5 + 8 + 11) / 4 = 26/4 = 6.5
μ₂ = (3 + 6 + 9 + 12) / 4 = 30/4 = 7.5

μ = [5.5, 6.5, 7.5]
```

**Step 2: Compute variance σ² per feature**
```
σ₀² = [(1-5.5)² + (4-5.5)² + (7-5.5)² + (10-5.5)²] / 4
    = [20.25 + 2.25 + 2.25 + 20.25] / 4 = 11.25

σ₁² = [(2-6.5)² + (5-6.5)² + (8-6.5)² + (11-6.5)²] / 4
    = [20.25 + 2.25 + 2.25 + 20.25] / 4 = 11.25

σ₂² = [(3-7.5)² + (6-7.5)² + (9-7.5)² + (12-7.5)²] / 4
    = [20.25 + 2.25 + 2.25 + 20.25] / 4 = 11.25

σ² = [11.25, 11.25, 11.25]
√(σ² + ε) ≈ [3.354, 3.354, 3.354] (with ε=1e-8)
```

**Step 3: Normalize**
```
X_norm = (X - μ) / √(σ² + ε)

Column 0: [(1-5.5)/3.354, (4-5.5)/3.354, (7-5.5)/3.354, (10-5.5)/3.354]
        = [-1.341, -0.447, 0.447, 1.341]

Column 1: [(2-6.5)/3.354, (5-6.5)/3.354, (8-6.5)/3.354, (11-6.5)/3.354]
        = [-1.341, -0.447, 0.447, 1.341]

Column 2: [(3-7.5)/3.354, (6-7.5)/3.354, (9-7.5)/3.354, (12-7.5)/3.354]
        = [-1.341, -0.447, 0.447, 1.341]

X_norm = [[-1.341, -1.341, -1.341],
          [-0.447, -0.447, -0.447],
          [ 0.447,  0.447,  0.447],
          [ 1.341,  1.341,  1.341]]
```

**Step 4: Scale and shift (γ=1, β=0)**
```
Y = γ · X_norm + β = 1 · X_norm + 0 = X_norm

Y = [[-1.341, -1.341, -1.341],
     [-0.447, -0.447, -0.447],
     [ 0.447,  0.447,  0.447],
     [ 1.341,  1.341,  1.341]]
```

**Verification:**
- Mean per feature: [0, 0, 0] ✓
- Variance per feature: [1, 1, 1] ✓

---

### Solution 5.2: BatchNorm Backward

**Given:** dL/dY (upstream gradient)
**Need:** dL/dX, dL/dγ, dL/dβ

**1. dL/dβ (easy):**
```python
# Y = γ·X_norm + β
# ∂Y/∂β = 1
dL_dβ = np.sum(dL_dY, axis=0)  # sum over batch dimension
```

**2. dL/dγ (easy):**
```python
# Y = γ·X_norm + β
# ∂Y/∂γ = X_norm
dL_dγ = np.sum(dL_dY * X_norm, axis=0)  # element-wise multiply, then sum
```

**3. dL/dX (complex - chain rule through normalization):**
```python
# Y = γ·X_norm + β
# X_norm = (X - μ) / √(σ² + ε)

# Step 1: dL/dX_norm
dL_dX_norm = dL_dY * γ

# Step 2: dL/dX through normalization
N = batch_size
dL_dσ² = np.sum(dL_dX_norm * (X - μ) * (-0.5) * (σ² + ε)**(-1.5), axis=0)
dL_dμ = np.sum(dL_dX_norm * (-1 / np.sqrt(σ² + ε)), axis=0) + \
        dL_dσ² * np.sum(-2 * (X - μ), axis=0) / N

dL_dX = dL_dX_norm / np.sqrt(σ² + ε) + \
        dL_dσ² * 2 * (X - μ) / N + \
        dL_dμ / N
```

**4. Why is BatchNorm backward complex?**
- Each sample's normalization depends on ALL samples in batch (mean, variance)
- Chain rule involves multiple paths (through μ, σ², and direct)
- Must account for batch statistics in gradients
- Computationally more expensive than forward pass

**Simplified intuition:**
- γ, β gradients: straightforward chain rule
- X gradients: Must propagate through batch statistics (mean and variance computed from all samples)

---

### Solution 5.3: Why BatchNorm Works

**1. Internal Covariate Shift problem:**
- During training, layer inputs' distributions change as previous layers update
- Each layer must adapt to constantly shifting input distribution
- BatchNorm stabilizes these distributions (zero mean, unit variance)

**2. Smoother loss landscape:**
- BatchNorm makes optimization landscape more "Lipschitz smooth"
- Gradients more predictable, less sensitivity to learning rate
- Can use larger learning rates (10-100x) without diverging
- Faster convergence

**3. Regularization effect:**
- Adds noise during training (each sample normalized by batch statistics)
- Similar to dropout: different "view" of data each batch
- Reduces overfitting (can even reduce need for dropout)
- Acts as implicit data augmentation

**4. Train vs Test behavior:**
```python
# Training: use batch statistics
μ_batch = np.mean(X_batch, axis=0)
σ²_batch = np.var(X_batch, axis=0)
X_norm = (X_batch - μ_batch) / np.sqrt(σ²_batch + ε)

# Test: use running statistics (accumulated during training)
X_norm = (X_test - μ_running) / np.sqrt(σ²_running + ε)

# Running statistics updated during training:
μ_running = momentum * μ_running + (1 - momentum) * μ_batch
σ²_running = momentum * σ²_running + (1 - momentum) * σ²_batch
```

**Why different at test time?**
- Test may have batch size 1 (can't compute batch statistics)
- Need consistent behavior regardless of batch composition
- Running average provides stable, representative statistics

---

## Part 6: CNN Architectures

### Solution 6.1: LeNet-5

**Input: 28×28×1 (MNIST)**

**Architecture:**
1. Conv 5×5, 6 filters → 24×24×6 (no padding)
2. Pool 2×2, stride=2 → 12×12×6
3. Conv 5×5, 16 filters → 8×8×16
4. Pool 2×2, stride=2 → 4×4×16
5. Flatten → 256
6. FC 120 → 120
7. FC 84 → 84
8. FC 10 → 10

**Output shapes:**
- After Conv1: (28-5+1) × (28-5+1) × 6 = **24×24×6**
- After Pool1: 24/2 × 24/2 × 6 = **12×12×6**
- After Conv2: (12-5+1) × (12-5+1) × 16 = **8×8×16**
- After Pool2: 8/2 × 8/2 × 16 = **4×4×16** = 256 flattened
- After FC1: **120**
- After FC2: **84**
- After FC3: **10** (output classes)

**Total parameters:**
- Conv1: (5×5×1 + 1) × 6 = 156
- Conv2: (5×5×6 + 1) × 16 = 2,416
- FC1: 256 × 120 + 120 = 30,840
- FC2: 120 × 84 + 84 = 10,164
- FC3: 84 × 10 + 10 = 850
- **Total: 44,426 parameters**

**Note:** Most parameters in fully-connected layers!

---

### Solution 6.2: VGG Block

**1. Stack two 3×3 convs: receptive field?**
- First 3×3: RF = 3
- Second 3×3: RF = 3 + (3-1) = **5×5**

**2. Compare params: two 3×3 vs one 5×5**
- Two 3×3: 2 × (3×3×C²) = **18C²** parameters
- One 5×5: 5×5×C² = **25C²** parameters
- Savings: 28% fewer parameters with two 3×3!

**3. Why 3×3 better?**
- **Fewer parameters**: More efficient
- **More nonlinearity**: Two ReLUs instead of one
- **Deeper network**: More expressive power
- **Better features**: Multiple nonlinear transformations

**4. VGG block implementation:**
```python
class VGGBlock:
    def __init__(self, in_channels, out_channels):
        self.conv1 = Conv2D(in_channels, out_channels, 3, padding=1)
        self.conv2 = Conv2D(out_channels, out_channels, 3, padding=1)
        self.pool = MaxPool2D(2, stride=2)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = self.pool(x)
        return x
```

---

### Solution 6.3: ResNet Skip Connection

**Residual block:** F(x) = Conv(Conv(x)) + x

**Given:** Input 16×16×64, Output 16×16×64

**1. Implementation:**
```python
class ResidualBlock:
    def __init__(self, channels):
        self.conv1 = Conv2D(channels, channels, 3, padding=1)
        self.bn1 = BatchNorm2D(channels)
        self.conv2 = Conv2D(channels, channels, 3, padding=1)
        self.bn2 = BatchNorm2D(channels)

    def forward(self, x):
        identity = x  # Save input

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Add skip connection
        out = out + identity
        out = relu(out)

        return out
```

**2. Forward computation:**
```
x: 16×16×64
→ Conv 3×3, 64: 16×16×64
→ BN + ReLU: 16×16×64
→ Conv 3×3, 64: 16×16×64
→ BN: 16×16×64
→ Add x (skip): 16×16×64
→ ReLU: 16×16×64
```

**3. Why does skip connection help training?**

**Gradient flow:**
- Without skip: gradient passes through all layers, may vanish
- With skip: gradient splits into two paths
  - Direct path: dL/dx (always flows)
  - Residual path: dL/dF(x)

```python
# Backward pass
dL/dx = dL/dF · dF/dx + dL/dx(identity)
      = dL/dF · dF/dx + dL/dx  # Always has direct term!
```

**Benefits:**
- **Gradient highway**: Always has "1" component in gradient
- **Easier to learn identity**: Network can set F(x)≈0 if identity is optimal
- **No vanishing gradients**: Even in 100+ layer networks
- **Better optimization**: Smoother loss landscape

**Key insight:** Skip connections allow networks to be arbitrarily deep without vanishing gradients. The gradient always has a direct path to early layers.

---

## Challenge Problems

### Challenge 1: Implement Conv2D from Scratch

**Complete implementation with forward and backward:**

```python
import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialization
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * \
                 np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros(out_channels)

        self.cache = None

    def forward(self, X):
        """
        X: (batch, in_channels, H, W)
        Returns: (batch, out_channels, H_out, W_out)
        """
        batch_size, C_in, H_in, W_in = X.shape
        assert C_in == self.in_channels

        # Apply padding
        if self.padding > 0:
            X = np.pad(X, ((0,0), (0,0),
                          (self.padding, self.padding),
                          (self.padding, self.padding)),
                      mode='constant')

        _, _, H_pad, W_pad = X.shape

        # Calculate output dimensions
        H_out = (H_pad - self.kernel_size) // self.stride + 1
        W_out = (W_pad - self.kernel_size) // self.stride + 1

        # Initialize output
        out = np.zeros((batch_size, self.out_channels, H_out, W_out))

        # Convolution
        for n in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride

                        # Extract region
                        region = X[n, :,
                                  h_start:h_start+self.kernel_size,
                                  w_start:w_start+self.kernel_size]

                        # Convolve
                        out[n, c_out, h, w] = np.sum(region * self.W[c_out]) + self.b[c_out]

        self.cache = X
        return out

    def backward(self, dout):
        """
        dout: (batch, out_channels, H_out, W_out)
        Returns: dX (batch, in_channels, H, W)
        """
        X = self.cache
        batch_size, C_in, H_pad, W_pad = X.shape
        _, C_out, H_out, W_out = dout.shape

        # Initialize gradients
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(X)

        # Compute gradients
        for n in range(batch_size):
            for c_out in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride

                        # Extract region
                        region = X[n, :,
                                  h_start:h_start+self.kernel_size,
                                  w_start:w_start+self.kernel_size]

                        # Gradient w.r.t weights
                        dW[c_out] += dout[n, c_out, h, w] * region

                        # Gradient w.r.t bias
                        db[c_out] += dout[n, c_out, h, w]

                        # Gradient w.r.t input
                        dX[n, :,
                           h_start:h_start+self.kernel_size,
                           w_start:w_start+self.kernel_size] += \
                            dout[n, c_out, h, w] * self.W[c_out]

        # Average over batch
        self.dW = dW / batch_size
        self.db = db / batch_size

        # Remove padding from dX
        if self.padding > 0:
            dX = dX[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return dX

    def numerical_gradient_check(self, X, eps=1e-5):
        """Verify backward pass with numerical gradients"""
        # Forward
        out = self.forward(X)
        loss = np.sum(out**2)  # Simple loss

        # Analytical gradient
        dout = 2 * out
        dX_analytical = self.backward(dout)

        # Numerical gradient for one weight
        w_idx = (0, 0, 0, 0)  # Check first weight

        self.W[w_idx] += eps
        out_plus = self.forward(X)
        loss_plus = np.sum(out_plus**2)

        self.W[w_idx] -= 2*eps
        out_minus = self.forward(X)
        loss_minus = np.sum(out_minus**2)

        self.W[w_idx] += eps  # Reset

        dW_numerical = (loss_plus - loss_minus) / (2*eps)
        dW_analytical = self.dW[w_idx]

        relative_error = abs(dW_numerical - dW_analytical) / \
                        (abs(dW_numerical) + abs(dW_analytical) + 1e-8)

        print(f"Numerical gradient: {dW_numerical:.6f}")
        print(f"Analytical gradient: {dW_analytical:.6f}")
        print(f"Relative error: {relative_error:.2e}")

        return relative_error < 1e-5

# Test
conv = Conv2D(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
X = np.random.randn(2, 3, 5, 5)

# Forward
out = conv.forward(X)
print(f"Output shape: {out.shape}")  # (2, 8, 5, 5)

# Backward
dout = np.random.randn(*out.shape)
dX = conv.backward(dout)
print(f"dX shape: {dX.shape}")  # (2, 3, 5, 5)

# Gradient check
print("\nGradient check:")
conv.numerical_gradient_check(X)
```

---

### Challenge 2: Train Tiny CNN on MNIST

**Complete training pipeline:**

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.reshape(-1, 1, 28, 28) / 255.0  # Normalize
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# One-hot encode labels
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train_onehot = one_hot(y_train)
y_test_onehot = one_hot(y_test)

# Layers (simplified implementations)
class SimpleCNN:
    def __init__(self):
        # Conv1: 1→8 channels, 3×3
        self.conv1 = Conv2D(1, 8, 3, padding=1)
        # Pool1: 28×28→14×14
        # Conv2: 8→16 channels, 3×3
        self.conv2 = Conv2D(8, 16, 3, padding=1)
        # Pool2: 14×14→7×7
        # FC: 16×7×7=784 → 10
        self.fc = np.random.randn(784, 10) * 0.01
        self.fc_bias = np.zeros(10)

    def forward(self, X, training=True):
        # Conv1 + ReLU + Pool
        out = self.conv1.forward(X)
        out = np.maximum(0, out)  # ReLU
        out = maxpool(out, 2)  # 28→14

        # Conv2 + ReLU + Pool
        out = self.conv2.forward(out)
        out = np.maximum(0, out)  # ReLU
        out = maxpool(out, 2)  # 14→7

        # Flatten + FC
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)
        out = out @ self.fc + self.fc_bias

        # Softmax
        exp = np.exp(out - np.max(out, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        return probs

    def backward(self, X, y, probs):
        # Softmax + cross-entropy gradient
        dout = probs - y  # (batch, 10)

        # FC backward
        dfc = flatten_output.T @ dout
        dfc_bias = np.sum(dout, axis=0)
        dout = dout @ self.fc.T

        # Reshape back
        dout = dout.reshape(conv2_pool_shape)

        # Continue backward through conv2, pool, conv1...
        # (full implementation omitted for brevity)

        return dfc, dfc_bias

# Training loop
model = SimpleCNN()
learning_rate = 0.01
momentum = 0.9
batch_size = 32
epochs = 10

velocity_fc = np.zeros_like(model.fc)

for epoch in range(epochs):
    # Shuffle data
    indices = np.random.permutation(len(X_train))

    for i in range(0, len(X_train), batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = X_train[batch_indices]
        y_batch = y_train_onehot[batch_indices]

        # Forward
        probs = model.forward(X_batch)

        # Loss
        loss = -np.mean(np.sum(y_batch * np.log(probs + 1e-8), axis=1))

        # Backward
        dfc, dfc_bias = model.backward(X_batch, y_batch, probs)

        # SGD with momentum
        velocity_fc = momentum * velocity_fc - learning_rate * dfc
        model.fc += velocity_fc
        model.fc_bias -= learning_rate * dfc_bias

    # Evaluate
    train_probs = model.forward(X_train[:1000])
    train_acc = np.mean(np.argmax(train_probs, axis=1) == y_train[:1000])

    test_probs = model.forward(X_test)
    test_acc = np.mean(np.argmax(test_probs, axis=1) == y_test)

    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")

print(f"\nFinal test accuracy: {test_acc:.3f}")
# Target: >95% (achievable with this architecture)
```

**Tips for >95% accuracy:**
- Use proper initialization (He for ReLU)
- Learning rate schedule (reduce after 5 epochs)
- Data augmentation (random shifts, rotations)
- Add BatchNorm layers
- Use Adam optimizer instead of SGD+momentum

---

**You've completed the CNN exercises! You now understand:**
- ✅ 2D convolution mechanics (hand calculations + implementation)
- ✅ Pooling operations (max, average, global)
- ✅ CNN forward pass (multi-layer architectures)
- ✅ CNN backpropagation (conv, pool, full network)
- ✅ Batch normalization (forward and backward)
- ✅ Modern architectures (LeNet, VGG, ResNet)
- ✅ Complete CNN training pipeline

**Next:** Move on to Transformer exercises or start implementing classic papers!
