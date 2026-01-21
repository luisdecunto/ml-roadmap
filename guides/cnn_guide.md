# CNN from Scratch - Coding Guide

**Time:** 15-20 hours
**Difficulty:** Advanced
**Prerequisites:** Python, NumPy, neural networks, backpropagation

## What You'll Build

Implement a complete Convolutional Neural Network from scratch:
1. 1D convolution (warmup on signals)
2. 2D convolution (images)
3. Pooling layers (max and average)
4. Backpropagation through conv and pool layers
5. Complete CNN architecture
6. **Final Project:** Train on MNIST (>90% accuracy)

---

## Project Setup

```bash
mkdir cnn-from-scratch
cd cnn-from-scratch

# Create files
touch conv1d.py
touch conv2d.py
touch pooling.py
touch conv_backprop.py
touch simple_cnn.py
touch requirements.txt
```

### requirements.txt
```
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
scikit-image>=0.21.0  # For test images
```

---

## Part 1: 1D Convolution (Warmup)

### Theory

**Convolution** is a sliding window operation:
```
Signal:  [1, 2, 3, 4, 5]
Kernel:  [1, 0, -1]  (edge detector)

Step 1:  [1, 2, 3] → 1*1 + 2*0 + 3*(-1) = -2
Step 2:     [2, 3, 4] → 1*2 + 0*3 + (-1)*4 = -2
Step 3:        [3, 4, 5] → 1*3 + 0*4 + (-1)*5 = -2

Output: [-2, -2, -2]
```

**Key Parameters:**
- **Stride**: Step size (larger stride = smaller output)
- **Padding**: Add zeros around input (preserve size)

**Output Size Formula:**
```
output_length = (input_length + 2*padding - kernel_length) // stride + 1
```

### Implementation

```python
# conv1d.py
import numpy as np
import matplotlib.pyplot as plt

def conv1d(signal, kernel, stride=1, padding=0):
    """
    1D convolution operation.

    Args:
        signal: (length,) - Input signal
        kernel: (kernel_size,) - Convolution kernel
        stride: Step size for sliding window
        padding: Zero padding on both sides

    Returns:
        output: Convolved signal
    """
    # Apply padding
    if padding > 0:
        signal = np.pad(signal, padding, mode='constant')

    # Calculate output size
    output_length = (len(signal) - len(kernel)) // stride + 1
    output = np.zeros(output_length)

    # Slide kernel over signal
    for i in range(output_length):
        start = i * stride
        end = start + len(kernel)
        # Element-wise multiply and sum (dot product)
        output[i] = np.sum(signal[start:end] * kernel)

    return output


# Test cases
if __name__ == "__main__":
    print("=" * 60)
    print("1D Convolution Tests")
    print("=" * 60)

    # Test 1: Basic edge detection
    signal = np.array([1, 2, 3, 4, 5])
    kernel = np.array([1, 0, -1])
    result = conv1d(signal, kernel)
    print(f"\nTest 1: Edge Detection")
    print(f"Signal: {signal}")
    print(f"Kernel: {kernel}")
    print(f"Output: {result}")
    print(f"Expected: [-2, -2, -2]")
    assert np.allclose(result, [-2, -2, -2])
    print("✓ Test 1 passed!")

    # Test 2: With stride
    result_stride = conv1d(signal, kernel, stride=2)
    print(f"\nTest 2: Stride=2")
    print(f"Output: {result_stride}")
    print(f"Expected: [-2, -2]")
    assert np.allclose(result_stride, [-2, -2])
    print("✓ Test 2 passed!")

    # Test 3: With padding
    result_pad = conv1d(signal, kernel, padding=1)
    print(f"\nTest 3: Padding=1")
    print(f"Output: {result_pad}")
    print(f"Expected: [1, -2, -2, -2, -5]")
    assert np.allclose(result_pad, [1, -2, -2, -2, -5])
    print("✓ Test 3 passed!")

    print("\n" + "=" * 60)
    print("All tests passed!")
```

### Test It!

```bash
python conv1d.py
```

Expected: All 3 tests pass.

---

## Part 2: 2D Convolution

### Theory

2D convolution extends 1D to images:

```
Image (3x3):          Kernel (2x2):
[1 2 3]               [1  0]
[4 5 6]               [0 -1]
[7 8 9]

Output position (0,0):
[1 2] * [1  0] = 1*1 + 2*0 + 4*0 + 5*(-1) = -4
[4 5]   [0 -1]

Output position (0,1):
[2 3] * [1  0] = 2*1 + 3*0 + 5*0 + 6*(-1) = -4
[5 6]   [0 -1]

... and so on
```

**Common Kernels:**
- **Sobel X**: Detects vertical edges
- **Sobel Y**: Detects horizontal edges
- **Blur**: Smoothing (average kernel)
- **Sharpen**: Enhances edges

### Implementation

```python
# conv2d.py
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

def conv2d(image, kernel, stride=1, padding=0):
    """
    2D convolution for single channel.

    Args:
        image: (H, W) - Input image
        kernel: (kH, kW) - Convolution kernel
        stride: Step size
        padding: Zero padding

    Returns:
        output: (out_H, out_W) - Convolved image
    """
    # Handle stride as tuple
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    # Apply padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant')

    H, W = image.shape
    kH, kW = kernel.shape

    # Calculate output size
    out_H = (H - kH) // stride_h + 1
    out_W = (W - kW) // stride_w + 1

    output = np.zeros((out_H, out_W))

    # Slide kernel over image
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride_h
            h_end = h_start + kH
            w_start = j * stride_w
            w_end = w_start + kW

            # Extract region and convolve
            region = image[h_start:h_end, w_start:w_end]
            output[i, j] = np.sum(region * kernel)

    return output


def conv2d_multichannel(image, kernel, stride=1, padding=0):
    """
    2D convolution with multiple input/output channels.

    Args:
        image: (in_channels, H, W)
        kernel: (out_channels, in_channels, kH, kW)
        stride: Step size
        padding: Zero padding

    Returns:
        output: (out_channels, out_H, out_W)
    """
    out_channels, in_channels, kH, kW = kernel.shape
    _, H, W = image.shape

    # Apply padding
    if padding > 0:
        image = np.pad(image, ((0, 0), (padding, padding), (padding, padding)))
        H, W = image.shape[1], image.shape[2]

    # Calculate output size
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1

    output = np.zeros((out_channels, out_H, out_W))

    # For each output channel
    for oc in range(out_channels):
        # Sum over all input channels
        for ic in range(in_channels):
            # Convolve this input channel with corresponding kernel
            output[oc] += conv2d(image[ic], kernel[oc, ic], stride, 0)

    return output


# Common kernels
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

SOBEL_Y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

SHARPEN = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

BLUR = np.ones((5, 5)) / 25


# Test cases
if __name__ == "__main__":
    print("=" * 60)
    print("2D Convolution Tests")
    print("=" * 60)

    # Test 1: Basic 2x2 kernel
    image = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    kernel = np.array([[1, 0],
                      [0, -1]])

    result = conv2d(image, kernel)
    expected = np.array([[-4, -4],
                        [-4, -4]])

    print(f"\nTest 1: Basic 2x2 Convolution")
    print(f"Image shape: {image.shape}")
    print(f"Kernel shape: {kernel.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Output:\n{result}")
    print(f"Expected:\n{expected}")
    assert np.allclose(result, expected)
    print("✓ Test 1 passed!")

    # Test 2: Stride
    image2 = np.arange(25).reshape(5, 5)
    kernel2 = np.ones((2, 2))
    result_stride = conv2d(image2, kernel2, stride=2)
    print(f"\nTest 2: Stride=2")
    print(f"Image shape: {image2.shape}")
    print(f"Output shape: {result_stride.shape}")
    assert result_stride.shape == (2, 2)
    print("✓ Test 2 passed!")

    # Test 3: Multi-channel
    print(f"\nTest 3: Multi-Channel Convolution")
    image_mc = np.random.randn(3, 5, 5)
    kernel_mc = np.random.randn(2, 3, 3, 3)
    result_mc = conv2d_multichannel(image_mc, kernel_mc)
    print(f"Input shape: {image_mc.shape} (3 channels)")
    print(f"Kernel shape: {kernel_mc.shape} (2 out, 3 in)")
    print(f"Output shape: {result_mc.shape}")
    assert result_mc.shape == (2, 3, 3)
    print("✓ Test 3 passed!")

    print("\n" + "=" * 60)
    print("All tests passed!")
```

### Test It!

```bash
python conv2d.py
```

Expected: All 3 tests pass.

---

## Part 3: Pooling Layers

### Theory

**Pooling** reduces spatial dimensions (downsampling):

**Max Pooling (2x2):**
```
Input (4x4):        Output (2x2):
[1  2  3  4]        [6  8]
[5  6  7  8]   →    [14 16]
[9  10 11 12]
[13 14 15 16]

Takes maximum in each 2x2 window
```

**Why Pooling?**
1. Reduces computation
2. Provides translation invariance
3. Prevents overfitting

### Implementation

```python
# pooling.py
import numpy as np

def max_pool2d(image, pool_size=2, stride=None):
    """
    Max pooling operation.

    Args:
        image: (H, W) or (C, H, W)
        pool_size: Size of pooling window
        stride: Step size (default: same as pool_size)

    Returns:
        output: Downsampled image
    """
    if stride is None:
        stride = pool_size

    # Handle single vs multi-channel
    if image.ndim == 2:
        H, W = image.shape
        out_H = (H - pool_size) // stride + 1
        out_W = (W - pool_size) // stride + 1
        output = np.zeros((out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                h_start = i * stride
                h_end = h_start + pool_size
                w_start = j * stride
                w_end = w_start + pool_size

                region = image[h_start:h_end, w_start:w_end]
                output[i, j] = np.max(region)

        return output
    else:
        # Multi-channel: pool each channel independently
        C, H, W = image.shape
        out_H = (H - pool_size) // stride + 1
        out_W = (W - pool_size) // stride + 1
        output = np.zeros((C, out_H, out_W))

        for c in range(C):
            output[c] = max_pool2d(image[c], pool_size, stride)

        return output


def avg_pool2d(image, pool_size=2, stride=None):
    """Average pooling operation."""
    if stride is None:
        stride = pool_size

    if image.ndim == 2:
        H, W = image.shape
        out_H = (H - pool_size) // stride + 1
        out_W = (W - pool_size) // stride + 1
        output = np.zeros((out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                h_start = i * stride
                h_end = h_start + pool_size
                w_start = j * stride
                w_end = w_start + pool_size

                region = image[h_start:h_end, w_start:w_end]
                output[i, j] = np.mean(region)

        return output
    else:
        C, H, W = image.shape
        out_H = (H - pool_size) // stride + 1
        out_W = (W - pool_size) // stride + 1
        output = np.zeros((C, out_H, out_W))

        for c in range(C):
            output[c] = avg_pool2d(image[c], pool_size, stride)

        return output


# Test cases
if __name__ == "__main__":
    print("=" * 60)
    print("Pooling Tests")
    print("=" * 60)

    # Test 1: Max pooling
    image = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]])
    result = max_pool2d(image, pool_size=2)
    expected = np.array([[6, 8],
                        [14, 16]])

    print(f"\nTest 1: Max Pooling 2x2")
    print(f"Input:\n{image}")
    print(f"Output:\n{result}")
    print(f"Expected:\n{expected}")
    assert np.allclose(result, expected)
    print("✓ Test 1 passed!")

    # Test 2: Average pooling
    result_avg = avg_pool2d(image, pool_size=2)
    expected_avg = np.array([[3.5, 5.5],
                            [11.5, 13.5]])
    print(f"\nTest 2: Average Pooling 2x2")
    print(f"Output:\n{result_avg}")
    print(f"Expected:\n{expected_avg}")
    assert np.allclose(result_avg, expected_avg)
    print("✓ Test 2 passed!")

    # Test 3: Multi-channel
    image_mc = np.random.randn(3, 8, 8)
    result_mc = max_pool2d(image_mc, pool_size=2)
    print(f"\nTest 3: Multi-Channel Pooling")
    print(f"Input shape: {image_mc.shape}")
    print(f"Output shape: {result_mc.shape}")
    assert result_mc.shape == (3, 4, 4)
    print("✓ Test 3 passed!")

    print("\n" + "=" * 60)
    print("All tests passed!")
```

### Test It!

```bash
python pooling.py
```

Expected: All 3 tests pass.

---

## Part 4: Backpropagation

### Theory

**Key Insight:** Gradients flow backward through each layer.

**Conv2D Backward:**
- Gradient w.r.t. kernel: Correlate input with output gradient
- Gradient w.r.t. input: Convolve output gradient with flipped kernel

**Max Pooling Backward:**
- Gradient flows only through the max element in each window

**Numerical Gradient Checking:**
```python
numerical_grad = (loss(θ + ε) - loss(θ - ε)) / (2ε)
```
Should match analytical gradient (error < 1e-5).

### Implementation

```python
# conv_backprop.py
import numpy as np

def conv2d_backward(dL_dout, image, kernel, stride=1, padding=0):
    """
    Backward pass for 2D convolution.

    Args:
        dL_dout: Gradient w.r.t. output (out_H, out_W)
        image: Input image (H, W)
        kernel: Convolution kernel (kH, kW)

    Returns:
        dL_dinput: Gradient w.r.t. input (H, W)
        dL_dkernel: Gradient w.r.t. kernel (kH, kW)
    """
    kH, kW = kernel.shape
    out_H, out_W = dL_dout.shape

    # Apply padding to image if needed
    if padding > 0:
        image_padded = np.pad(image, padding, mode='constant')
    else:
        image_padded = image

    # --- Gradient w.r.t. kernel ---
    dL_dkernel = np.zeros_like(kernel)

    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            h_end = h_start + kH
            w_start = j * stride
            w_end = w_start + kW

            region = image_padded[h_start:h_end, w_start:w_end]
            dL_dkernel += dL_dout[i, j] * region

    # --- Gradient w.r.t. input ---
    dL_dinput_padded = np.zeros_like(image_padded)
    kernel_flipped = np.flip(kernel)

    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            h_end = h_start + kH
            w_start = j * stride
            w_end = w_start + kW

            dL_dinput_padded[h_start:h_end, w_start:w_end] += \
                dL_dout[i, j] * kernel_flipped

    # Remove padding if needed
    if padding > 0:
        dL_dinput = dL_dinput_padded[padding:-padding, padding:-padding]
    else:
        dL_dinput = dL_dinput_padded

    return dL_dinput, dL_dkernel


def max_pool2d_backward(dL_dout, image, pool_size=2, stride=None):
    """
    Backward pass for max pooling.

    Gradient flows only through the max element in each window.
    """
    if stride is None:
        stride = pool_size

    dL_dinput = np.zeros_like(image)
    out_H, out_W = dL_dout.shape

    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size

            region = image[h_start:h_end, w_start:w_end]

            # Find position of max element
            max_idx = np.unravel_index(np.argmax(region), region.shape)

            # Gradient flows only through max element
            dL_dinput[h_start + max_idx[0], w_start + max_idx[1]] += dL_dout[i, j]

    return dL_dinput


# Gradient checking
def numerical_gradient_conv2d(image, kernel, epsilon=1e-5):
    """Compute numerical gradient for verification."""
    from conv2d import conv2d

    kH, kW = kernel.shape
    numerical_grad = np.zeros_like(kernel)

    for i in range(kH):
        for j in range(kW):
            kernel[i, j] += epsilon
            out_plus = conv2d(image, kernel)
            loss_plus = np.sum(out_plus)

            kernel[i, j] -= 2 * epsilon
            out_minus = conv2d(image, kernel)
            loss_minus = np.sum(out_minus)

            kernel[i, j] += epsilon

            numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

    return numerical_grad


if __name__ == "__main__":
    from conv2d import conv2d
    from pooling import max_pool2d

    print("=" * 60)
    print("Gradient Checking")
    print("=" * 60)

    # Test conv2d gradient
    np.random.seed(42)
    image = np.random.randn(5, 5)
    kernel = np.random.randn(3, 3)

    output = conv2d(image, kernel)
    dL_dout = np.ones_like(output)

    dL_dinput, dL_dkernel = conv2d_backward(dL_dout, image, kernel)
    dL_dkernel_numerical = numerical_gradient_conv2d(image, kernel)

    diff = np.abs(dL_dkernel - dL_dkernel_numerical).max()
    print(f"\nConv2D Gradient Check:")
    print(f"  Max difference: {diff:.2e}")

    if diff < 1e-5:
        print("  ✓ Gradient check PASSED!")
    else:
        print("  ✗ Gradient check FAILED!")

    # Test max pooling gradient
    image2 = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12],
                      [13, 14, 15, 16]], dtype=float)

    output2 = max_pool2d(image2, pool_size=2)
    dL_dout2 = np.ones_like(output2)
    dL_dinput2 = max_pool2d_backward(dL_dout2, image2, pool_size=2)

    expected = np.array([[0, 0, 0, 0],
                        [0, 1, 0, 1],
                        [0, 0, 0, 0],
                        [0, 1, 0, 1]], dtype=float)

    print(f"\nMax Pooling Gradient Check:")
    print(f"  Gradient:\n{dL_dinput2}")
    print(f"  Expected:\n{expected}")

    if np.allclose(dL_dinput2, expected):
        print("  ✓ Gradient check PASSED!")
    else:
        print("  ✗ Gradient check FAILED!")
```

### Test It!

```bash
python conv_backprop.py
```

Expected: Both gradient checks pass.

---

## Part 5: Complete CNN on MNIST

### Architecture

```
Input (1x28x28)
    ↓
Conv (8 filters, 3x3) → ReLU
    ↓ (8x26x26)
MaxPool (2x2)
    ↓ (8x13x13)
Flatten → (1352,)
    ↓
FC (10 classes) → Softmax
    ↓
Output (10,)
```

### Implementation

```python
# simple_cnn.py
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class SimpleCNN:
    """Simple CNN for MNIST."""

    def __init__(self):
        # Conv: 1 -> 8 filters, 3x3
        self.conv_kernel = np.random.randn(8, 1, 3, 3) * 0.01
        self.conv_bias = np.zeros(8)

        # FC: 1352 -> 10
        self.fc_weights = np.random.randn(1352, 10) * 0.01
        self.fc_bias = np.zeros(10)

        self.cache = {}

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def conv2d_single(self, image, kernel, bias):
        """Simple 2D convolution for single image."""
        _, in_channels, kH, kW = kernel.shape
        _, H, W = image.shape

        out_channels = kernel.shape[0]
        out_H = H - kH + 1
        out_W = W - kW + 1
        output = np.zeros((out_channels, out_H, out_W))

        for oc in range(out_channels):
            for ic in range(in_channels):
                for i in range(out_H):
                    for j in range(out_W):
                        region = image[ic, i:i+kH, j:j+kW]
                        output[oc, i, j] += np.sum(region * kernel[oc, ic])
            output[oc] += bias[oc]

        return output

    def max_pool2d_simple(self, image, pool_size=2):
        """Simple 2x2 max pooling."""
        C, H, W = image.shape
        out_H = H // pool_size
        out_W = W // pool_size
        output = np.zeros((C, out_H, out_W))

        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    region = image[c, i*2:i*2+2, j*2:j*2+2]
                    output[c, i, j] = np.max(region)

        return output

    def forward(self, x):
        """Forward pass."""
        self.cache['x'] = x

        # Conv + ReLU
        conv_out = self.conv2d_single(x, self.conv_kernel, self.conv_bias)
        self.cache['conv_out'] = conv_out
        relu_out = self.relu(conv_out)

        # Pool
        pool_out = self.max_pool2d_simple(relu_out)
        self.cache['pool_out'] = pool_out

        # Flatten
        flattened = pool_out.reshape(-1)

        # FC + Softmax
        fc_out = flattened @ self.fc_weights + self.fc_bias
        probs = self.softmax(fc_out)

        return probs

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs)


# Load MNIST
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data.values, mnist.target.values.astype(int)

# Normalize and reshape
X = X / 255.0
X = X.reshape(-1, 1, 28, 28)

# Small subset for testing
X = X[:1000]
y = y[:1000]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)}")
print(f"Test set: {len(X_test)}")

# Create and test CNN
cnn = SimpleCNN()

print("\nTesting CNN forward pass...")
sample = X_train[0]
pred = cnn.predict(sample)
print(f"Sample shape: {sample.shape}")
print(f"Prediction: {pred}")
print(f"True label: {y_train[0]}")

# Test on a few samples
correct = 0
for i in range(min(50, len(X_test))):
    pred = cnn.predict(X_test[i])
    if pred == y_test[i]:
        correct += 1

accuracy = correct / min(50, len(X_test))
print(f"\nRandom initialization accuracy: {accuracy:.2%}")
print("(Expected: ~10% for random weights)")
```

### Test It!

```bash
python simple_cnn.py
```

Expected: Forward pass works, random accuracy ~10%.

---

## Quick Reference

### Convolution Output Size
```python
out_H = (H + 2*pad - kH) // stride + 1
out_W = (W + 2*pad - kW) // stride + 1
```

### Shape Tracking (MNIST Example)
```
Input:         (1, 28, 28)
After Conv 3x3: (8, 26, 26)   # 8 filters
After Pool 2x2: (8, 13, 13)   # halved
Flatten:        (1352,)        # 8 * 13 * 13
After FC:       (10,)          # 10 classes
```

### Common Kernels
```python
# Edge detection
SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

# Blur
BLUR = np.ones((5, 5)) / 25

# Sharpen
SHARPEN = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
```

---

## Part 6: Training Loop with Backpropagation

Now let's add training! We need to implement backward passes for each layer and connect them.

### Training Architecture

```python
# train_cnn.py
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class TrainableCNN:
    def __init__(self, lr=0.001):
        # Conv: 1 -> 8 filters, 3x3
        self.conv_kernel = np.random.randn(8, 1, 3, 3) * 0.01
        self.conv_bias = np.zeros(8)

        # FC: 1352 -> 10
        self.fc_weights = np.random.randn(1352, 10) * 0.01
        self.fc_bias = np.zeros(10)

        self.lr = lr
        self.cache = {}

    def relu(self, x):
        return np.maximum(0, x)

    def relu_backward(self, dout, x):
        """ReLU gradient: 1 if x > 0, else 0"""
        dx = dout.copy()
        dx[x <= 0] = 0
        return dx

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def cross_entropy_loss(self, probs, label):
        """Cross-entropy loss"""
        return -np.log(probs[label] + 1e-8)  # Add epsilon for numerical stability

    def forward(self, x, label=None):
        """Forward pass with caching for backprop"""
        self.cache['x'] = x

        # Conv + ReLU
        conv_out = self.conv_forward(x)
        self.cache['conv_out'] = conv_out

        relu_out = self.relu(conv_out)
        self.cache['relu_out'] = relu_out

        # Pool
        pool_out = self.pool_forward(relu_out)
        self.cache['pool_out'] = pool_out

        # Flatten
        flattened = pool_out.reshape(-1)
        self.cache['flattened'] = flattened

        # FC + Softmax
        fc_out = flattened @ self.fc_weights + self.fc_bias
        probs = self.softmax(fc_out)

        if label is not None:
            loss = self.cross_entropy_loss(probs, label)
            return probs, loss
        return probs

    def conv_forward(self, image):
        """Conv forward using your conv2d function"""
        from conv2d import conv2d
        out_channels = self.conv_kernel.shape[0]
        results = []

        for out_c in range(out_channels):
            kernel_2d = self.conv_kernel[out_c, 0]
            result = conv2d(image[0], kernel_2d) + self.conv_bias[out_c]
            results.append(result)

        return np.stack(results, axis=0)

    def pool_forward(self, image):
        """Pool forward using your max_pool2d function"""
        from pooling import max_pool2d
        return max_pool2d(image, pool_size=2)

    def backward(self, probs, label):
        """Full backward pass"""
        # Gradient of loss w.r.t. softmax output
        dprobs = probs.copy()
        dprobs[label] -= 1  # Softmax + Cross-entropy gradient

        # FC backward
        dflattened = dprobs @ self.fc_weights.T
        dfc_weights = np.outer(self.cache['flattened'], dprobs)
        dfc_bias = dprobs

        # Unflatten
        pool_out_shape = self.cache['pool_out'].shape
        dpool_out = dflattened.reshape(pool_out_shape)

        # Pool backward
        drelu_out = self.pool_backward(dpool_out)

        # ReLU backward
        dconv_out = self.relu_backward(drelu_out, self.cache['conv_out'])

        # Conv backward
        dconv_kernel, dconv_bias = self.conv_backward(dconv_out)

        # Update weights
        self.fc_weights -= self.lr * dfc_weights
        self.fc_bias -= self.lr * dfc_bias
        self.conv_kernel -= self.lr * dconv_kernel
        self.conv_bias -= self.lr * dconv_bias

    def pool_backward(self, dout):
        """Max pool backward - gradient flows only through max positions"""
        from pooling import max_pool2d

        relu_out = self.cache['relu_out']
        drelu_out = np.zeros_like(relu_out)

        pool_size = 2
        out_channels, out_H, out_W = dout.shape

        for c in range(out_channels):
            for i in range(out_H):
                for j in range(out_W):
                    h_start = i * pool_size
                    h_end = h_start + pool_size
                    w_start = j * pool_size
                    w_end = w_start + pool_size

                    # Find max position in this pool region
                    region = relu_out[c, h_start:h_end, w_start:w_end]
                    max_idx = np.unravel_index(np.argmax(region), region.shape)

                    # Route gradient to max position
                    drelu_out[c, h_start + max_idx[0], w_start + max_idx[1]] += dout[c, i, j]

        return drelu_out

    def conv_backward(self, dout):
        """Conv backward - compute kernel and bias gradients"""
        from conv2d import conv2d

        image = self.cache['x'][0]  # (28, 28)
        out_channels = self.conv_kernel.shape[0]

        dkernel = np.zeros_like(self.conv_kernel)
        dbias = np.zeros_like(self.conv_bias)

        for out_c in range(out_channels):
            # Kernel gradient via correlation
            dkernel[out_c, 0] = conv2d(image, dout[out_c])

            # Bias gradient is sum of upstream gradients
            dbias[out_c] = np.sum(dout[out_c])

        return dkernel, dbias

    def train_step(self, x, label):
        """Single training step"""
        probs, loss = self.forward(x, label)
        self.backward(probs, label)

        acc = 1 if np.argmax(probs) == label else 0
        return loss, acc

# Training script
print('Loading MNIST...')
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data.values, mnist.target.values.astype(int)
X = X / 255.0
X = X.reshape(-1, 1, 28, 28)

# Use subset for faster training
X = X[:5000]
y = y[:5000]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f'Training set: {len(X_train)}')
print(f'Test set: {len(X_test)}')

# Train CNN
cnn = TrainableCNN(lr=0.005)

print("\nTraining CNN...")
losses = []
accuracies = []

for epoch in range(3):
    print(f'\n--- Epoch {epoch + 1} ---')

    # Shuffle training data
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    epoch_loss = 0
    epoch_acc = 0

    for i, (x, label) in enumerate(zip(X_train_shuffled, y_train_shuffled)):
        loss, acc = cnn.train_step(x, label)
        epoch_loss += loss
        epoch_acc += acc

        if (i + 1) % 500 == 0:
            avg_loss = epoch_loss / (i + 1)
            avg_acc = epoch_acc / (i + 1) * 100
            print(f'  [{i + 1}/{len(X_train)}] Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%')

    # Test accuracy
    test_correct = 0
    for x, label in zip(X_test, y_test):
        probs = cnn.forward(x)
        if np.argmax(probs) == label:
            test_correct += 1

    test_acc = test_correct / len(X_test) * 100
    print(f'  Test Accuracy: {test_acc:.2f}%')
```

### Test It!

```bash
python train_cnn.py
```

Expected output:
```
--- Epoch 1 ---
  [500/4000] Loss: 2.1234, Acc: 25.00%
  [1000/4000] Loss: 1.8456, Acc: 35.50%
  ...
  Test Accuracy: 65.00%

--- Epoch 2 ---
  ...
  Test Accuracy: 78.00%

--- Epoch 3 ---
  ...
  Test Accuracy: 85.00%
```

---

## Part 7: Visualizing Learned Filters

Visualize what your CNN learned!

```python
# visualize_filters.py
import numpy as np
import matplotlib.pyplot as plt
from train_cnn import TrainableCNN
from sklearn.datasets import fetch_openml

# Load trained model (train it first!)
print('Loading MNIST and training CNN...')
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.values / 255.0
X = X.reshape(-1, 1, 28, 28)
y = mnist.target.values.astype(int)

# Train briefly
cnn = TrainableCNN(lr=0.005)
for i in range(min(1000, len(X))):
    cnn.train_step(X[i], y[i])

# Visualize filters
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle('Learned Convolutional Filters', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < cnn.conv_kernel.shape[0]:
        # Extract filter for this channel
        kernel = cnn.conv_kernel[i, 0]  # Shape: (3, 3)

        # Plot
        im = ax.imshow(kernel, cmap='gray', interpolation='nearest')
        ax.set_title(f'Filter {i + 1}')
        ax.axis('off')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('learned_filters.png', dpi=150, bbox_inches='tight')
print("Saved learned_filters.png")
plt.show()

# Visualize feature maps
def visualize_activations(cnn, image, label):
    """Visualize conv layer activations"""
    conv_out = cnn.conv_forward(image)
    relu_out = cnn.relu(conv_out)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(f'Feature Maps for Digit {label}', fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < conv_out.shape[0]:
            ax.imshow(relu_out[i], cmap='viridis')
            ax.set_title(f'Channel {i + 1}')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'activations_digit_{label}.png', dpi=150, bbox_inches='tight')
    print(f"Saved activations_digit_{label}.png")
    plt.show()

# Visualize for a few digits
for digit in [0, 1, 2]:
    idx = np.where(y == digit)[0][0]
    visualize_activations(cnn, X[idx], digit)
```

### Test It!

```bash
python visualize_filters.py
```

Expected: Generates `learned_filters.png` showing 8 filters and activation maps for different digits.

---

## Part 8: MNIST Project - Achieve >98% Accuracy

Final challenge: Build a better CNN and reach >98% accuracy!

### Improvements to Try

1. **Deeper Architecture**
```python
# 2 conv layers instead of 1
Conv(1->16, 3x3) -> ReLU -> Pool
Conv(16->32, 3x3) -> ReLU -> Pool
FC(1568->128) -> ReLU
FC(128->10) -> Softmax
```

2. **Better Initialization**
```python
# He initialization for ReLU layers
conv_kernel = np.random.randn(16, 1, 3, 3) * np.sqrt(2.0 / (1 * 3 * 3))
```

3. **Learning Rate Schedule**
```python
def get_lr(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 10:
        return 0.005
    else:
        return 0.001
```

4. **Data Augmentation** (optional)
```python
def random_shift(image, max_shift=2):
    """Randomly shift image by max_shift pixels"""
    shift_h = np.random.randint(-max_shift, max_shift + 1)
    shift_w = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(np.roll(image, shift_h, axis=1), shift_w, axis=2)
```

5. **Batch Training** (optional)
```python
# Train on mini-batches of size 32
batch_size = 32
for i in range(0, len(X_train), batch_size):
    X_batch = X_train[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]
    # Average gradients over batch
```

### Challenge Template

```python
# mnist_challenge.py
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class ImprovedCNN:
    def __init__(self):
        # TODO: Implement deeper architecture
        # Conv1: 1 -> 16 filters
        # Conv2: 16 -> 32 filters
        # FC1: 1568 -> 128
        # FC2: 128 -> 10
        pass

    # TODO: Implement all forward/backward methods
    # Remember to cache intermediate values!

# Load FULL MNIST (70,000 samples)
print('Loading full MNIST...')
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.values / 255.0
X = X.reshape(-1, 1, 28, 28)
y = mnist.target.values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42)

# Train for 10-15 epochs
# Target: >98% test accuracy

print(f'Final Test Accuracy: {test_acc:.2f}%')
print('Target: >98%')
```

### Success Criteria

- ✅ Train on full MNIST (60k training, 10k test)
- ✅ Achieve >98% test accuracy
- ✅ Implement using only NumPy (your functions!)
- ✅ Complete within reasonable time (~30 minutes training max)

**Bonus:** Compare with PyTorch LeNet baseline to validate your implementation!

---

## Common Issues & Debugging

### 1. Shape Mismatch
**Symptoms:** "operands could not be broadcast together"
**Fix:**
- Print shapes at each step
- Verify output size formula
- Check stride/padding values

### 2. Gradient Check Fails
**Symptoms:** Numerical vs analytical gradient differs
**Fix:**
- Check backward pass implementation
- Verify kernel is flipped for input gradient
- Use smaller epsilon (1e-7) if needed

### 3. Slow Performance
**Symptoms:** Takes forever to run
**Fix:**
- Use smaller dataset for testing
- Vectorize operations where possible
- Use smaller batch size

---

## Resources

**Video Tutorials:**
- [CS231n Lecture 5: CNNs](http://cs231n.stanford.edu/slides/2024/lecture_5.pdf)
- [3Blue1Brown: CNNs](https://www.youtube.com/watch?v=aircAruvnKk)

**Reading:**
- [Understanding Deep Learning - Ch 11](https://udlbook.github.io/udlbook/)
- [Victor Zhou's CNN Tutorial](https://victorzhou.com/blog/intro-to-cnns-part-1/)

**Reference Implementations:**
- [CS231n Assignment 2](http://cs231n.github.io/assignments2024/assignment2/)

---

## Next Steps

After completing this guide:
1. ✅ Implement full training loop with backprop
2. Try deeper architectures (2+ conv layers)
3. Add batch normalization
4. Implement data augmentation
5. Train on full MNIST (>95% accuracy goal)

**Target:** >90% accuracy on MNIST with your from-scratch CNN!
