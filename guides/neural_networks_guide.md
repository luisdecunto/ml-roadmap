# Neural Networks from Scratch - Complete Guide

**Time:** 8-10 hours
**Difficulty:** Intermediate-Advanced
**Prerequisites:** Python, NumPy, calculus basics

See full implementation at: https://github.com/karpathy/micrograd (reference)

## Complete Implementation

All code files included. Run `python train.py` for MNIST training (>95% accuracy in 50 epochs).

**Files:** `neuron.py`, `activations.py`, `network.py`, `train.py`

---

## Quick Reference

### Activations
```python
# ReLU
def relu(z): return np.maximum(0, z)
def relu_derivative(z): return (z > 0).astype(float)

# Sigmoid
def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_derivative(z): s = sigmoid(z); return s * (1-s)

# Softmax
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

### Forward Pass (2-layer network)
```python
# Layer 1: Input -> Hidden (ReLU)
z1 = X @ W1 + b1
a1 = relu(z1)

# Layer 2: Hidden -> Output (Softmax)
z2 = a1 @ W2 + b2
a2 = softmax(z2)
```

### Backward Pass
```python
# Output layer
dz2 = y_pred - y_true
dW2 = (a1.T @ dz2) / batch_size
db2 = np.mean(dz2, axis=0, keepdims=True)

# Hidden layer
da1 = dz2 @ W2.T
dz1 = da1 * relu_derivative(z1)
dW1 = (X.T @ dz1) / batch_size
db1 = np.mean(dz1, axis=0, keepdims=True)
```

### Gradient Checking
```python
numerical_grad = (loss(θ + ε) - loss(θ - ε)) / (2ε)
relative_error = |numerical_grad - analytical_grad| / (|numerical| + |analytical|)
# Should be < 1e-5
```

---

## Resources

- [3Blue1Brown Neural Networks Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [CS231n Backprop Notes](http://cs231n.github.io/optimization-2/)
- [Understanding Deep Learning Book](https://udlbook.github.io/udlbook/)
