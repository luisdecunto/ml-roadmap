# Neural Networks from Scratch - Coding Guide

**Time:** 8-10 hours
**Difficulty:** Intermediate-Advanced
**Prerequisites:** Python, NumPy, calculus basics, backpropagation understanding

## What You'll Build

Implement a complete feedforward neural network from scratch:
1. Single neuron with activation functions
2. Activation functions (Sigmoid, ReLU, Tanh, Softmax)
3. 2-layer neural network (forward pass)
4. Backpropagation algorithm
5. Gradient checking
6. Training loop with mini-batch gradient descent
7. **Final Project:** Train on MNIST (>95% accuracy)

---

## Project Setup

```bash
mkdir neural-network-from-scratch
cd neural-network-from-scratch

# Create files
touch neuron.py
touch activations.py
touch network.py
touch train.py
touch test_network.py
touch requirements.txt
```

### requirements.txt
```
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0  # For dataset loading only
```

---

## Part 1: Single Neuron and Activations

### Theory

A **neuron** computes:
```
z = w·x + b    (linear combination)
a = σ(z)       (activation function)
```

Where:
- `w` = weights vector
- `x` = input vector
- `b` = bias (scalar)
- `σ` = activation function

**Common Activation Functions:**

1. **Sigmoid:** σ(z) = 1/(1 + e^(-z))
   - Range: (0, 1)
   - Used for: Binary classification output
   - Derivative: σ'(z) = σ(z)(1 - σ(z))

2. **ReLU** (Rectified Linear Unit): σ(z) = max(0, z)
   - Range: [0, ∞)
   - Used for: Hidden layers (most common)
   - Derivative: σ'(z) = 1 if z > 0, else 0

3. **Tanh:** σ(z) = (e^z - e^(-z))/(e^z + e^(-z))
   - Range: (-1, 1)
   - Used for: Hidden layers (zero-centered)
   - Derivative: σ'(z) = 1 - tanh²(z)

4. **Softmax:** σ(z)_i = e^(z_i) / Σ e^(z_j)
   - Range: (0, 1), outputs sum to 1
   - Used for: Multi-class classification output
   - Returns probability distribution

### Implementation

```python
# activations.py
import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function.

    Args:
        z: pre-activation (can be scalar, vector, or matrix)
    Returns:
        Activation in range (0, 1)
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """
    Derivative of sigmoid.

    Args:
        z: pre-activation
    Returns:
        Gradient dσ/dz
    """
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    """
    ReLU activation function.

    Args:
        z: pre-activation
    Returns:
        max(0, z)
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """
    Derivative of ReLU.

    Args:
        z: pre-activation
    Returns:
        1 if z > 0, else 0
    """
    return (z > 0).astype(float)


def tanh(z):
    """Tanh activation function."""
    return np.tanh(z)


def tanh_derivative(z):
    """Derivative of tanh."""
    return 1 - np.tanh(z) ** 2


def softmax(z):
    """
    Softmax activation for multi-class classification.

    Args:
        z: pre-activation (shape: [batch_size, num_classes])
    Returns:
        Probability distribution (each row sums to 1)
    """
    # Numerical stability: subtract max
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# neuron.py
import numpy as np
from activations import sigmoid, relu

class Neuron:
    """Single neuron with configurable activation."""

    def __init__(self, input_size, activation='sigmoid'):
        """
        Initialize neuron with random weights.

        Args:
            input_size: Number of inputs
            activation: 'sigmoid' or 'relu'
        """
        self.w = np.random.randn(input_size) * 0.1
        self.b = 0.0
        self.activation = activation

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input vector (shape: [input_size])
        Returns:
            Activation output (scalar)
        """
        z = np.dot(self.w, x) + self.b

        if self.activation == 'sigmoid':
            return sigmoid(z)
        elif self.activation == 'relu':
            return relu(z)
        else:
            return z  # Linear


# Test single neuron
if __name__ == "__main__":
    print("Testing Single Neuron")
    print("="*60)

    # Create neuron
    neuron = Neuron(input_size=3, activation='sigmoid')

    # Test input
    x = np.array([1.0, 2.0, 3.0])

    print(f"Weights: {neuron.w}")
    print(f"Bias: {neuron.b}")
    print(f"Input: {x}")
    print(f"Output: {neuron.forward(x):.4f}")
```

### Test It!

```bash
python neuron.py
```

Expected output: Activation value between 0 and 1 (sigmoid).

---

## Part 2: 2-Layer Neural Network (Forward Pass)

### Theory

**Network Architecture:**
```
Input (784) -> Hidden (128, ReLU) -> Output (10, Softmax)
```

**Forward Pass:**
```
z1 = X @ W1 + b1       # Linear transformation
a1 = ReLU(z1)          # Hidden activations

z2 = a1 @ W2 + b2      # Linear transformation
a2 = Softmax(z2)       # Output probabilities
```

**Shapes:**
- Input X: [batch_size, 784]
- W1: [784, 128], b1: [128]
- a1: [batch_size, 128]
- W2: [128, 10], b2: [10]
- a2: [batch_size, 10]

### Implementation

```python
# network.py
import numpy as np
from activations import relu, sigmoid, softmax

class NeuralNetwork:
    """2-layer feedforward neural network."""

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize network with random weights.

        Args:
            input_size: Input dimension (e.g., 784 for MNIST)
            hidden_size: Number of hidden units
            output_size: Number of classes (e.g., 10 for MNIST)
        """
        # He initialization for ReLU layers
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # Cache for backprop
        self.cache = {}

    def forward(self, X):
        """
        Forward pass through network.

        Args:
            X: Input batch [batch_size, input_size]
        Returns:
            Output probabilities [batch_size, output_size]
        """
        # Layer 1: Input -> Hidden (ReLU)
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)

        # Layer 2: Hidden -> Output (Softmax)
        z2 = a1 @ self.W2 + self.b2
        a2 = softmax(z2)

        # Save for backward pass
        self.cache = {
            'X': X,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'a2': a2
        }

        return a2

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: Input batch
        Returns:
            Predicted class indices
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)


# Test forward pass
if __name__ == "__main__":
    print("Testing 2-Layer Neural Network (Forward Pass)")
    print("="*60)

    # Create network
    net = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)

    # Create dummy input (batch of 5 examples)
    X = np.random.randn(5, 784)

    # Forward pass
    output = net.forward(X)

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output probabilities (first example):")
    print(output[0])
    print(f"Sum of probabilities: {np.sum(output[0]):.4f}")
    print(f"Predicted class: {np.argmax(output[0])}")
```

### Test It!

```bash
python network.py
```

Expected: Output shape [5, 10], probabilities sum to 1.0.

---

## Part 3: Backpropagation

### Theory

**Goal:** Compute gradients ∂L/∂W and ∂L/∂b for all layers.

**Loss Function** (Cross-Entropy):
```
L = -Σ y_true * log(y_pred)
```

**Backpropagation Equations:**

**Output Layer:**
```
dz2 = a2 - y_true              # Softmax + Cross-Entropy gradient
dW2 = (a1^T @ dz2) / batch_size
db2 = mean(dz2, axis=0)
```

**Hidden Layer:**
```
da1 = dz2 @ W2^T               # Gradient flowing back
dz1 = da1 * ReLU'(z1)          # Element-wise multiply with activation derivative
dW1 = (X^T @ dz1) / batch_size
db1 = mean(dz1, axis=0)
```

**Key Insight:** Backprop is just the chain rule applied layer by layer!

### Implementation

Add to `network.py`:

```python
def backward(self, y_true):
    """
    Backward pass (backpropagation).

    Args:
        y_true: True labels (one-hot encoded) [batch_size, output_size]
    Returns:
        Dictionary of gradients
    """
    batch_size = y_true.shape[0]

    # Retrieve cached values from forward pass
    X = self.cache['X']
    z1 = self.cache['z1']
    a1 = self.cache['a1']
    a2 = self.cache['a2']

    # Output layer gradients
    # For softmax + cross-entropy, gradient is simply: y_pred - y_true
    dz2 = a2 - y_true
    dW2 = (a1.T @ dz2) / batch_size
    db2 = np.mean(dz2, axis=0, keepdims=True)

    # Hidden layer gradients
    da1 = dz2 @ self.W2.T
    dz1 = da1 * (z1 > 0).astype(float)  # ReLU derivative
    dW1 = (X.T @ dz1) / batch_size
    db1 = np.mean(dz1, axis=0, keepdims=True)

    return {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }


def update_weights(self, gradients, learning_rate):
    """
    Update weights using gradient descent.

    Args:
        gradients: Dictionary of gradients from backward()
        learning_rate: Step size
    """
    self.W1 -= learning_rate * gradients['dW1']
    self.b1 -= learning_rate * gradients['db1']
    self.W2 -= learning_rate * gradients['dW2']
    self.b2 -= learning_rate * gradients['db2']


def compute_loss(self, y_true, y_pred):
    """
    Compute cross-entropy loss.

    Args:
        y_true: True labels (one-hot)
        y_pred: Predicted probabilities
    Returns:
        Average loss (scalar)
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # Cross-entropy loss
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss
```

---

## Part 4: Gradient Checking

### Theory

**Numerical Gradient:**
```
∂L/∂θ ≈ (L(θ + ε) - L(θ - ε)) / (2ε)
```

Where ε is a small value (e.g., 1e-5).

**Relative Error:**
```
error = |grad_numerical - grad_analytical| / (|grad_numerical| + |grad_analytical|)
```

If error < 1e-5, your backprop is correct!

### Implementation

```python
# test_network.py
import numpy as np
from network import NeuralNetwork

def gradient_check(net, X, y, epsilon=1e-5):
    """
    Verify backprop gradients with numerical gradients.

    Args:
        net: Neural network instance
        X: Input batch (small, e.g., 5 examples)
        y: True labels (one-hot)
        epsilon: Small value for finite differences
    Returns:
        Maximum relative error
    """
    # Get analytical gradients
    y_pred = net.forward(X)
    gradients = net.backward(y)

    # Check W1 gradient (check a few elements)
    print("Checking W1 gradients...")
    max_error = 0.0

    for i in range(min(5, net.W1.shape[0])):
        for j in range(min(5, net.W1.shape[1])):
            # Compute numerical gradient
            old_val = net.W1[i, j]

            net.W1[i, j] = old_val + epsilon
            y_plus = net.forward(X)
            loss_plus = net.compute_loss(y, y_plus)

            net.W1[i, j] = old_val - epsilon
            y_minus = net.forward(X)
            loss_minus = net.compute_loss(y, y_minus)

            net.W1[i, j] = old_val  # Restore

            grad_numerical = (loss_plus - loss_minus) / (2 * epsilon)
            grad_analytical = gradients['dW1'][i, j]

            # Relative error
            error = abs(grad_numerical - grad_analytical) / (abs(grad_numerical) + abs(grad_analytical) + 1e-8)
            max_error = max(max_error, error)

            if error > 1e-4:
                print(f"  W1[{i},{j}]: numerical={grad_numerical:.6f}, analytical={grad_analytical:.6f}, error={error:.6e}")

    print(f"Max relative error: {max_error:.6e}")

    if max_error < 1e-5:
        print("✓ Gradient check PASSED!")
    else:
        print("✗ Gradient check FAILED!")

    return max_error


if __name__ == "__main__":
    print("="*60)
    print("Gradient Checking")
    print("="*60)

    # Small network for testing
    net = NeuralNetwork(input_size=10, hidden_size=5, output_size=3)

    # Small batch
    X = np.random.randn(3, 10)
    y = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    gradient_check(net, X, y)
```

### Test It!

```bash
python test_network.py
```

Expected: "Gradient check PASSED!" with error < 1e-5.

---

## Part 5: Training on XOR Problem

### Theory

**XOR Problem:**
```
Input   Output
[0, 0]    0
[0, 1]    1
[1, 0]    1
[1, 1]    0
```

This is not linearly separable - requires hidden layer!

### Implementation

```python
# train_xor.py
import numpy as np
import matplotlib.pyplot as plt
from network import NeuralNetwork

# XOR dataset
X_train = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])

y_train = np.array([[1, 0],  # Class 0
                    [0, 1],  # Class 1
                    [0, 1],  # Class 1
                    [1, 0]]) # Class 0

# Create network
net = NeuralNetwork(input_size=2, hidden_size=4, output_size=2)

# Training loop
epochs = 5000
learning_rate = 0.5
losses = []

for epoch in range(epochs):
    # Forward pass
    y_pred = net.forward(X_train)

    # Compute loss
    loss = net.compute_loss(y_train, y_pred)
    losses.append(loss)

    # Backward pass
    grads = net.backward(y_train)

    # Update weights
    net.update_weights(grads, learning_rate)

    if (epoch + 1) % 1000 == 0:
        predictions = net.predict(X_train)
        true_labels = np.argmax(y_train, axis=1)
        accuracy = np.mean(predictions == true_labels)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.2%}")

# Final test
print("\n" + "="*60)
print("Final Predictions:")
print("="*60)
predictions = net.predict(X_train)
for i, x in enumerate(X_train):
    print(f"Input: {x} -> Predicted: {predictions[i]}, True: {np.argmax(y_train[i])}")

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss (XOR Problem)')
plt.grid(True, alpha=0.3)
plt.savefig('xor_loss.png')
print("\nSaved loss curve to 'xor_loss.png'")

# Plot decision boundary
def plot_decision_boundary(net):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = net.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=np.argmax(y_train, axis=1),
                s=200, cmap='RdYlBu', edgecolors='black', linewidths=2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('XOR Decision Boundary')
    plt.grid(True, alpha=0.3)
    plt.savefig('xor_boundary.png')
    print("Saved decision boundary to 'xor_boundary.png'")

plot_decision_boundary(net)
```

### Test It!

```bash
python train_xor.py
```

Expected: 100% accuracy, clear decision boundary separating XOR classes.

---

## Part 6: Training on MNIST

### Implementation

```python
# train_mnist.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from network import NeuralNetwork

print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist["data"], mnist["target"]

# Convert to numpy arrays
X = np.array(X, dtype=float)
y = np.array(y, dtype=int)

# Normalize
X = X / 255.0

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42
)

# One-hot encode labels
def one_hot(y, num_classes=10):
    n = y.shape[0]
    y_onehot = np.zeros((n, num_classes))
    y_onehot[np.arange(n), y] = 1
    return y_onehot

y_train_onehot = one_hot(y_train)
y_test_onehot = one_hot(y_test)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Create network
net = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)

# Training parameters
epochs = 50
batch_size = 128
learning_rate = 0.1

train_losses = []
test_accuracies = []

print("\nTraining...")
print("="*60)

for epoch in range(epochs):
    # Shuffle training data
    indices = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train_onehot[indices]

    # Mini-batch training
    epoch_loss = 0
    num_batches = X_train.shape[0] // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size

        X_batch = X_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]

        # Forward + backward + update
        y_pred = net.forward(X_batch)
        loss = net.compute_loss(y_batch, y_pred)
        grads = net.backward(y_batch)
        net.update_weights(grads, learning_rate)

        epoch_loss += loss

    # Average loss for epoch
    avg_loss = epoch_loss / num_batches
    train_losses.append(avg_loss)

    # Test accuracy
    test_pred = net.predict(X_test)
    test_acc = np.mean(test_pred == y_test)
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Test Acc: {test_acc:.4f}")

print("\n" + "="*60)
print(f"Final Test Accuracy: {test_accuracies[-1]:.4f}")
print("="*60)

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)

ax2.plot(test_accuracies)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Test Accuracy')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mnist_training.png')
print("\nSaved training curves to 'mnist_training.png'")

# Visualize predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, X_test.shape[0])
    image = X_test[idx].reshape(28, 28)
    pred = net.predict(X_test[idx:idx+1])[0]
    true = y_test[idx]

    ax.imshow(image, cmap='gray')
    color = 'green' if pred == true else 'red'
    ax.set_title(f'Pred: {pred}, True: {true}', color=color)
    ax.axis('off')

plt.tight_layout()
plt.savefig('mnist_predictions.png')
print("Saved predictions to 'mnist_predictions.png'")
```

### Test It!

```bash
python train_mnist.py
```

**Expected:**
- Training time: ~5-10 minutes
- Final accuracy: >95%
- Clear learning curve showing decreasing loss

---

## Quick Reference

### Complete Forward Pass
```python
# Layer 1: Input -> Hidden (ReLU)
z1 = X @ W1 + b1
a1 = relu(z1)

# Layer 2: Hidden -> Output (Softmax)
z2 = a1 @ W2 + b2
a2 = softmax(z2)
```

### Complete Backward Pass
```python
# Output layer
dz2 = a2 - y_true
dW2 = (a1.T @ dz2) / batch_size
db2 = np.mean(dz2, axis=0, keepdims=True)

# Hidden layer
da1 = dz2 @ W2.T
dz1 = da1 * (z1 > 0).astype(float)  # ReLU derivative
dW1 = (X.T @ dz1) / batch_size
db1 = np.mean(dz1, axis=0, keepdims=True)
```

### Weight Update
```python
W1 -= learning_rate * dW1
b1 -= learning_rate * db1
W2 -= learning_rate * dW2
b2 -= learning_rate * db2
```

---

## Common Issues & Debugging

### 1. Exploding/Vanishing Gradients
**Symptoms:** Loss becomes NaN or doesn't decrease
**Fix:**
- Use proper weight initialization (He for ReLU)
- Lower learning rate
- Use gradient clipping

### 2. Low Accuracy
**Symptoms:** Stuck at ~10% on MNIST
**Fix:**
- Check gradient computation (use gradient checking)
- Verify shapes match expected dimensions
- Ensure labels are one-hot encoded
- Try different learning rates (0.01, 0.1, 0.5)

### 3. Slow Training
**Symptoms:** Takes forever to converge
**Fix:**
- Increase batch size
- Increase learning rate (carefully)
- Add more hidden units
- Use mini-batch training

---

## Resources

**Video Tutorials:**
- [3Blue1Brown Neural Networks Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Andrej Karpathy: Backpropagation](https://www.youtube.com/watch?v=i94OvYb6noo)

**Reading:**
- [CS231n Backprop Notes](http://cs231n.github.io/optimization-2/)
- [Understanding Deep Learning Book - Ch 3](https://udlbook.github.io/udlbook/)
- [Neural Networks from Scratch in Python](https://nnfs.io/)

**Reference Implementations:**
- [Karpathy's micrograd](https://github.com/karpathy/micrograd)
- [Sentdex NNfSiX](https://github.com/Sentdex/NNfSiX)

---

## Next Steps

After completing this guide:
1. ✅ Complete [Neural Networks Exercises](exercises/neural_networks_exercises.html)
2. Try different architectures (3 layers, wider networks)
3. Implement momentum and Adam optimizer
4. Add regularization (L2, Dropout)
5. Move to CNNs (Module 7)

**Target:** >95% accuracy on MNIST with your from-scratch implementation!
