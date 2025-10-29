# Neural Networks Solutions - Module 6

Comprehensive solutions with step-by-step mathematical work and code implementations.

---

## Part 1: Single Neuron

### Exercise 1.1 Solution: Linear Combination

Inputs: x = [2, 3, -1]
Weights: w = [0.5, -0.3, 0.8]
Bias: b = 0.1

**Step 1: Compute weighted sum**
```
z = w·x + b = w₁x₁ + w₂x₂ + w₃x₃ + b
```

**Step 2: Calculate**
```
z = (0.5)(2) + (-0.3)(3) + (0.8)(-1) + 0.1
  = 1.0 - 0.9 - 0.8 + 0.1
  = -0.6
```

**Interpretation:** The pre-activation value is -0.6. This will be passed through an activation function to get the neuron's output.

**NumPy implementation:**
```python
import numpy as np

# Inputs, weights, bias
x = np.array([2, 3, -1])
w = np.array([0.5, -0.3, 0.8])
b = 0.1

# Weighted sum
z = np.dot(w, x) + b
print(f"Pre-activation: z = {z:.4f}")

# Alternative: using @ operator
z_alt = w @ x + b
print(f"Verification: z = {z_alt:.4f}")
```

---

### Exercise 1.2 Solution: Sigmoid Activation

Pre-activation: z = -0.6 (from Exercise 1.1)

**Sigmoid formula:** σ(z) = 1/(1 + e^(-z))

**Step 1: Calculate**
```
σ(-0.6) = 1/(1 + e^0.6)
        = 1/(1 + 1.8221)
        = 1/2.8221
        = 0.3543
```

**Step 2: Interpretation**
- Input range: (-∞, ∞)
- Output range: (0, 1)
- Output 0.3543 can be interpreted as a probability
- Values near 0.5 indicate uncertainty

**NumPy implementation:**
```python
def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

z = -0.6
output = sigmoid(z)
print(f"Sigmoid({z}) = {output:.4f}")

# Test with different values
test_values = [-5, -2, -0.6, 0, 0.6, 2, 5]
for z in test_values:
    print(f"σ({z:5.1f}) = {sigmoid(z):.4f}")

# Visualize
import matplotlib.pyplot as plt

z_range = np.linspace(-10, 10, 100)
y = sigmoid(z_range)

plt.figure(figsize=(8, 5))
plt.plot(z_range, y, 'b-', linewidth=2)
plt.axhline(0.5, color='r', linestyle='--', alpha=0.3)
plt.axvline(0, color='r', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('Sigmoid Activation Function')
plt.show()
```

---

### Exercise 1.3 Solution: Complete Neuron Calculation

Inputs: x = [1, 2], Weights: w = [0.4, -0.6], Bias: b = 0.5
Activation: ReLU(z) = max(0, z)

**Step 1: Weighted sum**
```
z = w·x + b
  = (0.4)(1) + (-0.6)(2) + 0.5
  = 0.4 - 1.2 + 0.5
  = -0.3
```

**Step 2: Apply ReLU**
```
ReLU(-0.3) = max(0, -0.3) = 0
```

**Step 3: If input was x = [3, 1]**
```
z = (0.4)(3) + (-0.6)(1) + 0.5
  = 1.2 - 0.6 + 0.5
  = 1.1

ReLU(1.1) = max(0, 1.1) = 1.1
```

**NumPy implementation:**
```python
def relu(z):
    """ReLU activation function"""
    return np.maximum(0, z)

def neuron_forward(x, w, b, activation='relu'):
    """Complete forward pass through a single neuron"""
    # Pre-activation
    z = np.dot(w, x) + b

    # Activation
    if activation == 'relu':
        a = relu(z)
    elif activation == 'sigmoid':
        a = sigmoid(z)
    elif activation == 'linear':
        a = z
    else:
        raise ValueError(f"Unknown activation: {activation}")

    return a, z  # Return both output and pre-activation

# Test case 1
x1 = np.array([1, 2])
w = np.array([0.4, -0.6])
b = 0.5

output1, z1 = neuron_forward(x1, w, b, 'relu')
print(f"Input {x1}: z = {z1:.4f}, output = {output1:.4f}")

# Test case 2
x2 = np.array([3, 1])
output2, z2 = neuron_forward(x2, w, b, 'relu')
print(f"Input {x2}: z = {z2:.4f}, output = {output2:.4f}")
```

---

## Part 2: Activation Functions

### Exercise 2.1 Solution: Sigmoid and Derivative

**Sigmoid:** σ(z) = 1/(1 + e^(-z))

**Derivative:** σ'(z) = σ(z)(1 - σ(z))

**Proof:**
```
Let σ(z) = 1/(1 + e^(-z))

Using quotient rule:
σ'(z) = d/dz[1/(1 + e^(-z))]
      = [0·(1+e^(-z)) - 1·(-e^(-z))] / (1+e^(-z))²
      = e^(-z) / (1+e^(-z))²

Factor out:
      = [1/(1+e^(-z))] · [e^(-z)/(1+e^(-z))]
      = σ(z) · [e^(-z)/(1+e^(-z))]

Note: e^(-z)/(1+e^(-z)) = 1 - 1/(1+e^(-z)) = 1 - σ(z)

Therefore: σ'(z) = σ(z)(1 - σ(z)) ✓
```

**NumPy implementation:**
```python
def sigmoid_derivative(z):
    """Derivative of sigmoid: σ'(z) = σ(z)(1 - σ(z))"""
    s = sigmoid(z)
    return s * (1 - s)

# Test at specific points
test_points = [-2, -1, 0, 1, 2]
print("z     | σ(z)   | σ'(z)")
print("------|--------|--------")
for z in test_points:
    s = sigmoid(z)
    ds = sigmoid_derivative(z)
    print(f"{z:5.1f} | {s:.4f} | {ds:.4f}")

# Maximum derivative is at z=0
z = 0
print(f"\nMaximum derivative at z=0: σ'(0) = {sigmoid_derivative(0):.4f}")

# Visualize sigmoid and its derivative
z_range = np.linspace(-6, 6, 200)
sig = sigmoid(z_range)
sig_deriv = sigmoid_derivative(z_range)

plt.figure(figsize=(10, 5))
plt.plot(z_range, sig, 'b-', linewidth=2, label='σ(z)')
plt.plot(z_range, sig_deriv, 'r--', linewidth=2, label="σ'(z)")
plt.grid(True, alpha=0.3)
plt.xlabel('z')
plt.ylabel('Value')
plt.title('Sigmoid and Its Derivative')
plt.legend()
plt.show()
```

---

### Exercise 2.2 Solution: ReLU and Derivative

**ReLU:** f(z) = max(0, z) = {z if z > 0; 0 if z ≤ 0}

**Derivative:** f'(z) = {1 if z > 0; 0 if z ≤ 0}

**At z = 0:** Technically undefined, but we typically set f'(0) = 0

**NumPy implementation:**
```python
def relu(z):
    """ReLU activation"""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU"""
    return (z > 0).astype(float)

# Test
test_values = [-2, -1, -0.1, 0, 0.1, 1, 2]
print("z     | ReLU(z) | ReLU'(z)")
print("------|---------|----------")
for z in test_values:
    f = relu(z)
    df = relu_derivative(z)
    print(f"{z:5.1f} | {f:7.1f} | {df:8.1f}")

# Visualize
z_range = np.linspace(-3, 3, 200)
relu_vals = relu(z_range)
relu_deriv_vals = relu_derivative(z_range)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(z_range, relu_vals, 'b-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('z')
plt.ylabel('ReLU(z)')
plt.title('ReLU Function')

plt.subplot(1, 2, 2)
plt.plot(z_range, relu_deriv_vals, 'r-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('z')
plt.ylabel("ReLU'(z)")
plt.title('ReLU Derivative')
plt.tight_layout()
plt.show()
```

---

### Exercise 2.3 Solution: Tanh and Derivative

**Tanh:** tanh(z) = (e^z - e^(-z))/(e^z + e^(-z))

**Derivative:** tanh'(z) = 1 - tanh²(z)

**Relation to sigmoid:** tanh(z) = 2σ(2z) - 1

**NumPy implementation:**
```python
def tanh(z):
    """Hyperbolic tangent"""
    return np.tanh(z)

def tanh_derivative(z):
    """Derivative of tanh: 1 - tanh²(z)"""
    t = np.tanh(z)
    return 1 - t**2

# Test
test_values = [-2, -1, 0, 1, 2]
print("z     | tanh(z) | tanh'(z)")
print("------|---------|----------")
for z in test_values:
    t = tanh(z)
    dt = tanh_derivative(z)
    print(f"{z:5.1f} | {t:7.4f} | {dt:8.4f}")

# Compare with sigmoid
print("\nComparison with sigmoid:")
for z in test_values:
    t = tanh(z)
    s = 2 * sigmoid(2*z) - 1
    print(f"z={z}: tanh(z)={t:.4f}, 2σ(2z)-1={s:.4f}")

# Visualize
z_range = np.linspace(-3, 3, 200)
tanh_vals = tanh(z_range)
tanh_deriv_vals = tanh_derivative(z_range)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(z_range, tanh_vals, 'b-', linewidth=2, label='tanh(z)')
plt.plot(z_range, sigmoid(z_range), 'r--', linewidth=2, label='σ(z)')
plt.grid(True, alpha=0.3)
plt.xlabel('z')
plt.ylabel('Value')
plt.title('Tanh vs Sigmoid')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(z_range, tanh_deriv_vals, 'b-', linewidth=2, label="tanh'(z)")
plt.plot(z_range, sigmoid_derivative(z_range), 'r--', linewidth=2, label="σ'(z)")
plt.grid(True, alpha=0.3)
plt.xlabel('z')
plt.ylabel('Derivative')
plt.title('Derivatives Comparison')
plt.legend()
plt.tight_layout()
plt.show()
```

---

### Exercise 2.4 Solution: Softmax Function

Logits: z = [2.0, 1.0, 0.1]

**Softmax formula:** softmax(zᵢ) = e^(zᵢ) / ∑ⱼ e^(zⱼ)

**Step 1: Compute exponentials**
```
e^z = [e^2.0, e^1.0, e^0.1]
    = [7.389, 2.718, 1.105]
```

**Step 2: Sum of exponentials**
```
∑ e^z = 7.389 + 2.718 + 1.105 = 11.212
```

**Step 3: Normalize**
```
softmax(z) = [7.389/11.212, 2.718/11.212, 1.105/11.212]
           = [0.659, 0.242, 0.099]
```

**Verification:** 0.659 + 0.242 + 0.099 = 1.000 ✓

**NumPy implementation:**
```python
def softmax(z):
    """Numerically stable softmax"""
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def softmax_derivative(z, i, j):
    """Jacobian of softmax: ∂yᵢ/∂zⱼ"""
    s = softmax(z)
    if i == j:
        return s[i] * (1 - s[i])
    else:
        return -s[i] * s[j]

# Test
z = np.array([2.0, 1.0, 0.1])
probs = softmax(z)

print("Logits:", z)
print("Softmax probabilities:", probs)
print("Sum:", np.sum(probs))

# Test with different scales
print("\nEffect of scaling:")
for scale in [0.1, 1.0, 10.0]:
    z_scaled = z * scale
    probs_scaled = softmax(z_scaled)
    print(f"Scale {scale}: {probs_scaled}")
    print(f"  Max prob: {np.max(probs_scaled):.4f}")

# Visualize softmax Jacobian
z_test = np.array([1.0, 0.5, 0.0])
s = softmax(z_test)
n = len(z_test)
jacobian = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        jacobian[i, j] = softmax_derivative(z_test, i, j)

print("\nSoftmax Jacobian:")
print(jacobian)
```

---

## Part 3: Forward Pass

### Exercise 3.1 Solution: 2-Layer Network Forward Pass

**Architecture:**
- Input: x = [1, 2]
- Hidden layer: 3 neurons, ReLU activation
- Output layer: 2 neurons, no activation (logits)

**Weights and biases:**
```
W1 (2×3): [[0.1, 0.2, -0.1],
           [0.3, -0.2, 0.4]]
b1 (3,): [0.1, 0.0, -0.1]

W2 (3×2): [[0.5, -0.3],
           [0.2, 0.4],
           [-0.1, 0.2]]
b2 (2,): [0.1, -0.1]
```

**Step 1: Hidden layer**
```
z1 = W1ᵀx + b1

z1[0] = 0.1×1 + 0.3×2 + 0.1 = 0.1 + 0.6 + 0.1 = 0.8
z1[1] = 0.2×1 + (-0.2)×2 + 0.0 = 0.2 - 0.4 + 0.0 = -0.2
z1[2] = (-0.1)×1 + 0.4×2 + (-0.1) = -0.1 + 0.8 - 0.1 = 0.6

a1 = ReLU(z1) = [0.8, 0, 0.6]
```

**Step 2: Output layer**
```
z2 = W2ᵀa1 + b2

z2[0] = 0.5×0.8 + 0.2×0 + (-0.1)×0.6 + 0.1
      = 0.4 + 0 - 0.06 + 0.1 = 0.44

z2[1] = (-0.3)×0.8 + 0.4×0 + 0.2×0.6 + (-0.1)
      = -0.24 + 0 + 0.12 - 0.1 = -0.22
```

**Output:** z2 = [0.44, -0.22]

**NumPy implementation:**
```python
class TwoLayerNetwork:
    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1  # Shape: (input_dim, hidden_dim)
        self.b1 = b1  # Shape: (hidden_dim,)
        self.W2 = W2  # Shape: (hidden_dim, output_dim)
        self.b2 = b2  # Shape: (output_dim,)

    def forward(self, x):
        """Forward pass through the network"""
        # Hidden layer
        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)

        # Output layer
        z2 = a1 @ self.W2 + self.b2

        # Store for backprop
        self.cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2}

        return z2

# Initialize network
W1 = np.array([[0.1, 0.2, -0.1],
               [0.3, -0.2, 0.4]])
b1 = np.array([0.1, 0.0, -0.1])

W2 = np.array([[0.5, -0.3],
               [0.2, 0.4],
               [-0.1, 0.2]])
b2 = np.array([0.1, -0.1])

net = TwoLayerNetwork(W1, b1, W2, b2)

# Forward pass
x = np.array([1, 2])
output = net.forward(x)

print("Input:", x)
print("Hidden activations:", net.cache['a1'])
print("Output:", output)
```

---

### Exercise 3.2 Solution: Batch Forward Pass

Same network, batch of 3 samples:
```
X = [[1, 2],
     [0, 1],
     [-1, 1]]
```

**Matrix form:**
```
Z1 = XW1 + b1  (broadcasting)
A1 = ReLU(Z1)
Z2 = A1W2 + b2
```

**NumPy implementation:**
```python
def batch_forward(net, X):
    """Forward pass for a batch of inputs"""
    # X shape: (batch_size, input_dim)

    # Hidden layer
    Z1 = X @ net.W1 + net.b1  # Broadcasting
    A1 = relu(Z1)

    # Output layer
    Z2 = A1 @ net.W2 + net.b2

    return Z2, {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2}

# Batch of inputs
X_batch = np.array([[1, 2],
                    [0, 1],
                    [-1, 1]])

outputs, cache = batch_forward(net, X_batch)

print("Batch inputs shape:", X_batch.shape)
print("Hidden activations shape:", cache['A1'].shape)
print("Outputs shape:", outputs.shape)
print("\nOutputs:")
print(outputs)

# Process each sample individually to verify
print("\nVerification (individual processing):")
for i, x in enumerate(X_batch):
    output_single = net.forward(x)
    print(f"Sample {i}: {output_single}")
```

---

### Exercise 3.3 Solution: Network with Multiple Layers

**3-layer network:**
- Input: 2 → Hidden1: 4 (ReLU) → Hidden2: 3 (ReLU) → Output: 2 (softmax)

**NumPy implementation:**
```python
class MultiLayerNetwork:
    def __init__(self, layer_sizes, activations):
        """
        layer_sizes: list of layer dimensions [input, hidden1, hidden2, ..., output]
        activations: list of activation functions for each layer
        """
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.weights = []
        self.biases = []

        # Initialize weights and biases (Xavier initialization)
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """Forward pass through all layers"""
        self.cache = {'A0': X}  # Input is activation of layer 0
        A = X

        for i, (W, b, activation) in enumerate(zip(self.weights, self.biases, self.activations)):
            Z = A @ W + b
            self.cache[f'Z{i+1}'] = Z

            if activation == 'relu':
                A = relu(Z)
            elif activation == 'sigmoid':
                A = sigmoid(Z)
            elif activation == 'tanh':
                A = tanh(Z)
            elif activation == 'softmax':
                A = softmax(Z) if Z.ndim == 1 else np.array([softmax(z) for z in Z])
            elif activation == 'linear':
                A = Z
            else:
                raise ValueError(f"Unknown activation: {activation}")

            self.cache[f'A{i+1}'] = A

        return A

# Create 3-layer network
layer_sizes = [2, 4, 3, 2]
activations = ['relu', 'relu', 'softmax']
net_deep = MultiLayerNetwork(layer_sizes, activations)

# Forward pass
x = np.array([1.0, 2.0])
output = net_deep.forward(x)

print("Network architecture:", layer_sizes)
print("Activations:", activations)
print("\nForward pass:")
for i in range(len(layer_sizes)):
    if i == 0:
        print(f"Layer {i} (input): {net_deep.cache['A0']}")
    else:
        print(f"Layer {i} (hidden): {net_deep.cache[f'A{i}']}")
print(f"Output (probabilities): {output}")
print(f"Sum: {np.sum(output):.6f}")
```

---

## Part 4: Backpropagation

### Exercise 4.1 Solution: Output Layer Gradient

**Setup:**
- Output: ŷ = [0.3, 0.7] (softmax probabilities)
- True label: y = [0, 1] (one-hot)
- Loss: Cross-entropy

**Step 1: Loss function**
```
L = -∑ yᵢ·log(ŷᵢ) = -log(0.7) ≈ 0.357
```

**Step 2: Gradient ∂L/∂z (before softmax)**

For softmax + cross-entropy, the gradient simplifies to:
```
∂L/∂z = ŷ - y = [0.3, 0.7] - [0, 1] = [0.3, -0.3]
```

**Derivation:**
```
L = -∑ yₖ·log(softmax(z)ₖ)

∂L/∂zᵢ = -∑ₖ yₖ · ∂log(softmax(z)ₖ)/∂zᵢ

For softmax + cross-entropy:
∂L/∂zᵢ = softmax(z)ᵢ - yᵢ = ŷᵢ - yᵢ
```

**NumPy implementation:**
```python
def softmax_cross_entropy_backward(y_pred, y_true):
    """
    Gradient of softmax cross-entropy loss
    Returns: ∂L/∂z = ŷ - y
    """
    return y_pred - y_true

# Example
y_pred = np.array([0.3, 0.7])
y_true = np.array([0, 1])

# Loss
loss = -np.sum(y_true * np.log(y_pred + 1e-10))
print(f"Cross-entropy loss: {loss:.4f}")

# Gradient
grad = softmax_cross_entropy_backward(y_pred, y_true)
print(f"Gradient ∂L/∂z: {grad}")
```

---

### Exercise 4.2 Solution: Backprop Through Hidden Layer

**Setup from Exercise 3.1:**
- Hidden activations: a1 = [0.8, 0, 0.6]
- Output gradient: ∂L/∂z2 = [0.3, -0.3]
- W2 = [[0.5, -0.3], [0.2, 0.4], [-0.1, 0.2]]

**Step 1: Gradient w.r.t. W2**
```
∂L/∂W2 = a1ᵀ · ∂L/∂z2

∂L/∂W2 = [0.8]   · [0.3, -0.3]
         [0  ]
         [0.6]

       = [[0.24, -0.24],
          [0,     0   ],
          [0.18, -0.18]]
```

**Step 2: Gradient w.r.t. b2**
```
∂L/∂b2 = ∂L/∂z2 = [0.3, -0.3]
```

**Step 3: Gradient w.r.t. a1 (backprop to hidden)**
```
∂L/∂a1 = ∂L/∂z2 · W2ᵀ

       = [0.3, -0.3] · [[0.5,  0.2, -0.1],
                         [-0.3, 0.4,  0.2]]

       = [0.3×0.5 + (-0.3)×(-0.3),
          0.3×0.2 + (-0.3)×0.4,
          0.3×(-0.1) + (-0.3)×0.2]

       = [0.15 + 0.09, 0.06 - 0.12, -0.03 - 0.06]
       = [0.24, -0.06, -0.09]
```

**Step 4: Gradient w.r.t. z1 (through ReLU)**
```
∂L/∂z1 = ∂L/∂a1 ⊙ ReLU'(z1)

z1 was [0.8, -0.2, 0.6], so ReLU'(z1) = [1, 0, 1]

∂L/∂z1 = [0.24, -0.06, -0.09] ⊙ [1, 0, 1]
       = [0.24, 0, -0.09]

(Gradient is blocked where ReLU was inactive!)
```

**NumPy implementation:**
```python
def backprop_layer(dL_dz_next, a_prev, W, z, activation='relu'):
    """
    Backprop through one layer

    Args:
        dL_dz_next: gradient from next layer (∂L/∂z)
        a_prev: activations from previous layer
        W: weights of current layer
        z: pre-activation values
        activation: activation function type

    Returns:
        dL_dW: gradient w.r.t. weights
        dL_db: gradient w.r.t. biases
        dL_da_prev: gradient to pass to previous layer
    """
    batch_size = a_prev.shape[0] if a_prev.ndim > 1 else 1

    # Gradients for weights and biases
    if a_prev.ndim == 1:
        dL_dW = np.outer(a_prev, dL_dz_next)
        dL_db = dL_dz_next
    else:
        dL_dW = a_prev.T @ dL_dz_next / batch_size
        dL_db = np.mean(dL_dz_next, axis=0)

    # Gradient to previous layer
    dL_da_prev = dL_dz_next @ W.T

    return dL_dW, dL_db, dL_da_prev

# From Exercise 3.1
a1 = np.array([0.8, 0, 0.6])
dL_dz2 = np.array([0.3, -0.3])
W2 = np.array([[0.5, -0.3],
               [0.2, 0.4],
               [-0.1, 0.2]])

# Backprop through output layer
dL_dW2, dL_db2, dL_da1 = backprop_layer(dL_dz2, a1, W2, None)

print("∂L/∂W2:")
print(dL_dW2)
print("\n∂L/∂b2:", dL_db2)
print("\n∂L/∂a1:", dL_da1)

# Backprop through ReLU
z1 = np.array([0.8, -0.2, 0.6])
dL_dz1 = dL_da1 * relu_derivative(z1)
print("\n∂L/∂z1:", dL_dz1)
```

---

### Exercise 4.3 Solution: Complete Backpropagation

Full backprop through the 2-layer network from Exercise 3.1

**NumPy implementation:**
```python
class TwoLayerNetworkWithBackprop:
    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    def forward(self, x):
        """Forward pass"""
        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2

        self.cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2}
        return z2

    def backward(self, dL_dz2):
        """
        Backward pass

        Args:
            dL_dz2: gradient of loss w.r.t. output logits

        Returns:
            Dictionary of gradients
        """
        x = self.cache['x']
        z1 = self.cache['z1']
        a1 = self.cache['a1']

        # Output layer gradients
        dL_dW2 = np.outer(a1, dL_dz2)
        dL_db2 = dL_dz2
        dL_da1 = dL_dz2 @ self.W2.T

        # Hidden layer gradients (through ReLU)
        dL_dz1 = dL_da1 * relu_derivative(z1)
        dL_dW1 = np.outer(x, dL_dz1)
        dL_db1 = dL_dz1

        return {
            'dW1': dL_dW1,
            'db1': dL_db1,
            'dW2': dL_dW2,
            'db2': dL_db2
        }

# Test
net_bp = TwoLayerNetworkWithBackprop(W1, b1, W2, b2)

# Forward pass
x = np.array([1, 2])
z2 = net_bp.forward(x)
y_pred = softmax(z2)

# True label
y_true = np.array([1, 0])

# Loss
loss = -np.sum(y_true * np.log(y_pred + 1e-10))

# Backward pass
dL_dz2 = y_pred - y_true
grads = net_bp.backward(dL_dz2)

print("Forward pass output:", y_pred)
print("Loss:", loss)
print("\nGradients:")
for key, value in grads.items():
    print(f"{key}:")
    print(value)
```

---

### Exercise 4.4 Solution: Gradient Checking

Verify backprop implementation using numerical gradients

**Numerical gradient:** (f(θ + ε) - f(θ - ε)) / (2ε)

**NumPy implementation:**
```python
def numerical_gradient(net, x, y_true, param_name, epsilon=1e-5):
    """
    Compute numerical gradient for a parameter

    Args:
        net: network object
        x: input
        y_true: true label
        param_name: 'W1', 'b1', 'W2', or 'b2'
        epsilon: small value for finite difference

    Returns:
        Numerical gradient (same shape as parameter)
    """
    def compute_loss():
        z2 = net.forward(x)
        y_pred = softmax(z2)
        return -np.sum(y_true * np.log(y_pred + 1e-10))

    param = getattr(net, param_name)
    grad_numerical = np.zeros_like(param)

    # Iterate over all elements
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_value = param[idx]

        # f(θ + ε)
        param[idx] = old_value + epsilon
        loss_plus = compute_loss()

        # f(θ - ε)
        param[idx] = old_value - epsilon
        loss_minus = compute_loss()

        # Numerical gradient
        grad_numerical[idx] = (loss_plus - loss_minus) / (2 * epsilon)

        # Restore
        param[idx] = old_value
        it.iternext()

    return grad_numerical

def gradient_check(net, x, y_true, grads_analytical, tolerance=1e-5):
    """
    Check analytical gradients against numerical gradients

    Returns:
        Dictionary with relative errors for each parameter
    """
    errors = {}

    for param_name in ['W1', 'b1', 'W2', 'b2']:
        grad_num = numerical_gradient(net, x, y_true, param_name)
        grad_ana = grads_analytical[f'd{param_name}']

        # Compute relative error
        numerator = np.linalg.norm(grad_num - grad_ana)
        denominator = np.linalg.norm(grad_num) + np.linalg.norm(grad_ana)
        relative_error = numerator / (denominator + 1e-10)

        errors[param_name] = relative_error

        print(f"\n{param_name}:")
        print(f"  Numerical gradient:\n{grad_num}")
        print(f"  Analytical gradient:\n{grad_ana}")
        print(f"  Relative error: {relative_error:.2e}")

        if relative_error < tolerance:
            print(f"  ✓ PASS (< {tolerance})")
        else:
            print(f"  ✗ FAIL (>= {tolerance})")

    return errors

# Perform gradient checking
errors = gradient_check(net_bp, x, y_true, grads, tolerance=1e-5)
```

---

## Part 5: Training

### Exercise 5.1 Solution: SGD Update Step

**Setup:**
- Current W = [[0.5, -0.3], [0.2, 0.4]]
- Gradient ∂L/∂W = [[0.1, -0.05], [0.08, 0.12]]
- Learning rate α = 0.1

**SGD update:** W_new = W_old - α·∂L/∂W

**Step 1: Calculate update**
```
W_new = [[0.5, -0.3],    - 0.1 × [[0.1, -0.05],
         [0.2,  0.4]]              [0.08, 0.12]]

      = [[0.5, -0.3],    - [[0.01, -0.005],
         [0.2,  0.4]]      [0.008, 0.012]]

      = [[0.49, -0.295],
         [0.192, 0.388]]
```

**NumPy implementation:**
```python
def sgd_update(params, grads, learning_rate):
    """
    SGD parameter update

    Args:
        params: dictionary of parameters
        grads: dictionary of gradients
        learning_rate: step size

    Returns:
        Updated parameters
    """
    updated_params = {}
    for key in params:
        updated_params[key] = params[key] - learning_rate * grads[f'd{key}']
    return updated_params

# Test
W = np.array([[0.5, -0.3],
              [0.2, 0.4]])
dW = np.array([[0.1, -0.05],
               [0.08, 0.12]])

learning_rate = 0.1
W_new = W - learning_rate * dW

print("Original W:")
print(W)
print("\nGradient ∂L/∂W:")
print(dW)
print("\nUpdated W:")
print(W_new)
print("\nChange in W:")
print(W_new - W)
```

---

### Exercise 5.2 Solution: Mini-batch Training Loop

**NumPy implementation:**
```python
def train_network(net, X_train, y_train, epochs=100, batch_size=32, learning_rate=0.01):
    """
    Train network using mini-batch SGD

    Args:
        net: network object
        X_train: training inputs (n_samples, input_dim)
        y_train: training labels (n_samples, output_dim)
        epochs: number of training epochs
        batch_size: size of mini-batches
        learning_rate: step size

    Returns:
        losses: list of average losses per epoch
    """
    n_samples = X_train.shape[0]
    losses = []

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_losses = []

        # Mini-batch loop
        for i in range(0, n_samples, batch_size):
            # Get mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Forward pass
            batch_grads = {'dW1': 0, 'db1': 0, 'dW2': 0, 'db2': 0}
            batch_loss = 0

            for x, y_true in zip(X_batch, y_batch):
                # Forward
                z2 = net.forward(x)
                y_pred = softmax(z2)

                # Loss
                loss = -np.sum(y_true * np.log(y_pred + 1e-10))
                batch_loss += loss

                # Backward
                dL_dz2 = y_pred - y_true
                grads = net.backward(dL_dz2)

                # Accumulate gradients
                for key in batch_grads:
                    batch_grads[key] += grads[key]

            # Average gradients over batch
            for key in batch_grads:
                batch_grads[key] /= len(X_batch)

            # Update parameters
            net.W1 -= learning_rate * batch_grads['dW1']
            net.b1 -= learning_rate * batch_grads['db1']
            net.W2 -= learning_rate * batch_grads['dW2']
            net.b2 -= learning_rate * batch_grads['db2']

            epoch_losses.append(batch_loss / len(X_batch))

        # Average loss for epoch
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return losses

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000
X_train = np.random.randn(n_samples, 2)
# Binary classification: class based on x1 + x2
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
# One-hot encode
y_train_onehot = np.eye(2)[y_train]

# Train
net_train = TwoLayerNetworkWithBackprop(W1, b1, W2, b2)
losses = train_network(net_train, X_train, y_train_onehot,
                      epochs=50, batch_size=32, learning_rate=0.1)

# Plot learning curve
plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True, alpha=0.3)
plt.show()
```

---

### Exercise 5.3 Solution: Evaluation and Accuracy

**NumPy implementation:**
```python
def evaluate_network(net, X_test, y_test):
    """
    Evaluate network on test set

    Args:
        net: trained network
        X_test: test inputs
        y_test: test labels (one-hot or integer)

    Returns:
        accuracy: classification accuracy
        predictions: predicted class labels
    """
    predictions = []
    correct = 0

    for x, y_true in zip(X_test, y_test):
        # Forward pass
        z2 = net.forward(x)
        y_pred = softmax(z2)

        # Predicted class
        pred_class = np.argmax(y_pred)
        predictions.append(pred_class)

        # True class
        if y_true.ndim > 0 and len(y_true) > 1:
            true_class = np.argmax(y_true)
        else:
            true_class = int(y_true)

        if pred_class == true_class:
            correct += 1

    accuracy = correct / len(X_test)
    return accuracy, np.array(predictions)

# Generate test set
n_test = 200
X_test = np.random.randn(n_test, 2)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
y_test_onehot = np.eye(2)[y_test]

# Evaluate
accuracy, predictions = evaluate_network(net_train, X_test, y_test_onehot)
print(f"\nTest Accuracy: {accuracy:.2%}")

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)

# Visualize decision boundary
def plot_decision_boundary(net, X, y):
    """Plot decision boundary of trained network"""
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = []
    for x1, x2 in zip(xx.ravel(), yy.ravel()):
        z2 = net.forward(np.array([x1, x2]))
        pred = np.argmax(softmax(z2))
        Z.append(pred)
    Z = np.array(Z).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'Decision Boundary (Accuracy: {accuracy:.2%})')
    plt.colorbar()
    plt.show()

plot_decision_boundary(net_train, X_test, y_test)
```

---

## Part 6: Debugging NNs

### Exercise 6.1 Solution: Vanishing Gradients

**Problem:** Gradients become very small in deep networks with sigmoid/tanh

**Analysis:**
```python
def analyze_vanishing_gradients(n_layers=10):
    """
    Simulate gradient flow through deep network with sigmoid

    Returns gradient magnitude at each layer
    """
    # Initialize random weights
    weights = [np.random.randn(10, 10) * 0.5 for _ in range(n_layers)]

    # Forward pass with sigmoid
    x = np.random.randn(10)
    activations = [x]
    z_values = []

    for W in weights:
        z = activations[-1] @ W
        a = sigmoid(z)
        z_values.append(z)
        activations.append(a)

    # Backward pass
    grad = np.ones(10)  # Initial gradient
    grad_norms = [np.linalg.norm(grad)]

    for i in range(n_layers - 1, -1, -1):
        # Gradient through sigmoid
        grad = grad * sigmoid_derivative(z_values[i])
        # Gradient through weights
        grad = grad @ weights[i].T
        grad_norms.append(np.linalg.norm(grad))

    return list(reversed(grad_norms))

# Test
grad_norms_sigmoid = analyze_vanishing_gradients(n_layers=10)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(grad_norms_sigmoid, 'o-')
plt.xlabel('Layer (0 = output)')
plt.ylabel('Gradient Norm')
plt.title('Vanishing Gradients with Sigmoid')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# Compare with ReLU
def analyze_relu_gradients(n_layers=10):
    """Same but with ReLU"""
    weights = [np.random.randn(10, 10) * 0.01 for _ in range(n_layers)]

    x = np.random.randn(10)
    activations = [x]
    z_values = []

    for W in weights:
        z = activations[-1] @ W
        a = relu(z)
        z_values.append(z)
        activations.append(a)

    grad = np.ones(10)
    grad_norms = [np.linalg.norm(grad)]

    for i in range(n_layers - 1, -1, -1):
        grad = grad * relu_derivative(z_values[i])
        grad = grad @ weights[i].T
        grad_norms.append(np.linalg.norm(grad))

    return list(reversed(grad_norms))

grad_norms_relu = analyze_relu_gradients(n_layers=10)

plt.subplot(1, 2, 2)
plt.plot(grad_norms_relu, 'o-')
plt.xlabel('Layer (0 = output)')
plt.ylabel('Gradient Norm')
plt.title('Gradient Flow with ReLU')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Gradient norm at first layer:")
print(f"  Sigmoid: {grad_norms_sigmoid[0]:.2e}")
print(f"  ReLU: {grad_norms_relu[0]:.2e}")
```

**Solutions:**
1. Use ReLU instead of sigmoid/tanh
2. Use batch normalization
3. Use residual connections
4. Use better initialization (Xavier/He)

---

### Exercise 6.2 Solution: Weight Initialization

**Compare different initialization strategies:**

**NumPy implementation:**
```python
def test_initialization(init_type, n_layers=5, n_neurons=100):
    """
    Test different initialization strategies

    Args:
        init_type: 'zeros', 'small', 'large', 'xavier', 'he'
    """
    activations = []
    x = np.random.randn(1000, n_neurons)  # Batch of inputs

    for layer in range(n_layers):
        # Initialize weights based on strategy
        if init_type == 'zeros':
            W = np.zeros((n_neurons, n_neurons))
        elif init_type == 'small':
            W = np.random.randn(n_neurons, n_neurons) * 0.01
        elif init_type == 'large':
            W = np.random.randn(n_neurons, n_neurons) * 1.0
        elif init_type == 'xavier':
            # Xavier: std = sqrt(2 / (fan_in + fan_out))
            W = np.random.randn(n_neurons, n_neurons) * np.sqrt(2.0 / (2 * n_neurons))
        elif init_type == 'he':
            # He: std = sqrt(2 / fan_in)
            W = np.random.randn(n_neurons, n_neurons) * np.sqrt(2.0 / n_neurons)
        else:
            raise ValueError(f"Unknown init type: {init_type}")

        # Forward pass
        z = x @ W
        x = np.tanh(z)  # Use tanh for Xavier
        # x = relu(z)  # Use ReLU for He

        activations.append(x)

    return activations

# Test different initializations
init_types = ['zeros', 'small', 'large', 'xavier', 'he']
fig, axes = plt.subplots(1, len(init_types), figsize=(15, 3))

for idx, init_type in enumerate(init_types):
    activations = test_initialization(init_type)

    # Plot activation distributions
    axes[idx].set_title(f'{init_type.capitalize()} Init')
    for i, act in enumerate(activations):
        axes[idx].hist(act.flatten(), bins=50, alpha=0.5, label=f'Layer {i+1}')
    axes[idx].set_xlabel('Activation Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].set_xlim([-2, 2])

plt.tight_layout()
plt.show()

# Print statistics
for init_type in init_types:
    activations = test_initialization(init_type)
    print(f"\n{init_type.upper()} initialization:")
    for i, act in enumerate(activations):
        print(f"  Layer {i+1}: mean={np.mean(act):.4f}, std={np.std(act):.4f}")
```

**Key insights:**
- **Zeros:** All neurons compute same thing (symmetry problem)
- **Small random:** Activations vanish in deep networks
- **Large random:** Activations saturate
- **Xavier:** Good for tanh/sigmoid (maintains variance)
- **He:** Good for ReLU (accounts for killed neurons)

---

### Exercise 6.3 Solution: Batch Normalization

**Batch normalization: normalize activations across mini-batch**

**NumPy implementation:**
```python
class BatchNorm:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        """
        Batch Normalization layer

        Args:
            num_features: number of features (neurons)
            epsilon: small constant for numerical stability
            momentum: momentum for running statistics
        """
        self.epsilon = epsilon
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones(num_features)  # Scale
        self.beta = np.zeros(num_features)  # Shift

        # Running statistics (for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x, training=True):
        """
        Forward pass

        Args:
            x: input (batch_size, num_features)
            training: whether in training mode

        Returns:
            Normalized and scaled output
        """
        if training:
            # Batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)

            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)

            # Update running statistics
            self.running_mean = (self.momentum * self.running_mean +
                                (1 - self.momentum) * batch_mean)
            self.running_var = (self.momentum * self.running_var +
                               (1 - self.momentum) * batch_var)

            # Cache for backward pass
            self.cache = {
                'x': x,
                'x_norm': x_norm,
                'batch_mean': batch_mean,
                'batch_var': batch_var
            }
        else:
            # Use running statistics
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        # Scale and shift
        out = self.gamma * x_norm + self.beta

        return out

    def backward(self, dout):
        """
        Backward pass

        Args:
            dout: gradient from next layer

        Returns:
            dx: gradient to pass to previous layer
            dgamma: gradient for gamma
            dbeta: gradient for beta
        """
        x = self.cache['x']
        x_norm = self.cache['x_norm']
        batch_mean = self.cache['batch_mean']
        batch_var = self.cache['batch_var']

        batch_size = x.shape[0]

        # Gradients for gamma and beta
        dgamma = np.sum(dout * x_norm, axis=0)
        dbeta = np.sum(dout, axis=0)

        # Gradient for x
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - batch_mean) *
                     -0.5 * (batch_var + self.epsilon)**(-1.5), axis=0)
        dmean = (np.sum(dx_norm * -1.0 / np.sqrt(batch_var + self.epsilon), axis=0) +
                dvar * np.sum(-2.0 * (x - batch_mean), axis=0) / batch_size)

        dx = (dx_norm / np.sqrt(batch_var + self.epsilon) +
              dvar * 2.0 * (x - batch_mean) / batch_size +
              dmean / batch_size)

        return dx, dgamma, dbeta

# Test
bn = BatchNorm(num_features=3)

# Training mode
x_train = np.random.randn(4, 3) * 5 + 10  # Mean ~10, std ~5
print("Input (training):")
print(x_train)
print(f"  Mean: {np.mean(x_train, axis=0)}")
print(f"  Std: {np.std(x_train, axis=0)}")

out_train = bn.forward(x_train, training=True)
print("\nOutput (after BN):")
print(out_train)
print(f"  Mean: {np.mean(out_train, axis=0)}")
print(f"  Std: {np.std(out_train, axis=0)}")

# Test backward pass
dout = np.random.randn(4, 3)
dx, dgamma, dbeta = bn.backward(dout)
print("\nGradients:")
print(f"  dx shape: {dx.shape}")
print(f"  dgamma: {dgamma}")
print(f"  dbeta: {dbeta}")
```

---

## Challenge Problems

### Challenge 1 Solution: Implement Dropout

**Dropout:** Randomly set activations to 0 during training

**NumPy implementation:**
```python
class Dropout:
    def __init__(self, drop_prob=0.5):
        """
        Dropout layer

        Args:
            drop_prob: probability of dropping a neuron
        """
        self.drop_prob = drop_prob

    def forward(self, x, training=True):
        """
        Forward pass

        Args:
            x: input activations
            training: whether in training mode

        Returns:
            Output with dropout applied (if training)
        """
        if training:
            # Create dropout mask
            self.mask = (np.random.rand(*x.shape) > self.drop_prob).astype(float)

            # Apply mask and scale (inverted dropout)
            out = x * self.mask / (1 - self.drop_prob)

            return out
        else:
            # No dropout during inference
            return x

    def backward(self, dout):
        """
        Backward pass

        Args:
            dout: gradient from next layer

        Returns:
            Gradient with same dropout mask applied
        """
        # Apply same mask
        dx = dout * self.mask / (1 - self.drop_prob)
        return dx

# Test dropout
dropout = Dropout(drop_prob=0.5)

# Training mode
x = np.random.randn(5, 4)
print("Input:")
print(x)

out_train = dropout.forward(x, training=True)
print("\nWith dropout (training):")
print(out_train)
print(f"Fraction kept: {np.sum(out_train != 0) / out_train.size:.2%}")

# Inference mode
out_test = dropout.forward(x, training=False)
print("\nWithout dropout (inference):")
print(out_test)
print("Same as input?", np.allclose(x, out_test))

# Test backward
dout = np.random.randn(5, 4)
dx = dropout.backward(dout)
print("\nBackward pass:")
print(f"Gradient shape: {dx.shape}")
print(f"Non-zero gradients: {np.sum(dx != 0)}")
```

---

### Challenge 2 Solution: Learning Rate Scheduling

**Implement various learning rate schedules**

**NumPy implementation:**
```python
class LearningRateScheduler:
    def __init__(self, initial_lr, schedule_type='constant', **kwargs):
        """
        Learning rate scheduler

        Args:
            initial_lr: starting learning rate
            schedule_type: 'constant', 'step', 'exponential', 'cosine'
            kwargs: additional parameters for specific schedules
        """
        self.initial_lr = initial_lr
        self.schedule_type = schedule_type
        self.kwargs = kwargs
        self.step_count = 0

    def get_lr(self):
        """Get current learning rate"""
        if self.schedule_type == 'constant':
            return self.initial_lr

        elif self.schedule_type == 'step':
            # Decay by factor every N steps
            decay_rate = self.kwargs.get('decay_rate', 0.5)
            decay_steps = self.kwargs.get('decay_steps', 100)
            return self.initial_lr * (decay_rate ** (self.step_count // decay_steps))

        elif self.schedule_type == 'exponential':
            # Exponential decay
            decay_rate = self.kwargs.get('decay_rate', 0.95)
            return self.initial_lr * (decay_rate ** self.step_count)

        elif self.schedule_type == 'cosine':
            # Cosine annealing
            T_max = self.kwargs.get('T_max', 1000)
            eta_min = self.kwargs.get('eta_min', 0)
            return eta_min + (self.initial_lr - eta_min) * (
                1 + np.cos(np.pi * self.step_count / T_max)) / 2

        elif self.schedule_type == 'warmup':
            # Linear warmup then decay
            warmup_steps = self.kwargs.get('warmup_steps', 100)
            if self.step_count < warmup_steps:
                return self.initial_lr * (self.step_count / warmup_steps)
            else:
                return self.initial_lr

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def step(self):
        """Increment step counter"""
        self.step_count += 1

# Visualize different schedules
schedules = {
    'Constant': LearningRateScheduler(0.1, 'constant'),
    'Step Decay': LearningRateScheduler(0.1, 'step', decay_rate=0.5, decay_steps=100),
    'Exponential': LearningRateScheduler(0.1, 'exponential', decay_rate=0.995),
    'Cosine': LearningRateScheduler(0.1, 'cosine', T_max=300),
    'Warmup': LearningRateScheduler(0.1, 'warmup', warmup_steps=50)
}

plt.figure(figsize=(12, 5))
for name, scheduler in schedules.items():
    lrs = []
    for _ in range(300):
        lrs.append(scheduler.get_lr())
        scheduler.step()
    plt.plot(lrs, label=name, linewidth=2)

plt.xlabel('Training Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedules')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Summary

**Key Concepts:**

1. **Single Neuron:** Linear combination + activation function
   - z = w·x + b
   - a = activation(z)

2. **Activation Functions:**
   - Sigmoid: smooth, (0, 1), vanishing gradients
   - ReLU: simple, fast, dead neurons
   - Tanh: (-1, 1), zero-centered
   - Softmax: probability distribution

3. **Forward Pass:**
   - Layer-by-layer computation
   - Cache values for backprop
   - Batch processing for efficiency

4. **Backpropagation:**
   - Chain rule for gradients
   - ∂L/∂W = aᵀ · ∂L/∂z
   - ∂L/∂a_prev = ∂L/∂z · Wᵀ

5. **Training:**
   - SGD: θ ← θ - α·∇L
   - Mini-batches for efficiency
   - Monitor loss and accuracy

6. **Common Issues:**
   - Vanishing gradients: use ReLU, batch norm
   - Overfitting: use dropout, regularization
   - Poor initialization: use Xavier/He

**Practical Tips:**
- Start with ReLU activation
- Use He initialization for ReLU
- Add batch normalization for deep networks
- Use dropout for regularization
- Monitor gradient norms during training
- Start with small learning rate
- Use learning rate scheduling for better convergence
