# Neural Networks Solutions - Module 6

Comprehensive solutions with step-by-step mathematical work and code implementations.

**Important:** These solutions match the exercises in [neural_networks_exercises.html](../exercises/neural_networks_exercises.html). Try solving them yourself first!

---

## Part 1: Single Neuron (30 min)

### Exercise 1.1 Solution: Linear Neuron

**Given:**
- Weights: w = [0.5, -0.3, 0.2]
- Bias: b = 0.1
- Input: x = [1.0, 2.0, 3.0]

**Step 1: Calculate z = w·x + b**

```
z = w₁x₁ + w₂x₂ + w₃x₃ + b
  = (0.5)(1.0) + (-0.3)(2.0) + (0.2)(3.0) + 0.1
  = 0.5 - 0.6 + 0.6 + 0.1
  = 0.6
```

**Step 2: Linear activation (f(z) = z)**

For linear activation, the output is simply z:
```
output = z = 0.6
```

**Step 3: Count parameters**

```
Parameters = weights + bias = 3 + 1 = 4 parameters
```

**NumPy Implementation:**

```python
import numpy as np

# Given values
w = np.array([0.5, -0.3, 0.2])
b = 0.1
x = np.array([1.0, 2.0, 3.0])

# Calculate z
z = np.dot(w, x) + b
print(f"z = {z:.4f}")  # z = 0.6000

# Linear activation
output = z
print(f"Output (linear): {output:.4f}")

# Count parameters
num_params = len(w) + 1  # weights + bias
print(f"Number of parameters: {num_params}")
```

---

### Exercise 1.2 Solution: Sigmoid Neuron

**Using same neuron from Exercise 1.1:**
- z = 0.6 (calculated above)
- Sigmoid: σ(z) = 1/(1 + e^(-z))

**Step 1: Calculate z = w·x + b**

Already calculated: z = 0.6

**Step 2: Calculate output a = σ(z)**

```
σ(0.6) = 1/(1 + e^(-0.6))
       = 1/(1 + e^(-0.6))
       = 1/(1 + 0.5488)
       = 1/1.5488
       = 0.6457
```

**Step 3: Calculate derivative σ'(z) = σ(z)(1 - σ(z))**

```
σ'(0.6) = 0.6457 × (1 - 0.6457)
        = 0.6457 × 0.3543
        = 0.2288
```

**Step 4: Why is this derivative useful for backpropagation?**

The derivative tells us how sensitive the output is to changes in the input. During backpropagation:
- We multiply gradients flowing backward by σ'(z)
- This "scales" the gradient based on how steep the sigmoid curve is at that point
- When z is very large or very small, σ'(z) ≈ 0 (vanishing gradient problem)
- The convenient form σ'(z) = σ(z)(1 - σ(z)) means we only need to compute sigmoid once!

**NumPy Implementation:**

```python
def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of sigmoid: σ'(z) = σ(z)(1 - σ(z))"""
    s = sigmoid(z)
    return s * (1 - s)

# From Exercise 1.1
z = 0.6

# Calculate sigmoid output
a = sigmoid(z)
print(f"Sigmoid output: a = {a:.4f}")  # 0.6457

# Calculate derivative
derivative = sigmoid_derivative(z)
print(f"Sigmoid derivative: σ'(z) = {derivative:.4f}")  # 0.2288

# Verify using the convenient form
derivative_alt = a * (1 - a)
print(f"Verification: σ(z)(1-σ(z)) = {derivative_alt:.4f}")
```

---

### Exercise 1.3 Solution: ReLU Neuron

**Using same weights from Exercise 1.1:**
- w = [0.5, -0.3, 0.2], b = 0.1
- ReLU: f(z) = max(0, z)

**Step 1: Calculate output for x = [1, 2, 3]**

```
z = w·x + b = 0.6 (from Exercise 1.1)
ReLU(0.6) = max(0, 0.6) = 0.6
```

**Step 2: Calculate output for x = [-1, -2, -3]**

```
z = (0.5)(-1) + (-0.3)(-2) + (0.2)(-3) + 0.1
  = -0.5 + 0.6 - 0.6 + 0.1
  = -0.4

ReLU(-0.4) = max(0, -0.4) = 0
```

**Step 3: What is the derivative of ReLU?**

```
ReLU'(z) = { 1  if z > 0
           { 0  if z ≤ 0

For z = 0.6: ReLU'(0.6) = 1
For z = -0.4: ReLU'(-0.4) = 0
```

**Step 4: What happens when z < 0? (dead neuron)**

When z < 0:
- ReLU output is 0
- Gradient is 0
- No learning happens for this neuron on this example
- If a neuron always gets z < 0, it becomes "dead" (never updates)
- This is called the "dying ReLU" problem
- Solution: Use Leaky ReLU or other variants

**NumPy Implementation:**

```python
def relu(z):
    """ReLU activation function"""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU"""
    return (z > 0).astype(float)

# Test case 1: x = [1, 2, 3]
x1 = np.array([1.0, 2.0, 3.0])
z1 = np.dot(w, x1) + b
output1 = relu(z1)
print(f"Input {x1}: z = {z1:.4f}, ReLU(z) = {output1:.4f}")
print(f"  Derivative: ReLU'(z) = {relu_derivative(z1):.0f}")

# Test case 2: x = [-1, -2, -3]
x2 = np.array([-1.0, -2.0, -3.0])
z2 = np.dot(w, x2) + b
output2 = relu(z2)
print(f"Input {x2}: z = {z2:.4f}, ReLU(z) = {output2:.4f}")
print(f"  Derivative: ReLU'(z) = {relu_derivative(z2):.0f}")
print(f"  This neuron is DEAD for this input (no gradient)!")
```

---

## Part 2: Activation Functions (35 min)

### Exercise 2.1 Solution: Comparing Activations

**Given:** z = [-2, -1, 0, 1, 2]

**Step 1: Calculate all activation functions**

**Sigmoid:** σ(z) = 1/(1 + e^(-z))
```
σ(-2) = 0.1192
σ(-1) = 0.2689
σ(0)  = 0.5000
σ(1)  = 0.7311
σ(2)  = 0.8808
```

**Tanh:** tanh(z) = (e^z - e^(-z))/(e^z + e^(-z))
```
tanh(-2) = -0.9640
tanh(-1) = -0.7616
tanh(0)  = 0.0000
tanh(1)  = 0.7616
tanh(2)  = 0.9640
```

**ReLU:** max(0, z)
```
ReLU(-2) = 0
ReLU(-1) = 0
ReLU(0)  = 0
ReLU(1)  = 1
ReLU(2)  = 2
```

**Leaky ReLU:** max(0.01z, z)
```
Leaky ReLU(-2) = -0.02
Leaky ReLU(-1) = -0.01
Leaky ReLU(0)  = 0.00
Leaky ReLU(1)  = 1.00
Leaky ReLU(2)  = 2.00
```

**Step 3: What are the ranges?**

- **Sigmoid:** (0, 1) - Always positive, bounded
- **Tanh:** (-1, 1) - Zero-centered, bounded
- **ReLU:** [0, ∞) - Unbounded above, always non-negative
- **Leaky ReLU:** (-∞, ∞) - Unbounded both directions

**NumPy Implementation:**

```python
def tanh(z):
    return np.tanh(z)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

z_values = np.array([-2, -1, 0, 1, 2])

print("z      | Sigmoid | Tanh    | ReLU  | Leaky ReLU")
print("-------|---------|---------|-------|------------")
for z in z_values:
    sig = sigmoid(z)
    tan = tanh(z)
    rel = relu(z)
    leaky = leaky_relu(z)
    print(f"{z:6.1f} | {sig:7.4f} | {tan:7.4f} | {rel:5.1f} | {leaky:10.2f}")

# Visualize
import matplotlib.pyplot as plt

z_range = np.linspace(-3, 3, 100)

plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.plot(z_range, sigmoid(z_range), 'b-', linewidth=2)
plt.title('Sigmoid')
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(0, color='k', linestyle='--', alpha=0.3)

plt.subplot(1, 4, 2)
plt.plot(z_range, tanh(z_range), 'g-', linewidth=2)
plt.title('Tanh')
plt.xlabel('z')
plt.ylabel('tanh(z)')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(0, color='k', linestyle='--', alpha=0.3)

plt.subplot(1, 4, 3)
plt.plot(z_range, relu(z_range), 'r-', linewidth=2)
plt.title('ReLU')
plt.xlabel('z')
plt.ylabel('ReLU(z)')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(0, color='k', linestyle='--', alpha=0.3)

plt.subplot(1, 4, 4)
plt.plot(z_range, leaky_relu(z_range), 'm-', linewidth=2)
plt.title('Leaky ReLU')
plt.xlabel('z')
plt.ylabel('Leaky ReLU(z)')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### Exercise 2.2 Solution: Activation Derivatives

**Calculate derivatives at z = 1:**

**Step 1: d/dz sigmoid(z)**

Formula: σ'(z) = σ(z)(1 - σ(z))

```
σ(1) = 0.7311
σ'(1) = 0.7311 × (1 - 0.7311)
      = 0.7311 × 0.2689
      = 0.1966
```

**Step 2: d/dz tanh(z) = 1 - tanh²(z)**

```
tanh(1) = 0.7616
tanh'(1) = 1 - (0.7616)²
         = 1 - 0.5800
         = 0.4200
```

**Step 3: d/dz ReLU(z)**

```
ReLU'(1) = 1  (since 1 > 0)
```

**Step 4: Which has vanishing gradient problem?**

**Sigmoid and Tanh** have the vanishing gradient problem:

- **Sigmoid:** Maximum derivative is 0.25 at z=0
  - For z = ±5, σ'(z) ≈ 0.0066 (very small!)
  - In deep networks, gradients multiply: (0.25)^10 = 0.00000095

- **Tanh:** Maximum derivative is 1.0 at z=0
  - Better than sigmoid, but still saturates
  - For z = ±3, tanh'(z) ≈ 0.01

- **ReLU:** No vanishing gradient for z > 0
  - Gradient is always 1 or 0
  - Problem: "dying ReLU" when z < 0

**NumPy Implementation:**

```python
def tanh_derivative(z):
    return 1 - np.tanh(z)**2

z = 1

# Calculate all derivatives
sig_deriv = sigmoid_derivative(z)
tanh_deriv = tanh_derivative(z)
relu_deriv = relu_derivative(z)

print(f"At z = {z}:")
print(f"  d/dz sigmoid(z) = {sig_deriv:.4f}")
print(f"  d/dz tanh(z)    = {tanh_deriv:.4f}")
print(f"  d/dz ReLU(z)    = {relu_deriv:.0f}")

# Show vanishing gradient problem
print("\nVanishing gradient demonstration:")
z_extreme = np.array([-5, -3, -1, 0, 1, 3, 5])
print("\nz     | σ'(z)   | tanh'(z)")
print("------|---------|----------")
for z in z_extreme:
    print(f"{z:5.0f} | {sigmoid_derivative(z):7.4f} | {tanh_derivative(z):8.4f}")

# Visualize derivatives
z_range = np.linspace(-6, 6, 200)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(z_range, sigmoid_derivative(z_range), 'b-', linewidth=2)
plt.title("Sigmoid Derivative")
plt.xlabel('z')
plt.ylabel("σ'(z)")
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(z_range, tanh_derivative(z_range), 'g-', linewidth=2)
plt.title("Tanh Derivative")
plt.xlabel('z')
plt.ylabel("tanh'(z)")
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(z_range, relu_derivative(z_range), 'r-', linewidth=2)
plt.title("ReLU Derivative")
plt.xlabel('z')
plt.ylabel("ReLU'(z)")
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### Exercise 2.3 Solution: Softmax

**Given:** Logits z = [2.0, 1.0, 0.1]

**Step 1: Calculate softmax: p(i) = exp(z_i) / Σexp(z_j)**

```
exp(z) = [exp(2.0), exp(1.0), exp(0.1)]
       = [7.389, 2.718, 1.105]

Sum = 7.389 + 2.718 + 1.105 = 11.212

softmax(z) = [7.389/11.212, 2.718/11.212, 1.105/11.212]
           = [0.659, 0.242, 0.099]
```

**Step 2: Verify outputs sum to 1**

```
Sum = 0.659 + 0.242 + 0.099 = 1.000 ✓
```

**Step 3: Which class has highest probability?**

Class 0 has the highest probability (0.659 or 65.9%), corresponding to the largest logit (2.0).

**Step 4: What happens if you add constant to all logits?**

**Key insight:** Softmax is **translation invariant**!

```
If z' = z + c for all elements:

softmax(z'_i) = exp(z_i + c) / Σexp(z_j + c)
              = exp(z_i)·exp(c) / (exp(c)·Σexp(z_j))
              = exp(z_i) / Σexp(z_j)
              = softmax(z_i)

The exp(c) cancels out!
```

This property is used for **numerical stability**: subtract max(z) before computing softmax to avoid overflow.

**NumPy Implementation:**

```python
def softmax(z):
    """Numerically stable softmax"""
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

# Given logits
z = np.array([2.0, 1.0, 0.1])

# Calculate softmax
probs = softmax(z)
print("Logits:", z)
print("Softmax probabilities:", probs)
print("Sum:", np.sum(probs))
print(f"Highest probability: Class {np.argmax(probs)} ({probs[np.argmax(probs)]:.3f})")

# Test translation invariance
c = 10.0
z_shifted = z + c
probs_shifted = softmax(z_shifted)

print(f"\nAfter adding {c} to all logits:")
print("New logits:", z_shifted)
print("Softmax probabilities:", probs_shifted)
print("Same as before?", np.allclose(probs, probs_shifted))

# Without numerical stability, this would overflow!
try:
    unstable = np.exp(z_shifted) / np.sum(np.exp(z_shifted))
    print("Unstable softmax:", unstable)
except:
    print("Unstable softmax overflowed!")
```

---

### Exercise 2.4 Solution: Why Non-linearity?

**Given:** Two-layer network with linear activations

**Architecture:**
```
Input x = [1, 1]
Hidden layer: W₁ = [[1, 2], [3, 4]], b₁ = [0, 0]
Output layer: W₂ = [[1], [2]], b₂ = [0]
```

**Step 1: Compute output for x = [1, 1]**

```
z₁ = W₁x + b₁
   = [[1, 2], [3, 4]] @ [1, 1] + [0, 0]
   = [1+2, 3+4]
   = [3, 7]

a₁ = z₁ (linear activation) = [3, 7]

z₂ = W₂a₁ + b₂
   = [[1], [2]]ᵀ @ [3, 7] + 0
   = 1×3 + 2×7
   = 3 + 14
   = 17

output = z₂ (linear activation) = 17
```

**Step 2: Show this equals single layer: W₂W₁x**

```
W₂W₁ = [[1, 2]] @ [[1, 2], [3, 4]]
     = [1×1 + 2×3, 1×2 + 2×4]
     = [1+6, 2+8]
     = [7, 10]

W₂W₁x = [7, 10] @ [1, 1]
      = 7 + 10
      = 17  ✓ Same result!
```

**Step 3: Why do we need non-linear activations?**

**Key insight:** Composing linear functions gives another linear function!

Without non-linearity:
- No matter how many layers you stack, it's equivalent to a single layer
- Cannot learn complex patterns like XOR, circles, etc.
- Universal approximation theorem requires non-linearity

**With non-linearity:**
- Networks can learn any continuous function (universal approximation)
- Can represent decision boundaries of arbitrary complexity
- Each layer learns increasingly abstract features

**NumPy Implementation:**

```python
# Network with linear activations
W1 = np.array([[1, 2], [3, 4]])
b1 = np.array([0, 0])
W2 = np.array([[1, 2]])
b2 = np.array([0])
x = np.array([1, 1])

# Two-layer forward pass
z1 = W1.T @ x + b1
a1 = z1  # Linear activation
z2 = W2 @ a1 + b2
output_2layer = z2

print("Two-layer network output:", output_2layer)

# Equivalent single layer
W_combined = W2 @ W1.T
b_combined = b2
output_1layer = W_combined @ x + b_combined

print("Single layer output:", output_1layer)
print("Same?", np.allclose(output_2layer, output_1layer))

# Demonstrate XOR cannot be learned with linear model
def plot_xor():
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR labels

    plt.figure(figsize=(6, 6))
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', s=100, label='Class 0', marker='o')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', s=100, label='Class 1', marker='s')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('XOR Problem (No Single Line Can Separate These!)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.show()

plot_xor()
print("\nA linear model cannot solve XOR!")
print("We need non-linear activations to create curved decision boundaries.")
```

---

## Part 3: Forward Pass (30 min)

### Exercise 3.1 Solution: Two-Layer Network

**Given:**
- Architecture: 2 → 3 → 1
- Input: x = [1.0, 2.0]
- Weights and biases provided in exercise

**Note:** The exercise should specify the exact weights. Using example weights:

```
W₁ = [[0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6]]  (2×3 matrix)
b₁ = [0.1, 0.2, 0.3]    (3,)

W₂ = [[1.0],
      [2.0],
      [3.0]]            (3×1 matrix)
b₂ = [0.5]              (1,)
```

**Step 1: Calculate z₁ = W₁ᵀx + b₁**

```
z₁ = [[0.1, 0.4],    ᵀ  [1.0]   [0.1]
      [0.2, 0.5],    ×  [2.0] + [0.2]
      [0.3, 0.6]]              [0.3]

   = [0.1×1.0 + 0.4×2.0 + 0.1,
      0.2×1.0 + 0.5×2.0 + 0.2,
      0.3×1.0 + 0.6×2.0 + 0.3]

   = [0.1 + 0.8 + 0.1,
      0.2 + 1.0 + 0.2,
      0.3 + 1.2 + 0.3]

   = [1.0, 1.4, 1.8]
```

**Step 2: Calculate a₁ = σ(z₁)**

```
a₁ = [σ(1.0), σ(1.4), σ(1.8)]
   = [0.7311, 0.8021, 0.8581]
```

**Step 3: Calculate z₂ = W₂ᵀa₁ + b₂**

```
z₂ = [1.0, 2.0, 3.0] @ [0.7311, 0.8021, 0.8581] + 0.5
   = 1.0×0.7311 + 2.0×0.8021 + 3.0×0.8581 + 0.5
   = 0.7311 + 1.6042 + 2.5743 + 0.5
   = 5.4096
```

**Step 4: Calculate a₂ = σ(z₂)**

```
a₂ = σ(5.4096) = 0.9955
```

**NumPy Implementation:**

```python
# Network parameters
W1 = np.array([[0.1, 0.2, 0.3],
               [0.4, 0.5, 0.6]])
b1 = np.array([0.1, 0.2, 0.3])

W2 = np.array([[1.0],
               [2.0],
               [3.0]])
b2 = np.array([0.5])

x = np.array([1.0, 2.0])

# Forward pass
print("Forward pass through 2-layer network:")
print("="*50)

# Layer 1
z1 = x @ W1 + b1
print(f"z₁ = {z1}")

a1 = sigmoid(z1)
print(f"a₁ = σ(z₁) = {a1}")

# Layer 2
z2 = a1 @ W2 + b2
print(f"z₂ = {z2}")

a2 = sigmoid(z2)
print(f"a₂ = σ(z₂) = {a2}")

print(f"\nFinal output: {a2[0]:.4f}")
```

---

### Exercise 3.2 Solution: Batch Processing

**Given:** Same network, batch X = [[1.0, 2.0], [0.5, 1.5]]

**Step 1: Calculate Z₁ = XW₁ + b₁ (shape?)**

```
X shape: (2, 2) - 2 samples, 2 features
W₁ shape: (2, 3) - 2 inputs, 3 hidden neurons
b₁ shape: (3,) - 3 biases

Z₁ = XW₁ + b₁
Z₁ shape: (2, 3) - 2 samples, 3 hidden activations

Z₁ = [[1.0, 2.0],  @  [[0.1, 0.2, 0.3],   + [0.1, 0.2, 0.3]
      [0.5, 1.5]]       [0.4, 0.5, 0.6]]

Sample 1: [1.0×0.1 + 2.0×0.4, 1.0×0.2 + 2.0×0.5, 1.0×0.3 + 2.0×0.6] + b₁
        = [0.9, 1.2, 1.5] + [0.1, 0.2, 0.3]
        = [1.0, 1.4, 1.8]

Sample 2: [0.5×0.1 + 1.5×0.4, 0.5×0.2 + 1.5×0.5, 0.5×0.3 + 1.5×0.6] + b₁
        = [0.65, 0.85, 1.05] + [0.1, 0.2, 0.3]
        = [0.75, 1.05, 1.35]

Z₁ = [[1.0, 1.4, 1.8],
      [0.75, 1.05, 1.35]]
```

**Step 2: Calculate A₁ = σ(Z₁)**

```
A₁ = [[σ(1.0), σ(1.4), σ(1.8)],
      [σ(0.75), σ(1.05), σ(1.35)]]

   = [[0.7311, 0.8021, 0.8581],
      [0.6792, 0.7408, 0.7942]]
```

**Step 3: Calculate final outputs**

```
Z₂ = A₁W₂ + b₂

Sample 1: [0.7311, 0.8021, 0.8581] @ [1.0, 2.0, 3.0]ᵀ + 0.5
        = 0.7311 + 1.6042 + 2.5743 + 0.5 = 5.4096

Sample 2: [0.6792, 0.7408, 0.7942] @ [1.0, 2.0, 3.0]ᵀ + 0.5
        = 0.6792 + 1.4816 + 2.3826 + 0.5 = 5.0434

Z₂ = [[5.4096],
      [5.0434]]

A₂ = σ(Z₂) = [[0.9955],
              [0.9936]]
```

**Step 4: Why is batching useful?**

1. **Computational efficiency:** Matrix operations on GPUs/CPUs are highly optimized
2. **Better gradient estimates:** Average gradient over multiple samples is more stable
3. **Memory efficiency:** Reuse intermediate results, better cache utilization
4. **Regularization effect:** Noise in batch gradients can help escape local minima
5. **Practical:** Most datasets don't fit in memory at once

**NumPy Implementation:**

```python
# Batch of inputs
X_batch = np.array([[1.0, 2.0],
                     [0.5, 1.5]])

print("Batch forward pass:")
print("="*50)

# Layer 1
Z1 = X_batch @ W1 + b1  # Broadcasting handles the bias addition
print(f"Z₁ shape: {Z1.shape}")
print(f"Z₁ =\n{Z1}")

A1 = sigmoid(Z1)
print(f"\nA₁ = σ(Z₁) =\n{A1}")

# Layer 2
Z2 = A1 @ W2 + b2
print(f"\nZ₂ shape: {Z2.shape}")
print(f"Z₂ =\n{Z2}")

A2 = sigmoid(Z2)
print(f"\nA₂ = σ(Z₂) =\n{A2}")

# Compare with single sample processing
print("\nVerification (processing samples individually):")
for i, x in enumerate(X_batch):
    z1_single = x @ W1 + b1
    a1_single = sigmoid(z1_single)
    z2_single = a1_single @ W2 + b2
    a2_single = sigmoid(z2_single)
    print(f"Sample {i}: {a2_single[0]:.4f}")
```

---

### Exercise 3.3 Solution: Deep Network

**Given:** 3-layer network: 2 → 4 → 3 → 1

**Step 1: How many weight matrices?**

3 weight matrices (one per layer transition)

**Step 2: What are the dimensions of each?**

```
W₁: (2, 4) - maps 2 inputs to 4 hidden neurons
W₂: (4, 3) - maps 4 neurons to 3 neurons
W₃: (3, 1) - maps 3 neurons to 1 output
```

**Step 3: Total number of parameters?**

```
Weights:
  W₁: 2 × 4 = 8
  W₂: 4 × 3 = 12
  W₃: 3 × 1 = 3
  Total weights: 8 + 12 + 3 = 23

Biases:
  b₁: 4
  b₂: 3
  b₃: 1
  Total biases: 4 + 3 + 1 = 8

Total parameters: 23 + 8 = 31 parameters
```

**Step 4: Write forward pass equations**

```
Layer 1:
  z₁ = xW₁ + b₁
  a₁ = σ(z₁)

Layer 2:
  z₂ = a₁W₂ + b₂
  a₂ = σ(z₂)

Layer 3 (output):
  z₃ = a₂W₃ + b₃
  a₃ = σ(z₃)  or  a₃ = z₃ (if regression)

Output: a₃
```

**NumPy Implementation:**

```python
# Initialize random weights for 2 → 4 → 3 → 1 network
np.random.seed(42)

W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros(4)

W2 = np.random.randn(4, 3) * 0.5
b2 = np.zeros(3)

W3 = np.random.randn(3, 1) * 0.5
b3 = np.zeros(1)

# Count parameters
total_params = (
    W1.size + b1.size +
    W2.size + b2.size +
    W3.size + b3.size
)

print("3-Layer Network Architecture: 2 → 4 → 3 → 1")
print("="*50)
print(f"W₁ shape: {W1.shape}, parameters: {W1.size}")
print(f"b₁ shape: {b1.shape}, parameters: {b1.size}")
print(f"W₂ shape: {W2.shape}, parameters: {W2.size}")
print(f"b₂ shape: {b2.shape}, parameters: {b2.size}")
print(f"W₃ shape: {W3.shape}, parameters: {W3.size}")
print(f"b₃ shape: {b3.shape}, parameters: {b3.size}")
print(f"\nTotal parameters: {total_params}")

# Forward pass
x = np.array([1.0, 2.0])

z1 = x @ W1 + b1
a1 = sigmoid(z1)
print(f"\nLayer 1: z₁ shape {z1.shape}, a₁ shape {a1.shape}")

z2 = a1 @ W2 + b2
a2 = sigmoid(z2)
print(f"Layer 2: z₂ shape {z2.shape}, a₂ shape {a2.shape}")

z3 = a2 @ W3 + b3
a3 = sigmoid(z3)
print(f"Layer 3: z₃ shape {z3.shape}, a₃ shape {a3.shape}")

print(f"\nFinal output: {a3[0]:.4f}")
```

---

## Part 4: Backpropagation (45 min)

### Exercise 4.1 Solution: Single Neuron Gradient

**Given:**
- Model: y = σ(wx + b)
- Loss: L = ½(y - t)²
- Values: w = 0.5, b = 0.1, x = 2.0, t = 1.0

**Step 1: Forward pass - calculate y and loss L**

```
z = wx + b = 0.5 × 2.0 + 0.1 = 1.0 + 0.1 = 1.1

y = σ(1.1) = 1/(1 + e^(-1.1))
           = 1/(1 + 0.3329)
           = 1/1.3329
           = 0.7503

L = ½(y - t)²
  = ½(0.7503 - 1.0)²
  = ½(-0.2497)²
  = ½(0.0624)
  = 0.0312
```

**Step 2: Calculate ∂L/∂y, ∂y/∂z, ∂z/∂w**

```
∂L/∂y = y - t = 0.7503 - 1.0 = -0.2497

∂y/∂z = σ'(z) = σ(z)(1 - σ(z))
      = 0.7503 × (1 - 0.7503)
      = 0.7503 × 0.2497
      = 0.1873

∂z/∂w = x = 2.0
```

**Step 3: Chain rule - ∂L/∂w = ∂L/∂y · ∂y/∂z · ∂z/∂w**

```
∂L/∂w = ∂L/∂y · ∂y/∂z · ∂z/∂w
      = (-0.2497) × 0.1873 × 2.0
      = -0.0935
```

**Step 4: Calculate ∂L/∂b similarly**

```
∂z/∂b = 1

∂L/∂b = ∂L/∂y · ∂y/∂z · ∂z/∂b
      = (-0.2497) × 0.1873 × 1
      = -0.0468
```

**NumPy Implementation:**

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Given values
w = 0.5
b = 0.1
x = 2.0
t = 1.0

# Forward pass
z = w * x + b
y = sigmoid(z)
loss = 0.5 * (y - t)**2

print(f"Forward pass:")
print(f"  z = {z:.4f}")
print(f"  y = σ(z) = {y:.4f}")
print(f"  L = ½(y-t)² = {loss:.4f}")

# Backward pass - compute gradients
dL_dy = y - t
dy_dz = sigmoid_derivative(z)
dz_dw = x
dz_db = 1

dL_dw = dL_dy * dy_dz * dz_dw
dL_db = dL_dy * dy_dz * dz_db

print(f"\nBackward pass:")
print(f"  ∂L/∂y = {dL_dy:.4f}")
print(f"  ∂y/∂z = {dy_dz:.4f}")
print(f"  ∂z/∂w = {dz_dw:.4f}")
print(f"  ∂L/∂w = {dL_dw:.4f}")
print(f"  ∂L/∂b = {dL_db:.4f}")
```

---

### Exercise 4.2 Solution: Two-Layer Backprop

**Given:**
- Architecture: 1 → 2 → 1 (all sigmoid)
- x = [1.0], t = 1.0
- W₁ = [[0.5], [0.3]], b₁ = [0.1, 0.2]
- W₂ = [[0.4, 0.6]], b₂ = [0.1]
- Loss: L = ½(y - t)²

**Step 1: Forward pass**

```
z₁ = W₁x + b₁
   = [[0.5], [0.3]] × [1.0] + [0.1, 0.2]
   = [0.5, 0.3] + [0.1, 0.2]
   = [0.6, 0.5]

a₁ = σ(z₁) = [σ(0.6), σ(0.5)]
           = [0.6457, 0.6225]

z₂ = W₂a₁ + b₂
   = [[0.4, 0.6]] @ [0.6457, 0.6225] + [0.1]
   = [0.4×0.6457 + 0.6×0.6225] + [0.1]
   = [0.2583 + 0.3735] + [0.1]
   = [0.7318]

y = σ(z₂) = σ(0.7318) = 0.6751

L = ½(y - t)² = ½(0.6751 - 1.0)² = ½(0.1055) = 0.0528
```

**Step 2: Loss gradient**

```
∂L/∂y = y - t = 0.6751 - 1.0 = -0.3249
```

**Step 3: Output layer backward (δ₂)**

```
δ₂ = (y - t) · σ'(z₂)
   = -0.3249 × [σ(z₂)(1 - σ(z₂))]
   = -0.3249 × [0.6751 × 0.3249]
   = -0.3249 × 0.2193
   = -0.0713
```

**Step 4: Gradients for W₂ and b₂**

```
∂L/∂W₂ = δ₂ · a₁ᵀ
       = [-0.0713] @ [0.6457, 0.6225]ᵀ
       = [[-0.0460, -0.0444]]

∂L/∂b₂ = δ₂ = [-0.0713]
```

**Step 5: Hidden layer backward (δ₁)**

```
δ₁ = (W₂ᵀδ₂) ⊙ σ'(z₁)
   = [[0.4], [0.6]] × [-0.0713] ⊙ [σ'(0.6), σ'(0.5)]
   = [-0.0285, -0.0428] ⊙ [0.2289, 0.2350]
   = [-0.0065, -0.0101]
```

**Step 6: Gradients for W₁ and b₁**

```
∂L/∂W₁ = δ₁ · xᵀ
       = [[-0.0065], [-0.0101]] @ [1.0]
       = [[-0.0065], [-0.0101]]

∂L/∂b₁ = δ₁ = [-0.0065, -0.0101]
```

**NumPy Implementation:**

```python
# Network parameters
W1 = np.array([[0.5], [0.3]])  # (2, 1)
b1 = np.array([0.1, 0.2])      # (2,)
W2 = np.array([[0.4, 0.6]])    # (1, 2)
b2 = np.array([0.1])           # (1,)

x = np.array([1.0])
t = np.array([1.0])

# Forward pass
z1 = W1 @ x + b1
a1 = sigmoid(z1)

z2 = W2 @ a1 + b2
y = sigmoid(z2)

loss = 0.5 * (y - t)**2

print("Forward pass:")
print(f"  z₁ = {z1}")
print(f"  a₁ = {a1}")
print(f"  z₂ = {z2}")
print(f"  y = {y}")
print(f"  L = {loss[0]:.4f}")

# Backward pass
# Output layer
delta2 = (y - t) * sigmoid_derivative(z2)
dL_dW2 = np.outer(delta2, a1)
dL_db2 = delta2

# Hidden layer
delta1 = (W2.T @ delta2) * sigmoid_derivative(z1)
dL_dW1 = np.outer(delta1, x)
dL_db1 = delta1

print("\nBackward pass:")
print(f"  δ₂ = {delta2}")
print(f"  ∂L/∂W₂ = {dL_dW2}")
print(f"  ∂L/∂b₂ = {dL_db2}")
print(f"  δ₁ = {delta1}")
print(f"  ∂L/∂W₁ = {dL_dW1}")
print(f"  ∂L/∂b₁ = {dL_db1}")
```

---

### Exercise 4.3 Solution: Backprop with ReLU

**Using same architecture as 4.2 but with ReLU:**

Replace sigmoid with ReLU in hidden layer. Forward pass will differ, and gradients will be simpler!

**Key difference:** ReLU'(z) = 1 if z > 0, else 0

**NumPy Implementation:**

```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Forward with ReLU in hidden layer
z1 = W1 @ x + b1
a1 = relu(z1)  # ReLU instead of sigmoid

z2 = W2 @ a1 + b2
y = sigmoid(z2)  # Output still sigmoid

loss = 0.5 * (y - t)**2

print("Forward with ReLU:")
print(f"  z₁ = {z1}")
print(f"  a₁ (ReLU) = {a1}")
print(f"  y = {y}")
print(f"  L = {loss[0]:.4f}")

# Backward pass
delta2 = (y - t) * sigmoid_derivative(z2)
dL_dW2_relu = np.outer(delta2, a1)
dL_db2_relu = delta2

# Hidden layer - ReLU derivative
delta1_relu = (W2.T @ delta2) * relu_derivative(z1)
dL_dW1_relu = np.outer(delta1_relu, x)
dL_db1_relu = delta1_relu

print("\nComparison:")
print(f"Sigmoid hidden: δ₁ = {delta1}")
print(f"ReLU hidden:    δ₁ = {delta1_relu}")
print("\nReLU gradients are simpler (either 0 or 1)!")
```

---

### Exercise 4.4 Solution: Vanishing Gradient

**Given:** Deep network with 10 sigmoid layers

**Step 1: Maximum sigmoid derivative**

```
σ'(z) = σ(z)(1 - σ(z))
Max occurs at z = 0: σ'(0) = 0.5 × 0.5 = 0.25
```

**Step 2: Gradient at first layer**

```
If each layer has gradient ≈ 0.25:
Gradient at layer 1 = (0.25)^10 = 9.5 × 10^(-7) ≈ 0.000001
```

**Step 3: Why is this problematic?**

- Gradients become extremely small (vanish)
- First layers learn very slowly or not at all
- Network can't learn long-term dependencies
- Training stalls

**Step 4: How do ReLU and skip connections help?**

**ReLU:**
- Gradient is 1 for z > 0 (no decay!)
- No saturation for positive values
- Faster convergence

**Skip connections (ResNets):**
- Gradient can flow directly: ∂L/∂x = ∂L/∂output × (1 + ∂F/∂x)
- Always has path with gradient = 1
- Enables training of very deep networks (100+ layers)

**NumPy Demonstration:**

```python
# Simulate gradient flow through 10 layers
n_layers = 10

# Sigmoid network
grad_sigmoid = 1.0
for i in range(n_layers):
    grad_sigmoid *= 0.25  # Max sigmoid derivative

print(f"Gradient after {n_layers} sigmoid layers: {grad_sigmoid:.2e}")

# ReLU network
grad_relu = 1.0
for i in range(n_layers):
    grad_relu *= 1.0  # ReLU derivative (for active neurons)

print(f"Gradient after {n_layers} ReLU layers: {grad_relu:.2e}")

print(f"\nRatio: {grad_relu / grad_sigmoid:.2e}x better!")
```

---

## Part 5: Training (35 min)

### Exercise 5.1 Solution: Gradient Descent Update

**Given:**
- Gradients: ∂L/∂w = 0.3, ∂L/∂b = 0.2
- Initial: w = 0.5, b = 0.1
- Learning rate: α = 0.1

**Step 1: Update rule**

```
w_new = w - α · ∂L/∂w
b_new = b - α · ∂L/∂b
```

**Step 2: Calculate new w and b**

```
w_new = 0.5 - 0.1 × 0.3 = 0.5 - 0.03 = 0.47
b_new = 0.1 - 0.1 × 0.2 = 0.1 - 0.02 = 0.08
```

**Step 3: Perform 5 iterations**

Assuming gradients stay constant (they wouldn't in practice):

```
Iteration 1: w = 0.47, b = 0.08
Iteration 2: w = 0.44, b = 0.06
Iteration 3: w = 0.41, b = 0.04
Iteration 4: w = 0.38, b = 0.02
Iteration 5: w = 0.35, b = 0.00
```

**Step 4: Does loss decrease?**

Yes! Since we're moving in the negative gradient direction, we're going downhill on the loss surface.

**NumPy Implementation:**

```python
w = 0.5
b = 0.1
alpha = 0.1
grad_w = 0.3
grad_b = 0.2

print("Gradient Descent Updates:")
print(f"{'Iter':<6} {'w':<8} {'b':<8} {'Δw':<10} {'Δb':<10}")
print("-" * 42)

for i in range(6):
    print(f"{i:<6} {w:<8.4f} {b:<8.4f} {-alpha*grad_w:<10.4f} {-alpha*grad_b:<10.4f}")

    if i < 5:
        w = w - alpha * grad_w
        b = b - alpha * grad_b
```

---

### Exercise 5.2 Solution: Mini-batch Training

**Given:** Dataset with 100 samples, batch size 10

**Step 1: How many batches per epoch?**

```
Number of batches = Total samples / Batch size
                  = 100 / 10
                  = 10 batches per epoch
```

**Step 2: Weight updates per epoch?**

```
Updates per epoch = Number of batches = 10 updates
```

**Step 3: Compare with batch GD (1 update/epoch)**

```
Batch GD:
- 1 update per epoch (use all 100 samples)
- More accurate gradient
- Slower (must process all data before update)
- More memory required

Mini-batch (10 samples):
- 10 updates per epoch
- 10x more frequent updates
- Faster convergence (usually)
- Better memory efficiency
```

**Step 4: Compare with SGD (100 updates/epoch)**

```
SGD (batch size = 1):
- 100 updates per epoch
- Very noisy gradients
- Can escape local minima
- Fastest per-update
- May not converge to exact minimum

Mini-batch strikes a balance:
- More stable than SGD
- Faster than batch GD
- Best of both worlds!
```

**Summary Table:**

```
Method          | Batch Size | Updates/Epoch | Gradient Noise | Speed
----------------|------------|---------------|----------------|-------
Batch GD        | 100        | 1             | Low            | Slow
Mini-batch GD   | 10         | 10            | Medium         | Fast
SGD             | 1          | 100           | High           | Fastest
```

---

### Exercise 5.3 Solution: Learning Rate Effects

**Given:** w = 0.5, ∂L/∂w = 0.1

**Step 1: Update with α = 0.01**

```
w_new = w - α · ∂L/∂w
      = 0.5 - 0.01 × 0.1
      = 0.5 - 0.001
      = 0.499
```
Small step, safe but slow.

**Step 2: Update with α = 1.0**

```
w_new = 0.5 - 1.0 × 0.1
      = 0.5 - 0.1
      = 0.4
```
Reasonable step size.

**Step 3: Update with α = 10.0**

```
w_new = 0.5 - 10.0 × 0.1
      = 0.5 - 1.0
      = -0.5
```
Too large! May overshoot the minimum.

**Step 4: Which learning rate works best?**

**α = 0.01:** Too small
- Very slow convergence
- Needs many iterations
- Safe but inefficient

**α = 1.0:** Good choice
- Reasonable progress per step
- Balanced speed and stability

**α = 10.0:** Too large
- May overshoot minimum
- Can cause divergence
- Loss might oscillate or increase

**Step 5: What if gradient = -0.1 instead?**

```
α = 0.01: w_new = 0.5 - 0.01 × (-0.1) = 0.501 (moves opposite direction)
α = 1.0:  w_new = 0.5 - 1.0 × (-0.1) = 0.6
α = 10.0: w_new = 0.5 - 10.0 × (-0.1) = 1.5
```

Negative gradient means we're on the other side of the minimum - we move in the positive direction!

**NumPy Implementation:**

```python
w = 0.5
grad = 0.1

learning_rates = [0.01, 1.0, 10.0]

print("Learning Rate Effects:")
print(f"{'α':<10} {'w_new':<10} {'Step size':<12} {'Assessment'}")
print("-" * 50)

for alpha in learning_rates:
    w_new = w - alpha * grad
    step = abs(w_new - w)

    if alpha < 0.1:
        assessment = "Too slow"
    elif alpha > 5:
        assessment = "Too fast (risk overshoot)"
    else:
        assessment = "Good"

    print(f"{alpha:<10} {w_new:<10.3f} {step:<12.3f} {assessment}")

# With negative gradient
print("\nWith negative gradient (-0.1):")
grad_neg = -0.1
for alpha in learning_rates:
    w_new = w - alpha * grad_neg
    print(f"α = {alpha:<5}: w_new = {w_new:.3f}")
```

---

## Part 6: Debugging NNs (35 min)

### Exercise 6.1 Solution: Gradient Checking

**Given:**
- y = σ(wx), L = ½(y - t)²
- w = 0.5, x = 2.0, t = 1.0, ε = 1e-5

**Step 1: Compute analytical gradient using chain rule**

```
z = wx = 0.5 × 2.0 = 1.0
y = σ(1.0) = 0.7311
L = ½(0.7311 - 1.0)² = 0.0362

∂L/∂y = y - t = 0.7311 - 1.0 = -0.2689
∂y/∂z = σ'(z) = 0.7311 × (1 - 0.7311) = 0.1966
∂z/∂w = x = 2.0

∂L/∂w = ∂L/∂y · ∂y/∂z · ∂z/∂w
      = -0.2689 × 0.1966 × 2.0
      = -0.1058
```

**Step 2: Compute numerical gradient**

```
def L(w):
    z = w * x
    y = σ(z)
    return ½(y - t)²

w⁺ = w + ε = 0.5 + 0.00001 = 0.50001
w⁻ = w - ε = 0.5 - 0.00001 = 0.49999

L(w⁺) = ½(σ(0.50001 × 2) - 1)² = 0.036146
L(w⁻) = ½(σ(0.49999 × 2) - 1)² = 0.036252

Numerical gradient = (L(w⁺) - L(w⁻)) / (2ε)
                   = (0.036146 - 0.036252) / (2 × 0.00001)
                   = -0.00106 / 0.00002
                   = -0.1058
```

**Step 3: Calculate relative error**

```
Relative error = |analytical - numerical| / (|analytical| + |numerical|)
               = |−0.1058 - (−0.1058)| / (|−0.1058| + |−0.1058|)
               = 0 / 0.2116
               = 0.0000

(In practice, might be ~1e-9 due to floating point precision)
```

**Step 4: Verify correctness**

```
Error < 1e-7? ✓ YES!

The analytical gradient is correct!
```

**NumPy Implementation:**

```python
def compute_loss(w, x, t):
    z = w * x
    y = sigmoid(z)
    return 0.5 * (y - t)**2

# Analytical gradient
w = 0.5
x = 2.0
t = 1.0

z = w * x
y = sigmoid(z)

dL_dy = y - t
dy_dz = sigmoid_derivative(z)
dz_dw = x

grad_analytical = dL_dy * dy_dz * dz_dw

# Numerical gradient
epsilon = 1e-5
w_plus = w + epsilon
w_minus = w - epsilon

loss_plus = compute_loss(w_plus, x, t)
loss_minus = compute_loss(w_minus, x, t)

grad_numerical = (loss_plus - loss_minus) / (2 * epsilon)

# Relative error
relative_error = abs(grad_analytical - grad_numerical) / (abs(grad_analytical) + abs(grad_numerical))

print("Gradient Checking:")
print(f"  Analytical gradient: {grad_analytical:.6f}")
print(f"  Numerical gradient:  {grad_numerical:.6f}")
print(f"  Relative error:      {relative_error:.2e}")

if relative_error < 1e-7:
    print("  ✓ PASS - Gradients are correct!")
else:
    print("  ✗ FAIL - Check your backprop implementation!")
```

---

### Exercise 6.2 Solution: Detecting Bugs

**Step 1: Loss is NaN → What's wrong?**

**Possible causes:**
- **Exploding gradients:** Learning rate too high, weights become huge
- **Numerical overflow:** exp() in softmax with large logits
- **Division by zero:** Computing log(0) in cross-entropy
- **NaN in input data**

**Fixes:**
- Use gradient clipping
- Subtract max in softmax for numerical stability
- Add small epsilon to log: log(p + 1e-10)
- Check data for NaNs/Infs

**Step 2: Loss doesn't decrease → Check?**

**Debugging checklist:**
1. **Learning rate:** Try smaller values (0.001, 0.0001)
2. **Gradients:** Check if they're zero (dead neurons?)
3. **Implementation bugs:** Gradient sign, weight update formula
4. **Data preprocessing:** Normalize inputs
5. **Initialization:** Too large or too small weights
6. **Activation functions:** Dead ReLUs?

**Quick test:** Overfit on single batch - if it doesn't work, there's a bug!

**Step 3: Training 99%, validation 60% → Problem?**

**This is overfitting!**

**Why it happens:**
- Model memorizes training data
- Doesn't generalize to new data
- Too complex for dataset size

**Solutions:**
- **Regularization:** Add L2/L1 penalty
- **Dropout:** Randomly disable neurons during training
- **Early stopping:** Stop when validation loss stops improving
- **More data:** Best solution if possible
- **Data augmentation:** Create variations of existing data
- **Simpler model:** Reduce layers/neurons

**Step 4: Loss oscillates wildly → Fix?**

**Possible causes:**
- **Learning rate too high:** Taking huge steps
- **Batch size too small:** Very noisy gradients (batch=1 or 2)
- **Bad initialization:** Weights too large

**Fixes:**
- **Reduce learning rate:** Try 0.1x current value
- **Increase batch size:** Try 32, 64, 128
- **Use learning rate scheduling:** Decay over time
- **Add gradient clipping:** Prevent extreme updates
- **Use momentum/Adam:** Smoother updates

**Visualization:**

```python
import matplotlib.pyplot as plt

# Example loss curves
epochs = np.arange(50)

# Good training
loss_good = 2.0 * np.exp(-epochs/10) + 0.1

# Oscillating (LR too high)
loss_oscillate = 2.0 * np.exp(-epochs/10) + 0.5 * np.sin(epochs/2)

# Not decreasing (bug or LR too small)
loss_flat = 2.0 + 0.1 * np.random.randn(50)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(epochs, loss_good)
plt.title("Good Training ✓")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(epochs, loss_oscillate)
plt.title("Oscillating Loss ✗\n(LR too high)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(epochs, loss_flat)
plt.title("Not Decreasing ✗\n(Bug or LR too small)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### Exercise 6.3 Solution: Sanity Checks

**Before training, perform these checks:**

**Step 1: Overfit single batch (10 samples) - should reach ~0 loss**

```python
# Test on tiny batch
X_tiny = X_train[:10]
y_tiny = y_train[:10]

# Train for many epochs
for epoch in range(1000):
    loss = train_step(X_tiny, y_tiny)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Should reach near-zero loss
# If not, there's a bug in your implementation!
```

**Why this works:** With only 10 samples, the network should memorize them perfectly. If it can't, your training code has a bug.

**Step 2: Check loss on random predictions**

```python
# Completely random predictions
y_random = np.random.randint(0, num_classes, size=len(y_test))

# For binary classification with cross-entropy
expected_loss = -np.log(1/num_classes)  # Should be ~0.69 for 2 classes

print(f"Random predictions loss: {compute_loss(y_random):.4f}")
print(f"Expected (theoretical): {expected_loss:.4f}")

# Initial network loss should be close to random
```

**Why this matters:** If your initial loss is very different from random chance, something is wrong with initialization or loss calculation.

**Step 3: Disable regularization - loss should decrease**

```python
# Without regularization
model = MyNetwork(l2_lambda=0.0, dropout=0.0)
loss_no_reg = train(model, X_train, y_train)

# With regularization
model_reg = MyNetwork(l2_lambda=0.01, dropout=0.5)
loss_with_reg = train(model_reg, X_train, y_train)

# Training loss without regularization should be lower
assert loss_no_reg < loss_with_reg, "Regularization should increase training loss!"
```

**Step 4: Try tiny dataset - should memorize perfectly**

```python
# Use only 20 samples
X_tiny = X_train[:20]
y_tiny = y_train[:20]

# Train until perfect
model = train_until_perfect(X_tiny, y_tiny, max_epochs=5000)

accuracy = evaluate(model, X_tiny, y_tiny)
assert accuracy > 0.99, "Should perfectly memorize 20 samples!"

print(f"Memorized {len(X_tiny)} samples with {accuracy:.1%} accuracy ✓")
```

**Complete Sanity Check Script:**

```python
def sanity_checks(model, X_train, y_train, X_test, y_test):
    """Run all sanity checks before serious training"""

    print("="*50)
    print("SANITY CHECKS")
    print("="*50)

    # Check 1: Random predictions
    print("\n1. Random Prediction Loss:")
    y_random = np.random.randint(0, 2, size=len(y_test))
    random_acc = np.mean(y_random == y_test)
    print(f"   Random accuracy: {random_acc:.2%}")
    print(f"   Expected: ~50% for binary")

    # Check 2: Initial loss
    print("\n2. Initial Network Loss:")
    initial_loss = model.compute_loss(X_test[:100], y_test[:100])
    expected = -np.log(0.5)  # For binary
    print(f"   Initial loss: {initial_loss:.4f}")
    print(f"   Expected (~): {expected:.4f}")

    # Check 3: Overfit tiny batch
    print("\n3. Overfitting Tiny Batch:")
    X_tiny = X_train[:10]
    y_tiny = y_train[:10]

    for epoch in range(500):
        loss = model.train_step(X_tiny, y_tiny)
        if epoch % 100 == 0:
            acc = np.mean(model.predict(X_tiny) == y_tiny)
            print(f"   Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.1%}")

    if loss < 0.01:
        print("   ✓ PASS - Can overfit tiny batch")
    else:
        print("   ✗ FAIL - Cannot overfit tiny batch (bug!)")

    print("\n" + "="*50)

# Run checks
sanity_checks(model, X_train, y_train, X_test, y_test)
```

---

## Summary: Key Takeaways

**Backpropagation:**
- Chain rule is fundamental: ∂L/∂w = ∂L/∂y · ∂y/∂z · ∂z/∂w
- Work backwards from loss to inputs
- Cache forward pass values for backward pass
- Vanishing gradients are real - use ReLU!

**Training:**
- SGD: θ ← θ - α·∇L
- Mini-batches balance speed and stability
- Learning rate is critical - start small (0.001)
- Monitor both training and validation loss

**Debugging:**
- Always do gradient checking first
- Overfit tiny batch to verify implementation
- Check sanity: initial loss, random predictions
- Use proper initialization (Xavier/He)

**Common Issues:**
- Loss is NaN → Check for overflow, use gradient clipping
- Loss doesn't decrease → Lower learning rate, check gradients
- Overfitting → Regularization, dropout, more data
- Oscillating loss → Lower learning rate, increase batch size

Now you have complete solutions for all exercises! Practice these by hand to build deep intuition.
