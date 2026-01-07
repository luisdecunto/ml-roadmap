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

**[Continuing in next message with Parts 4-6...]**

---

## Quick Reference: Forward Pass Summary

**Single sample:**
```python
z = x @ W + b
a = activation(z)
```

**Batch of samples:**
```python
Z = X @ W + b  # Broadcasting handles batch dimension
A = activation(Z)
```

**Parameter counting:**
```
Total params = Σ(weights) + Σ(biases)
For layer i→j: weights = i×j, biases = j
```

Try the remaining exercises (Parts 4-6) on your own! The solutions follow the same pattern of detailed step-by-step calculations.
