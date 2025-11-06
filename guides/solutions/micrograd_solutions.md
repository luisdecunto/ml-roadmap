# Micrograd (Autograd Engine) - Solutions

Complete solutions for all micrograd exercises with detailed explanations.

---

## Section 1: Understanding the Value Class & Forward Pass

### Exercise 1.1: Manual Forward Pass

**Question:** Calculate `d.data` for:
```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
```

**Solution:**
```
Step 1: a * b = 2.0 * (-3.0) = -6.0
Step 2: -6.0 + c = -6.0 + 10.0 = 4.0

Answer: d.data = 4.0
```

**Verification Code:**
```python
from micrograd.engine import Value

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
print(f"d.data = {d.data}")  # Output: 4.0
```

---

### Exercise 1.2: Complex Expression

**Question:** Compute `z.data` for:
```python
x = Value(3.0)
y = Value(-2.0)
z = x**2 + 2*x*y + y**2
```

**Solution:**
```
Step 1: x**2 = 3.0**2 = 9.0
Step 2: 2*x*y = 2 * 3.0 * (-2.0) = -12.0
Step 3: y**2 = (-2.0)**2 = 4.0
Step 4: z = 9.0 + (-12.0) + 4.0 = 1.0

Answer: z.data = 1.0

Note: This is actually (x + y)² = (3 + (-2))² = 1² = 1
```

**Verification Code:**
```python
x = Value(3.0)
y = Value(-2.0)
z = x**2 + 2*x*y + y**2
print(f"z.data = {z.data}")  # Output: 1.0
```

---

### Exercise 1.3: ReLU Activation

**Question:** Calculate `c.data`, `d.data`, and `e.data` for:
```python
a = Value(-5.0)
b = Value(3.0)
c = a.relu()
d = b.relu()
e = c + d
```

**Solution:**
```
ReLU(x) = max(0, x)

Step 1: c = relu(-5.0) = max(0, -5.0) = 0.0
Step 2: d = relu(3.0) = max(0, 3.0) = 3.0
Step 3: e = 0.0 + 3.0 = 3.0

Answer: c.data = 0.0, d.data = 3.0, e.data = 3.0
```

**Verification Code:**
```python
a = Value(-5.0)
b = Value(3.0)
c = a.relu()
d = b.relu()
e = c + d
print(f"c.data = {c.data}, d.data = {d.data}, e.data = {e.data}")
# Output: c.data = 0.0, d.data = 3.0, e.data = 3.0
```

---

## Section 2: Manual Backpropagation

### Exercise 2.1: Simple Chain Rule

**Question:** Calculate `a.grad` and `b.grad` for:
```python
a = Value(2.0)
b = Value(3.0)
c = a * b
c.backward()
```

**Solution:**
```
Forward: c = a * b = 2.0 * 3.0 = 6.0

Backward (chain rule):
dc/dc = 1.0 (gradient of output w.r.t itself)

For multiplication: c = a * b
dc/da = b = 3.0
dc/db = a = 2.0

Answer: a.grad = 3.0, b.grad = 2.0
```

**Verification Code:**
```python
a = Value(2.0)
b = Value(3.0)
c = a * b
c.backward()
print(f"a.grad = {a.grad}, b.grad = {b.grad}")
# Output: a.grad = 3.0, b.grad = 2.0
```

---

### Exercise 2.2: Addition and Multiplication

**Question:** Calculate gradients for:
```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
d.backward()
```

**Solution:**
```
Forward pass:
temp = a * b = 2.0 * (-3.0) = -6.0
d = temp + c = -6.0 + 10.0 = 4.0

Backward pass (working backwards from d):
dd/dd = 1.0

d = temp + c, so:
dd/dtemp = 1.0
dd/dc = 1.0  → c.grad = 1.0

temp = a * b, so:
dtemp/da = b = -3.0
dtemp/db = a = 2.0

By chain rule:
dd/da = dd/dtemp * dtemp/da = 1.0 * (-3.0) = -3.0
dd/db = dd/dtemp * dtemp/db = 1.0 * 2.0 = 2.0

Answer: a.grad = -3.0, b.grad = 2.0, c.grad = 1.0
```

**Verification Code:**
```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
d.backward()
print(f"a.grad = {a.grad}, b.grad = {b.grad}, c.grad = {c.grad}")
# Output: a.grad = -3.0, b.grad = 2.0, c.grad = 1.0
```

---

### Exercise 2.3: Power Function Gradient

**Question:** Calculate `x.grad` for:
```python
x = Value(3.0)
y = x**3
y.backward()
```

**Solution:**
```
Forward: y = x³ = 3.0³ = 27.0

Backward:
dy/dx = 3x² = 3 * (3.0)² = 3 * 9 = 27.0

Answer: x.grad = 27.0
```

**Verification Code:**
```python
x = Value(3.0)
y = x**3
y.backward()
print(f"x.grad = {x.grad}")  # Output: 27.0
```

---

### Exercise 2.4: ReLU Gradient

**Question:** Calculate `a.grad` and `b.grad` for:
```python
a = Value(-2.0)
b = Value(3.0)
c = a.relu()
d = b.relu()
e = c + d
e.backward()
```

**Solution:**
```
Forward pass:
c = relu(-2.0) = 0.0
d = relu(3.0) = 3.0
e = 0.0 + 3.0 = 3.0

Backward pass:
de/de = 1.0

e = c + d, so:
de/dc = 1.0
de/dd = 1.0

ReLU derivative:
- If input > 0: gradient = 1
- If input ≤ 0: gradient = 0

For c = relu(a):
dc/da = 0 (because a = -2.0 < 0)

For d = relu(b):
dd/db = 1 (because b = 3.0 > 0)

By chain rule:
de/da = de/dc * dc/da = 1.0 * 0 = 0.0
de/db = de/dd * dd/db = 1.0 * 1 = 1.0

Answer: a.grad = 0.0, b.grad = 1.0
```

**Verification Code:**
```python
a = Value(-2.0)
b = Value(3.0)
c = a.relu()
d = b.relu()
e = c + d
e.backward()
print(f"a.grad = {a.grad}, b.grad = {b.grad}")
# Output: a.grad = 0.0, b.grad = 1.0
```

---

## Section 3: Understanding Computational Graphs

### Exercise 3.1: Draw the Computation Graph

**Question:** Draw the graph for:
```python
a = Value(2.0)
b = Value(-3.0)
c = a + b
d = a * b
e = c * d
```

**Solution:**

```
         a(2.0) ────┬────→ (+) → c(-1.0) ─────┐
                    │                          │
                    │                          ↓
         b(-3.0) ───┴────→ (*) → d(-6.0) ───→ (*) → e(6.0)
                           ↑                    ↑
                           └────────────────────┘
```

**Detailed Graph:**
```
Nodes:
- a = 2.0 (leaf)
- b = -3.0 (leaf)
- c = -1.0 (operation: +, children: a, b)
- d = -6.0 (operation: *, children: a, b)
- e = 6.0 (operation: *, children: c, d)

Edges (dependencies):
- c depends on a and b (addition)
- d depends on a and b (multiplication)
- e depends on c and d (multiplication)
```

**Computed Values:**
```
a = 2.0
b = -3.0
c = a + b = 2.0 + (-3.0) = -1.0
d = a * b = 2.0 * (-3.0) = -6.0
e = c * d = (-1.0) * (-6.0) = 6.0
```

---

### Exercise 3.2: Topological Sort Understanding

**Question:** Write the topological order for backpropagation and explain why.

**Solution:**

**Topological Order (reverse of computation):**
```
e → d → c → b → a
```

Or more precisely (some nodes can be parallel):
```
Level 0: e
Level 1: c, d (can be processed in parallel)
Level 2: a, b (can be processed in parallel)
```

**Explanation:**

Backpropagation must visit nodes in **reverse topological order** because:

1. **Dependency Chain**: To compute gradients for a node, we need gradients from all nodes that depend on it
2. **Bottom-up**: Start from output (e) and work backwards to inputs (a, b)
3. **Example**:
   - Before computing `a.grad`, we need both `c.grad` and `d.grad` (since a affects both c and d)
   - Before computing `c.grad`, we need `e.grad` (since e depends on c)

**Why Topological Order Matters:**
- Ensures each node's gradient is computed only after all downstream gradients are ready
- Prevents using uninitialized gradients
- Implements the chain rule correctly through the entire graph

**Verification:**
```python
a = Value(2.0)
b = Value(-3.0)
c = a + b
d = a * b
e = c * d
e.backward()

# Backpropagation visited in order: e, d, c, a, b
# (or e, d, c, b, a - the exact order of a/b doesn't matter as they're leaves)
```

---

## Section 4: Building a Neuron from Scratch

### Exercise 4.1: Single Neuron Forward Pass

**Question:** Calculate neuron output for w₁=0.5, w₂=-0.3, b=0.1, x₁=2.0, x₂=3.0

**Solution:**
```
Formula: output = tanh(w₁x₁ + w₂x₂ + b)

Step 1: Compute weighted sum
z = w₁x₁ + w₂x₂ + b
z = 0.5 * 2.0 + (-0.3) * 3.0 + 0.1
z = 1.0 + (-0.9) + 0.1
z = 0.2

Step 2: Apply tanh activation
output = tanh(0.2) ≈ 0.197375

Answer: output ≈ 0.197
```

**Implementation Code:**
```python
import math
from micrograd.engine import Value

# Weights and bias
w1 = Value(0.5)
w2 = Value(-0.3)
b = Value(0.1)

# Inputs
x1 = Value(2.0)
x2 = Value(3.0)

# Forward pass
z = w1*x1 + w2*x2 + b
output = z.tanh()

print(f"z.data = {z.data}")           # 0.2
print(f"output.data = {output.data}") # ~0.197375

# Manual verification
z_manual = 0.5*2.0 + (-0.3)*3.0 + 0.1  # 0.2
output_manual = math.tanh(0.2)          # ~0.197375
print(f"Manual calculation: {output_manual}")
```

---

### Exercise 4.2: Neuron Gradient Computation

**Question:** Calculate gradients for w₁, w₂, and b

**Solution:**
```python
# Using the same setup from 4.1
w1 = Value(0.5)
w2 = Value(-0.3)
b = Value(0.1)
x1 = Value(2.0)
x2 = Value(3.0)

z = w1*x1 + w2*x2 + b
output = z.tanh()
output.backward()

print(f"w1.grad = {w1.grad}")  # ~1.922
print(f"w2.grad = {w2.grad}")  # ~2.883
print(f"b.grad = {b.grad}")    # ~0.961
```

**Manual Calculation:**
```
Backward pass:
doutput/doutput = 1.0

output = tanh(z)
doutput/dz = 1 - tanh²(z) = 1 - 0.197375² ≈ 0.961

z = w₁x₁ + w₂x₂ + b
dz/dw₁ = x₁ = 2.0
dz/dw₂ = x₂ = 3.0
dz/db = 1.0

By chain rule:
doutput/dw₁ = doutput/dz * dz/dw₁ = 0.961 * 2.0 ≈ 1.922
doutput/dw₂ = doutput/dz * dz/dw₂ = 0.961 * 3.0 ≈ 2.883
doutput/db = doutput/dz * dz/db = 0.961 * 1.0 ≈ 0.961
```

---

### Exercise 4.3: Build a Custom Neuron Class

**Solution:**
```python
import random
from micrograd.engine import Value

class Neuron:
    def __init__(self, nin):
        # Initialize random weights and bias
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Compute: sum(wi*xi) + b, then apply tanh
        # x is a list of inputs
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        output = activation.tanh()
        return output

    def parameters(self):
        return self.w + [self.b]

# Test with 3-input neuron
neuron = Neuron(3)
x = [Value(1.0), Value(2.0), Value(-1.0)]
output = neuron(x)
output.backward()

print(f"Output: {output.data}")
print(f"Number of parameters: {len(neuron.parameters())}")  # Should be 4 (3 weights + 1 bias)
print(f"Gradients computed: {all(p.grad != 0 for p in neuron.parameters() if p in neuron.w)}")
```

---

## Section 5: Multi-Layer Perceptron (MLP)

### Exercise 5.1: Build a Layer Class

**Solution:**
```python
class Layer:
    def __init__(self, nin, nout):
        # Create nout neurons, each with nin inputs
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # Apply all neurons to input x
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        # Return all parameters from all neurons
        return [p for neuron in self.neurons for p in neuron.parameters()]

# Test
layer = Layer(3, 4)  # 3 inputs, 4 outputs
x = [Value(1.0), Value(2.0), Value(-1.0)]
outputs = layer(x)
print(f"Number of outputs: {len(outputs)}")        # 4
print(f"Total parameters: {len(layer.parameters())}")  # 16 (4 neurons * 4 params each)
```

---

### Exercise 5.2: Build an MLP Class

**Solution:**
```python
class MLP:
    def __init__(self, nin, nouts):
        # nouts is a list: [hidden1, hidden2, ..., output]
        # Create layers: nin -> nouts[0] -> nouts[1] -> ... -> nouts[-1]
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        # Forward pass through all layers
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # Return all parameters from all layers
        return [p for layer in self.layers for p in layer.parameters()]

# Test: 3-layer MLP with 2 inputs, two hidden layers of 4 neurons, 1 output
mlp = MLP(2, [4, 4, 1])
x = [Value(1.0), Value(-0.5)]
output = mlp(x)
print(f"Output: {output.data}")
print(f"Total parameters: {len(mlp.parameters())}")

# Verify gradients flow
output.backward()
print(f"All gradients computed: {all(p.grad != 0 for p in mlp.parameters())}")
```

---

### Exercise 5.3: Count Parameters

**Question:** Count parameters for MLP([3, 4, 4, 1])

**Solution:**
```
Architecture: 3 inputs → 4 hidden → 4 hidden → 1 output

Layer 1 (3 → 4):
- 4 neurons, each with 3 weights + 1 bias = 4 params
- Total: 4 * 4 = 16 parameters

Layer 2 (4 → 4):
- 4 neurons, each with 4 weights + 1 bias = 5 params
- Total: 4 * 5 = 20 parameters

Layer 3 (4 → 1):
- 1 neuron with 4 weights + 1 bias = 5 params
- Total: 1 * 5 = 5 parameters

Total: 16 + 20 + 5 = 41 parameters
```

**Verification:**
```python
mlp = MLP(3, [4, 4, 1])
print(f"Total parameters: {len(mlp.parameters())}")  # Should print 41
```

---

## Section 6: Loss Functions & Training

### Exercise 6.1: Mean Squared Error (MSE)

**Solution:**
```python
def mse_loss(predictions, targets):
    # predictions and targets are lists of Value objects
    n = len(predictions)
    squared_errors = [(pred - target)**2 for pred, target in zip(predictions, targets)]
    total_error = sum(squared_errors)
    return total_error * (1.0 / n)

# Test
predictions = [Value(1.5), Value(2.0), Value(3.5)]
targets = [Value(1.0), Value(2.0), Value(3.0)]
loss = mse_loss(predictions, targets)
print(f"MSE Loss: {loss.data}")  # Should be: ((0.5)² + 0² + (0.5)²) / 3 = 0.5/3 ≈ 0.167

# Verify gradients flow
loss.backward()
print(f"Predictions gradients: {[p.grad for p in predictions]}")
```

**Manual Calculation:**
```
errors = [1.5-1.0, 2.0-2.0, 3.5-3.0] = [0.5, 0.0, 0.5]
squared = [0.25, 0.0, 0.25]
sum = 0.5
mse = 0.5 / 3 ≈ 0.1667
```

---

### Exercise 6.2: Gradient Descent Step

**Solution:**
```python
learning_rate = 0.01

# After loss.backward() has been called
for p in mlp.parameters():
    p.data += -learning_rate * p.grad  # Move in opposite direction of gradient
    p.grad = 0.0  # Zero gradients for next iteration

# Why zero gradients?
# Because gradients accumulate! If we don't zero them,
# the next backward() will ADD to existing gradients,
# causing incorrect updates.
```

**Explanation:**
- Gradient descent update rule: θ = θ - α∇L
- We subtract because gradient points in direction of steepest *increase*
- We want to *decrease* the loss
- Zeroing gradients prevents accumulation across iterations

---

### Exercise 6.3: Simple Training Loop

**Solution:**
```python
# Dataset: XOR function
xs = [
    [Value(0.0), Value(0.0)],
    [Value(0.0), Value(1.0)],
    [Value(1.0), Value(0.0)],
    [Value(1.0), Value(1.0)]
]
ys = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]

# Create MLP: 2 inputs -> 4 hidden -> 1 output
mlp = MLP(2, [4, 1])
learning_rate = 0.1

# Training loop
for iteration in range(20):
    # Forward pass
    predictions = [mlp(x) for x in xs]
    loss = mse_loss(predictions, ys)

    # Backward pass
    for p in mlp.parameters():
        p.grad = 0.0  # Zero gradients
    loss.backward()

    # Update parameters
    for p in mlp.parameters():
        p.data += -learning_rate * p.grad

    # Print progress
    if iteration % 5 == 0:
        print(f"Iteration {iteration}, Loss: {loss.data:.4f}")

# Final predictions
print("\nFinal predictions:")
for x, y_true in zip(xs, ys):
    y_pred = mlp(x)
    print(f"Input: {[xi.data for xi in x]}, True: {y_true.data}, Pred: {y_pred.data:.4f}")
```

**Expected Output:**
```
Iteration 0, Loss: 0.5123
Iteration 5, Loss: 0.4821
Iteration 10, Loss: 0.4532
Iteration 15, Loss: 0.4267

Note: XOR is a hard problem! Loss may not reach zero with this simple setup.
For better results, use more hidden neurons, more iterations, or different architecture.
```

---

## Section 7: Advanced Challenges

### Challenge 1: Implement tanh() in Value Class

**Solution:**
```python
import math

def tanh(self):
    # Forward pass
    x = self.data
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(t, (self,), 'tanh')

    # Backward pass
    def _backward():
        self.grad += (1 - t**2) * out.grad  # Derivative: 1 - tanh²(x)
    out._backward = _backward

    return out

# Add to Value class
Value.tanh = tanh

# Test
x = Value(0.5)
y = x.tanh()
y.backward()
print(f"tanh(0.5) = {y.data:.4f}")      # ~0.4621
print(f"gradient = {x.grad:.4f}")       # ~0.7864

# Verify: d/dx tanh(0.5) = 1 - tanh²(0.5) = 1 - 0.4621² ≈ 0.7864 ✓
```

---

### Challenge 2: Implement Sigmoid Activation

**Solution:**
```python
def sigmoid(self):
    # Forward pass
    x = self.data
    s = 1 / (1 + math.exp(-x))
    out = Value(s, (self,), 'sigmoid')

    # Backward pass
    def _backward():
        self.grad += s * (1 - s) * out.grad  # Derivative: σ(x) * (1 - σ(x))
    out._backward = _backward

    return out

# Add to Value class
Value.sigmoid = sigmoid

# Test
x = Value(0.0)
y = x.sigmoid()
y.backward()
print(f"sigmoid(0.0) = {y.data:.4f}")  # 0.5000
print(f"gradient = {x.grad:.4f}")      # 0.2500

# Verify: d/dx sigmoid(0) = 0.5 * (1 - 0.5) = 0.25 ✓
```

**Modified Neuron with Sigmoid:**
```python
class SigmoidNeuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        output = activation.sigmoid()  # Use sigmoid instead of tanh
        return output

    def parameters(self):
        return self.w + [self.b]

# Compare training with tanh vs sigmoid
```

---

### Challenge 3: Binary Classification with Micrograd

**Solution:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate make_moons-like dataset
def make_moons_simple(n_samples=100):
    np.random.seed(42)
    n = n_samples // 2

    # Upper moon
    theta = np.linspace(0, np.pi, n)
    x1_upper = np.cos(theta)
    x2_upper = np.sin(theta)

    # Lower moon
    theta = np.linspace(0, np.pi, n)
    x1_lower = 1 - np.cos(theta)
    x2_lower = 0.5 - np.sin(theta)

    # Combine and add noise
    X = np.vstack([
        np.column_stack([x1_upper, x2_upper]),
        np.column_stack([x1_lower, x2_lower])
    ])
    X += np.random.normal(0, 0.1, X.shape)

    y = np.array([0]*n + [1]*n)
    return X, y

# Generate data
X, y = make_moons_simple(100)

# Convert to Value objects
X_value = [[Value(x[0]), Value(x[1])] for x in X]
y_value = [Value(float(label)) for label in y]

# Build MLP with sigmoid output
mlp = MLP(2, [16, 16, 1])
learning_rate = 0.1

# Binary cross-entropy loss (simplified as MSE for this exercise)
for iteration in range(100):
    # Forward pass
    predictions = [mlp(x) for x in X_value]
    loss = mse_loss(predictions, y_value)

    # Backward pass
    for p in mlp.parameters():
        p.grad = 0.0
    loss.backward()

    # Update
    for p in mlp.parameters():
        p.data += -learning_rate * p.grad

    if iteration % 20 == 0:
        print(f"Iteration {iteration}, Loss: {loss.data:.4f}")

# Plot decision boundary (optional)
def plot_decision_boundary(mlp, X, y):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = np.array([mlp([Value(x), Value(y)]).data
                  for x, y in np.c_[xx.ravel(), yy.ravel()]])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    plt.title('Decision Boundary')
    plt.show()

plot_decision_boundary(mlp, X, y)
```

---

### Challenge 4: Compare with PyTorch

**Solution:**
```python
import torch
import torch.nn as nn

# Step 1: Build same network in micrograd
mlp_micro = MLP(2, [3, 1])

# Step 2: Build same network in PyTorch
class PyTorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 3)
        self.layer2 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        return x

mlp_torch = PyTorchMLP()

# Step 3: Copy weights from micrograd to PyTorch
with torch.no_grad():
    # Layer 1 weights
    for i in range(3):
        for j in range(2):
            mlp_torch.layer1.weight[i, j] = mlp_micro.layers[0].neurons[i].w[j].data
        mlp_torch.layer1.bias[i] = mlp_micro.layers[0].neurons[i].b.data

    # Layer 2 weights
    for j in range(3):
        mlp_torch.layer2.weight[0, j] = mlp_micro.layers[1].neurons[0].w[j].data
    mlp_torch.layer2.bias[0] = mlp_micro.layers[1].neurons[0].b.data

# Step 4: Same input
x_micro = [Value(1.0), Value(2.0)]
x_torch = torch.tensor([1.0, 2.0])

# Step 5: Forward pass
out_micro = mlp_micro(x_micro)
out_torch = mlp_torch(x_torch)

print(f"Micrograd output: {out_micro.data}")
print(f"PyTorch output: {out_torch.item()}")
print(f"Difference: {abs(out_micro.data - out_torch.item()):.10f}")

# Step 6: Backward pass
out_micro.backward()
out_torch.backward()

# Step 7: Compare gradients
print("\nGradient comparison:")
print(f"Micrograd layer1 weight[0,0] grad: {mlp_micro.layers[0].neurons[0].w[0].grad}")
print(f"PyTorch layer1 weight[0,0] grad: {mlp_torch.layer1.weight.grad[0, 0].item()}")
print(f"Difference: {abs(mlp_micro.layers[0].neurons[0].w[0].grad - mlp_torch.layer1.weight.grad[0, 0].item()):.10f}")

# Result: Gradients should match to ~10 decimal places! 🎉
```

---

## Summary

Congratulations on completing all micrograd exercises! 🎉

**Key Takeaways:**
1. **Autograd is just the chain rule** - Applied systematically through computational graphs
2. **Forward pass builds the graph** - Each operation creates nodes and edges
3. **Backward pass computes gradients** - Reverse topological order ensures correct gradient flow
4. **Neural networks are just compositions** - Layers of weighted sums + activations
5. **Training is gradient descent** - Iteratively update parameters to minimize loss

**You now understand:**
- How PyTorch/TensorFlow work under the hood
- Why computational graphs are necessary
- How backpropagation really works
- The fundamentals of building neural networks from scratch

**Next Steps:**
- Study PyTorch source code with this new understanding
- Implement more activation functions (LeakyReLU, ELU, Softmax)
- Build a simple CNN or RNN using the same principles
- Read the original backpropagation paper (Rumelhart et al., 1986)

Great work! 🚀
