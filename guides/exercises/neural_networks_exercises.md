# Neural Networks Exercises - Module 6

**Time:** 4-5 hours
**Difficulty:** Intermediate-Advanced
**Materials needed:** Paper, pencil, calculator, NumPy

Complete these exercises by hand first, then implement in NumPy. Solutions are in `guides/solutions/neural_networks_solutions.md`

---

## Part 1: Single Neuron (30 min)

### Exercise 1.1: Linear Neuron
Single neuron with weights w = [0.5, -0.3, 0.2], bias b = 0.1:

Input: x = [1.0, 2.0, 3.0]

1. Calculate z = w·x + b (weighted sum)
2. For linear activation (f(z) = z), what is the output?
3. How many parameters does this neuron have?
4. Implement in NumPy

### Exercise 1.2: Sigmoid Neuron
Same neuron as 1.1 but with sigmoid activation σ(z) = 1/(1 + e^(-z)):

1. Calculate z = w·x + b
2. Calculate output a = σ(z)
3. Calculate derivative σ'(z) = σ(z)(1 - σ(z))
4. Why is this derivative useful for backpropagation?

### Exercise 1.3: ReLU Neuron
Same weights, with ReLU activation f(z) = max(0, z):

1. Calculate output for x = [1, 2, 3]
2. Calculate output for x = [-1, -2, -3]
3. What is the derivative of ReLU?
4. What happens when z < 0? (dead neuron problem)

---

## Part 2: Activation Functions (35 min)

### Exercise 2.1: Comparing Activations
For z = [-2, -1, 0, 1, 2], calculate:

1. Sigmoid: σ(z) = 1/(1 + e^(-z))
2. Tanh: tanh(z) = (e^z - e^(-z))/(e^z + e^(-z))
3. ReLU: max(0, z)
4. Leaky ReLU: max(0.01z, z)

Plot or sketch each. What are the ranges?

### Exercise 2.2: Activation Derivatives
Calculate derivatives at z = 1:

1. d/dz sigmoid(z)
2. d/dz tanh(z) = 1 - tanh²(z)
3. d/dz ReLU(z)
4. Which has vanishing gradient problem?

### Exercise 2.3: Softmax
Logits z = [2.0, 1.0, 0.1]:

1. Calculate softmax: p(i) = exp(z_i) / Σexp(z_j)
2. Verify outputs sum to 1
3. Which class has highest probability?
4. What happens if you add constant to all logits?

### Exercise 2.4: Why Non-linearity?
Two-layer network with linear activations:

Layer 1: W₁ = [[1, 2], [3, 4]], b₁ = [0, 0]
Layer 2: W₂ = [[1, 1]], b₂ = [0]

1. Compute output for x = [1, 1]
2. Show this equals single layer: W₂W₁x
3. Why do we need non-linear activations?

---

## Part 3: Forward Pass (30 min)

### Exercise 3.1: Two-Layer Network
Network architecture:
- Input: 2 neurons
- Hidden: 3 neurons (sigmoid)
- Output: 1 neuron (sigmoid)

Weights:
- W₁ = [[0.5, -0.2], [0.3, 0.4], [-0.1, 0.6]], b₁ = [0.1, -0.2, 0.3]
- W₂ = [[0.7, -0.3, 0.5]], b₂ = [0.2]

Input: x = [1.0, 2.0]

Calculate step-by-step:
1. z₁ = W₁x + b₁
2. a₁ = σ(z₁)
3. z₂ = W₂a₁ + b₂
4. a₂ = σ(z₂) (final output)

### Exercise 3.2: Batch Processing
Same network as 3.1, batch of 2 inputs:
- X = [[1.0, 2.0], [0.5, 1.5]] (shape: 2×2)

1. Calculate Z₁ = XW₁ᵀ + b₁ (shape?)
2. Calculate A₁ = σ(Z₁)
3. Calculate final outputs
4. Why is batching useful?

### Exercise 3.3: Deep Network
3-layer network: 2 → 4 → 3 → 1

1. How many weight matrices?
2. What are the dimensions of each?
3. Total number of parameters?
4. Write forward pass equations

---

## Part 4: Backpropagation (45 min)

### Exercise 4.1: Single Neuron Gradient
Neuron: y = σ(wx + b), loss: L = ½(y - t)²

Given: w = 0.5, b = 0.1, x = 2.0, t = 1.0

1. Forward pass: calculate y
2. Calculate loss L
3. Calculate ∂L/∂y = (y - t)
4. Calculate ∂y/∂z where z = wx + b
5. Calculate ∂z/∂w = x
6. Chain rule: ∂L/∂w = ∂L/∂y · ∂y/∂z · ∂z/∂w

### Exercise 4.2: Two-Layer Backprop
Network: x → h (2 neurons) → y (1 neuron), all sigmoid

Weights: W₁ = [[0.5, 0.3], [0.2, 0.4]], W₂ = [[0.6, 0.7]]
Input: x = [1, 2], target: t = 1

Forward pass:
1. z₁ = W₁x, a₁ = σ(z₁)
2. z₂ = W₂a₁, y = σ(z₂)

Backward pass:
3. δ₂ = (y - t) · σ'(z₂)
4. ∂L/∂W₂ = δ₂ · a₁ᵀ
5. δ₁ = (W₂ᵀδ₂) ⊙ σ'(z₁)
6. ∂L/∂W₁ = δ₁ · xᵀ

### Exercise 4.3: Backprop with ReLU
Replace sigmoid with ReLU in Exercise 4.2:

1. Forward pass with ReLU
2. ReLU derivative: 1 if z > 0, else 0
3. Backward pass
4. Compare gradients with sigmoid version

### Exercise 4.4: Vanishing Gradient
Deep network with 10 layers, all sigmoid:

1. If each layer's gradient is ∂a/∂z ≈ 0.25 (sigmoid derivative max)
2. What is gradient at first layer? (0.25)^10
3. Why is this problematic?
4. How do ReLU and skip connections help?

---

## Part 5: Training (35 min)

### Exercise 5.1: Gradient Descent Update
Single neuron, MSE loss:

After computing gradients: ∂L/∂w = 0.3, ∂L/∂b = 0.2
Current: w = 0.5, b = 0.1
Learning rate: α = 0.1

1. Update rule: w_new = w - α · ∂L/∂w
2. Calculate new w and b
3. Perform 5 iterations (assume same gradients)
4. Does loss decrease?

### Exercise 5.2: Mini-batch Training
Dataset: 100 samples, batch size 10

1. How many batches per epoch?
2. Weight updates per epoch?
3. Compare with batch GD (1 update/epoch)
4. Compare with SGD (100 updates/epoch)

### Exercise 5.3: Learning Rate Effects
Initial weights close to optimum, gradient ≈ 0.1:

1. Update with α = 0.01
2. Update with α = 1.0
3. Update with α = 10.0
4. Which learning rate works best? Why?

---

## Part 6: Debugging NNs (35 min)

### Exercise 6.1: Gradient Checking
Network with w = 0.5, loss function L(w):

1. Compute analytical gradient: ∂L/∂w = 0.3
2. Compute numerical gradient: (L(w + ε) - L(w - ε))/(2ε) with ε = 1e-5
3. Calculate relative error: |analytical - numerical| / |analytical + numerical|
4. If error < 1e-7, gradients correct!

### Exercise 6.2: Detecting Bugs
Symptoms and causes:

1. Loss is NaN → What's wrong? (learning rate too high, exploding gradients)
2. Loss doesn't decrease → Check? (learning rate, gradients, data normalization)
3. Training acc 99%, val acc 60% → Problem? (overfitting)
4. Loss oscillates wildly → Fix? (reduce learning rate, smaller batches)

### Exercise 6.3: Sanity Checks
Before training full network:

1. Overfit single batch (10 samples) - should reach ~0 loss
2. Check loss on random predictions - should be ~-log(1/C) for C classes
3. Disable regularization - loss should decrease
4. Try tiny dataset (5 samples) - should memorize perfectly

---

## Challenge Problems (Optional)

### Challenge 1: XOR Problem
XOR truth table:
```
Input    Output
[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0
```

1. Show single layer (linear) can't solve XOR
2. Design 2-layer network that solves it
3. Initialize weights and run forward pass
4. Verify outputs match truth table

### Challenge 2: Implement Full Network
Implement from scratch (NumPy only):

1. Forward pass for arbitrary layers
2. Backward pass (backpropagation)
3. Train on simple dataset (make_moons or make_circles)
4. Achieve >90% accuracy

---

## NumPy Implementation

```python
import numpy as np

# Exercise 1.1 - Linear Neuron
w = np.array([0.5, -0.3, 0.2])
b = 0.1
x = np.array([1.0, 2.0, 3.0])
z = np.dot(w, x) + b
print(f"Linear output: {z}")

# Exercise 1.2 - Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

a = sigmoid(z)
print(f"Sigmoid output: {a}")
print(f"Sigmoid derivative: {sigmoid_derivative(z)}")

# Exercise 2.1 - Activation Functions
z = np.array([-2, -1, 0, 1, 2])
print("Sigmoid:", sigmoid(z))
print("Tanh:", np.tanh(z))
print("ReLU:", np.maximum(0, z))
print("Leaky ReLU:", np.maximum(0.01 * z, z))

# Exercise 2.3 - Softmax
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Subtract max for numerical stability
    return exp_z / np.sum(exp_z)

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Softmax: {probs}, Sum: {np.sum(probs)}")

# Exercise 3.1 - Two-Layer Network
W1 = np.array([[0.5, -0.2], [0.3, 0.4], [-0.1, 0.6]])
b1 = np.array([0.1, -0.2, 0.3])
W2 = np.array([[0.7, -0.3, 0.5]])
b2 = np.array([0.2])

x = np.array([1.0, 2.0])

# Forward pass
z1 = np.dot(W1, x) + b1
a1 = sigmoid(z1)
z2 = np.dot(W2, a1) + b2
a2 = sigmoid(z2)

print(f"Hidden layer: {a1}")
print(f"Output: {a2}")

# Exercise 4.1 - Single Neuron Gradient
w, b, x, t = 0.5, 0.1, 2.0, 1.0

# Forward
z = w * x + b
y = sigmoid(z)
L = 0.5 * (y - t)**2

# Backward
dL_dy = (y - t)
dy_dz = sigmoid_derivative(z)
dz_dw = x

dL_dw = dL_dy * dy_dz * dz_dw
dL_db = dL_dy * dy_dz

print(f"Loss: {L:.4f}")
print(f"∂L/∂w: {dL_dw:.4f}, ∂L/∂b: {dL_db:.4f}")

# Exercise 4.2 - Two-Layer Backpropagation
class SimpleNN:
    def __init__(self):
        self.W1 = np.array([[0.5, 0.3], [0.2, 0.4]])
        self.W2 = np.array([[0.6, 0.7]])
        self.b1 = np.array([0, 0])
        self.b2 = np.array([0])

    def forward(self, x):
        self.x = x
        self.z1 = np.dot(self.W1, x)
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.W2, self.a1)
        self.y = sigmoid(self.z2)
        return self.y

    def backward(self, t, lr=0.1):
        # Output layer
        delta2 = (self.y - t) * sigmoid_derivative(self.z2)
        dW2 = np.outer(delta2, self.a1)
        db2 = delta2

        # Hidden layer
        delta1 = np.dot(self.W2.T, delta2) * sigmoid_derivative(self.z1)
        dW1 = np.outer(delta1, self.x)
        db1 = delta1

        # Update
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        return dW1, dW2

nn = SimpleNN()
x = np.array([1, 2])
t = np.array([1])

for i in range(100):
    y = nn.forward(x)
    loss = 0.5 * (y - t)**2
    nn.backward(t)
    if i % 20 == 0:
        print(f"Iter {i}: Loss = {loss[0]:.4f}, Output = {y[0]:.4f}")

# Challenge 2 - Full Network
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.01
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        self.activations = [X]
        self.zs = []

        A = X
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            Z = np.dot(A, W.T) + b
            A = sigmoid(Z)
            self.zs.append(Z)
            self.activations.append(A)

        # Output layer
        Z = np.dot(A, self.weights[-1].T) + self.biases[-1]
        A = sigmoid(Z)
        self.zs.append(Z)
        self.activations.append(A)

        return A

    def backward(self, X, y, lr=0.1):
        m = X.shape[0]

        # Output layer gradient
        delta = (self.activations[-1] - y) * sigmoid_derivative(self.zs[-1])

        dW = np.dot(delta.T, self.activations[-2]) / m
        db = np.sum(delta, axis=0) / m

        self.weights[-1] -= lr * dW
        self.biases[-1] -= lr * db

        # Hidden layers
        for l in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[l+1]) * sigmoid_derivative(self.zs[l])
            dW = np.dot(delta.T, self.activations[l]) / m
            db = np.sum(delta, axis=0) / m
            self.weights[l] -= lr * dW
            self.biases[l] -= lr * db

# Test on XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 4, 1])
for epoch in range(1000):
    predictions = nn.forward(X)
    loss = np.mean((predictions - y)**2)
    nn.backward(X, y, lr=1.0)
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

print("\nFinal predictions:")
print(nn.forward(X))
```

---

## Tips for Success

1. **Work through by hand first** - Understanding beats memorization
2. **Check dimensions** - Matrix multiplication shapes must match
3. **Use chain rule systematically** - Write out each derivative
4. **Numerical gradient checking** - Verify backprop implementation
5. **Start small** - 2-3 neurons, then scale up
6. **Vectorize** - Batch operations much faster
7. **Debug incrementally** - Test each layer separately
8. **Visualize** - Draw computation graphs
