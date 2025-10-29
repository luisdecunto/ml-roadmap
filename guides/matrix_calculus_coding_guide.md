# Matrix Calculus Coding Guide

**Time:** 6-8 hours
**Difficulty:** Intermediate
**Prerequisites:** Linear algebra basics, Python/NumPy

## What You'll Build

By the end of this guide, you'll have implemented:
1. **Numerical gradient checker** - Verify analytical gradients
2. **Jacobian computation** - For vector-valued functions
3. **QR decomposition** - Using Gram-Schmidt process

These are fundamental tools for implementing and debugging neural networks.

---

## Part 1: Numerical Gradient Checking (2 hours)

### Theory

**Why numerical gradients?**
- Verify your analytical gradient implementations
- Catch bugs early before they compound
- Essential for debugging backpropagation

**Finite difference approximation:**
```
f'(x) ≈ [f(x + h) - f(x - h)] / (2h)
```

For small h (typically 1e-5), this approximates the true gradient.

### Implementation

**Step 1: Basic gradient checker**

```python
import numpy as np

def numerical_gradient(f, x, h=1e-5):
    """
    Compute numerical gradient of f at x using finite differences.

    Args:
        f: Function that takes x and returns scalar
        x: Point to evaluate gradient (can be array)
        h: Step size for finite differences

    Returns:
        grad: Numerical gradient (same shape as x)
    """
    grad = np.zeros_like(x, dtype=float)

    # Iterate over all indices in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index

        # Save original value
        old_value = x[idx]

        # Evaluate f(x + h)
        x[idx] = old_value + h
        fxh_plus = f(x)

        # Evaluate f(x - h)
        x[idx] = old_value - h
        fxh_minus = f(x)

        # Compute numerical gradient
        grad[idx] = (fxh_plus - fxh_minus) / (2 * h)

        # Restore original value
        x[idx] = old_value
        it.iternext()

    return grad
```

**Step 2: Relative error computation**

```python
def relative_error(analytical, numerical):
    """
    Compute relative error between analytical and numerical gradients.

    Relative error = |a - n| / max(|a|, |n|)

    Good if < 1e-7 (exact match)
    OK if < 1e-5 (probably correct)
    Bad if > 1e-3 (likely bug)
    """
    numerator = np.abs(analytical - numerical)
    denominator = np.maximum(np.abs(analytical), np.abs(numerical))

    # Avoid division by zero
    denominator = np.maximum(denominator, 1e-10)

    return numerator / denominator
```

**Step 3: Full gradient checking function**

```python
def check_gradient(f, grad_f, x, tolerance=1e-5):
    """
    Check analytical gradient against numerical gradient.

    Args:
        f: Function f(x) -> scalar
        grad_f: Gradient function grad_f(x) -> gradient array
        x: Point to check
        tolerance: Maximum acceptable relative error

    Returns:
        passed: Boolean indicating if check passed
        max_error: Maximum relative error found
    """
    # Compute gradients
    analytical = grad_f(x)
    numerical = numerical_gradient(f, x)

    # Compute relative errors
    errors = relative_error(analytical, numerical)
    max_error = np.max(errors)

    # Check if passed
    passed = max_error < tolerance

    print(f"Gradient check: {'PASSED' if passed else 'FAILED'}")
    print(f"Max relative error: {max_error:.2e}")

    if not passed:
        print(f"\nAnalytical gradient:\n{analytical}")
        print(f"\nNumerical gradient:\n{numerical}")
        print(f"\nRelative errors:\n{errors}")

    return passed, max_error
```

### Example Usage

```python
# Example 1: Quadratic function
def f(x):
    """f(x) = x^T @ x (sum of squares)"""
    return np.sum(x ** 2)

def grad_f(x):
    """Gradient: 2x"""
    return 2 * x

# Test
x = np.random.randn(5)
passed, error = check_gradient(f, grad_f, x)
# Should pass with error < 1e-7

# Example 2: More complex function
def f2(W):
    """f(W) = ||W||_F^2 (Frobenius norm squared)"""
    return np.sum(W ** 2)

def grad_f2(W):
    """Gradient: 2W"""
    return 2 * W

W = np.random.randn(3, 4)
passed, error = check_gradient(f2, grad_f2, W)
```

### Common Pitfalls

1. **Step size too large**: Use h=1e-5, not 1e-2
2. **Forgot to restore x**: Always restore original value after perturbing
3. **Shape mismatch**: Ensure analytical and numerical gradients have same shape
4. **In-place operations**: Be careful with operations that modify x

---

## Part 2: Jacobian Computation (2 hours)

### Theory

**Jacobian matrix** for function f: R^n → R^m:

```
J[i,j] = ∂f_i / ∂x_j
```

Shape: (m, n) where m = output dim, n = input dim

**Example:**
```
f([x1, x2]) = [x1^2 + x2, 2*x1*x2]

J = [[2*x1,  1    ],
     [2*x2,  2*x1 ]]
```

### Implementation

**Step 1: Numerical Jacobian**

```python
def numerical_jacobian(f, x, h=1e-5):
    """
    Compute Jacobian matrix of f at x using finite differences.

    Args:
        f: Function R^n -> R^m
        x: Input vector of shape (n,)
        h: Step size

    Returns:
        J: Jacobian matrix of shape (m, n)
    """
    n = x.shape[0]

    # Evaluate f at x to get output dimension
    fx = f(x)
    m = fx.shape[0] if fx.ndim > 0 else 1

    # Initialize Jacobian
    J = np.zeros((m, n))

    # Compute each column using finite differences
    for j in range(n):
        # Perturb x[j]
        x_plus = x.copy()
        x_plus[j] += h

        x_minus = x.copy()
        x_minus[j] -= h

        # Compute partial derivative
        J[:, j] = (f(x_plus) - f(x_minus)) / (2 * h)

    return J
```

**Step 2: Example functions with analytical Jacobians**

```python
def example_function_1(x):
    """f(x) = [x[0]^2 + x[1], 2*x[0]*x[1]]"""
    return np.array([x[0]**2 + x[1], 2*x[0]*x[1]])

def jacobian_1(x):
    """Analytical Jacobian of example_function_1"""
    J = np.array([[2*x[0], 1],
                  [2*x[1], 2*x[0]]])
    return J

# Test
x = np.array([3.0, 4.0])
J_numerical = numerical_jacobian(example_function_1, x)
J_analytical = jacobian_1(x)

print("Numerical Jacobian:")
print(J_numerical)
print("\nAnalytical Jacobian:")
print(J_analytical)
print("\nRelative error:")
print(relative_error(J_analytical, J_numerical))
```

**Step 3: Jacobian for neural network layer**

```python
def linear_layer(x, W, b):
    """
    Linear layer: f(x) = Wx + b

    Args:
        x: Input vector (n,)
        W: Weight matrix (m, n)
        b: Bias vector (m,)

    Returns:
        y: Output vector (m,)
    """
    return W @ x + b

def linear_layer_jacobian(x, W, b):
    """
    Jacobian of linear layer.

    For f(x) = Wx + b, Jacobian is just W!
    """
    return W

# Test
n, m = 5, 3
x = np.random.randn(n)
W = np.random.randn(m, n)
b = np.random.randn(m)

# Wrapper for numerical Jacobian
def f(x_input):
    return linear_layer(x_input, W, b)

J_numerical = numerical_jacobian(f, x)
J_analytical = linear_layer_jacobian(x, W, b)

print("Max error:", np.max(relative_error(J_analytical, J_numerical)))
# Should be < 1e-7
```

### Applications

**1. Sensitivity analysis**: How much does output change with input?
**2. Neural network backpropagation**: Chain rule uses Jacobians
**3. Optimization**: Second-order methods need Jacobian

---

## Part 3: QR Decomposition (3 hours)

### Theory

**QR Decomposition:** A = QR where
- Q is orthogonal (Q^T Q = I)
- R is upper triangular

**Used for:**
- Solving least squares problems
- Computing eigenvalues (QR algorithm)
- Stable numerical computation

**Gram-Schmidt Process:**
1. Start with columns of A: [a1, a2, ..., an]
2. Orthogonalize:
   - u1 = a1
   - u2 = a2 - proj_{u1}(a2)
   - u3 = a3 - proj_{u1}(a3) - proj_{u2}(a3)
   - ...
3. Normalize: qi = ui / ||ui||

### Implementation

**Step 1: Classical Gram-Schmidt**

```python
def gram_schmidt(A):
    """
    Classical Gram-Schmidt orthogonalization.

    Args:
        A: Input matrix (m, n) where m >= n

    Returns:
        Q: Orthogonal matrix (m, n)
        R: Upper triangular matrix (n, n)
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        # Start with j-th column of A
        v = A[:, j].copy()

        # Subtract projections onto previous columns
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v = v - R[i, j] * Q[:, i]

        # Normalize
        R[j, j] = np.linalg.norm(v)

        if R[j, j] < 1e-10:
            raise ValueError(f"Column {j} is linearly dependent")

        Q[:, j] = v / R[j, j]

    return Q, R
```

**Step 2: Modified Gram-Schmidt (more numerically stable)**

```python
def modified_gram_schmidt(A):
    """
    Modified Gram-Schmidt (more stable than classical).

    Args:
        A: Input matrix (m, n)

    Returns:
        Q: Orthogonal matrix (m, n)
        R: Upper triangular matrix (n, n)
    """
    m, n = A.shape
    Q = A.copy().astype(float)
    R = np.zeros((n, n))

    for j in range(n):
        # Compute norm before orthogonalization
        R[j, j] = np.linalg.norm(Q[:, j])

        if R[j, j] < 1e-10:
            raise ValueError(f"Column {j} is linearly dependent")

        # Normalize
        Q[:, j] = Q[:, j] / R[j, j]

        # Orthogonalize subsequent columns
        for k in range(j + 1, n):
            R[j, k] = Q[:, j] @ Q[:, k]
            Q[:, k] = Q[:, k] - R[j, k] * Q[:, j]

    return Q, R
```

**Step 3: Verification**

```python
def verify_qr(A, Q, R, tolerance=1e-10):
    """
    Verify QR decomposition is correct.

    Checks:
    1. A = QR
    2. Q^T Q = I (Q is orthogonal)
    3. R is upper triangular
    """
    m, n = A.shape

    # Check 1: A = QR
    reconstruction_error = np.max(np.abs(A - Q @ R))
    print(f"Reconstruction error ||A - QR||: {reconstruction_error:.2e}")

    # Check 2: Q^T Q = I
    QTQ = Q.T @ Q
    orthogonality_error = np.max(np.abs(QTQ - np.eye(n)))
    print(f"Orthogonality error ||Q^T Q - I||: {orthogonality_error:.2e}")

    # Check 3: R is upper triangular
    R_lower = np.tril(R, -1)  # Strictly lower triangular part
    triangular_error = np.max(np.abs(R_lower))
    print(f"Triangular error (max |R_lower|): {triangular_error:.2e}")

    # Overall check
    passed = (reconstruction_error < tolerance and
              orthogonality_error < tolerance and
              triangular_error < tolerance)

    print(f"\nQR decomposition: {'PASSED' if passed else 'FAILED'}")

    return passed

# Test
A = np.random.randn(5, 3)
Q, R = modified_gram_schmidt(A)
verify_qr(A, Q, R)
```

### Application: Least Squares

```python
def solve_least_squares_qr(A, b):
    """
    Solve least squares problem min ||Ax - b||^2 using QR.

    Solution: x = R^{-1} Q^T b

    Args:
        A: Design matrix (m, n) where m >= n
        b: Target vector (m,)

    Returns:
        x: Solution vector (n,)
    """
    Q, R = modified_gram_schmidt(A)

    # Compute Q^T b
    QTb = Q.T @ b

    # Solve Rx = Q^T b using back substitution
    x = np.linalg.solve(R, QTb)

    return x

# Example: Fit polynomial to noisy data
np.random.seed(42)

# Generate data: y = 2 + 3x + 0.5x^2 + noise
x_data = np.linspace(-1, 1, 20)
y_true = 2 + 3*x_data + 0.5*x_data**2
y_noisy = y_true + 0.1 * np.random.randn(20)

# Design matrix for quadratic fit
A = np.column_stack([np.ones_like(x_data), x_data, x_data**2])

# Solve using QR
coeffs = solve_least_squares_qr(A, y_noisy)
print(f"Fitted coefficients: {coeffs}")
print(f"True coefficients: [2, 3, 0.5]")

# Compare with NumPy
coeffs_numpy = np.linalg.lstsq(A, y_noisy, rcond=None)[0]
print(f"NumPy lstsq: {coeffs_numpy}")
print(f"Difference: {np.max(np.abs(coeffs - coeffs_numpy)):.2e}")
```

---

## Complete Example: Gradient Checking for Simple Network

```python
class SimpleNetwork:
    """Two-layer network for demonstration"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)

    def forward(self, x):
        """Forward pass"""
        self.x = x
        self.h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        self.y = self.h @ self.W2 + self.b2
        return self.y

    def backward(self, dy):
        """Backward pass - compute gradients"""
        # Output layer
        dW2 = self.h.T @ dy
        db2 = np.sum(dy, axis=0)
        dh = dy @ self.W2.T

        # ReLU backward
        dh[self.h <= 0] = 0

        # Hidden layer
        dW1 = self.x.T @ dh
        db1 = np.sum(dh, axis=0)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def get_params_vector(self):
        """Flatten all parameters into single vector"""
        return np.concatenate([
            self.W1.ravel(),
            self.b1.ravel(),
            self.W2.ravel(),
            self.b2.ravel()
        ])

    def set_params_vector(self, params):
        """Set parameters from flattened vector"""
        idx = 0

        # W1
        size = self.W1.size
        self.W1 = params[idx:idx+size].reshape(self.W1.shape)
        idx += size

        # b1
        size = self.b1.size
        self.b1 = params[idx:idx+size]
        idx += size

        # W2
        size = self.W2.size
        self.W2 = params[idx:idx+size].reshape(self.W2.shape)
        idx += size

        # b2
        size = self.b2.size
        self.b2 = params[idx:idx+size]

def check_network_gradients():
    """Check gradients of simple network"""
    # Create network
    net = SimpleNetwork(input_dim=3, hidden_dim=5, output_dim=2)

    # Sample data
    x = np.random.randn(4, 3)  # 4 samples
    y_true = np.random.randn(4, 2)

    # Define loss function
    def loss_fn(params):
        net.set_params_vector(params)
        y_pred = net.forward(x)
        return np.mean((y_pred - y_true) ** 2)

    # Get analytical gradients
    y_pred = net.forward(x)
    dy = 2 * (y_pred - y_true) / y_true.shape[0]
    grads = net.backward(dy)

    # Flatten analytical gradients
    grad_analytical = np.concatenate([
        grads['W1'].ravel(),
        grads['b1'].ravel(),
        grads['W2'].ravel(),
        grads['b2'].ravel()
    ])

    # Compute numerical gradient
    params = net.get_params_vector()
    grad_numerical = numerical_gradient(loss_fn, params)

    # Check
    error = relative_error(grad_analytical, grad_numerical)
    max_error = np.max(error)

    print(f"Gradient check for network:")
    print(f"Max relative error: {max_error:.2e}")
    print(f"Result: {'PASSED' if max_error < 1e-5 else 'FAILED'}")

    return max_error < 1e-5

# Run gradient check
check_network_gradients()
```

---

## Verification Checklist

- [ ] Numerical gradient checker works for scalar and vector functions
- [ ] Relative error computation handles edge cases
- [ ] Jacobian computation matches analytical for test functions
- [ ] QR decomposition satisfies A = QR
- [ ] Q is orthogonal (Q^T Q = I)
- [ ] R is upper triangular
- [ ] Least squares solution using QR matches NumPy
- [ ] Network gradient check passes with error < 1e-5

---

## Tips

1. **Always check gradients** before training
2. **Use small random inputs** for testing
3. **Modified Gram-Schmidt** is more stable than classical
4. **Save intermediate values** during forward pass for backward pass
5. **Check shapes** at every step

---

## Resources

- [CS231n: Gradient Checking](http://cs231n.github.io/neural-networks-3/#gradcheck)
- [Numerical Recipes](http://numerical.recipes/) - Advanced numerical methods
- Gilbert Strang: Linear Algebra - QR decomposition chapter
- [Matrix Calculus](http://www.matrixcalculus.org/) - For verifying derivatives

---

**Time breakdown:**
- Gradient checking: 2 hours
- Jacobian computation: 2 hours
- QR decomposition: 3 hours
- Testing and verification: 1 hour
