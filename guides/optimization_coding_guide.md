# Optimization Algorithms - Coding Guide

**Time:** 6-8 hours
**Difficulty:** Intermediate
**Prerequisites:** Python, NumPy, calculus basics

## What You'll Build

Implement optimization algorithms from scratch:
1. Gradient Descent (vanilla)
2. Stochastic Gradient Descent (SGD)
3. Mini-batch SGD
4. SGD with Momentum
5. RMSprop
6. Adam optimizer
7. **Project:** Compare all optimizers on MNIST

---

## Project Setup

```bash
mkdir optimization-from-scratch
cd optimization-from-scratch

# Create files
touch optimizers.py
touch test_optimizers.py
touch visualize.py
touch mnist_comparison.py
touch requirements.txt
```

### requirements.txt
```
numpy==1.24.0
matplotlib==3.7.0
scipy==1.11.0
```

---

## Part 1: Gradient Descent Basics

### Theory

Gradient Descent updates parameters using:
```
θ_{t+1} = θ_t - α ∇f(θ_t)
```

Where:
- θ: parameters
- α: learning rate
- ∇f: gradient of loss function

### Implementation: Vanilla Gradient Descent

```python
# optimizers.py
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, grad_f, x0, learning_rate=0.01, num_iterations=100,
                     tolerance=1e-6, verbose=True):
    """
    Vanilla Gradient Descent

    Args:
        f: objective function to minimize
        grad_f: gradient of f
        x0: initial point (can be scalar or array)
        learning_rate: step size
        num_iterations: max iterations
        tolerance: stop if gradient norm < tolerance

    Returns:
        x_history: trajectory of x values
        f_history: trajectory of function values
    """
    x = np.array(x0, dtype=float)
    x_history = [x.copy()]
    f_history = [f(x)]

    for i in range(num_iterations):
        # Compute gradient
        grad = grad_f(x)

        # Check convergence
        if np.linalg.norm(grad) < tolerance:
            if verbose:
                print(f"Converged at iteration {i}")
            break

        # Update parameters
        x = x - learning_rate * grad

        # Record history
        x_history.append(x.copy())
        f_history.append(f(x))

        if verbose and (i % 10 == 0 or i == num_iterations - 1):
            print(f"Iter {i:3d}: f(x) = {f(x):8.4f}, ||grad|| = {np.linalg.norm(grad):8.4f}")

    return np.array(x_history), np.array(f_history)


# Test on simple quadratic: f(x,y) = x^2 + y^2
def quadratic(x):
    """f(x) = x^2 + y^2"""
    return np.sum(x**2)

def quadratic_grad(x):
    """∇f(x) = [2x, 2y]"""
    return 2 * x

# Run optimization
x0 = np.array([5.0, 5.0])
x_hist, f_hist = gradient_descent(quadratic, quadratic_grad, x0,
                                   learning_rate=0.1, num_iterations=50)

print(f"\nFinal x: {x_hist[-1]}")
print(f"Final f(x): {f_hist[-1]}")
```

### Visualization

```python
# visualize.py
import numpy as np
import matplotlib.pyplot as plt

def visualize_optimization_2d(f, x_history, title="Optimization Path"):
    """
    Visualize optimization trajectory on 2D contour plot

    Args:
        f: objective function (must accept (x,y) or array)
        x_history: array of shape (iterations, 2)
        title: plot title
    """
    # Create grid for contour plot
    x_hist = x_history[:, 0]
    y_hist = x_history[:, 1]

    x_min, x_max = x_hist.min() - 1, x_hist.max() + 1
    y_min, y_max = y_hist.min() - 1, y_hist.max() + 1

    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Compute function values on grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    # Plot
    plt.figure(figsize=(10, 8))

    # Contour plot
    levels = np.logspace(-1, 2, 20)
    contour = plt.contour(X, Y, Z, levels=levels, alpha=0.3)
    plt.clabel(contour, inline=True, fontsize=8)

    # Optimization path
    plt.plot(x_hist, y_hist, 'ro-', linewidth=2, markersize=8,
             label='Optimization Path', alpha=0.7)
    plt.plot(x_hist[0], y_hist[0], 'g*', markersize=20, label='Start')
    plt.plot(x_hist[-1], y_hist[-1], 'r*', markersize=20, label='End')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

# Example usage
if __name__ == "__main__":
    from optimizers import gradient_descent, quadratic, quadratic_grad

    x0 = np.array([5.0, 5.0])
    x_hist, f_hist = gradient_descent(quadratic, quadratic_grad, x0,
                                       learning_rate=0.1, num_iterations=50,
                                       verbose=False)

    visualize_optimization_2d(quadratic, x_hist, "Gradient Descent on f(x,y) = x² + y²")
```

---

## Part 2: Stochastic Gradient Descent (SGD)

### Theory

Instead of computing gradient on ALL data, use one random sample:
```
θ_{t+1} = θ_t - α ∇f_i(θ_t)
```

Where i is randomly sampled index.

### Implementation

```python
# optimizers.py (add this)

class SGD:
    """Stochastic Gradient Descent optimizer"""

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def step(self, params, grads):
        """
        Update parameters

        Args:
            params: list of parameter arrays
            grads: list of gradient arrays (same shape as params)
        """
        for param, grad in zip(params, grads):
            param -= self.lr * grad

# Test on simple linear regression
def sgd_linear_regression():
    """Train linear regression y = wx + b using SGD"""

    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y_true = 3 * X + 2 + 0.1 * np.random.randn(100, 1)

    # Initialize parameters
    w = np.random.randn(1, 1)
    b = np.zeros((1, 1))

    # Optimizer
    optimizer = SGD(learning_rate=0.01)

    # Training loop
    num_epochs = 100
    losses = []

    for epoch in range(num_epochs):
        # Shuffle data
        indices = np.random.permutation(len(X))

        epoch_loss = 0
        for i in indices:
            # Get single sample
            xi = X[i:i+1]
            yi = y_true[i:i+1]

            # Forward pass
            y_pred = xi @ w + b
            loss = ((y_pred - yi) ** 2).mean()
            epoch_loss += loss

            # Backward pass (compute gradients)
            error = y_pred - yi
            grad_w = xi.T @ error / len(xi)
            grad_b = error.mean()

            # Update parameters
            optimizer.step([w, b], [grad_w, np.array([[grad_b]])])

        avg_loss = epoch_loss / len(X)
        losses.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}, w = {w[0,0]:.4f}, b = {b[0,0]:.4f}")

    print(f"\nFinal: w = {w[0,0]:.4f} (true: 3.0), b = {b[0,0]:.4f} (true: 2.0)")

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SGD Training Loss')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    sgd_linear_regression()
```

---

## Part 3: Mini-batch SGD

### Implementation

```python
# optimizers.py (add this)

class MiniBatchSGD:
    """Mini-batch SGD with configurable batch size"""

    def __init__(self, learning_rate=0.01, batch_size=32):
        self.lr = learning_rate
        self.batch_size = batch_size

    def step(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.lr * grad

def minibatch_training(X, y, batch_size=32, num_epochs=50, learning_rate=0.01):
    """
    Train with mini-batches

    Args:
        X: input data (N, features)
        y: target data (N, 1)
        batch_size: size of each mini-batch
        num_epochs: number of training epochs
        learning_rate: step size
    """
    # Initialize
    w = np.random.randn(X.shape[1], 1) * 0.01
    b = np.zeros((1, 1))

    optimizer = MiniBatchSGD(learning_rate=learning_rate, batch_size=batch_size)

    N = len(X)
    losses = []

    for epoch in range(num_epochs):
        # Shuffle data
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_loss = 0
        num_batches = 0

        # Process mini-batches
        for i in range(0, N, batch_size):
            # Get batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Forward pass
            y_pred = X_batch @ w + b
            loss = ((y_pred - y_batch) ** 2).mean()
            epoch_loss += loss
            num_batches += 1

            # Backward pass
            error = y_pred - y_batch
            grad_w = X_batch.T @ error / len(X_batch)
            grad_b = error.mean()

            # Update
            optimizer.step([w, b], [grad_w, np.array([[grad_b]])])

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}")

    return w, b, losses

# Compare different batch sizes
def compare_batch_sizes():
    """Compare SGD with different batch sizes"""

    # Generate data
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    w_true = np.array([[1], [2], [-1], [0.5], [3]])
    y = X @ w_true + 1 + 0.1 * np.random.randn(1000, 1)

    batch_sizes = [1, 32, 128, 1000]  # 1=SGD, 1000=GD

    plt.figure(figsize=(12, 6))

    for bs in batch_sizes:
        _, _, losses = minibatch_training(X, y, batch_size=bs, num_epochs=50,
                                          learning_rate=0.01)
        label = f"Batch size = {bs}" if bs < 1000 else "Full batch (GD)"
        plt.plot(losses, label=label, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Effect of Batch Size on Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    compare_batch_sizes()
```

---

## Part 4: SGD with Momentum

### Theory

Momentum accumulates gradients:
```
v_t = β * v_{t-1} + (1-β) * ∇f(θ_t)
θ_{t+1} = θ_t - α * v_t
```

Helps escape local minima and accelerate convergence.

### Implementation

```python
# optimizers.py (add this)

class SGDMomentum:
    """SGD with Momentum"""

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = None

    def step(self, params, grads):
        # Initialize velocities on first call
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]

        # Update with momentum
        for i, (param, grad, velocity) in enumerate(zip(params, grads, self.velocities)):
            # v = β*v + (1-β)*grad
            velocity[:] = self.momentum * velocity + (1 - self.momentum) * grad

            # θ = θ - α*v
            param -= self.lr * velocity

# Test on Rosenbrock function (has narrow valley)
def rosenbrock(x):
    """Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    """Gradient of Rosenbrock"""
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])

def compare_momentum():
    """Compare SGD with and without momentum on Rosenbrock"""

    x0 = np.array([0.0, 0.0])
    num_iterations = 1000
    lr = 0.001

    # Without momentum
    x_hist_sgd = [x0.copy()]
    x = x0.copy()
    for i in range(num_iterations):
        grad = rosenbrock_grad(x)
        x = x - lr * grad
        x_hist_sgd.append(x.copy())

    # With momentum
    x_hist_mom = [x0.copy()]
    x = x0.copy()
    v = np.zeros(2)
    beta = 0.9
    for i in range(num_iterations):
        grad = rosenbrock_grad(x)
        v = beta * v + (1 - beta) * grad
        x = x - lr * v
        x_hist_mom.append(x.copy())

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Create contour
    x_range = np.linspace(-0.5, 1.5, 200)
    y_range = np.linspace(-0.5, 1.5, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rosenbrock(np.array([X[i, j], Y[i, j]]))

    for ax, x_hist, title in [(ax1, x_hist_sgd, "Vanilla SGD"),
                               (ax2, x_hist_mom, "SGD + Momentum")]:
        ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), alpha=0.3)
        x_hist = np.array(x_hist)
        ax.plot(x_hist[:, 0], x_hist[:, 1], 'r-', linewidth=1, alpha=0.7)
        ax.plot(x_hist[0, 0], x_hist[0, 1], 'go', markersize=10, label='Start')
        ax.plot(x_hist[-1, 0], x_hist[-1, 1], 'r*', markersize=15, label='End')
        ax.plot(1, 1, 'b*', markersize=15, label='Optimum')
        ax.set_title(f"{title}\nFinal: ({x_hist[-1, 0]:.3f}, {x_hist[-1, 1]:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_momentum()
```

---

## Part 5: RMSprop

### Theory

RMSprop uses adaptive learning rates:
```
s_t = β * s_{t-1} + (1-β) * (∇f)²
θ_{t+1} = θ_t - α * ∇f / (√s_t + ε)
```

### Implementation

```python
# optimizers.py (add this)

class RMSprop:
    """RMSprop optimizer"""

    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.s = None

    def step(self, params, grads):
        # Initialize squared gradients
        if self.s is None:
            self.s = [np.zeros_like(p) for p in params]

        for i, (param, grad, si) in enumerate(zip(params, grads, self.s)):
            # Accumulate squared gradient
            si[:] = self.beta * si + (1 - self.beta) * (grad ** 2)

            # Update parameters
            param -= self.lr * grad / (np.sqrt(si) + self.epsilon)
```

---

## Part 6: Adam Optimizer

### Theory

Adam combines momentum + RMSprop:
```
m_t = β₁ * m_{t-1} + (1-β₁) * ∇f      # First moment (momentum)
v_t = β₂ * v_{t-1} + (1-β₂) * (∇f)²   # Second moment (RMSprop)

m̂_t = m_t / (1 - β₁^t)                 # Bias correction
v̂_t = v_t / (1 - β₂^t)

θ_{t+1} = θ_t - α * m̂_t / (√v̂_t + ε)
```

### Implementation

```python
# optimizers.py (add this)

class Adam:
    """Adam optimizer"""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        # Initialize moments
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1

        for i, (param, grad, mi, vi) in enumerate(zip(params, grads, self.m, self.v)):
            # Update biased first moment
            mi[:] = self.beta1 * mi + (1 - self.beta1) * grad

            # Update biased second moment
            vi[:] = self.beta2 * vi + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = mi / (1 - self.beta1 ** self.t)
            v_hat = vi / (1 - self.beta2 ** self.t)

            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

---

## Part 7: MNIST Comparison Project

```python
# mnist_comparison.py

import numpy as np
import matplotlib.pyplot as plt
from optimizers import SGD, SGDMomentum, RMSprop, Adam

def load_mnist_subset():
    """Load a small subset of MNIST for quick testing"""
    # You can use sklearn or download manually
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    # Load MNIST
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')

    # Normalize
    X = X / 255.0

    # Convert labels to integers
    y = y.astype(int)

    # Use subset for faster training
    X_train, X_test, y_train, y_test = train_test_split(
        X[:10000], y[:10000], test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

def softmax(z):
    """Numerically stable softmax"""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    """Cross-entropy loss"""
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

def train_model(X_train, y_train, X_test, y_test, optimizer, num_epochs=20, batch_size=128):
    """
    Train simple neural network on MNIST

    Returns:
        train_losses: training loss history
        test_accs: test accuracy history
    """
    # Network architecture: 784 -> 128 -> 10
    input_size = 784
    hidden_size = 128
    output_size = 10

    # Initialize parameters
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))

    params = [W1, b1, W2, b2]

    # Convert labels to one-hot
    y_train_onehot = np.zeros((len(y_train), 10))
    y_train_onehot[np.arange(len(y_train)), y_train] = 1

    N = len(X_train)
    train_losses = []
    test_accs = []

    for epoch in range(num_epochs):
        # Shuffle
        indices = np.random.permutation(N)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_onehot[indices]

        epoch_loss = 0
        num_batches = 0

        # Mini-batches
        for i in range(0, N, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Forward pass
            z1 = X_batch @ W1 + b1
            a1 = np.maximum(0, z1)  # ReLU
            z2 = a1 @ W2 + b2
            a2 = softmax(z2)

            # Loss
            loss = cross_entropy_loss(a2, y_batch)
            epoch_loss += loss
            num_batches += 1

            # Backward pass
            dz2 = a2 - y_batch
            dW2 = a1.T @ dz2 / len(X_batch)
            db2 = np.sum(dz2, axis=0, keepdims=True) / len(X_batch)

            da1 = dz2 @ W2.T
            dz1 = da1 * (z1 > 0)  # ReLU derivative
            dW1 = X_batch.T @ dz1 / len(X_batch)
            db1 = np.sum(dz1, axis=0, keepdims=True) / len(X_batch)

            # Update
            optimizer.step(params, [dW1, db1, dW2, db2])

        # Record training loss
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # Compute test accuracy
        z1_test = X_test @ W1 + b1
        a1_test = np.maximum(0, z1_test)
        z2_test = a1_test @ W2 + b2
        a2_test = softmax(z2_test)
        predictions = np.argmax(a2_test, axis=1)
        accuracy = np.mean(predictions == y_test)
        test_accs.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, Test Acc = {accuracy*100:.2f}%")

    return train_losses, test_accs

def compare_all_optimizers():
    """Compare all optimizers on MNIST"""

    print("Loading MNIST...")
    X_train, X_test, y_train, y_test = load_mnist_subset()

    optimizers = {
        'SGD': SGD(learning_rate=0.1),
        'SGD+Momentum': SGDMomentum(learning_rate=0.1, momentum=0.9),
        'RMSprop': RMSprop(learning_rate=0.001),
        'Adam': Adam(learning_rate=0.001)
    }

    results = {}

    for name, optimizer in optimizers.items():
        print(f"\n{'='*60}")
        print(f"Training with {name}")
        print('='*60)

        losses, accs = train_model(X_train, y_train, X_test, y_test,
                                   optimizer, num_epochs=20, batch_size=128)
        results[name] = {'losses': losses, 'accs': accs}

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for name, data in results.items():
        ax1.plot(data['losses'], label=name, linewidth=2)
        ax2.plot(data['accs'], label=name, linewidth=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Test Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=150)
    print("\n✓ Saved comparison plot to 'optimizer_comparison.png'")
    plt.show()

    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for name, data in results.items():
        print(f"{name:15s}: Final Loss = {data['losses'][-1]:.4f}, "
              f"Final Acc = {data['accs'][-1]*100:.2f}%")

if __name__ == "__main__":
    compare_all_optimizers()
```

---

## Verification Checklist

- [ ] Gradient descent converges on quadratic function
- [ ] SGD trains linear regression successfully
- [ ] Mini-batch sizes (1, 32, 128) show expected behavior
- [ ] Momentum helps on Rosenbrock function
- [ ] RMSprop and Adam implemented correctly
- [ ] MNIST comparison runs and shows Adam/RMSprop perform best
- [ ] Visualizations generated successfully

---

## Expected Results

**On simple functions:**
- GD: Smooth, deterministic convergence
- SGD: Noisy but reaches solution
- Momentum: Faster on curved surfaces

**On MNIST:**
- SGD: ~92% accuracy, slow convergence
- SGD+Momentum: ~94% accuracy, faster
- RMSprop: ~95% accuracy, stable
- Adam: ~95-96% accuracy, fastest convergence

---

## Key Takeaways

1. **Learning rate matters** - Too high: diverges, too low: slow
2. **Batch size tradeoff** - Small: noisy gradients, large: smooth but slow
3. **Momentum helps** - Especially in ravines and saddle points
4. **Adaptive methods (Adam, RMSprop)** - Best general-purpose choice
5. **Bias correction** - Important in Adam for early iterations

---

## Resources

- [An Overview of Gradient Descent Optimization Algorithms](https://arxiv.org/abs/1609.04747) (Ruder 2016)
- [Adam paper](https://arxiv.org/abs/1412.6980)
- [CS231n Optimization Notes](http://cs231n.github.io/optimization-1/)
- [Understanding Deep Learning Ch 7](https://udlbook.github.io/udlbook/)
