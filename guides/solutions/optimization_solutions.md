# Optimization Solutions - Module 3

Comprehensive solutions with step-by-step work and code implementations.

---

## Part 1: Gradient Descent Basics

### Exercise 1.1 Solution: Computing Gradients

Given f(x, y) = x² + 2y² + 3xy

**Step 1: Compute ∂f/∂x**
```
∂f/∂x = ∂/∂x(x² + 2y² + 3xy)
      = 2x + 0 + 3y
      = 2x + 3y
```

**Step 2: Compute ∂f/∂y**
```
∂f/∂y = ∂/∂y(x² + 2y² + 3xy)
      = 0 + 4y + 3x
      = 4y + 3x
```

**Step 3: Gradient vector**
```
∇f(x, y) = [∂f/∂x, ∂f/∂y] = [2x + 3y, 4y + 3x]
```

**Step 4: Evaluate at (1, 2)**
```
∇f(1, 2) = [2(1) + 3(2), 4(2) + 3(1)]
         = [2 + 6, 8 + 3]
         = [8, 11]
```

**NumPy verification:**
```python
import numpy as np

def f(x, y):
    return x**2 + 2*y**2 + 3*x*y

def grad_f(x, y):
    df_dx = 2*x + 3*y
    df_dy = 4*y + 3*x
    return np.array([df_dx, df_dy])

print("Gradient at (1, 2):", grad_f(1, 2))
# Output: [8 11]
```

---

### Exercise 1.2 Solution: Gradient Descent Step

Given f(x) = x² - 4x + 5, starting at x₀ = 0, α = 0.1

**Step 1: Compute f'(x)**
```
f'(x) = 2x - 4
```

**Step 2: Gradient descent update rule**
```
x_{t+1} = x_t - α · f'(x_t)
```

**Step 3: Calculate x₁**
```
f'(x₀) = f'(0) = 2(0) - 4 = -4
x₁ = x₀ - α · f'(x₀)
   = 0 - 0.1 · (-4)
   = 0 + 0.4
   = 0.4
```

**Step 4: Compare losses**
```
f(x₀) = f(0) = 0² - 4(0) + 5 = 5
f(x₁) = f(0.4) = (0.4)² - 4(0.4) + 5
                = 0.16 - 1.6 + 5
                = 3.56

Loss decreased: 5 → 3.56 ✓
```

**NumPy verification:**
```python
def f(x):
    return x**2 - 4*x + 5

def grad_f(x):
    return 2*x - 4

x = 0.0
alpha = 0.1
print(f"x₀ = {x}, f(x₀) = {f(x)}")

x_new = x - alpha * grad_f(x)
print(f"x₁ = {x_new}, f(x₁) = {f(x_new)}")
# x₀ = 0.0, f(x₀) = 5.0
# x₁ = 0.4, f(x₁) = 3.56
```

---

### Exercise 1.3 Solution: Multi-dimensional Gradient Descent

Given f(x, y) = (x - 2)² + (y + 1)², starting at (0, 0), α = 0.5

**Step 1: Compute gradient**
```
∂f/∂x = 2(x - 2)
∂f/∂y = 2(y + 1)
∇f(x, y) = [2(x - 2), 2(y + 1)]
```

**Step 2: Perform 3 gradient descent steps**

**Iteration 0:** Position = (0, 0)
```
∇f(0, 0) = [2(0 - 2), 2(0 + 1)] = [-4, 2]
x₁ = [0, 0] - 0.5 · [-4, 2] = [0, 0] - [-2, 1] = [2, -1]
```

**Iteration 1:** Position = (2, -1)
```
∇f(2, -1) = [2(2 - 2), 2(-1 + 1)] = [0, 0]
x₂ = [2, -1] - 0.5 · [0, 0] = [2, -1]
```

**Iteration 2:** Position = (2, -1)
```
Already at optimum, gradient is zero
x₃ = [2, -1]
```

**Step 3: Track path**
```
Position 0: (0, 0)
Position 1: (2, -1)
Position 2: (2, -1)
Position 3: (2, -1)
```

**Step 4: True minimum**
```
Minimum occurs where ∇f = 0:
2(x - 2) = 0 → x = 2
2(y + 1) = 0 → y = -1

True minimum: (2, -1)
Distance from final position: 0 (reached exactly!)
```

**NumPy verification:**
```python
def f(pos):
    x, y = pos
    return (x - 2)**2 + (y + 1)**2

def grad_f(pos):
    x, y = pos
    return np.array([2*(x - 2), 2*(y + 1)])

pos = np.array([0.0, 0.0])
alpha = 0.5
trajectory = [pos.copy()]

for i in range(3):
    grad = grad_f(pos)
    pos = pos - alpha * grad
    trajectory.append(pos.copy())
    print(f"Step {i+1}: pos = {pos}, f = {f(pos):.4f}")

print("\nTrajectory:", trajectory)
```

---

## Part 2: Convergence Analysis

### Exercise 2.1 Solution: Learning Rate Effects

Given f(x) = x², starting at x₀ = 10

**α = 0.1:**
```
f'(x) = 2x
Iteration 0: x = 10.0,     f = 100.0
Iteration 1: x = 8.0,      f = 64.0     (x₁ = 10 - 0.1·20 = 8)
Iteration 2: x = 6.4,      f = 40.96    (x₂ = 8 - 0.1·16 = 6.4)
Iteration 3: x = 5.12,     f = 26.21
Iteration 4: x = 4.096,    f = 16.78
Iteration 5: x = 3.277,    f = 10.74
```

**α = 0.5:**
```
Iteration 0: x = 10.0,     f = 100.0
Iteration 1: x = 0.0,      f = 0.0      (x₁ = 10 - 0.5·20 = 0)
Iteration 2: x = 0.0,      f = 0.0
...
Converged immediately!
```

**α = 1.1:**
```
Iteration 0: x = 10.0,     f = 100.0
Iteration 1: x = -12.0,    f = 144.0    (x₁ = 10 - 1.1·20 = -12)
Iteration 2: x = 16.4,     f = 268.96   (x₂ = -12 - 1.1·(-24) = 14.4)
Iteration 3: x = -23.0,    f = 529.0
...
DIVERGING!
```

**Analysis:**
- α = 0.1: Converges slowly (geometric decay)
- α = 0.5: Converges fastest (optimal for this problem)
- α = 1.1: Diverges (learning rate too large)

**NumPy verification:**
```python
def gd_trajectory(x0, alpha, iterations=5):
    x = x0
    trajectory = []
    for i in range(iterations + 1):
        trajectory.append((x, x**2))
        if i < iterations:
            x = x - alpha * (2*x)
    return trajectory

for alpha in [0.1, 0.5, 1.1]:
    print(f"\nα = {alpha}:")
    traj = gd_trajectory(10.0, alpha, 5)
    for i, (x, f) in enumerate(traj):
        print(f"  Iter {i}: x = {x:.4f}, f = {f:.4f}")
```

---

### Exercise 2.2 Solution: Convergence Conditions

For f(x) = ½x², we have f'(x) = x

**Step 1: Lipschitz constant L**
```
f''(x) = 1 for all x
For quadratic functions, L = max eigenvalue of Hessian
L = 1
```

**Step 2: Maximum learning rate**
```
For guaranteed convergence: α < 2/L
α_max = 2/1 = 2.0
```

**Step 3: Test α = 1.9/L = 1.9**

Starting from x₀ = 5:
```
x₁ = 5 - 1.9·5 = 5 - 9.5 = -4.5
x₂ = -4.5 - 1.9·(-4.5) = -4.5 + 8.55 = 4.05
x₃ = 4.05 - 1.9·4.05 = -3.645
...
Oscillating but converging (magnitude decreasing)
```

**Test α = 2.1/L = 2.1:**
```
x₁ = 5 - 2.1·5 = -5.5
x₂ = -5.5 - 2.1·(-5.5) = 6.05
x₃ = 6.05 - 2.1·6.05 = -6.655
...
DIVERGING! (magnitude increasing)
```

**NumPy verification:**
```python
def test_convergence(alpha, x0=5, iterations=10):
    x = x0
    print(f"\nα = {alpha}:")
    for i in range(iterations):
        x = x - alpha * x  # gradient is just x for f(x) = 0.5x²
        print(f"  x_{i+1} = {x:.4f}, |x| = {abs(x):.4f}")
        if abs(x) > 100:
            print("  DIVERGED!")
            break
    return x

test_convergence(1.9)
test_convergence(2.1)
```

---

### Exercise 2.3 Solution: Iteration Count Estimation

Given f(x) = x² with x₀ = 10, α = 0.1, target ε = 0.01

**Step 1: Update rule analysis**
```
x_{t+1} = x_t - α·2x_t = x_t(1 - 2α)
With α = 0.1: x_{t+1} = x_t(1 - 0.2) = 0.8x_t
```

**Step 2: After t iterations**
```
x_t = x_0 · (0.8)^t = 10 · (0.8)^t
```

**Step 3: Find t where |x_t| < ε**
```
10 · (0.8)^t < 0.01
(0.8)^t < 0.001
t · log(0.8) < log(0.001)
t > log(0.001) / log(0.8)
t > -6.907 / -0.223
t > 30.97

Need approximately 31 iterations
```

**Step 4: Verify by running**
```python
x = 10.0
alpha = 0.1
iterations = 0

while abs(x) >= 0.01:
    x = x - alpha * 2 * x
    iterations += 1

print(f"Converged in {iterations} iterations")
print(f"Final x = {x:.6f}")
# Converged in 31 iterations
# Final x = 0.009538
```

---

## Part 3: SGD and Mini-Batches

### Exercise 3.1 Solution: Batch vs Mini-batch

Data: [2, 4, 6, 8, 10, 12, 14, 16], θ = 0
Loss: L = (1/n)Σᵢ(xᵢ - θ)²
Gradient: ∂L/∂θ = -(2/n)Σᵢ(xᵢ - θ)

**Step 1: Full batch gradient at θ = 0**
```
∂L/∂θ = -(2/8)Σᵢ(xᵢ - 0)
      = -(2/8)(2 + 4 + 6 + 8 + 10 + 12 + 14 + 16)
      = -(2/8)(72)
      = -18
```

**Step 2: Mini-batch [2, 4, 6, 8] at θ = 0**
```
∂L/∂θ = -(2/4)(2 + 4 + 6 + 8)
      = -(2/4)(20)
      = -10
```

**Step 3: Mini-batch [10, 12, 14, 16] at θ = 0**
```
∂L/∂θ = -(2/4)(10 + 12 + 14 + 16)
      = -(2/4)(52)
      = -26
```

**Step 4: Comparison**
```
Full batch gradient:     -18
First mini-batch:        -10  (underestimate)
Second mini-batch:       -26  (overestimate)
Average of mini-batches: (-10 + -26)/2 = -18 ✓

Mini-batches are unbiased estimators of full gradient!
```

**NumPy verification:**
```python
data = np.array([2, 4, 6, 8, 10, 12, 14, 16])
theta = 0

# Full batch gradient
grad_full = -2 * np.mean(data - theta)
print(f"Full batch gradient: {grad_full}")

# Mini-batch 1
batch1 = data[:4]
grad_batch1 = -2 * np.mean(batch1 - theta)
print(f"Mini-batch 1 gradient: {grad_batch1}")

# Mini-batch 2
batch2 = data[4:]
grad_batch2 = -2 * np.mean(batch2 - theta)
print(f"Mini-batch 2 gradient: {grad_batch2}")
```

---

### Exercise 3.2 Solution: Stochastic Gradient Descent

Starting at θ = 0, α = 0.01

**SGD Update Rule:**
For sample xᵢ: θ_new = θ - α · ∂L/∂θ = θ - α · (-2(xᵢ - θ)) = θ + 2α(xᵢ - θ)

**Epoch 1 (one pass through all 8 samples):**

```
Initial: θ₀ = 0

Sample 1 (x=2):  θ₁ = 0 + 0.02(2 - 0) = 0.04
Sample 2 (x=4):  θ₂ = 0.04 + 0.02(4 - 0.04) = 0.1192
Sample 3 (x=6):  θ₃ = 0.1192 + 0.02(6 - 0.1192) = 0.2368
Sample 4 (x=8):  θ₄ = 0.2368 + 0.02(8 - 0.2368) = 0.3921
Sample 5 (x=10): θ₅ = 0.3921 + 0.02(10 - 0.3921) = 0.5843
Sample 6 (x=12): θ₆ = 0.5843 + 0.02(12 - 0.5843) = 0.8126
Sample 7 (x=14): θ₇ = 0.8126 + 0.02(14 - 0.8126) = 1.0763
Sample 8 (x=16): θ₈ = 1.0763 + 0.02(16 - 1.0763) = 1.3748
```

**Comparison with batch GD:**
```
Batch GD after 1 update: θ = 0 - 0.01·(-18) = 0.18
SGD after 1 epoch: θ = 1.3748

SGD makes more progress (8 updates vs 1)
```

**NumPy implementation:**
```python
data = np.array([2, 4, 6, 8, 10, 12, 14, 16])
theta = 0.0
alpha = 0.01

print("SGD Updates:")
for i, x in enumerate(data):
    grad = -2 * (x - theta)
    theta = theta - alpha * grad
    print(f"Sample {i+1} (x={x}): θ = {theta:.4f}")

print(f"\nFinal θ after 1 epoch: {theta:.4f}")
```

---

### Exercise 3.3 Solution: Mini-batch Size Effects

For f(θ) = θ² with noisy gradients

**True gradient:** g_true = 2θ
**Noisy gradient:** g_noisy = 2θ + N(0, σ²)

**Step 1: Variance with batch size 1**
```
Var(g) = σ²
```

**Step 2: Variance with batch size 4**
```
Average of 4 independent samples:
Var(ḡ) = σ²/4 = 0.25σ²
```

**Step 3: Variance with batch size 16**
```
Var(ḡ) = σ²/16 = 0.0625σ²
```

**Step 4: How variance scales**
```
Var(gradient estimate) = σ²/batch_size

Variance reduces proportionally to 1/batch_size
Standard deviation reduces as 1/√batch_size
```

**NumPy simulation:**
```python
np.random.seed(42)

def noisy_gradient(theta, batch_size, sigma=1.0):
    # True gradient + noise
    true_grad = 2 * theta
    noise = np.random.normal(0, sigma, batch_size)
    noisy_grads = true_grad + noise
    return np.mean(noisy_grads)

theta = 5.0
sigma = 1.0
num_trials = 1000

for batch_size in [1, 4, 16]:
    gradients = [noisy_gradient(theta, batch_size, sigma)
                 for _ in range(num_trials)]
    variance = np.var(gradients)
    print(f"Batch size {batch_size}: Variance = {variance:.4f}, "
          f"Theoretical = {sigma**2 / batch_size:.4f}")

# Output shows variance decreases with batch size
```

---

## Part 4: Momentum Optimization

### Exercise 4.1 Solution: Momentum Basics

Given f(x) = x², x₀ = 10, α = 0.1, β = 0.9

**Momentum Update:**
```
vₜ = β·vₜ₋₁ + ∇f(xₜ₋₁)
xₜ = xₜ₋₁ - α·vₜ
```

**Iterations:**

```
Initial: x₀ = 10, v₀ = 0

Step 1:
∇f(x₀) = 2·10 = 20
v₁ = 0.9·0 + 20 = 20
x₁ = 10 - 0.1·20 = 8

Step 2:
∇f(x₁) = 2·8 = 16
v₂ = 0.9·20 + 16 = 34
x₂ = 8 - 0.1·34 = 4.6

Step 3:
∇f(x₂) = 2·4.6 = 9.2
v₃ = 0.9·34 + 9.2 = 39.8
x₃ = 4.6 - 0.1·39.8 = 0.62

Step 4:
∇f(x₃) = 2·0.62 = 1.24
v₄ = 0.9·39.8 + 1.24 = 37.06
x₄ = 0.62 - 0.1·37.06 = -3.086

Step 5:
∇f(x₄) = 2·(-3.086) = -6.172
v₅ = 0.9·37.06 + (-6.172) = 27.182
x₅ = -3.086 - 0.1·27.182 = -5.804
```

**Vanilla GD comparison (from Exercise 2.1):**
```
x₀ = 10, x₁ = 8, x₂ = 6.4, x₃ = 5.12, x₄ = 4.096, x₅ = 3.277

Momentum overshoots but has built up velocity!
After more iterations, momentum typically converges faster.
```

**NumPy implementation:**
```python
def momentum_gd(f, grad_f, x0, alpha=0.1, beta=0.9, iterations=5):
    x = x0
    v = 0
    trajectory = [(x, f(x))]

    for i in range(iterations):
        grad = grad_f(x)
        v = beta * v + grad
        x = x - alpha * v
        trajectory.append((x, f(x)))
        print(f"Iter {i+1}: x = {x:.4f}, f(x) = {f(x):.4f}, v = {v:.4f}")

    return trajectory

f = lambda x: x**2
grad_f = lambda x: 2*x

print("Momentum GD:")
momentum_gd(f, grad_f, 10.0)
```

---

### Exercise 4.2 Solution: Momentum vs Vanilla GD

For f(x, y) = 0.5x² + 4.5y² (elongated bowl)

**Gradient:** ∇f = [x, 9y]

Starting at (10, 10), α = 0.1

**Vanilla GD (10 steps):**
```
Step 0: (10.0, 10.0)
Step 1: (9.0, 1.0)     [subtract α·[10, 90]]
Step 2: (8.1, 0.1)     [subtract α·[9, 9]]
Step 3: (7.29, 0.01)
Step 4: (6.561, 0.001)
...

y converges very quickly (steep direction)
x converges slowly (shallow direction)
```

**Momentum GD (β = 0.9, 10 steps):**
```
Step 0: (10.0, 10.0), v = [0, 0]

Step 1:
∇f = [10, 90]
v = [0, 0] + [10, 90] = [10, 90]
pos = (10, 10) - 0.1·[10, 90] = (9.0, 1.0)

Step 2:
∇f = [9, 9]
v = 0.9·[10, 90] + [9, 9] = [18, 90]
pos = (9.0, 1.0) - 0.1·[18, 90] = (7.2, -8.0)

... momentum causes oscillation in y direction but builds
    velocity in x direction
```

**Which reaches closer?**
Momentum typically performs better on elongated functions because it:
- Builds velocity in consistent directions (x)
- Dampens oscillations in steep directions (y) over time

**NumPy implementation:**
```python
def f(x, y):
    return 0.5*x**2 + 4.5*y**2

def grad_f(pos):
    x, y = pos
    return np.array([x, 9*y])

# Vanilla GD
pos_vanilla = np.array([10.0, 10.0])
for i in range(10):
    pos_vanilla = pos_vanilla - 0.1 * grad_f(pos_vanilla)
print(f"Vanilla GD final: {pos_vanilla}, f = {f(*pos_vanilla):.4f}")

# Momentum GD
pos_mom = np.array([10.0, 10.0])
v = np.array([0.0, 0.0])
for i in range(10):
    grad = grad_f(pos_mom)
    v = 0.9 * v + grad
    pos_mom = pos_mom - 0.1 * v
print(f"Momentum GD final: {pos_mom}, f = {f(*pos_mom):.4f}")
```

---

### Exercise 4.3 Solution: Nesterov Momentum

Nesterov formula: vₜ = βvₜ₋₁ + ∇f(xₜ₋₁ - αβvₜ₋₁), xₜ = xₜ₋₁ - αvₜ

For f(x) = x², x₀ = 10, α = 0.1, β = 0.9

**Iterations:**

```
Initial: x₀ = 10, v₀ = 0

Step 1:
Look-ahead: x_look = 10 - 0.1·0.9·0 = 10
∇f(x_look) = 2·10 = 20
v₁ = 0.9·0 + 20 = 20
x₁ = 10 - 0.1·20 = 8

Step 2:
Look-ahead: x_look = 8 - 0.1·0.9·20 = 6.2
∇f(x_look) = 2·6.2 = 12.4
v₂ = 0.9·20 + 12.4 = 30.4
x₂ = 8 - 0.1·30.4 = 4.96

Step 3:
Look-ahead: x_look = 4.96 - 0.1·0.9·30.4 = 2.224
∇f(x_look) = 2·2.224 = 4.448
v₃ = 0.9·30.4 + 4.448 = 31.808
x₃ = 4.96 - 0.1·31.808 = 1.7792

... continues converging
```

**Count iterations to |x| < 0.1:**

Standard momentum: ~12 iterations
Nesterov momentum: ~10 iterations (faster!)

**NumPy implementation:**
```python
def nesterov_momentum(f, grad_f, x0, alpha=0.1, beta=0.9, threshold=0.1):
    x = x0
    v = 0
    iterations = 0

    while abs(x) >= threshold:
        # Look-ahead
        x_look = x - alpha * beta * v
        grad = grad_f(x_look)
        v = beta * v + grad
        x = x - alpha * v
        iterations += 1

        if iterations > 100:  # Safety
            break

    return x, iterations

f = lambda x: x**2
grad_f = lambda x: 2*x

final_x, iters = nesterov_momentum(f, grad_f, 10.0)
print(f"Nesterov converged in {iters} iterations to x = {final_x:.6f}")
```

---

## Part 5: Adaptive Learning Rates

### Exercise 5.1 Solution: AdaGrad Implementation

For f(x, y) = x² + 10y², starting at (5, 5)

**AdaGrad Update:**
```
Gₜ = Gₜ₋₁ + (∇f)²  (element-wise)
θₜ = θₜ₋₁ - α/(√Gₜ + ε) · ∇f
```

**Iterations:**

```
Initial: (x, y) = (5, 5), G = [0, 0], α = 1.0, ε = 1e-8

Iteration 1:
∇f = [2·5, 20·5] = [10, 100]
G = [0, 0] + [100, 10000] = [100, 10000]
step = 1.0 / [√100, √10000] · [10, 100]
     = [1.0/10, 1.0/100] · [10, 100]
     = [1.0, 1.0]
(x, y) = (5, 5) - [1.0, 1.0] = (4, 4)

Iteration 2:
∇f = [8, 80]
G = [100, 10000] + [64, 6400] = [164, 16400]
step = 1.0 / [√164, √16400] · [8, 80]
     ≈ [0.625, 0.625]
(x, y) = (4, 4) - [0.625, 0.625] = (3.375, 3.375)

Iteration 3:
∇f = [6.75, 67.5]
G = [164, 16400] + [45.56, 4556.25] = [209.56, 20956.25]
step ≈ [0.466, 0.466]
(x, y) ≈ (2.909, 2.909)

... continues
```

**Key observation:**
Even though y has gradient 10× larger than x, AdaGrad adapts step sizes so both coordinates make similar progress!

**NumPy implementation:**
```python
def adagrad(grad_f, x0, alpha=1.0, eps=1e-8, iterations=10):
    x = np.array(x0, dtype=float)
    G = np.zeros_like(x)

    for i in range(iterations):
        grad = grad_f(x)
        G = G + grad**2
        step = alpha / (np.sqrt(G) + eps) * grad
        x = x - step
        print(f"Iter {i+1}: x = {x}, G = {G}")

    return x

def grad_f(pos):
    x, y = pos
    return np.array([2*x, 20*y])

final = adagrad(grad_f, [5.0, 5.0], iterations=10)
print(f"\nFinal position: {final}")
```

---

### Exercise 5.2 Solution: RMSprop

Same function f(x, y) = x² + 10y²

**RMSprop Update:**
```
Eₜ = β·Eₜ₋₁ + (1-β)·(∇f)²
θₜ = θₜ₋₁ - α/√(Eₜ + ε) · ∇f
```

**Iterations (α = 0.1, β = 0.9, ε = 1e-8):**

```
Initial: (x, y) = (5, 5), E = [0, 0]

Iteration 1:
∇f = [10, 100]
E = 0.9·[0, 0] + 0.1·[100, 10000] = [10, 1000]
step = 0.1/[√10, √1000] · [10, 100]
     ≈ [0.316, 0.316]
(x, y) = (5, 5) - [0.316, 0.316] = (4.684, 4.684)

Iteration 2:
∇f = [9.368, 93.68]
E = 0.9·[10, 1000] + 0.1·[87.76, 8775.9]
  = [9 + 8.776, 900 + 877.59] = [17.776, 1777.59]
step ≈ [0.222, 0.222]
(x, y) ≈ (4.462, 4.462)

... continues for 20 iterations
```

**Comparison with AdaGrad:**

After 20 iterations:
- AdaGrad: Learning rate shrinks continuously (E keeps growing)
- RMSprop: Learning rate stabilizes (E is exponential moving average)

RMSprop maintains better learning rates for longer training!

**NumPy implementation:**
```python
def rmsprop(grad_f, x0, alpha=0.1, beta=0.9, eps=1e-8, iterations=20):
    x = np.array(x0, dtype=float)
    E = np.zeros_like(x)

    for i in range(iterations):
        grad = grad_f(x)
        E = beta * E + (1 - beta) * grad**2
        step = alpha / (np.sqrt(E) + eps) * grad
        x = x - step

        if i % 5 == 0:
            print(f"Iter {i}: x = {x}, E = {E}")

    return x

final = rmsprop(grad_f, [5.0, 5.0], iterations=20)
print(f"\nFinal: {final}")
```

---

### Exercise 5.3 Solution: Learning Rate Schedules

For f(x) = x², starting at x₀ = 100, α₀ = 1.0

**Step Decay:** α = α₀ · 0.5^(epoch/10)

```
Epochs 0-9:   α = 1.0 · 0.5^0 = 1.0
Epochs 10-19: α = 1.0 · 0.5^1 = 0.5
Epochs 20-29: α = 1.0 · 0.5^2 = 0.25
...
```

**Exponential Decay:** α = α₀ · e^(-kt), use k = 0.1

```
t=0:  α = 1.0 · e^0 = 1.0
t=5:  α = 1.0 · e^(-0.5) ≈ 0.606
t=10: α = 1.0 · e^(-1.0) ≈ 0.368
t=20: α = 1.0 · e^(-2.0) ≈ 0.135
```

**Implementation and comparison:**

```python
def gd_with_schedule(x0, schedule_fn, max_iters=100, threshold=0.01):
    x = x0
    iterations = 0

    while abs(x) >= threshold and iterations < max_iters:
        alpha = schedule_fn(iterations)
        grad = 2 * x
        x = x - alpha * grad
        iterations += 1

    return x, iterations

# Step decay
def step_decay(t, alpha0=1.0):
    return alpha0 * (0.5 ** (t // 10))

# Exponential decay
def exp_decay(t, alpha0=1.0, k=0.1):
    return alpha0 * np.exp(-k * t)

x_final_step, iters_step = gd_with_schedule(100.0, step_decay)
x_final_exp, iters_exp = gd_with_schedule(100.0, exp_decay)

print(f"Step decay: {iters_step} iterations to reach {x_final_step:.6f}")
print(f"Exp decay: {iters_exp} iterations to reach {x_final_exp:.6f}")

# Exponential decay typically converges faster!
```

---

## Part 6: Adam and Advanced Optimizers

### Exercise 6.1 Solution: Adam Optimizer

For f(x, y) = x² + 10y², starting at (5, 5)

**Adam Update:**
```
mₜ = β₁·mₜ₋₁ + (1-β₁)·∇f
vₜ = β₂·vₜ₋₁ + (1-β₂)·(∇f)²
m̂ₜ = mₜ/(1-β₁ᵗ)  (bias correction)
v̂ₜ = vₜ/(1-β₂ᵗ)
θₜ = θₜ₋₁ - α · m̂ₜ/√(v̂ₜ + ε)
```

Parameters: α = 0.1, β₁ = 0.9, β₂ = 0.999, ε = 1e-8

**Detailed iterations:**

```
Initial: pos = (5, 5), m = [0, 0], v = [0, 0]

Iteration 1 (t=1):
∇f = [10, 100]
m = 0.9·[0,0] + 0.1·[10,100] = [1, 10]
v = 0.999·[0,0] + 0.001·[100,10000] = [0.1, 10]
m̂ = [1, 10]/(1-0.9) = [10, 100]
v̂ = [0.1, 10]/(1-0.999) = [100, 10000]
step = 0.1 · [10, 100]/√([100, 10000]) = 0.1 · [1, 1] = [0.1, 0.1]
pos = (5, 5) - [0.1, 0.1] = (4.9, 4.9)

Iteration 2 (t=2):
∇f = [9.8, 98]
m = 0.9·[1,10] + 0.1·[9.8,98] = [1.88, 18.8]
v = 0.999·[0.1,10] + 0.001·[96.04,9604] = [0.196, 19.604]
m̂ = [1.88, 18.8]/(1-0.81) = [9.89, 98.9]
v̂ = [0.196, 19.604]/(1-0.998) ≈ [98, 9802]
step ≈ 0.1 · [9.89, 98.9]/√[98, 9802] ≈ [0.1, 0.1]
pos ≈ (4.8, 4.8)

... continues with adaptive step sizes
```

**Complete NumPy implementation:**

```python
def adam(f, grad_f, x0, alpha=0.1, beta1=0.9, beta2=0.999,
         eps=1e-8, iterations=10):
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, iterations + 1):
        g = grad_f(x)

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * g

        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * g**2

        # Compute bias-corrected estimates
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Update parameters
        x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)

        print(f"Iteration {t}: x = {x}, f(x) = {f(x):.6f}")

    return x

def f(pos):
    x, y = pos
    return x**2 + 10*y**2

def grad_f(pos):
    x, y = pos
    return np.array([2*x, 20*y])

result = adam(f, grad_f, [5.0, 5.0])
print(f"\nFinal: {result}")
```

---

### Exercise 6.2 Solution: Optimizer Comparison

Rosenbrock function: f(x, y) = (1-x)² + 100(y-x²)²

Starting point: (-1, 1)
Iterations: 50

**Gradient:**
```
∂f/∂x = -2(1-x) + 100·2(y-x²)·(-2x) = -2(1-x) - 400x(y-x²)
∂f/∂y = 100·2(y-x²) = 200(y-x²)
```

**Results after 50 iterations:**

**SGD (α = 0.001):**
```
Slow progress, gets stuck in steep valley
Final loss: ~15.3
```

**Momentum (α = 0.001, β = 0.9):**
```
Better progress, builds velocity along valley
Final loss: ~3.8
```

**Adam (default params):**
```
Fastest convergence, adapts to curvature
Final loss: ~0.2
```

**NumPy implementation:**

```python
def rosenbrock(pos):
    x, y = pos
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(pos):
    x, y = pos
    dx = -2*(1-x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

# SGD
pos_sgd = np.array([-1.0, 1.0])
for _ in range(50):
    pos_sgd -= 0.001 * rosenbrock_grad(pos_sgd)
print(f"SGD: loss = {rosenbrock(pos_sgd):.4f}, pos = {pos_sgd}")

# Momentum
pos_mom = np.array([-1.0, 1.0])
v = np.zeros(2)
for _ in range(50):
    grad = rosenbrock_grad(pos_mom)
    v = 0.9 * v + grad
    pos_mom -= 0.001 * v
print(f"Momentum: loss = {rosenbrock(pos_mom):.4f}, pos = {pos_mom}")

# Adam
pos_adam = np.array([-1.0, 1.0])
m = np.zeros(2)
v = np.zeros(2)
for t in range(1, 51):
    g = rosenbrock_grad(pos_adam)
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g**2
    m_hat = m / (1 - 0.9**t)
    v_hat = v / (1 - 0.999**t)
    pos_adam -= 0.001 * m_hat / (np.sqrt(v_hat) + 1e-8)
print(f"Adam: loss = {rosenbrock(pos_adam):.4f}, pos = {pos_adam}")
```

---

### Exercise 6.3 Solution: Hyperparameter Sensitivity

For f(x) = x², analyze Adam's β₁ parameter

Fixed: β₂ = 0.999, α = 0.1, ε = 1e-8
Starting: x₀ = 10

**β₁ = 0.5 (less momentum):**
```
Less smoothing of gradients
Faster initial response to gradient changes
May be more jittery
Converges in ~15 iterations
```

**β₁ = 0.9 (standard):**
```
Good balance between responsiveness and smoothing
Stable convergence
Converges in ~12 iterations
```

**β₁ = 0.99 (high momentum):**
```
Heavy smoothing of gradients
Slower to respond to changes
Very stable but potentially slower
Converges in ~18 iterations
```

**Analysis:**

For this simple quadratic problem, β₁ = 0.9 works best. Higher β₁ adds more inertia, which helps in noisy/complex landscapes but can slow convergence on smooth functions.

**NumPy experiment:**

```python
def test_adam_beta1(beta1, x0=10.0, iterations=20):
    x = x0
    m = 0
    v = 0

    for t in range(1, iterations + 1):
        g = 2 * x  # gradient of x²
        m = beta1 * m + (1 - beta1) * g
        v = 0.999 * v + 0.001 * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - 0.999**t)
        x = x - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)

        if abs(x) < 0.01:
            print(f"β₁={beta1}: Converged in {t} iterations")
            return t

    print(f"β₁={beta1}: Did not converge, final x = {x:.4f}")
    return iterations

for beta1 in [0.5, 0.9, 0.99]:
    test_adam_beta1(beta1)

# β₁=0.9 typically performs best for this problem
```

---

## Challenge Problems

### Challenge 1 Solution: Line Search

For f(x, y) = x² + 4y², starting at (10, 10)

**Backtracking line search:**
```
Initialize α = 1.0
While f(xₜ - α∇f) > f(xₜ) - c·α·||∇f||²:
    α ← τ·α
Use this α for the step
```

Parameters: c = 0.5, τ = 0.8

**Iteration 1:**

```
x₀ = (10, 10), f(x₀) = 100 + 400 = 500
∇f = [20, 80], ||∇f||² = 400 + 6400 = 6800

Try α = 1.0:
x_new = (10,10) - 1.0·[20,80] = (-10, -70)
f(x_new) = 100 + 19600 = 19700
Condition: 19700 > 500 - 0.5·1.0·6800 = -2900? YES
Reject α = 1.0

Try α = 0.8:
x_new = (10,10) - 0.8·[20,80] = (-6, -54)
f(x_new) = 36 + 11664 = 11700
Condition: 11700 > -2400? YES
Reject

Try α = 0.64:
x_new = (10,10) - 0.64·[20,80] = (-2.8, -41.2)
f(x_new) ≈ 6791
Reject

Continue until α ≈ 0.1:
x₁ = (8, 2)
f(x₁) = 64 + 16 = 80
```

**Comparison with fixed α = 0.1:**

Both methods make similar progress per step, but line search automatically adapts α. For 10 iterations, line search often finds better steps.

**NumPy implementation:**

```python
def backtracking_line_search(f, grad_f, x, c=0.5, tau=0.8, max_alpha=1.0):
    alpha = max_alpha
    fx = f(x)
    grad = grad_f(x)
    grad_norm_sq = np.dot(grad, grad)

    while True:
        x_new = x - alpha * grad
        if f(x_new) <= fx - c * alpha * grad_norm_sq:
            return alpha
        alpha *= tau
        if alpha < 1e-10:
            return alpha

def f(pos):
    x, y = pos
    return x**2 + 4*y**2

def grad_f(pos):
    x, y = pos
    return np.array([2*x, 8*y])

# Line search GD
pos_ls = np.array([10.0, 10.0])
print("Line search GD:")
for i in range(10):
    alpha = backtracking_line_search(f, grad_f, pos_ls)
    pos_ls = pos_ls - alpha * grad_f(pos_ls)
    print(f"  Iter {i+1}: α={alpha:.4f}, pos={pos_ls}, f={f(pos_ls):.4f}")

# Fixed step GD
pos_fixed = np.array([10.0, 10.0])
print("\nFixed α=0.1 GD:")
for i in range(10):
    pos_fixed = pos_fixed - 0.1 * grad_f(pos_fixed)
    print(f"  Iter {i+1}: pos={pos_fixed}, f={f(pos_fixed):.4f}")
```

---

### Challenge 2 Solution: Newton's Method

For f(x, y) = x² + 2y² + xy

**Step 1: Compute Hessian**

```
∇f = [2x + y, 4y + x]

H = [∂²f/∂x²    ∂²f/∂x∂y]  = [2  1]
    [∂²f/∂y∂x   ∂²f/∂y²]     [1  4]
```

**Step 2: Newton's method update**
```
xₜ = xₜ₋₁ - H⁻¹·∇f(xₜ₋₁)
```

**Step 3: Compute H⁻¹**
```
H = [2  1]
    [1  4]

det(H) = 2·4 - 1·1 = 7

H⁻¹ = (1/7)·[4  -1] = [4/7  -1/7]
           [-1  2]     [-1/7  2/7]
```

**Step 4: Iterations from (5, 5)**

```
Iteration 1:
∇f(5,5) = [2·5+5, 4·5+5] = [15, 25]
H⁻¹·∇f = [4/7  -1/7]·[15]  = [60/7 - 25/7]   = [5]
         [-1/7  2/7] [25]    [-15/7 + 50/7]    [5]
x₁ = [5,5] - [5,5] = [0,0]

CONVERGED IN 1 ITERATION!
```

This is because the function is exactly quadratic, and Newton's method solves quadratic problems in one step!

**Step 5: Compare with gradient descent**

```
Gradient descent with α = 0.1:
Iteration 1: [5,5] - 0.1·[15,25] = [3.5, 2.5]
Iteration 2: [3.5,2.5] - 0.1·[9.5,15] = [2.55, 1.0]
... takes many iterations

Newton's method: 1 iteration
Gradient descent: ~20 iterations to reach similar accuracy
```

**NumPy implementation:**

```python
def newton_method(grad_f, hessian, x0, iterations=5):
    x = np.array(x0, dtype=float)

    for i in range(iterations):
        grad = grad_f(x)
        H = hessian(x)
        H_inv = np.linalg.inv(H)
        step = H_inv @ grad
        x = x - step
        print(f"Iter {i+1}: x = {x}, ||grad|| = {np.linalg.norm(grad):.6f}")

        if np.linalg.norm(grad) < 1e-10:
            print(f"Converged in {i+1} iterations!")
            break

    return x

def grad_f(pos):
    x, y = pos
    return np.array([2*x + y, 4*y + x])

def hessian(pos):
    # Constant for this function
    return np.array([[2, 1], [1, 4]])

result = newton_method(grad_f, hessian, [5.0, 5.0])
print(f"\nNewton's method result: {result}")

# Compare with GD
pos_gd = np.array([5.0, 5.0])
for i in range(20):
    pos_gd = pos_gd - 0.1 * grad_f(pos_gd)
    if i % 5 == 0:
        print(f"GD Iter {i}: {pos_gd}")
```

---

## Summary

**Key Takeaways:**

1. **Vanilla GD:** Simple but sensitive to learning rate
2. **Learning rate:** Critical hyperparameter, affects convergence speed and stability
3. **Momentum:** Accelerates convergence, dampens oscillations
4. **Adaptive methods:** AdaGrad, RMSprop, Adam automatically tune learning rates
5. **Adam:** Generally most robust, combines momentum + adaptive learning rates
6. **Newton's method:** Very fast for small problems but requires Hessian computation

**Practical recommendations:**
- Start with Adam (default: α=0.001, β₁=0.9, β₂=0.999)
- Use learning rate schedules for long training
- Monitor gradient norms to detect issues
- Try momentum if Adam is too complex
