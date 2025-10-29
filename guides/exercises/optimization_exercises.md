# Optimization Exercises - Module 3

**Time:** 2-3 hours
**Difficulty:** Intermediate
**Materials needed:** Paper, pencil, calculator, NumPy

Complete these exercises by hand first, then verify with NumPy. Solutions are in `guides/solutions/optimization_solutions.md`

---

## Part 1: Gradient Descent Basics (25 min)

### Exercise 1.1: Computing Gradients
Given the function f(x, y) = x² + 2y² + 3xy:

1. Compute ∂f/∂x by hand
2. Compute ∂f/∂y by hand
3. Write the gradient vector ∇f(x, y)
4. Evaluate the gradient at point (1, 2)

### Exercise 1.2: Gradient Descent Step
Given f(x) = x² - 4x + 5 starting at x₀ = 0:

1. Compute f'(x)
2. Perform one gradient descent step with learning rate α = 0.1
3. Calculate the new position x₁
4. Compare f(x₀) and f(x₁) - did we decrease the loss?

### Exercise 1.3: Multi-dimensional Gradient Descent
For f(x, y) = (x - 2)² + (y + 1)²:

1. Compute the gradient ∇f
2. Starting from (0, 0), perform 3 gradient descent steps with α = 0.5
3. Track the path: list positions after each step
4. What is the true minimum? How close did you get?

---

## Part 2: Convergence Analysis (25 min)

### Exercise 2.1: Learning Rate Effects
Given f(x) = x² starting at x₀ = 10:

1. Perform 5 GD steps with α = 0.1 and record positions
2. Repeat with α = 0.5
3. Repeat with α = 1.1
4. Which learning rate converges fastest? Which diverges?

### Exercise 2.2: Convergence Conditions
For the quadratic f(x) = ½x²:

1. The gradient is f'(x) = x. What is the Lipschitz constant L?
2. What is the maximum learning rate for guaranteed convergence? (α < 2/L)
3. Test your answer: try α = 1.9/L and α = 2.1/L from x₀ = 5

### Exercise 2.3: Iteration Count Estimation
Given f(x) = x² with target accuracy ε = 0.01:

1. Starting from x₀ = 10 with α = 0.1, estimate iterations needed
2. Use formula: approximately log(ε/||x₀||) / log(1 - α)
3. Verify by actually running the iterations

---

## Part 3: SGD and Mini-Batches (25 min)

### Exercise 3.1: Batch vs Mini-batch
Given dataset with 8 samples, loss function L = (1/n)Σᵢ(xᵢ - θ)²:
- Data: [2, 4, 6, 8, 10, 12, 14, 16]

1. Compute full batch gradient at θ = 0
2. Compute mini-batch gradient using samples [2, 4, 6, 8] at θ = 0
3. Compute mini-batch gradient using samples [10, 12, 14, 16] at θ = 0
4. Compare the three gradients

### Exercise 3.2: Stochastic Gradient Descent
Using same data as 3.1, implement SGD:

1. Start at θ = 0, use α = 0.01
2. Perform one SGD epoch (8 updates, one per sample)
3. Record θ after each update
4. Compare final θ with batch GD result

### Exercise 3.3: Mini-batch Size Effects
For function f(θ) = θ² with noisy gradients:

1. Simulate gradient: true gradient ± random noise N(0, σ²)
2. Compare variance of gradient estimate for batch sizes: 1, 4, 16
3. How does variance scale with batch size?

---

## Part 4: Momentum Optimization (25 min)

### Exercise 4.1: Momentum Basics
Given f(x) = x² starting at x₀ = 10:

1. Implement momentum GD: vₜ = βvₜ₋₁ + ∇f, xₜ = xₜ₋₁ - αvₜ
2. Use α = 0.1, β = 0.9, v₀ = 0
3. Perform 5 steps and compare with vanilla GD
4. Plot or list the positions

### Exercise 4.2: Momentum vs Vanilla GD
For f(x, y) = 0.5x² + 4.5y² (elongated bowl):

1. Start at (10, 10), use α = 0.1
2. Run 10 steps of vanilla GD
3. Run 10 steps of momentum GD with β = 0.9
4. Which reaches closer to (0, 0)?

### Exercise 4.3: Nesterov Momentum
Implement Nesterov momentum for f(x) = x²:

1. Formula: vₜ = βvₜ₋₁ + ∇f(xₜ₋₁ - αβvₜ₋₁), xₜ = xₜ₋₁ - αvₜ
2. Start at x₀ = 10, use α = 0.1, β = 0.9
3. Compare convergence with standard momentum
4. Count iterations to reach |x| < 0.1

---

## Part 5: Adaptive Learning Rates (25 min)

### Exercise 5.1: AdaGrad Implementation
For f(x, y) = x² + 10y² starting at (5, 5):

1. Implement AdaGrad: θₜ = θₜ₋₁ - α/(√Gₜ + ε) · ∇f
   - Gₜ = Gₜ₋₁ + (∇f)²
2. Use α = 1.0, ε = 1e-8
3. Run 10 iterations
4. Observe how step sizes adapt differently for x and y

### Exercise 5.2: RMSprop
Implement RMSprop for same function:

1. Formula: Eₜ = βEₜ₋₁ + (1-β)(∇f)², θₜ = θₜ₋₁ - α/√(Eₜ + ε) · ∇f
2. Use α = 0.1, β = 0.9, ε = 1e-8
3. Compare with AdaGrad - does it avoid the shrinking learning rate problem?
4. Run 20 iterations and compare

### Exercise 5.3: Learning Rate Schedules
For f(x) = x²:

1. Implement step decay: α = α₀ · 0.5^(epoch/10)
2. Implement exponential decay: α = α₀ · e^(-kt)
3. Start at x₀ = 100, α₀ = 1.0
4. Compare convergence speed (iterations to reach |x| < 0.01)

---

## Part 6: Adam and Advanced Optimizers (25 min)

### Exercise 6.1: Adam Optimizer
Implement Adam for f(x, y) = x² + 10y²:

1. First moment: mₜ = β₁mₜ₋₁ + (1-β₁)∇f
2. Second moment: vₜ = β₂vₜ₋₁ + (1-β₂)(∇f)²
3. Bias correction: m̂ₜ = mₜ/(1-β₁ᵗ), v̂ₜ = vₜ/(1-β₂ᵗ)
4. Update: θₜ = θₜ₋₁ - α · m̂ₜ/√(v̂ₜ + ε)
5. Use α = 0.1, β₁ = 0.9, β₂ = 0.999, ε = 1e-8
6. Run 10 iterations from (5, 5)

### Exercise 6.2: Optimizer Comparison
For Rosenbrock function f(x, y) = (1-x)² + 100(y-x²)²:

1. Run 50 iterations of SGD (α = 0.001)
2. Run 50 iterations of Momentum (α = 0.001, β = 0.9)
3. Run 50 iterations of Adam (default params)
4. Start all at (-1, 1), compare final losses

### Exercise 6.3: Hyperparameter Sensitivity
For f(x) = x², analyze Adam's β₁:

1. Run Adam with β₁ = 0.5, 0.9, 0.99 (keep β₂ = 0.999)
2. Start at x₀ = 10, run 20 iterations each
3. How does β₁ affect convergence speed?
4. Which value works best for this problem?

---

## Challenge Problems (Optional)

### Challenge 1: Line Search
Implement backtracking line search for f(x, y) = x² + 4y²:

1. Start with α = 1.0
2. While f(xₜ - α∇f) > f(xₜ) - c·α·||∇f||², reduce α ← τα
3. Use c = 0.5, τ = 0.8
4. Compare with fixed learning rate over 10 iterations

### Challenge 2: Newton's Method
For f(x, y) = x² + 2y² + xy:

1. Compute Hessian matrix H (matrix of second derivatives)
2. Implement Newton's method: xₜ = xₜ₋₁ - H⁻¹∇f
3. Start at (5, 5), perform 5 iterations
4. Compare with gradient descent (how many fewer iterations?)

---

## NumPy Verification

```python
import numpy as np

# Exercise 1.1 Verification
def f(x, y):
    return x**2 + 2*y**2 + 3*x*y

def grad_f(x, y):
    df_dx = 2*x + 3*y
    df_dy = 4*y + 3*x
    return np.array([df_dx, df_dy])

print("Gradient at (1, 2):", grad_f(1, 2))

# Exercise 1.2 - Gradient Descent
def gd_step(x, alpha, grad_fn):
    return x - alpha * grad_fn(x)

x = 0.0
alpha = 0.1
grad = lambda x: 2*x - 4
x_new = gd_step(x, alpha, grad)
print(f"x0 = {x}, x1 = {x_new}")

# Exercise 4.1 - Momentum
def momentum_gd(f, grad_f, x0, alpha=0.1, beta=0.9, iterations=5):
    x = x0
    v = 0
    trajectory = [x]
    for _ in range(iterations):
        v = beta * v + grad_f(x)
        x = x - alpha * v
        trajectory.append(x)
    return trajectory

traj = momentum_gd(lambda x: x**2, lambda x: 2*x, 10.0)
print("Momentum trajectory:", traj)

# Exercise 6.1 - Adam
def adam(f, grad_f, x0, alpha=0.1, beta1=0.9, beta2=0.999, eps=1e-8, iterations=10):
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, iterations + 1):
        g = grad_f(*x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)
        print(f"Iteration {t}: x = {x}, f(x) = {f(*x):.6f}")

    return x

# Test Adam
result = adam(
    lambda x, y: x**2 + 10*y**2,
    lambda x, y: np.array([2*x, 20*y]),
    [5.0, 5.0]
)
```

---

## Tips for Success

1. **Start simple** - Understand vanilla GD before advanced optimizers
2. **Visualize** - Plot loss curves and optimization paths
3. **Check gradients** - Use numerical differentiation to verify
4. **Tune hyperparameters** - Small changes can have big effects
5. **Compare methods** - Same problem, different optimizers
6. **Watch for divergence** - If loss increases, learning rate too high
7. **Momentum intuition** - Think of a ball rolling down a hill
8. **Adaptive rates** - Different parameters may need different learning rates
