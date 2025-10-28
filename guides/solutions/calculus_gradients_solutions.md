# Calculus & Gradients - Solutions

**Time:** Reference for 3-4 hours of exercises
**Difficulty:** Intermediate

---

## Part 1: Scalar Derivatives - Solutions

### Problem 1: Basic derivatives

**a) f(x) = 3x² + 2x - 5**

f'(x) = d/dx(3x²) + d/dx(2x) - d/dx(5)
      = 3(2x) + 2 - 0
      = 6x + 2

**b) f(x) = x³ - 4x² + 7**

f'(x) = d/dx(x³) - d/dx(4x²) + d/dx(7)
      = 3x² - 4(2x) + 0
      = 3x² - 8x

**c) f(x) = 1/x = x⁻¹**

f'(x) = d/dx(x⁻¹)
      = -1·x⁻²
      = -1/x²

**d) f(x) = √x = x^(1/2)**

f'(x) = d/dx(x^(1/2))
      = (1/2)x^(-1/2)
      = 1/(2√x)

### Problem 2: At what rate is f(x) = x² changing at x = 3?

f'(x) = 2x
f'(3) = 2(3) = 6

**Answer:** The function is changing at a rate of 6 units per unit of x at x = 3.

### Problem 3: Product Rule

**f(x) = (2x + 1)(x² - 3)**

Using product rule: (uv)' = u'v + uv'

Let u = 2x + 1,  u' = 2
Let v = x² - 3,  v' = 2x

f'(x) = u'v + uv'
      = 2(x² - 3) + (2x + 1)(2x)
      = 2x² - 6 + 4x² + 2x
      = 6x² + 2x - 6

### Problem 4: Chain Rule

**f(x) = (3x² + 1)⁵**

Using chain rule: d/dx[g(h(x))] = g'(h(x))·h'(x)

Let u = 3x² + 1,  u' = 6x

f'(x) = 5u⁴ · u'
      = 5(3x² + 1)⁴ · 6x
      = 30x(3x² + 1)⁴

---

## Part 2: Partial Derivatives - Solutions

### Problem 5: Compute partial derivatives

**f(x, y) = x²y + 3xy² - 2x + 5**

**∂f/∂x** (treat y as constant):
∂f/∂x = 2xy + 3y² - 2

**∂f/∂y** (treat x as constant):
∂f/∂y = x² + 6xy

### Problem 6: Evaluate partial derivatives at (2, 3)

From Problem 5:

**∂f/∂x|(2,3)** = 2(2)(3) + 3(3)² - 2
                = 12 + 27 - 2
                = 37

**∂f/∂y|(2,3)** = (2)² + 6(2)(3)
                = 4 + 36
                = 40

**Interpretation:** At point (2, 3):
- If x increases by 1 unit (y constant), f increases by ~37 units
- If y increases by 1 unit (x constant), f increases by ~40 units

### Problem 7: Second-order partial derivatives

**f(x, y) = x³y² + 2xy**

**First-order partials:**
∂f/∂x = 3x²y² + 2y
∂f/∂y = 2x³y + 2x

**Second-order partials:**

**∂²f/∂x²:**
∂²f/∂x² = ∂/∂x(3x²y² + 2y)
        = 6xy²

**∂²f/∂y²:**
∂²f/∂y² = ∂/∂y(2x³y + 2x)
        = 2x³

**∂²f/∂x∂y (mixed partial):**
∂²f/∂x∂y = ∂/∂x(2x³y + 2x)
          = 6x²y + 2

**Verify ∂²f/∂y∂x = ∂²f/∂x∂y:**
∂²f/∂y∂x = ∂/∂y(3x²y² + 2y)
          = 6x²y + 2 ✓

(Schwarz's theorem: mixed partials are equal for continuous functions)

---

## Part 3: Gradients - Solutions

### Problem 8: Compute gradient

**f(x, y) = x² + 2xy + y²**

∂f/∂x = 2x + 2y
∂f/∂y = 2x + 2y

**∇f = [∂f/∂x, ∂f/∂y] = [2x + 2y, 2x + 2y]**

**At (1, 2):**
∇f(1, 2) = [2(1) + 2(2), 2(1) + 2(2)]
         = [6, 6]

**Interpretation:** At point (1, 2), the direction of steepest ascent is [6, 6], meaning moving equally in x and y directions. The magnitude √(6² + 6²) = 6√2 ≈ 8.49 indicates how steep it is.

### Problem 9: Three-variable gradient

**f(x, y, z) = x²y + yz² - 3z**

∂f/∂x = 2xy
∂f/∂y = x² + z²
∂f/∂z = 2yz - 3

**∇f = [2xy, x² + z², 2yz - 3]**

**At (1, 2, 1):**
∇f(1, 2, 1) = [2(1)(2), (1)² + (1)², 2(2)(1) - 3]
            = [4, 2, 1]

### Problem 10: Gradient descent step

**Current:** w = 3
**∇L(w) = 2w** at w = 3 means ∇L(3) = 6
**Learning rate:** α = 0.1

**Gradient descent update:**
w_new = w_old - α·∇L(w_old)
      = 3 - 0.1(6)
      = 3 - 0.6
      = 2.4

**Next step:**
w_new = 2.4 - 0.1(2·2.4)
      = 2.4 - 0.48
      = 1.92

---

## Part 4: Chain Rule (Multivariate) - Solutions

### Problem 11: Chain rule for two variables

**z = x² + y²**
**x = 2t, y = 3t**

**dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt)**

∂z/∂x = 2x
∂z/∂y = 2y
dx/dt = 2
dy/dt = 3

**dz/dt = 2x·2 + 2y·3**
       = 4x + 6y
       = 4(2t) + 6(3t)
       = 8t + 18t
       = 26t

**At t = 1:**
dz/dt = 26(1) = 26

### Problem 12: Neural network backpropagation

**Network:** x → z = wx → a = σ(z) → L = (a - y)²

**Given:**
- w = 0.5
- x = 2
- y = 1 (target)
- σ(z) = 1/(1 + e^(-z))
- σ'(z) = σ(z)(1 - σ(z))

**Forward pass:**

z = wx = 0.5 × 2 = 1

a = σ(1) = 1/(1 + e^(-1)) = 1/(1 + 0.368) ≈ 0.731

L = (a - y)² = (0.731 - 1)² = (-0.269)² ≈ 0.072

**Backward pass (chain rule):**

**∂L/∂w = (∂L/∂a)(∂a/∂z)(∂z/∂w)**

**∂L/∂a:**
∂L/∂a = 2(a - y) = 2(0.731 - 1) = -0.538

**∂a/∂z:**
∂a/∂z = σ'(z) = σ(z)(1 - σ(z))
      = 0.731(1 - 0.731)
      = 0.731 × 0.269
      ≈ 0.197

**∂z/∂w:**
∂z/∂w = x = 2

**Combine:**
∂L/∂w = (-0.538)(0.197)(2)
      ≈ -0.212

**Interpretation:** The gradient is negative, meaning to reduce loss, we should increase w.

---

## Part 5: Jacobian Matrices - Solutions

### Problem 13: Jacobian of vector function

**f: ℝ² → ℝ²**
**f(x, y) = [x² + y, xy]**

Let f₁(x, y) = x² + y
Let f₂(x, y) = xy

**Jacobian J:**

J = [∂f₁/∂x  ∂f₁/∂y]
    [∂f₂/∂x  ∂f₂/∂y]

∂f₁/∂x = 2x,  ∂f₁/∂y = 1
∂f₂/∂x = y,   ∂f₂/∂y = x

**J = [2x  1]**
    **[y   x]**

**At (1, 2):**

**J(1, 2) = [2(1)  1] = [2  1]**
          **[2     1]   [2  1]**

### Problem 14: Three-variable Jacobian

**f(x, y, z) = [xy, yz, xz]**

Let f₁ = xy, f₂ = yz, f₃ = xz

**J = [∂f₁/∂x  ∂f₁/∂y  ∂f₁/∂z]   [y  x  0]**
    **[∂f₂/∂x  ∂f₂/∂y  ∂f₂/∂z] = [0  z  y]**
    **[∂f₃/∂x  ∂f₃/∂y  ∂f₃/∂z]   [z  0  x]**

---

## Part 6: Hessian Matrices - Solutions

### Problem 15: Compute Hessian

**f(x, y) = x³ + y³ + 3xy**

**First-order partials:**
∂f/∂x = 3x² + 3y
∂f/∂y = 3y² + 3x

**Second-order partials:**
∂²f/∂x² = 6x
∂²f/∂y² = 6y
∂²f/∂x∂y = 3
∂²f/∂y∂x = 3

**Hessian H:**

**H = [∂²f/∂x²    ∂²f/∂x∂y]   [6x  3]**
    **[∂²f/∂y∂x   ∂²f/∂y²  ] = [3   6y]**

**At (1, 1):**

**H(1, 1) = [6(1)  3 ] = [6  3]**
          **[3     6(1)]   [3  6]**

**Eigenvalues:**
det(H - λI) = 0
det([6-λ   3  ]) = 0
   ([3     6-λ])

(6-λ)² - 9 = 0
36 - 12λ + λ² - 9 = 0
λ² - 12λ + 27 = 0

λ = (12 ± √(144-108))/2 = (12 ± 6)/2

**λ₁ = 9, λ₂ = 3**

Both eigenvalues are positive → **local minimum** at (1, 1)

### Problem 16: Neural network curvature

**L(w) = w⁴ - 4w³ + 6w²**

**First derivative:**
dL/dw = 4w³ - 12w² + 12w

**Second derivative (Hessian for 1D):**
d²L/dw² = 12w² - 24w + 12

**At w = 1:**
d²L/dw²|_{w=1} = 12(1)² - 24(1) + 12
                = 12 - 24 + 12
                = 0

**Interpretation:** The Hessian is zero at w = 1, indicating an inflection point. Neither convex nor concave at this point. Gradient descent may be slow here.

---

## Part 7: ML-Specific Applications - Solutions

### Problem 17: Mean Squared Error gradient

**MSE(w, b) = (1/3)[(wx₁ + b - y₁)² + (wx₂ + b - y₂)² + (wx₃ + b - y₃)²]**

**Dataset:**
(x₁, y₁) = (1, 2)
(x₂, y₂) = (2, 4)
(x₃, y₃) = (3, 5)

**Current parameters:** w = 1, b = 0

**Predictions:**
ŷ₁ = 1(1) + 0 = 1
ŷ₂ = 1(2) + 0 = 2
ŷ₃ = 1(3) + 0 = 3

**Errors:**
e₁ = ŷ₁ - y₁ = 1 - 2 = -1
e₂ = ŷ₂ - y₂ = 2 - 4 = -2
e₃ = ŷ₃ - y₃ = 3 - 5 = -2

**∂MSE/∂w = (2/3)[e₁·x₁ + e₂·x₂ + e₃·x₃]**
          = (2/3)[(-1)(1) + (-2)(2) + (-2)(3)]
          = (2/3)[-1 - 4 - 6]
          = (2/3)(-11)
          = -22/3
          ≈ -7.33

**∂MSE/∂b = (2/3)[e₁ + e₂ + e₃]**
          = (2/3)[-1 - 2 - 2]
          = (2/3)(-5)
          = -10/3
          ≈ -3.33

**Gradient descent update (α = 0.1):**

w_new = w - α(∂MSE/∂w)
      = 1 - 0.1(-7.33)
      = 1 + 0.733
      = 1.733

b_new = b - α(∂MSE/∂b)
      = 0 - 0.1(-3.33)
      = 0 + 0.333
      = 0.333

### Problem 18: Binary Cross-Entropy gradient

**BCE = -[y log(ŷ) + (1 - y) log(1 - ŷ)]**

**Prediction:** ŷ = σ(z) = 1/(1 + e^(-z))
**Where:** z = wx
**Given:** w = 0.5, x = 2, y = 1

**Forward pass:**
z = 0.5 × 2 = 1
ŷ = σ(1) = 1/(1 + e^(-1)) ≈ 0.731

**∂BCE/∂ŷ:**
∂BCE/∂ŷ = -[y/ŷ - (1-y)/(1-ŷ)]
        = -[1/0.731 - 0/0.269]
        = -1.368

**∂ŷ/∂z:**
∂ŷ/∂z = ŷ(1 - ŷ)
      = 0.731(0.269)
      ≈ 0.197

**∂z/∂w:**
∂z/∂w = x = 2

**Chain rule:**
∂BCE/∂w = (∂BCE/∂ŷ)(∂ŷ/∂z)(∂z/∂w)
        = (-1.368)(0.197)(2)
        ≈ -0.539

**Simplified form:**
∂BCE/∂w = (ŷ - y)·x = (0.731 - 1)(2) = -0.538 ✓

### Problem 19: Softmax gradient

**Softmax:** p_i = e^(z_i) / Σ_j e^(z_j)

**Given:** z = [1, 2, 0.5]

**Compute softmax:**
e^1 = 2.718
e^2 = 7.389
e^0.5 = 1.649

Sum = 2.718 + 7.389 + 1.649 = 11.756

p₁ = 2.718/11.756 ≈ 0.231
p₂ = 7.389/11.756 ≈ 0.629
p₃ = 1.649/11.756 ≈ 0.140

**∂p₂/∂z₁:**
For i ≠ j: ∂p_i/∂z_j = -p_i·p_j

∂p₂/∂z₁ = -p₂·p₁
        = -(0.629)(0.231)
        ≈ -0.145

**Interpretation:** Increasing z₁ decreases p₂ because softmax is a probability distribution (sum = 1).

### Problem 20: Two-layer network backpropagation

**Architecture:**
x → h = σ(w₁x + b₁) → ŷ = σ(w₂h + b₂) → L = (ŷ - y)²

**Given:**
- w₁ = 0.5, b₁ = 0.1
- w₂ = 0.8, b₂ = 0.2
- x = 1, y = 1
- σ(z) = 1/(1 + e^(-z))

**Forward pass:**

z₁ = w₁x + b₁ = 0.5(1) + 0.1 = 0.6
h = σ(0.6) = 1/(1 + e^(-0.6)) ≈ 0.646

z₂ = w₂h + b₂ = 0.8(0.646) + 0.2 ≈ 0.717
ŷ = σ(0.717) ≈ 0.672

L = (0.672 - 1)² ≈ 0.108

**Backward pass:**

**∂L/∂ŷ:**
∂L/∂ŷ = 2(ŷ - y) = 2(0.672 - 1) = -0.656

**∂ŷ/∂z₂:**
∂ŷ/∂z₂ = ŷ(1 - ŷ) = 0.672(0.328) ≈ 0.220

**∂z₂/∂w₂:**
∂z₂/∂w₂ = h = 0.646

**∂L/∂w₂:**
∂L/∂w₂ = (∂L/∂ŷ)(∂ŷ/∂z₂)(∂z₂/∂w₂)
       = (-0.656)(0.220)(0.646)
       ≈ -0.093

**∂z₂/∂h:**
∂z₂/∂h = w₂ = 0.8

**∂h/∂z₁:**
∂h/∂z₁ = h(1 - h) = 0.646(0.354) ≈ 0.229

**∂z₁/∂w₁:**
∂z₁/∂w₁ = x = 1

**∂L/∂w₁:**
∂L/∂w₁ = (∂L/∂ŷ)(∂ŷ/∂z₂)(∂z₂/∂h)(∂h/∂z₁)(∂z₁/∂w₁)
       = (-0.656)(0.220)(0.8)(0.229)(1)
       ≈ -0.026

**Summary:**
- ∂L/∂w₂ ≈ -0.093
- ∂L/∂w₁ ≈ -0.026

Both gradients are negative, indicating we should increase both weights to reduce loss.

---

## Challenge Problems - Solutions

### Challenge 1: Implement Gradient Checker

```python
import numpy as np

def gradient_check(f, grad_f, x, epsilon=1e-7):
    """
    f: function that takes x and returns scalar
    grad_f: function that computes gradient analytically
    x: point to check gradient at
    epsilon: small perturbation
    """
    # Compute analytical gradient
    analytical_grad = grad_f(x)

    # Compute numerical gradient
    numerical_grad = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += epsilon

        x_minus = x.copy()
        x_minus[i] -= epsilon

        # Finite difference approximation
        numerical_grad[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)

    # Compute relative error
    numerator = np.linalg.norm(analytical_grad - numerical_grad)
    denominator = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)
    relative_error = numerator / denominator

    print(f"Analytical gradient: {analytical_grad}")
    print(f"Numerical gradient:  {numerical_grad}")
    print(f"Relative error:      {relative_error:.2e}")

    # If relative error < 1e-7, gradients are likely correct
    if relative_error < 1e-7:
        print("✓ Gradient check PASSED")
    else:
        print("✗ Gradient check FAILED")

    return relative_error

# Example usage
def f(x):
    """f(x1, x2) = x1^2 + x1*x2 + x2^2"""
    return x[0]**2 + x[0]*x[1] + x[1]**2

def grad_f(x):
    """Gradient of f"""
    return np.array([2*x[0] + x[1], x[0] + 2*x[1]])

x = np.array([1.0, 2.0])
gradient_check(f, grad_f, x)

# Output:
# Analytical gradient: [4. 5.]
# Numerical gradient:  [4. 5.]
# Relative error:      1.57e-10
# ✓ Gradient check PASSED
```

### Challenge 2: Momentum in Gradient Descent

**Gradient descent with momentum:**

v_t = β·v_{t-1} + (1 - β)·∇L(w_t)
w_{t+1} = w_t - α·v_t

**Given:**
- f(w) = w² - 4w + 3
- w₀ = 5
- α = 0.1
- β = 0.9
- v₀ = 0

**Gradient:** ∇f(w) = 2w - 4

**Step 1:**
∇f(5) = 2(5) - 4 = 6
v₁ = 0.9(0) + 0.1(6) = 0.6
w₁ = 5 - 0.1(0.6) = 5 - 0.06 = 4.94

**Step 2:**
∇f(4.94) = 2(4.94) - 4 = 5.88
v₂ = 0.9(0.6) + 0.1(5.88) = 0.54 + 0.588 = 1.128
w₂ = 4.94 - 0.1(1.128) = 4.94 - 0.1128 = 4.827

**Step 3:**
∇f(4.827) = 2(4.827) - 4 = 5.654
v₃ = 0.9(1.128) + 0.1(5.654) = 1.0152 + 0.5654 = 1.581
w₃ = 4.827 - 0.1(1.581) = 4.827 - 0.1581 = 4.669

**Step 4:**
∇f(4.669) = 2(4.669) - 4 = 5.338
v₄ = 0.9(1.581) + 0.1(5.338) = 1.423 + 0.534 = 1.957
w₄ = 4.669 - 0.1(1.957) = 4.669 - 0.1957 = 4.473

**Step 5:**
∇f(4.473) = 2(4.473) - 4 = 4.946
v₅ = 0.9(1.957) + 0.1(4.946) = 1.761 + 0.495 = 2.256
w₅ = 4.473 - 0.1(2.256) = 4.473 - 0.2256 = 4.247

**Comparison with regular gradient descent:**

Regular GD without momentum:
- w₁ = 5 - 0.1(6) = 4.4
- w₂ = 4.4 - 0.1(4.8) = 3.92
- w₃ = 3.92 - 0.1(3.84) = 3.536
- w₄ = 3.536 - 0.1(3.072) = 3.229
- w₅ = 3.229 - 0.1(2.458) = 2.983

**Observations:**
- Regular GD reaches closer to optimum (w* = 2) faster in this case
- Momentum builds up velocity, causing larger steps
- For this simple convex function, momentum overshoots
- Momentum shines in problems with noisy gradients or ravines

### Challenge 3: RMSprop Implementation

```python
import numpy as np

def rmsprop(f, grad_f, w_init, alpha=0.01, beta=0.9, epsilon=1e-8, n_iterations=100):
    """
    RMSprop optimizer

    f: objective function
    grad_f: gradient function
    w_init: initial parameters
    alpha: learning rate
    beta: decay rate for moving average
    epsilon: small constant to prevent division by zero
    """
    w = w_init.copy()
    s = np.zeros_like(w)  # Running average of squared gradients

    history_w = [w.copy()]
    history_loss = [f(w)]

    for t in range(n_iterations):
        # Compute gradient
        g = grad_f(w)

        # Update moving average of squared gradient
        s = beta * s + (1 - beta) * (g ** 2)

        # Update parameters
        w = w - alpha * g / (np.sqrt(s) + epsilon)

        # Record history
        history_w.append(w.copy())
        history_loss.append(f(w))

        if t % 10 == 0:
            print(f"Iteration {t}: w = {w}, loss = {f(w):.6f}")

    return w, history_w, history_loss

# Example: Minimize f(w1, w2) = w1^2 + 10*w2^2
def f(w):
    return w[0]**2 + 10*w[1]**2

def grad_f(w):
    return np.array([2*w[0], 20*w[1]])

w_init = np.array([5.0, 5.0])
w_final, history_w, history_loss = rmsprop(f, grad_f, w_init, alpha=0.1, n_iterations=50)

print(f"\nFinal w: {w_final}")
print(f"Final loss: {f(w_final):.6f}")

# Output shows RMSprop adapts learning rate per parameter,
# allowing faster convergence in w2 direction (steeper) while
# being more conservative in w1 direction (flatter)
```

---

## Verification with NumPy

```python
import numpy as np

# Verify Problem 11: Chain rule
def z_func(t):
    x = 2*t
    y = 3*t
    return x**2 + y**2

t_val = 1.0
h = 1e-7
numerical_derivative = (z_func(t_val + h) - z_func(t_val - h)) / (2*h)
analytical_derivative = 26 * t_val

print(f"Problem 11 verification:")
print(f"Numerical: {numerical_derivative:.6f}")
print(f"Analytical: {analytical_derivative:.6f}")
print(f"Match: {np.isclose(numerical_derivative, analytical_derivative)}\n")

# Verify Problem 15: Hessian eigenvalues
H = np.array([[6, 3], [3, 6]])
eigenvalues = np.linalg.eigvals(H)
print(f"Problem 15 verification:")
print(f"Eigenvalues: {eigenvalues}")
print(f"Both positive: {np.all(eigenvalues > 0)} (local minimum)\n")

# Verify Problem 18: BCE gradient
y = 1
y_hat = 0.731
x = 2
analytical_grad = (y_hat - y) * x
print(f"Problem 18 verification:")
print(f"Simplified BCE gradient: {analytical_grad:.3f}")
```

---

## Key Takeaways

1. **Chain rule is fundamental** - All backpropagation relies on it
2. **Gradient = direction of steepest ascent** - Negative gradient descends
3. **Second derivatives (Hessian) tell us about curvature** - Useful for optimization
4. **Always verify gradients numerically** when implementing new models
5. **Momentum and adaptive methods** (RMSprop, Adam) improve training
6. **For ML: ∂Loss/∂weight = (prediction - target) × input** is the most common pattern

Practice these concepts until computing gradients becomes second nature!
