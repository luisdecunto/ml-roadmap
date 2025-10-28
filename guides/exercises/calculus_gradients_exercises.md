# Matrix Calculus & Gradients Exercises - Module 2

**Time:** 3-4 hours
**Difficulty:** Intermediate
**Materials needed:** Paper, pencil, calculator

Complete these by hand to build intuition for backpropagation. Solutions in `guides/solutions/calculus_gradients_solutions.md`

---

## Part 1: Scalar Derivatives Review (30 min)

### Exercise 1.1: Basic Derivatives
Calculate derivatives by hand (show your work):

1. f(x) = 3x² + 2x - 5,  find f'(x)
2. f(x) = x³ - 4x² + x,  find f'(x)
3. f(x) = 1/x²,  find f'(x)
4. f(x) = e^(2x),  find f'(x)
5. f(x) = ln(x²),  find f'(x)

### Exercise 1.2: Chain Rule
Calculate using chain rule:

1. f(x) = (3x + 2)⁴,  find f'(x)
2. f(x) = e^(x²),  find f'(x)
3. f(x) = ln(2x + 1),  find f'(x)
4. f(x) = sin(3x²),  find f'(x)

### Exercise 1.3: Product and Quotient Rules
1. f(x) = x² · e^x,  find f'(x) using product rule
2. f(x) = x³ · ln(x),  find f'(x)
3. f(x) = (x² + 1)/(x - 1),  find f'(x) using quotient rule

---

## Part 2: Partial Derivatives (45 min)

### Exercise 2.1: Basic Partial Derivatives
For f(x, y) = x²y + 3xy² - 2x + y

Calculate:
1. ∂f/∂x (treat y as constant)
2. ∂f/∂y (treat x as constant)
3. Evaluate both at point (x, y) = (1, 2)

### Exercise 2.2: More Partial Derivatives
For f(x, y) = e^(xy) + x²y³

Calculate:
1. ∂f/∂x
2. ∂f/∂y
3. Evaluate at (x, y) = (0, 1)

### Exercise 2.3: Second-Order Partial Derivatives
For f(x, y) = x³y² - 2xy + 5

Calculate all second-order partials:
1. ∂²f/∂x²
2. ∂²f/∂y²
3. ∂²f/∂x∂y
4. ∂²f/∂y∂x
5. Verify that ∂²f/∂x∂y = ∂²f/∂y∂x (Clairaut's theorem)

---

## Part 3: Gradients (60 min)

### Exercise 3.1: Computing Gradients
For f(x, y) = x² + y² - 2x - 4y + 5

1. Calculate gradient: ∇f = [∂f/∂x, ∂f/∂y]ᵀ
2. Find the gradient at point (1, 2)
3. Find critical points (where ∇f = 0)
4. Is the critical point a minimum, maximum, or saddle point?

### Exercise 3.2: Gradient of Quadratic Form
For f(x) = xᵀAx where x = [x₁, x₂]ᵀ and A = [[2, 1], [1, 3]]

1. Expand f(x) in terms of x₁, x₂
2. Calculate ∂f/∂x₁
3. Calculate ∂f/∂x₂
4. Write the gradient ∇f(x)
5. Verify the formula: ∇f(x) = (A + Aᵀ)x

### Exercise 3.3: Gradient Descent Step
Given f(x, y) = x² + 4y² and starting point (x₀, y₀) = (4, 2):

1. Calculate gradient at (4, 2)
2. Using learning rate α = 0.1, calculate one gradient descent step:
   (x₁, y₁) = (x₀, y₀) - α∇f(x₀, y₀)
3. Calculate f(x₀, y₀) and f(x₁, y₁) - did we reduce the function?
4. Calculate the gradient at the new point (x₁, y₁)

---

## Part 4: Chain Rule for Multivariable Functions (60 min)

### Exercise 4.1: Simple Chain Rule
Let z = f(x, y) = x² + y² where x = 2t and y = 3t

Find dz/dt using chain rule:
dz/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt)

### Exercise 4.2: Backpropagation Example
Consider a simple neural network computation:

```
Input: x
Hidden: h = σ(wx + b)  where σ(z) = 1/(1 + e^(-z))
Output: y = h²
Loss: L = (y - t)²  where t is target
```

Given x = 2, w = 0.5, b = 1, t = 0.8:

1. **Forward pass:** Calculate h, y, L (show all steps)

2. **Backward pass:** Calculate gradients using chain rule:
   - ∂L/∂y
   - ∂y/∂h
   - ∂h/∂w
   - ∂h/∂b
   - ∂L/∂w = (∂L/∂y)(∂y/∂h)(∂h/∂w)
   - ∂L/∂b = (∂L/∂y)(∂y/∂h)(∂h/∂b)

Note: σ'(z) = σ(z)(1 - σ(z))

### Exercise 4.3: Vector Chain Rule
Given:
- z = f(y) = y₁² + y₂²
- y = g(x) = [2x₁ + x₂, x₁ - x₂]ᵀ

Find ∂z/∂x₁ and ∂z/∂x₂ using chain rule:
∂z/∂xᵢ = Σⱼ (∂z/∂yⱼ)(∂yⱼ/∂xᵢ)

---

## Part 5: Jacobian Matrices (45 min)

### Exercise 5.1: Computing Jacobian
For function f: ℝ² → ℝ³ defined by:
```
f₁(x₁, x₂) = x₁² + x₂
f₂(x₁, x₂) = x₁x₂
f₃(x₁, x₂) = x₁ + 2x₂²
```

Calculate the Jacobian matrix:
```
J = [[∂f₁/∂x₁, ∂f₁/∂x₂],
     [∂f₂/∂x₁, ∂f₂/∂x₂],
     [∂f₃/∂x₁, ∂f₃/∂x₂]]
```

Evaluate at point (1, 2).

### Exercise 5.2: Chain Rule with Jacobians
Given:
- z = f(y): ℝ² → ℝ where f(y₁, y₂) = y₁² + 2y₂²
- y = g(x): ℝ³ → ℝ² where g(x₁, x₂, x₃) = [x₁ + x₂, x₂x₃]ᵀ

1. Calculate ∇f (gradient of f)
2. Calculate Jacobian of g: Jg (2×3 matrix)
3. Calculate gradient of z with respect to x using: ∇ₓz = Jgᵀ∇f

---

## Part 6: Hessian Matrices (30 min)

### Exercise 6.1: Computing Hessian
For f(x, y) = x³ + y³ - 3xy

Calculate the Hessian matrix:
```
H = [[∂²f/∂x², ∂²f/∂x∂y],
     [∂²f/∂y∂x, ∂²f/∂y²]]
```

Evaluate at point (1, 1).

### Exercise 6.2: Analyzing Critical Points
For f(x, y) = x² - xy + y² + 2x - y

1. Find critical points (solve ∇f = 0)
2. Calculate Hessian at each critical point
3. Determine nature of each critical point using second derivative test:
   - If det(H) > 0 and ∂²f/∂x² > 0: local minimum
   - If det(H) > 0 and ∂²f/∂x² < 0: local maximum
   - If det(H) < 0: saddle point

---

## Part 7: ML-Specific Gradients (60 min)

### Exercise 7.1: Linear Regression Gradient
For linear regression: ŷ = wx + b
Loss: L = (y - ŷ)² = (y - wx - b)²

Given data point: x = 3, y = 7
Current parameters: w = 1.5, b = 2

1. Calculate predicted value ŷ
2. Calculate loss L
3. Calculate ∂L/∂w (show chain rule steps)
4. Calculate ∂L/∂b
5. Update parameters using α = 0.1:
   - w_new = w - α(∂L/∂w)
   - b_new = b - α(∂L/∂b)

### Exercise 7.2: Logistic Regression Gradient
For binary classification:
- Prediction: ŷ = σ(wx + b) where σ(z) = 1/(1 + e^(-z))
- Binary cross-entropy loss: L = -[y log(ŷ) + (1-y) log(1-ŷ)]

Given: x = 2, y = 1 (true class), w = 0.5, b = 0.5

1. Calculate z = wx + b
2. Calculate ŷ = σ(z)
3. Calculate loss L
4. Calculate ∂L/∂ŷ
5. Calculate ∂ŷ/∂z (remember: σ'(z) = σ(z)(1-σ(z)))
6. Calculate ∂z/∂w and ∂z/∂b
7. Use chain rule: ∂L/∂w = (∂L/∂ŷ)(∂ŷ/∂z)(∂z/∂w)
8. Calculate ∂L/∂b similarly

### Exercise 7.3: Softmax Gradient (Challenging)
For multi-class classification with 3 classes:
```
Logits: z = [z₁, z₂, z₃]ᵀ
Softmax: ŷᵢ = e^(zᵢ) / Σⱼe^(zⱼ)
Cross-entropy: L = -Σᵢ yᵢ log(ŷᵢ)
```

Given: z = [2, 1, 0.5], y = [1, 0, 0] (one-hot encoded, class 0 is correct)

1. Calculate softmax outputs ŷ₁, ŷ₂, ŷ₃
2. Calculate loss L
3. Show that ∂L/∂zᵢ = ŷᵢ - yᵢ (this is a remarkable simplification!)
4. Calculate gradients ∂L/∂z₁, ∂L/∂z₂, ∂L/∂z₃

---

## NumPy Verification

After solving by hand, verify with numerical gradients:

```python
import numpy as np

def numerical_gradient(f, x, h=1e-5):
    """Compute gradient numerically using finite differences"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        x_minus_h = x.copy()
        x_minus_h[i] -= h
        grad[i] = (f(x_plus_h) - f(x_minus_h)) / (2*h)
    return grad

# Example: verify gradient of f(x,y) = x² + y²
def f(x):
    return x[0]**2 + x[1]**2

x = np.array([1.0, 2.0])
numerical_grad = numerical_gradient(f, x)
analytical_grad = 2*x  # True gradient

print("Numerical:", numerical_grad)
print("Analytical:", analytical_grad)
print("Difference:", np.linalg.norm(numerical_grad - analytical_grad))
```

---

## Challenge Problems

### Challenge 1: Newton's Method
Implement one step of Newton's method for minimizing f(x, y) = x² + 4y²

Formula: x_new = x - H⁻¹∇f

Starting from (4, 2), calculate the next point.

### Challenge 2: Batch Gradient
Extend Exercise 7.1 to mini-batch of 3 points:
- (x₁, y₁) = (1, 3)
- (x₂, y₂) = (2, 5)
- (x₃, y₃) = (3, 7)

Calculate average gradient over the batch.

### Challenge 3: Derive Backprop for 2-Layer Network
```
x → h₁ = σ(W₁x + b₁) → h₂ = σ(W₂h₁ + b₂) → L = (h₂ - y)²
```

Derive ∂L/∂W₁, ∂L/∂b₁, ∂L/∂W₂, ∂L/∂b₂ using chain rule.

---

## Submission Checklist

- [ ] All partial derivatives calculated correctly
- [ ] Chain rule applied properly for backprop
- [ ] Gradients verified numerically in NumPy
- [ ] Understanding of gradient descent update rule
- [ ] Can explain backpropagation intuitively

---

## Key Concepts

1. **Gradient** points in direction of steepest ascent
2. **Negative gradient** points toward local minimum
3. **Chain rule** is the foundation of backpropagation
4. **Jacobian** generalizes derivative to vector functions
5. **Hessian** describes local curvature (second derivatives)
6. **Numerical gradients** are slower but useful for verification

Mastering these concepts is essential for understanding deep learning! 🧠
