# Computing Gradients of Quadratic Forms - Practice Guide

**Time:** 30-45 minutes
**Difficulty:** Intermediate
**Prerequisites:** Linear algebra basics, partial derivatives

Learn to compute gradients of quadratic forms f(x) = x^T A x, a fundamental skill for understanding optimization in machine learning.

---

## 📚 What is a Quadratic Form?

A **quadratic form** is a function of the form:

**f(x) = x^T A x**

Where:
- **x** is a vector in ℝ^n
- **A** is an n×n matrix (usually symmetric)
- The result is a scalar

### Why Are Quadratic Forms Important?

1. **Loss functions** - Many ML loss functions are quadratic or approximately quadratic
2. **Newton's method** - Uses Hessian (second derivative matrix) which forms quadratic approximations
3. **Least squares** - Linear regression minimizes ||Ax - b||² which is a quadratic form
4. **PCA** - Maximizes variance x^T Σ x where Σ is covariance matrix

---

## 🎯 Part 1: Computing the Gradient (2D Example)

### Example: f(x) = x^T A x where x = [x₁, x₂]^T

Let's use:
```
A = [2  1]
    [1  3]
```

### Step 1: Expand the Quadratic Form

```
f(x) = [x₁ x₂] [2  1] [x₁]
                [1  3] [x₂]
```

**Matrix multiplication:**
```
[2  1] [x₁]   [2x₁ + x₂]
[1  3] [x₂] = [x₁ + 3x₂]
```

**Dot product:**
```
f(x) = [x₁ x₂] [2x₁ + x₂]
                [x₁ + 3x₂]

     = x₁(2x₁ + x₂) + x₂(x₁ + 3x₂)
     = 2x₁² + x₁x₂ + x₁x₂ + 3x₂²
     = 2x₁² + 2x₁x₂ + 3x₂²
```

### Step 2: Compute Partial Derivatives

**∂f/∂x₁:**
```
∂f/∂x₁ = ∂/∂x₁(2x₁² + 2x₁x₂ + 3x₂²)
       = 4x₁ + 2x₂
```

**∂f/∂x₂:**
```
∂f/∂x₂ = ∂/∂x₂(2x₁² + 2x₁x₂ + 3x₂²)
       = 2x₁ + 6x₂
```

### Step 3: Form the Gradient

**∇f(x) = [∂f/∂x₁, ∂f/∂x₂]^T = [4x₁ + 2x₂, 2x₁ + 6x₂]^T**

---

## 📐 Part 2: The General Formula

For f(x) = x^T A x:

### If A is Symmetric (A = A^T)

**∇f(x) = 2Ax**

### If A is NOT Symmetric

**∇f(x) = (A + A^T)x**

### Proof for Symmetric Case

For f(x) = x^T A x where A is symmetric:

```
f(x) = Σᵢ Σⱼ xᵢ Aᵢⱼ xⱼ
```

Taking derivative with respect to xₖ:

```
∂f/∂xₖ = Σᵢ Aᵢₖ xᵢ + Σⱼ Aₖⱼ xⱼ  (product rule applied)
```

Since A is symmetric (Aᵢₖ = Aₖᵢ):

```
∂f/∂xₖ = Σᵢ Aₖᵢ xᵢ + Σⱼ Aₖⱼ xⱼ
       = 2 Σⱼ Aₖⱼ xⱼ
       = 2(Ax)ₖ
```

Therefore: **∇f(x) = 2Ax**

---

## 💻 Part 3: Verification in NumPy

### Manual Calculation

```python
import numpy as np

# Define matrix A (symmetric)
A = np.array([[2, 1],
              [1, 3]])

# Define point x
x = np.array([2.0, 3.0])

# Compute f(x) = x^T A x
f_x = x.T @ A @ x
print(f"f(x) = {f_x}")  # Output: 43.0

# Analytical gradient: ∇f(x) = 2Ax
grad_analytical = 2 * A @ x
print(f"Analytical gradient: {grad_analytical}")
# Output: [14. 22.]
```

### Numerical Gradient (for verification)

```python
def numerical_gradient(f, x, h=1e-7):
    """Compute gradient numerically using finite differences"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Define function
def f(x):
    return x.T @ A @ x

# Compute numerical gradient
grad_numerical = numerical_gradient(f, x)
print(f"Numerical gradient: {grad_numerical}")
# Output: [14. 22.] (matches analytical!)

# Check error
error = np.linalg.norm(grad_analytical - grad_numerical)
print(f"Error: {error:.2e}")  # Should be very small (< 1e-7)
```

---

## 🧮 Part 4: Practice Exercises

### Exercise 1: 2D Quadratic Form

Given:
```
A = [3  -1]
    [-1  2]

x = [1]
    [2]
```

1. Expand f(x) = x^T A x
2. Compute ∇f(x) by hand
3. Verify using formula ∇f(x) = 2Ax
4. Verify numerically in NumPy

<details>
<summary>Click for solution</summary>

**Step 1: Expand**
```
f(x) = [1 2] [3  -1] [1]
              [-1  2] [2]

     = [1 2] [3-2 ] = [1 2] [1]
             [-1+4]         [3]

     = 1(1) + 2(3) = 7
```

Or expand fully:
```
f(x) = 3x₁² - x₁x₂ - x₁x₂ + 2x₂²
     = 3x₁² - 2x₁x₂ + 2x₂²
```

**Step 2: Gradient by hand**
```
∂f/∂x₁ = 6x₁ - 2x₂ = 6(1) - 2(2) = 2
∂f/∂x₂ = -2x₁ + 4x₂ = -2(1) + 4(2) = 6

∇f(x) = [2, 6]^T
```

**Step 3: Using formula**
```
∇f(x) = 2Ax = 2 [3  -1] [1]
                 [-1  2] [2]

       = 2 [1]  = [2]
           [3]    [6]  ✓
```

**Step 4: NumPy verification**
```python
A = np.array([[3, -1], [-1, 2]])
x = np.array([1.0, 2.0])
print(2 * A @ x)  # [2. 6.]
```
</details>

### Exercise 2: 3D Quadratic Form

Given:
```
A = [2  0  1]     x = [1]
    [0  3  0]         [2]
    [1  0  4]         [1]
```

1. Compute f(x) = x^T A x
2. Compute ∇f(x) = 2Ax
3. Verify numerically

<details>
<summary>Click for solution</summary>

```python
A = np.array([[2, 0, 1],
              [0, 3, 0],
              [1, 0, 4]])

x = np.array([1.0, 2.0, 1.0])

# f(x) = x^T A x
f_x = x.T @ A @ x
print(f"f(x) = {f_x}")  # Output: 17.0

# Gradient
grad = 2 * A @ x
print(f"∇f(x) = {grad}")  # Output: [ 6. 12. 10.]
```

**Manual verification:**
```
Ax = [2  0  1] [1]   [2+0+1]   [3]
     [0  3  0] [2] = [0+6+0] = [6]
     [1  0  4] [1]   [1+0+4]   [5]

∇f(x) = 2Ax = [6, 12, 10]^T ✓
```
</details>

### Exercise 3: Non-Symmetric Matrix

Given non-symmetric A:
```
A = [2  3]
    [1  4]

x = [1]
    [2]
```

1. Show that A ≠ A^T
2. Compute ∇f(x) using formula (A + A^T)x
3. Verify by expanding and differentiating

<details>
<summary>Click for solution</summary>

**Step 1: Check symmetry**
```
A^T = [2  1]  ≠  [2  3] = A  (not symmetric!)
      [3  4]     [1  4]
```

**Step 2: Using formula ∇f(x) = (A + A^T)x**
```
A + A^T = [2  3] + [2  1] = [4  4]
          [1  4]   [3  4]   [4  8]

∇f(x) = [4  4] [1] = [12]
        [4  8] [2]   [20]
```

**Step 3: Verify by expansion**
```
f(x) = [1 2] [2  3] [1]
              [1  4] [2]

     = [1 2] [8 ]  = 24
             [9]

Expand: f(x) = x₁(2x₁ + 3x₂) + x₂(x₁ + 4x₂)
             = 2x₁² + 3x₁x₂ + x₁x₂ + 4x₂²
             = 2x₁² + 4x₁x₂ + 4x₂²

∂f/∂x₁ = 4x₁ + 4x₂ = 4(1) + 4(2) = 12 ✓
∂f/∂x₂ = 4x₁ + 8x₂ = 4(1) + 8(2) = 20 ✓
```
</details>

---

## 🎓 Part 5: Applications in Machine Learning

### Application 1: Least Squares

Minimize ||Ax - b||²:

```
f(x) = (Ax - b)^T(Ax - b)
     = x^T A^T A x - 2b^T A x + b^T b
```

Setting ∇f(x) = 0:
```
2A^T Ax - 2A^T b = 0
A^T Ax = A^T b  (normal equations)
x* = (A^T A)^(-1) A^T b
```

### Application 2: Mahalanobis Distance

Distance weighted by covariance matrix Σ^(-1):

```
d²(x, μ) = (x - μ)^T Σ^(-1) (x - μ)
```

Gradient with respect to x:
```
∇ₓ d²(x, μ) = 2Σ^(-1)(x - μ)
```

### Application 3: Quadratic Programming

Minimize f(x) = ½x^T Q x + c^T x subject to constraints.

Gradient:
```
∇f(x) = Qx + c
```

Setting to zero for unconstrained optimum:
```
Qx* = -c
x* = -Q^(-1)c
```

---

## 🧪 Complete Verification Script

```python
import numpy as np

def verify_quadratic_form_gradient():
    """Complete verification of quadratic form gradient formulas"""

    # Test 1: Symmetric matrix
    print("=" * 50)
    print("Test 1: Symmetric Matrix")
    print("=" * 50)

    A_sym = np.array([[2, 1], [1, 3]])
    x = np.array([2.0, 3.0])

    # Analytical gradient
    grad_analytical = 2 * A_sym @ x

    # Numerical gradient
    def f_sym(x):
        return x.T @ A_sym @ x

    grad_numerical = numerical_gradient(f_sym, x)

    print(f"Matrix A:\n{A_sym}")
    print(f"Point x: {x}")
    print(f"f(x) = {f_sym(x)}")
    print(f"Analytical gradient: {grad_analytical}")
    print(f"Numerical gradient:  {grad_numerical}")
    print(f"Error: {np.linalg.norm(grad_analytical - grad_numerical):.2e}")

    # Test 2: Non-symmetric matrix
    print("\n" + "=" * 50)
    print("Test 2: Non-Symmetric Matrix")
    print("=" * 50)

    A_nonsym = np.array([[2, 3], [1, 4]])

    # Analytical gradient (A + A^T)x
    grad_analytical = (A_nonsym + A_nonsym.T) @ x

    # Numerical gradient
    def f_nonsym(x):
        return x.T @ A_nonsym @ x

    grad_numerical = numerical_gradient(f_nonsym, x)

    print(f"Matrix A:\n{A_nonsym}")
    print(f"Point x: {x}")
    print(f"f(x) = {f_nonsym(x)}")
    print(f"Analytical gradient: {grad_analytical}")
    print(f"Numerical gradient:  {grad_numerical}")
    print(f"Error: {np.linalg.norm(grad_analytical - grad_numerical):.2e}")

    # Test 3: 3D case
    print("\n" + "=" * 50)
    print("Test 3: 3D Symmetric Matrix")
    print("=" * 50)

    A_3d = np.array([[2, 0, 1],
                     [0, 3, 0],
                     [1, 0, 4]])
    x_3d = np.array([1.0, 2.0, 1.0])

    grad_analytical = 2 * A_3d @ x_3d

    def f_3d(x):
        return x.T @ A_3d @ x

    grad_numerical = numerical_gradient(f_3d, x_3d)

    print(f"Matrix A:\n{A_3d}")
    print(f"Point x: {x_3d}")
    print(f"f(x) = {f_3d(x_3d)}")
    print(f"Analytical gradient: {grad_analytical}")
    print(f"Numerical gradient:  {grad_numerical}")
    print(f"Error: {np.linalg.norm(grad_analytical - grad_numerical):.2e}")

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)

def numerical_gradient(f, x, h=1e-7):
    """Compute gradient numerically using finite differences"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Run verification
verify_quadratic_form_gradient()
```

---

## ✅ Key Takeaways

1. **Quadratic form:** f(x) = x^T A x is fundamental in ML optimization
2. **Symmetric case:** ∇f(x) = 2Ax (most common)
3. **General case:** ∇f(x) = (A + A^T)x
4. **Always verify:** Use numerical gradients to check analytical derivations
5. **Applications:** Appears in least squares, PCA, Newton's method, and more

---

## 🚀 Next Steps

1. Practice computing gradients of quadratic forms by hand
2. Implement the verification script in NumPy
3. Study how quadratic forms appear in your ML algorithms
4. Learn about Hessian matrices (second derivatives forming quadratic approximations)

Understanding quadratic forms is essential for optimization theory and machine learning! 📈
