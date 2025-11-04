# Matrix Calculus & Gradients - Solutions (Module 2)

**Time:** Reference for 3-4 hours of exercises
**Difficulty:** Intermediate

Complete solutions to exercises in `guides/exercises/calculus_gradients_exercises.md`

---

## Part 1: Scalar Derivatives Review - Solutions

### Exercise 1.1: Basic Derivatives

**1. f(x) = 3x² + 2x - 5**

f'(x) = d/dx(3x²) + d/dx(2x) - d/dx(5)
     = 3(2x) + 2 - 0
     = **6x + 2**

**2. f(x) = x³ - 4x² + x**

f'(x) = d/dx(x³) - d/dx(4x²) + d/dx(x)
     = 3x² - 4(2x) + 1
     = **3x² - 8x + 1**

**3. f(x) = 1/x² = x⁻²**

f'(x) = d/dx(x⁻²)
     = -2x⁻³
     = **-2/x³**

**4. f(x) = e^(2x)**

Using chain rule:
f'(x) = e^(2x) · d/dx(2x)
     = e^(2x) · 2
     = **2e^(2x)**

**5. f(x) = ln(x²)**

Using chain rule:
f'(x) = (1/x²) · d/dx(x²)
     = (1/x²) · 2x
     = **2/x**

Alternatively: ln(x²) = 2ln(x), so f'(x) = 2/x ✓

---

### Exercise 1.2: Chain Rule

**1. f(x) = (3x + 2)⁴**

Let u = 3x + 2, then f = u⁴

f'(x) = d/du(u⁴) · du/dx
     = 4u³ · 3
     = **12(3x + 2)³**

**2. f(x) = e^(x²)**

Let u = x², then f = e^u

f'(x) = e^u · du/dx
     = e^(x²) · 2x
     = **2x · e^(x²)**

**3. f(x) = ln(2x + 1)**

Let u = 2x + 1, then f = ln(u)

f'(x) = (1/u) · du/dx
     = 1/(2x + 1) · 2
     = **2/(2x + 1)**

**4. f(x) = sin(3x²)**

Let u = 3x², then f = sin(u)

f'(x) = cos(u) · du/dx
     = cos(3x²) · 6x
     = **6x · cos(3x²)**

---

### Exercise 1.3: Product and Quotient Rules

**1. f(x) = x² · e^x** (Product rule)

u = x², u' = 2x
v = e^x, v' = e^x

f'(x) = u'v + uv'
     = 2x · e^x + x² · e^x
     = **e^x(2x + x²)**
     = **e^x · x(x + 2)**

**2. f(x) = x³ · ln(x)** (Product rule)

u = x³, u' = 3x²
v = ln(x), v' = 1/x

f'(x) = u'v + uv'
     = 3x² · ln(x) + x³ · (1/x)
     = 3x² ln(x) + x²
     = **x²(3ln(x) + 1)**

**3. f(x) = (x² + 1)/(x - 1)** (Quotient rule)

u = x² + 1, u' = 2x
v = x - 1, v' = 1

f'(x) = (u'v - uv')/v²
     = [2x(x - 1) - (x² + 1)(1)]/(x - 1)²
     = [2x² - 2x - x² - 1]/(x - 1)²
     = **(x² - 2x - 1)/(x - 1)²**

---

## Part 2: Partial Derivatives - Solutions

### Exercise 2.1: Basic Partial Derivatives

**f(x, y) = x²y + 3xy² - 2x + y**

**1. ∂f/∂x** (treat y as constant):
∂f/∂x = 2xy + 3y² - 2

**2. ∂f/∂y** (treat x as constant):
∂f/∂y = x² + 6xy + 1

**3. Evaluate at (1, 2):**

∂f/∂x|(1,2) = 2(1)(2) + 3(2)² - 2
            = 4 + 12 - 2
            = **14**

∂f/∂y|(1,2) = (1)² + 6(1)(2) + 1
            = 1 + 12 + 1
            = **14**

---

### Exercise 2.2: More Partial Derivatives

**f(x, y) = e^(xy) + x²y³**

**1. ∂f/∂x:**
∂f/∂x = e^(xy) · y + 2xy³
     = **ye^(xy) + 2xy³**

**2. ∂f/∂y:**
∂f/∂y = e^(xy) · x + 3x²y²
     = **xe^(xy) + 3x²y²**

**3. Evaluate at (0, 1):**

∂f/∂x|(0,1) = 1 · e^(0·1) + 2(0)(1)³
            = 1 · 1 + 0
            = **1**

∂f/∂y|(0,1) = 0 · e^(0·1) + 3(0)²(1)²
            = 0 + 0
            = **0**

---

### Exercise 2.3: Second-Order Partial Derivatives

**f(x, y) = x³y² - 2xy + 5**

**First-order partials:**
∂f/∂x = 3x²y² - 2y
∂f/∂y = 2x³y - 2x

**1. ∂²f/∂x²:**
∂²f/∂x² = ∂/∂x(3x²y² - 2y)
        = **6xy²**

**2. ∂²f/∂y²:**
∂²f/∂y² = ∂/∂y(2x³y - 2x)
        = **2x³**

**3. ∂²f/∂x∂y** (differentiate ∂f/∂x with respect to y):
∂²f/∂x∂y = ∂/∂y(3x²y² - 2y)
          = 6x²y - 2
          = **6x²y - 2**

**4. ∂²f/∂y∂x** (differentiate ∂f/∂y with respect to x):
∂²f/∂y∂x = ∂/∂x(2x³y - 2x)
          = 6x²y - 2
          = **6x²y - 2**

**5. Verification:**
∂²f/∂x∂y = ∂²f/∂y∂x = 6x²y - 2 ✓

This confirms Clairaut's theorem (mixed partials are equal for continuous functions).

---

## Part 3: Gradients - Solutions

### Exercise 3.1: Computing Gradients

**f(x, y) = x² + y² - 2x - 4y + 5**

**1. Calculate gradient:**

∂f/∂x = 2x - 2
∂f/∂y = 2y - 4

**∇f = [2x - 2, 2y - 4]ᵀ**

**2. Gradient at (1, 2):**

∇f(1, 2) = [2(1) - 2, 2(2) - 4]
         = **[0, 0]ᵀ**

**3. Critical points (where ∇f = 0):**

2x - 2 = 0  ⟹  x = 1
2y - 4 = 0  ⟹  y = 2

**Critical point: (1, 2)**

**4. Classify critical point:**

Hessian matrix:
H = [∂²f/∂x²    ∂²f/∂x∂y]   [2  0]
    [∂²f/∂y∂x   ∂²f/∂y²  ] = [0  2]

Both eigenvalues are 2 > 0 (or simply: det(H) = 4 > 0 and ∂²f/∂x² = 2 > 0)

**Answer: Local minimum at (1, 2)**

Function value: f(1, 2) = 1 + 4 - 2 - 8 + 5 = 0

---

### Exercise 3.2: Gradient of Quadratic Form

**f(x) = xᵀAx where x = [x₁, x₂]ᵀ and A = [[2, 1], [1, 3]]**

**1. Expand f(x):**

f(x) = [x₁ x₂] [2  1] [x₁]
                [1  3] [x₂]

     = [x₁ x₂] [2x₁ + x₂ ]
                [x₁ + 3x₂]

     = x₁(2x₁ + x₂) + x₂(x₁ + 3x₂)
     = 2x₁² + x₁x₂ + x₁x₂ + 3x₂²
     = **2x₁² + 2x₁x₂ + 3x₂²**

**2. ∂f/∂x₁:**
∂f/∂x₁ = 4x₁ + 2x₂

**3. ∂f/∂x₂:**
∂f/∂x₂ = 2x₁ + 6x₂

**4. Gradient:**
**∇f(x) = [4x₁ + 2x₂, 2x₁ + 6x₂]ᵀ**

**5. Verify formula ∇f(x) = (A + Aᵀ)x:**

A + Aᵀ = [2  1] + [2  1] = [4  2]
         [1  3]   [1  3]   [2  6]

(A + Aᵀ)x = [4  2] [x₁] = [4x₁ + 2x₂]
            [2  6] [x₂]   [2x₁ + 6x₂]

**Verified! ✓**

Note: For symmetric matrices (A = Aᵀ), this simplifies to ∇f(x) = 2Ax

---

### Exercise 3.3: Gradient Descent Step

**f(x, y) = x² + 4y²**
**Starting point: (x₀, y₀) = (4, 2)**
**Learning rate: α = 0.1**

**1. Gradient at (4, 2):**

∂f/∂x = 2x
∂f/∂y = 8y

∇f(4, 2) = [2(4), 8(2)]
         = **[8, 16]ᵀ**

**2. Gradient descent update:**

[x₁]   [x₀]       [∂f/∂x]
[y₁] = [y₀] - α · [∂f/∂y]

     = [4] - 0.1 · [8 ]
       [2]         [16]

     = [4] - [0.8]
       [2]   [1.6]

     = **[3.2, 0.4]ᵀ**

**3. Function values:**

f(4, 2) = (4)² + 4(2)² = 16 + 16 = **32**

f(3.2, 0.4) = (3.2)² + 4(0.4)² = 10.24 + 0.64 = **10.88**

**Yes, function decreased from 32 to 10.88! ✓**

**4. Gradient at new point (3.2, 0.4):**

∇f(3.2, 0.4) = [2(3.2), 8(0.4)]
             = **[6.4, 3.2]ᵀ**

Gradient magnitude decreased from √(8² + 16²) = 17.89 to √(6.4² + 3.2²) = 7.16

---

## Part 4: Chain Rule for Multivariable Functions - Solutions

### Exercise 4.1: Simple Chain Rule

**z = f(x, y) = x² + y²**
**x = 2t, y = 3t**

**Find dz/dt:**

dz/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt)

∂f/∂x = 2x
∂f/∂y = 2y
dx/dt = 2
dy/dt = 3

dz/dt = 2x · 2 + 2y · 3
      = 4x + 6y
      = 4(2t) + 6(3t)
      = 8t + 18t
      = **26t**

**Verification:** z = (2t)² + (3t)² = 4t² + 9t² = 13t²
So dz/dt = 26t ✓

---

### Exercise 4.2: Backpropagation Example

**Network:**
```
x → h = σ(wx + b)  where σ(z) = 1/(1 + e^(-z))
h → y = h²
y → L = (y - t)²  where t is target
```

**Given:** x = 2, w = 0.5, b = 1, t = 0.8

**1. Forward pass:**

**z = wx + b**
z = 0.5(2) + 1 = 2

**h = σ(z) = 1/(1 + e^(-z))**
h = 1/(1 + e^(-2))
h = 1/(1 + 0.1353)
h ≈ **0.8808**

**y = h²**
y = (0.8808)²
y ≈ **0.7758**

**L = (y - t)²**
L = (0.7758 - 0.8)²
L = (-0.0242)²
L ≈ **0.000586**

**2. Backward pass:**

**∂L/∂y:**
∂L/∂y = 2(y - t)
      = 2(0.7758 - 0.8)
      = 2(-0.0242)
      = **-0.0484**

**∂y/∂h:**
∂y/∂h = 2h
      = 2(0.8808)
      = **1.7616**

**∂h/∂w:**
∂h/∂z = σ'(z) = σ(z)(1 - σ(z))
      = 0.8808(1 - 0.8808)
      = 0.8808 × 0.1192
      = **0.1050**

∂z/∂w = x = 2

∂h/∂w = (∂h/∂z)(∂z/∂w)
      = 0.1050 × 2
      = **0.2100**

**∂h/∂b:**
∂z/∂b = 1

∂h/∂b = (∂h/∂z)(∂z/∂b)
      = 0.1050 × 1
      = **0.1050**

**Chain rule - ∂L/∂w:**
∂L/∂w = (∂L/∂y)(∂y/∂h)(∂h/∂w)
      = (-0.0484)(1.7616)(0.2100)
      = **-0.0179**

**Chain rule - ∂L/∂b:**
∂L/∂b = (∂L/∂y)(∂y/∂h)(∂h/∂b)
      = (-0.0484)(1.7616)(0.1050)
      = **-0.0090**

**Interpretation:** Both gradients are negative, so to reduce loss, we should *increase* both w and b.

---

### Exercise 4.3: Vector Chain Rule

**Given:**
- z = f(y) = y₁² + y₂²
- y = g(x) = [2x₁ + x₂, x₁ - x₂]ᵀ

**Find ∂z/∂x₁ and ∂z/∂x₂:**

**Chain rule:** ∂z/∂xᵢ = Σⱼ (∂z/∂yⱼ)(∂yⱼ/∂xᵢ)

**Compute gradients:**

∂z/∂y₁ = 2y₁
∂z/∂y₂ = 2y₂

∂y₁/∂x₁ = 2,  ∂y₁/∂x₂ = 1
∂y₂/∂x₁ = 1,  ∂y₂/∂x₂ = -1

**∂z/∂x₁:**
∂z/∂x₁ = (∂z/∂y₁)(∂y₁/∂x₁) + (∂z/∂y₂)(∂y₂/∂x₁)
       = 2y₁ · 2 + 2y₂ · 1
       = 4y₁ + 2y₂
       = 4(2x₁ + x₂) + 2(x₁ - x₂)
       = 8x₁ + 4x₂ + 2x₁ - 2x₂
       = **10x₁ + 2x₂**

**∂z/∂x₂:**
∂z/∂x₂ = (∂z/∂y₁)(∂y₁/∂x₂) + (∂z/∂y₂)(∂y₂/∂x₂)
       = 2y₁ · 1 + 2y₂ · (-1)
       = 2y₁ - 2y₂
       = 2(2x₁ + x₂) - 2(x₁ - x₂)
       = 4x₁ + 2x₂ - 2x₁ + 2x₂
       = **2x₁ + 4x₂**

---

## Part 5: Jacobian Matrices - Solutions

### Exercise 5.1: Computing Jacobian

**f: ℝ² → ℝ³ defined by:**
```
f₁(x₁, x₂) = x₁² + x₂
f₂(x₁, x₂) = x₁x₂
f₃(x₁, x₂) = x₁ + 2x₂²
```

**Jacobian matrix:**

J = [∂f₁/∂x₁  ∂f₁/∂x₂]   [2x₁    1  ]
    [∂f₂/∂x₁  ∂f₂/∂x₂] = [x₂     x₁ ]
    [∂f₃/∂x₁  ∂f₃/∂x₂]   [1      4x₂]

**Evaluate at (1, 2):**

**J(1, 2) = [2(1)   1  ]   [2  1]**
          **[2      1  ] = [2  1]**
          **[1    4(2)]   [1  8]**

---

### Exercise 5.2: Chain Rule with Jacobians

**Given:**
- z = f(y): ℝ² → ℝ where f(y₁, y₂) = y₁² + 2y₂²
- y = g(x): ℝ³ → ℝ² where g(x₁, x₂, x₃) = [x₁ + x₂, x₂x₃]ᵀ

**1. Calculate ∇f (gradient of f):**

∂f/∂y₁ = 2y₁
∂f/∂y₂ = 4y₂

**∇f = [2y₁, 4y₂]ᵀ** (2×1 vector)

**2. Calculate Jacobian of g:**

y₁ = x₁ + x₂
y₂ = x₂x₃

Jg = [∂y₁/∂x₁  ∂y₁/∂x₂  ∂y₁/∂x₃]   [1   1   0 ]
     [∂y₂/∂x₁  ∂y₂/∂x₂  ∂y₂/∂x₃] = [0   x₃  x₂]

**Jg is 2×3 matrix**

**3. Calculate ∇ₓz = Jgᵀ∇f:**

Jgᵀ = [1    0 ]
      [1    x₃]
      [0    x₂]

∇ₓz = [1    0 ] [2y₁]   [2y₁        ]
      [1    x₃] [4y₂] = [2y₁ + 4x₃y₂]
      [0    x₂]         [4x₂y₂      ]

Substituting y₁ = x₁ + x₂ and y₂ = x₂x₃:

**∇ₓz = [2(x₁ + x₂), 2(x₁ + x₂) + 4x₃(x₂x₃), 4x₂(x₂x₃)]ᵀ**
     **= [2x₁ + 2x₂, 2x₁ + 2x₂ + 4x₂x₃², 4x₂²x₃]ᵀ**

---

## Part 6: Hessian Matrices - Solutions

### Exercise 6.1: Computing Hessian

**f(x, y) = x³ + y³ - 3xy**

**First-order partials:**
∂f/∂x = 3x² - 3y
∂f/∂y = 3y² - 3x

**Second-order partials:**
∂²f/∂x² = 6x
∂²f/∂y² = 6y
∂²f/∂x∂y = -3
∂²f/∂y∂x = -3

**Hessian matrix:**

**H = [6x   -3]**
    **[-3   6y]**

**Evaluate at (1, 1):**

**H(1, 1) = [6(1)  -3]   [6  -3]**
          **[-3    6(1)] = [-3  6]**

---

### Exercise 6.2: Analyzing Critical Points

**f(x, y) = x² - xy + y² + 2x - y**

**1. Find critical points:**

∂f/∂x = 2x - y + 2 = 0
∂f/∂y = -x + 2y - 1 = 0

From second equation: x = 2y - 1

Substitute into first:
2(2y - 1) - y + 2 = 0
4y - 2 - y + 2 = 0
3y = 0
y = 0

Then: x = 2(0) - 1 = -1

**Critical point: (-1, 0)**

**2. Calculate Hessian:**

∂²f/∂x² = 2
∂²f/∂y² = 2
∂²f/∂x∂y = -1
∂²f/∂y∂x = -1

**H = [2   -1]**
    **[-1   2]**

**3. Classify critical point:**

det(H) = (2)(2) - (-1)(-1) = 4 - 1 = 3 > 0
∂²f/∂x² = 2 > 0

Since det(H) > 0 and ∂²f/∂x² > 0:

**Answer: Local minimum at (-1, 0)**

Function value: f(-1, 0) = 1 - 0 + 0 - 2 - 0 = -1

---

## Part 7: ML-Specific Gradients - Solutions

### Exercise 7.1: Linear Regression Gradient

**Model:** ŷ = wx + b
**Loss:** L = (y - ŷ)²
**Data:** x = 3, y = 7
**Parameters:** w = 1.5, b = 2
**Learning rate:** α = 0.1

**1. Calculate predicted value:**
ŷ = wx + b = 1.5(3) + 2 = 4.5 + 2 = **6.5**

**2. Calculate loss:**
L = (y - ŷ)² = (7 - 6.5)² = (0.5)² = **0.25**

**3. Calculate ∂L/∂w:**

L = (y - wx - b)²

Let u = y - wx - b, then L = u²

∂L/∂u = 2u
∂u/∂w = -x

**∂L/∂w = (∂L/∂u)(∂u/∂w)**
       = 2(y - wx - b)(-x)
       = -2x(y - wx - b)

At current values:
∂L/∂w = -2(3)(7 - 1.5(3) - 2)
      = -6(7 - 4.5 - 2)
      = -6(0.5)
      = **-3**

**4. Calculate ∂L/∂b:**

∂u/∂b = -1

**∂L/∂b = (∂L/∂u)(∂u/∂b)**
       = 2(y - wx - b)(-1)
       = -2(y - wx - b)

At current values:
∂L/∂b = -2(0.5)
      = **-1**

**5. Update parameters:**

w_new = w - α(∂L/∂w)
      = 1.5 - 0.1(-3)
      = 1.5 + 0.3
      = **1.8**

b_new = b - α(∂L/∂b)
      = 2 - 0.1(-1)
      = 2 + 0.1
      = **2.1**

**Verification:** New prediction: ŷ = 1.8(3) + 2.1 = 5.4 + 2.1 = 7.5 (closer to target 7!) ✓

---

### Exercise 7.2: Logistic Regression Gradient

**Model:** ŷ = σ(z) where z = wx + b and σ(z) = 1/(1 + e^(-z))
**Loss:** L = -[y log(ŷ) + (1-y) log(1-ŷ)]
**Given:** x = 2, y = 1, w = 0.5, b = 0.5

**1. Calculate z:**
z = wx + b = 0.5(2) + 0.5 = 1 + 0.5 = **1.5**

**2. Calculate ŷ:**
ŷ = σ(1.5) = 1/(1 + e^(-1.5))
  = 1/(1 + 0.2231)
  ≈ **0.8176**

**3. Calculate loss:**
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
  = -[1 · log(0.8176) + 0 · log(0.1824)]
  = -log(0.8176)
  ≈ **0.2014**

**4. Calculate ∂L/∂ŷ:**
∂L/∂ŷ = -[y/ŷ - (1-y)/(1-ŷ)]
      = -[1/0.8176 - 0/0.1824]
      = -1.2231
      ≈ **-1.223**

**5. Calculate ∂ŷ/∂z:**
σ'(z) = σ(z)(1 - σ(z))
      = 0.8176(1 - 0.8176)
      = 0.8176 × 0.1824
      ≈ **0.1491**

**6. Calculate ∂z/∂w and ∂z/∂b:**
∂z/∂w = x = **2**
∂z/∂b = **1**

**7. Chain rule - ∂L/∂w:**
∂L/∂w = (∂L/∂ŷ)(∂ŷ/∂z)(∂z/∂w)
      = (-1.223)(0.1491)(2)
      ≈ **-0.3648**

**8. Chain rule - ∂L/∂b:**
∂L/∂b = (∂L/∂ŷ)(∂ŷ/∂z)(∂z/∂b)
      = (-1.223)(0.1491)(1)
      ≈ **-0.1824**

**Simplified form (bonus):**
∂L/∂w = (ŷ - y) · x = (0.8176 - 1) · 2 = -0.3648 ✓
∂L/∂b = (ŷ - y) = 0.8176 - 1 = -0.1824 ✓

---

### Exercise 7.3: Softmax Gradient

**Given:** z = [2, 1, 0.5], y = [1, 0, 0] (class 0 is correct)

**1. Calculate softmax outputs:**

e^(z₁) = e^2 ≈ 7.389
e^(z₂) = e^1 ≈ 2.718
e^(z₃) = e^0.5 ≈ 1.649

Sum = 7.389 + 2.718 + 1.649 = 11.756

ŷ₁ = 7.389/11.756 ≈ **0.6285**
ŷ₂ = 2.718/11.756 ≈ **0.2312**
ŷ₃ = 1.649/11.756 ≈ **0.1403**

Verification: 0.6285 + 0.2312 + 0.1403 = 1.0000 ✓

**2. Calculate loss:**
L = -Σᵢ yᵢ log(ŷᵢ)
  = -(1 · log(0.6285) + 0 · log(0.2312) + 0 · log(0.1403))
  = -log(0.6285)
  ≈ **0.4644**

**3. Show that ∂L/∂zᵢ = ŷᵢ - yᵢ:**

For softmax + cross-entropy, the gradient simplifies beautifully:

∂L/∂zᵢ = ŷᵢ - yᵢ

This is a well-known result! The derivation involves:
- ∂L/∂ŷᵢ = -yᵢ/ŷᵢ
- ∂ŷᵢ/∂zⱼ = ŷᵢ(δᵢⱼ - ŷⱼ) where δᵢⱼ is Kronecker delta
- Chain rule over all outputs

The terms magically cancel to give the simple form!

**4. Calculate gradients:**

∂L/∂z₁ = ŷ₁ - y₁ = 0.6285 - 1 = **-0.3715**
∂L/∂z₂ = ŷ₂ - y₂ = 0.2312 - 0 = **0.2312**
∂L/∂z₃ = ŷ₃ - y₃ = 0.1403 - 0 = **0.1403**

**Interpretation:**
- Class 0 gradient is negative (we want to increase logit for correct class)
- Classes 1, 2 gradients are positive (we want to decrease logits for wrong classes)

---

## NumPy Verification

```python
import numpy as np

# Verify Exercise 4.1: Chain rule
def z_func(t):
    x = 2*t
    y = 3*t
    return x**2 + y**2

t = 1.0
h = 1e-7
numerical = (z_func(t + h) - z_func(t - h)) / (2*h)
analytical = 26 * t
print(f"Ex 4.1 - Numerical: {numerical:.6f}, Analytical: {analytical:.6f}")
# Output: 26.000000, 26.000000 ✓

# Verify Exercise 7.3: Softmax gradient
z = np.array([2.0, 1.0, 0.5])
y = np.array([1, 0, 0])

# Forward pass
exp_z = np.exp(z)
softmax = exp_z / np.sum(exp_z)
loss = -np.sum(y * np.log(softmax))

# Gradient
grad = softmax - y

print(f"Ex 7.3 - Softmax: {softmax}")
print(f"Ex 7.3 - Loss: {loss:.4f}")
print(f"Ex 7.3 - Gradient: {grad}")
# Output matches our hand calculations ✓
```

---

## Challenge Problems - Solutions

### Challenge 1: Newton's Method

**f(x, y) = x² + 4y²**
**Starting point:** (4, 2)

**Formula:** x_new = x - H⁻¹∇f

**Gradient:**
∇f = [2x, 8y]ᵀ

At (4, 2):
∇f(4, 2) = [8, 16]ᵀ

**Hessian:**
H = [∂²f/∂x²    ∂²f/∂x∂y]   [2  0]
    [∂²f/∂y∂x   ∂²f/∂y²  ] = [0  8]

**Inverse Hessian:**
H⁻¹ = [1/2   0  ]
      [0     1/8]

**Newton step:**
[x_new]   [4]       [1/2   0  ] [8 ]
[y_new] = [2] - [0     1/8] [16]

        = [4] - [4]
          [2]   [2]

        = **[0, 0]ᵀ**

**Answer: Newton's method finds the global minimum (0, 0) in one step!**

This is because f is quadratic, and Newton's method is exact for quadratic functions.

---

### Challenge 2: Batch Gradient

**Data:**
- (x₁, y₁) = (1, 3)
- (x₂, y₂) = (2, 5)
- (x₃, y₃) = (3, 7)

**Model:** ŷ = wx + b
**Loss for single point:** Lᵢ = (yᵢ - wxᵢ - b)²

**Current parameters:** w = 1.5, b = 2 (from Exercise 7.1)

**Gradient for each point:**

For point i:
∂Lᵢ/∂w = -2xᵢ(yᵢ - wxᵢ - b)
∂Lᵢ/∂b = -2(yᵢ - wxᵢ - b)

**Point 1:** (1, 3)
ŷ₁ = 1.5(1) + 2 = 3.5
∂L₁/∂w = -2(1)(3 - 3.5) = -2(1)(-0.5) = 1
∂L₁/∂b = -2(-0.5) = 1

**Point 2:** (2, 5)
ŷ₂ = 1.5(2) + 2 = 5
∂L₂/∂w = -2(2)(5 - 5) = 0
∂L₂/∂b = -2(0) = 0

**Point 3:** (3, 7)
ŷ₃ = 1.5(3) + 2 = 6.5
∂L₃/∂w = -2(3)(7 - 6.5) = -6(0.5) = -3
∂L₃/∂b = -2(0.5) = -1

**Average gradient (mini-batch):**

∂L/∂w = (1 + 0 + (-3))/3 = -2/3 ≈ **-0.667**
∂L/∂b = (1 + 0 + (-1))/3 = 0/3 = **0**

**Update (α = 0.1):**

w_new = 1.5 - 0.1(-0.667) = 1.5 + 0.0667 = **1.567**
b_new = 2 - 0.1(0) = **2**

---

### Challenge 3: Derive Backprop for 2-Layer Network

**Network:**
```
x → z₁ = W₁x + b₁ → h₁ = σ(z₁) → z₂ = W₂h₁ + b₂ → h₂ = σ(z₂) → L = (h₂ - y)²
```

**Notation:**
- σ'(z) = σ(z)(1 - σ(z))
- δ₂ = ∂L/∂z₂
- δ₁ = ∂L/∂z₁

**Backward pass:**

**1. ∂L/∂h₂:**
∂L/∂h₂ = 2(h₂ - y)

**2. ∂L/∂z₂ = δ₂:**
δ₂ = (∂L/∂h₂)(∂h₂/∂z₂)
   = 2(h₂ - y) · σ'(z₂)
   = 2(h₂ - y) · h₂(1 - h₂)

**3. ∂L/∂W₂:**
∂L/∂W₂ = δ₂ · h₁ᵀ

**4. ∂L/∂b₂:**
∂L/∂b₂ = δ₂

**5. ∂L/∂h₁:**
∂L/∂h₁ = W₂ᵀ · δ₂

**6. ∂L/∂z₁ = δ₁:**
δ₁ = (∂L/∂h₁) ⊙ σ'(z₁)
   = (W₂ᵀδ₂) ⊙ [h₁ ⊙ (1 - h₁)]

(⊙ denotes element-wise multiplication)

**7. ∂L/∂W₁:**
∂L/∂W₁ = δ₁ · xᵀ

**8. ∂L/∂b₁:**
∂L/∂b₁ = δ₁

**Summary:**
```
Forward: x → z₁ → h₁ → z₂ → h₂ → L
Backward: ∂L/∂W₁, ∂L/∂b₁ ← δ₁ ← δ₂ ← ∂L/∂h₂
```

This is the essence of backpropagation! 🧠

---

## Key Takeaways

1. **Chain rule** is fundamental - all backpropagation uses it
2. **Gradient** points in direction of steepest ascent; negative gradient descends
3. **For ML:** Most common pattern is ∂Loss/∂weight = (prediction - target) × input
4. **Jacobians** generalize gradients to vector-valued functions
5. **Hessians** describe curvature (convexity) of loss surface
6. **Always verify gradients numerically** when implementing new architectures!

Practice until computing gradients becomes second nature! 🚀
