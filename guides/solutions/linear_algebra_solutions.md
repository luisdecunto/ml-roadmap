# Linear Algebra Exercise Solutions - Module 1

**Complete solutions with step-by-step work**

---

## Part 1: Vectors and Vector Spaces

### Solution 1.1: Vector Operations

Given: **a** = [2, 3, -1], **b** = [1, -2, 4], **c** = [-3, 0, 2]

**1. a + b:**
```
a + b = [2, 3, -1] + [1, -2, 4]
      = [2+1, 3+(-2), -1+4]
      = [3, 1, 3]
```

**2. 2a - 3b:**
```
2a = 2[2, 3, -1] = [4, 6, -2]
3b = 3[1, -2, 4] = [3, -6, 12]

2a - 3b = [4, 6, -2] - [3, -6, 12]
        = [4-3, 6-(-6), -2-12]
        = [1, 12, -14]
```

**3. a · b (dot product):**
```
a · b = (2)(1) + (3)(-2) + (-1)(4)
      = 2 - 6 - 4
      = -8
```

**4. ||a|| (L2 norm):**
```
||a|| = √(a · a)
      = √(2² + 3² + (-1)²)
      = √(4 + 9 + 1)
      = √14
      ≈ 3.742
```

**5. ||b||₁ (L1 norm):**
```
||b||₁ = |1| + |-2| + |4|
       = 1 + 2 + 4
       = 7
```

### Solution 1.2: Linear Combinations

Find α, β such that: **c** = α**a** + β**b**

```
[-3, 0, 2] = α[2, 3, -1] + β[1, -2, 4]
[-3, 0, 2] = [2α + β, 3α - 2β, -α + 4β]

This gives us the system:
2α + β = -3     ... (1)
3α - 2β = 0     ... (2)
-α + 4β = 2     ... (3)

From equation (2):
3α = 2β
α = (2/3)β

Substitute into equation (1):
2(2/3)β + β = -3
(4/3)β + β = -3
(7/3)β = -3
β = -9/7

Then: α = (2/3)(-9/7) = -6/7

Verify with equation (3):
-(-6/7) + 4(-9/7) = 6/7 - 36/7 = -30/7 ≠ 2

Therefore: c CANNOT be expressed as a linear combination of a and b.
(The vectors a, b, c are not coplanar - c is not in span{a, b})
```

### Solution 1.3: Unit Vectors

Given **a** = [3, 4]

**1. Unit vector:**
```
||a|| = √(3² + 4²) = √(9 + 16) = √25 = 5

â = a / ||a||
  = [3, 4] / 5
  = [3/5, 4/5]
  = [0.6, 0.8]
```

**2. Verify magnitude is 1:**
```
||â|| = √((3/5)² + (4/5)²)
      = √(9/25 + 16/25)
      = √(25/25)
      = √1
      = 1 ✓
```

### Solution 1.4: Orthogonality

Given **u** = [1, 2, -1], **v** = [2, -1, 0]

**1. u · v:**
```
u · v = (1)(2) + (2)(-1) + (-1)(0)
      = 2 - 2 + 0
      = 0
```

**2. Are they orthogonal?**
```
Yes! Since u · v = 0, the vectors are orthogonal (perpendicular).
```

**3. Find vector w orthogonal to u:**
```
We need w such that u · w = 0

Let w = [a, b, c]
u · w = 1(a) + 2(b) + (-1)(c) = 0
So: a + 2b - c = 0
Or: c = a + 2b

One simple solution: Let a = 1, b = 0
Then c = 1

w = [1, 0, 1]

Verify: u · w = 1(1) + 2(0) + (-1)(1) = 1 + 0 - 1 = 0 ✓

Note: There are infinitely many solutions!
Other examples: [0, 1, 2], [2, -1, 0], etc.
```

---

## Part 2: Matrix Operations

### Solution 2.1: Matrix Addition and Scalar Multiplication

**1. A + B:**
```
A + B = [[2, -1,  3],   [[1,  2, -1],
         [0,  4, -2]]  +  [3, -1,  0]]

      = [[2+1, -1+2,  3+(-1)],
         [0+3,  4+(-1), -2+0]]

      = [[3,  1,  2],
         [3,  3, -2]]
```

**2. 3A - 2B:**
```
3A = [[6, -3,  9],
      [0, 12, -6]]

2B = [[2,  4, -2],
      [6, -2,  0]]

3A - 2B = [[6-2, -3-4,  9-(-2)],
           [0-6, 12-(-2), -6-0]]

        = [[4, -7, 11],
           [-6, 14, -6]]
```

### Solution 2.2: Matrix Multiplication

**1. AB:**
```
A = [[1, 2, 3],      B = [[2,  1],
     [4, 5, 6]]           [0, -1],
                          [3,  2]]

AB is 2×2

Element (1,1): (1)(2) + (2)(0) + (3)(3) = 2 + 0 + 9 = 11
Element (1,2): (1)(1) + (2)(-1) + (3)(2) = 1 - 2 + 6 = 5
Element (2,1): (4)(2) + (5)(0) + (6)(3) = 8 + 0 + 18 = 26
Element (2,2): (4)(1) + (5)(-1) + (6)(2) = 4 - 5 + 12 = 11

AB = [[11,  5],
      [26, 11]]
```

**2. Dimension of AB:**
```
A is 2×3, B is 3×2
AB is 2×2
```

**3. Can you compute BA?**
```
B is 3×2, A is 2×3

For BA, we need columns of B = rows of A
Columns of B: 2
Rows of A: 2
These match! ✓

BA is 3×3

BA = [[2,  1],   [[1, 2, 3],
      [0, -1], ×  [4, 5, 6]]
      [3,  2]]

Element (1,1): (2)(1) + (1)(4) = 2 + 4 = 6
Element (1,2): (2)(2) + (1)(5) = 4 + 5 = 9
Element (1,3): (2)(3) + (1)(6) = 6 + 6 = 12
Element (2,1): (0)(1) + (-1)(4) = 0 - 4 = -4
Element (2,2): (0)(2) + (-1)(5) = 0 - 5 = -5
Element (2,3): (0)(3) + (-1)(6) = 0 - 6 = -6
Element (3,1): (3)(1) + (2)(4) = 3 + 8 = 11
Element (3,2): (3)(2) + (2)(5) = 6 + 10 = 16
Element (3,3): (3)(3) + (2)(6) = 9 + 12 = 21

BA = [[6,   9, 12],
      [-4, -5, -6],
      [11, 16, 21]]
```

### Solution 2.3: Matrix-Vector Multiplication

```
Ax = [[2, -1,  3],   [[1],
      [1,  0, -2], ×  [2],
      [0,  4,  1]]    [-1]]

Element 1: (2)(1) + (-1)(2) + (3)(-1) = 2 - 2 - 3 = -3
Element 2: (1)(1) + (0)(2) + (-2)(-1) = 1 + 0 + 2 = 3
Element 3: (0)(1) + (4)(2) + (1)(-1) = 0 + 8 - 1 = 7

Ax = [[-3],
      [ 3],
      [ 7]]
```

### Solution 2.4: Transpose

**1. Aᵀ:**
```
A = [[1, 2, 3],
     [4, 5, 6]]

Aᵀ = [[1, 4],
      [2, 5],
      [3, 6]]
```

**2. (Aᵀ)ᵀ:**
```
(Aᵀ)ᵀ = [[1, 2, 3],
         [4, 5, 6]]

(Aᵀ)ᵀ = A ✓
```

**3. Verify (AB)ᵀ = BᵀAᵀ:**
```
From Exercise 2.2:
AB = [[11,  5],
      [26, 11]]

(AB)ᵀ = [[11, 26],
         [ 5, 11]]

Aᵀ = [[1, 4],
      [2, 5],
      [3, 6]]

Bᵀ = [[2, 0, 3],
      [1, -1, 2]]

BᵀAᵀ = [[2, 0, 3],   [[1, 4],
        [1, -1, 2]] ×  [2, 5],
                       [3, 6]]

Element (1,1): (2)(1) + (0)(2) + (3)(3) = 2 + 0 + 9 = 11
Element (1,2): (2)(4) + (0)(5) + (3)(6) = 8 + 0 + 18 = 26
Element (2,1): (1)(1) + (-1)(2) + (2)(3) = 1 - 2 + 6 = 5
Element (2,2): (1)(4) + (-1)(5) + (2)(6) = 4 - 5 + 12 = 11

BᵀAᵀ = [[11, 26],
        [ 5, 11]]

(AB)ᵀ = BᵀAᵀ ✓ Verified!
```

### Solution 2.5: Identity and Inverse

**1. 3×3 Identity matrix:**
```
I₃ = [[1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]]
```

**2. Verify AI = A:**
```
Let A = [[2, 3, 1],
         [1, 0, 2],
         [4, 1, 5]]

AI = [[2, 3, 1],   [[1, 0, 0],
      [1, 0, 2], ×  [0, 1, 0],
      [4, 1, 5]]    [0, 0, 1]]

    = [[2, 3, 1],
       [1, 0, 2],
       [4, 1, 5]]

AI = A ✓
```

**3. Find A⁻¹ for A = [[2, 1], [1, 1]]:**
```
We want A⁻¹ = [[a, b],  such that AA⁻¹ = I
               [c, d]]

[[2, 1],   [[a, b],   [[1, 0],
 [1, 1]] ×  [c, d]] =  [0, 1]]

This gives us:
2a + c = 1  ... (1)
2b + d = 0  ... (2)
a + c = 0   ... (3)
b + d = 1   ... (4)

From (1) and (3):
2a + c = 1
a + c = 0
Subtracting: a = 1
Then: c = -1

From (2) and (4):
2b + d = 0
b + d = 1
Subtracting: b = -1
Then: d = 2

A⁻¹ = [[ 1, -1],
       [-1,  2]]

Verify:
AA⁻¹ = [[2, 1],   [[ 1, -1],
        [1, 1]] ×  [-1,  2]]

     = [[2(1)+1(-1), 2(-1)+1(2)],
        [1(1)+1(-1), 1(-1)+1(2)]]

     = [[1, 0],
        [0, 1]] ✓
```

---

## Part 3: Eigenvalues and Eigenvectors

### Solution 3.1: Understanding Eigenvectors

Given A = [[3, 1], [0, 2]], v = [[1], [0]]

**1. Calculate Av:**
```
Av = [[3, 1],   [[1],
      [0, 2]] ×  [0]]

   = [[3(1) + 1(0)],
      [0(1) + 2(0)]]

   = [[3],
      [0]]
```

**2. Is v an eigenvector?**
```
Av = [[3],  = 3[[1],  = 3v
      [0]]     [0]]

Yes! v is an eigenvector with eigenvalue λ = 3
```

**3. Is w = [[1], [1]] an eigenvector?**
```
Aw = [[3, 1],   [[1],
      [0, 2]] ×  [1]]

   = [[3(1) + 1(1)],
      [0(1) + 2(1)]]

   = [[4],
      [2]]

Is [[4], [2]] a scalar multiple of [[1], [1]]?
[[4], [2]] = k[[1], [1]] would require:
4 = k and 2 = k
This is impossible! (4 ≠ 2)

No, w is NOT an eigenvector.
```

### Solution 3.2: Finding Eigenvalues

Find eigenvalues of A = [[4, 2], [1, 3]]

**Step 1: Set up characteristic equation det(A - λI) = 0**
```
A - λI = [[4, 2],   [[λ, 0],
          [1, 3]] -  [0, λ]]

       = [[4-λ,   2],
          [  1, 3-λ]]
```

**Step 2: Calculate determinant**
```
det(A - λI) = (4-λ)(3-λ) - (2)(1)
            = 12 - 4λ - 3λ + λ² - 2
            = λ² - 7λ + 10
```

**Step 3: Solve quadratic equation**
```
λ² - 7λ + 10 = 0

Using quadratic formula:
λ = (7 ± √(49 - 40)) / 2
  = (7 ± √9) / 2
  = (7 ± 3) / 2

λ₁ = (7 + 3)/2 = 5
λ₂ = (7 - 3)/2 = 2

Eigenvalues: λ₁ = 5, λ₂ = 2
```

### Solution 3.3: Finding Eigenvectors

**For λ₁ = 5:**
```
(A - 5I)v = 0

[[4-5,   2],   [[v₁],   [[0],
 [  1, 3-5]] ×  [v₂]] =  [0]]

[[-1,  2],   [[v₁],   [[0],
 [ 1, -2]] ×  [v₂]] =  [0]]

Equation 1: -v₁ + 2v₂ = 0  →  v₁ = 2v₂
Equation 2:  v₁ - 2v₂ = 0  →  v₁ = 2v₂ (same!)

Let v₂ = 1, then v₁ = 2

Eigenvector: v₁ = [[2],
                    [1]]
```

**For λ₂ = 2:**
```
(A - 2I)v = 0

[[4-2,   2],   [[v₁],   [[0],
 [  1, 3-2]] ×  [v₂]] =  [0]]

[[2, 2],   [[v₁],   [[0],
 [1, 1]] ×  [v₂]] =  [0]]

Equation 1: 2v₁ + 2v₂ = 0  →  v₁ = -v₂
Equation 2:  v₁ +  v₂ = 0  →  v₁ = -v₂ (same!)

Let v₂ = 1, then v₁ = -1

Eigenvector: v₂ = [[-1],
                    [ 1]]
```

### Solution 3.4: Diagonalization

Given A = [[3, 1], [0, 2]]

**1. Find eigenvalues:**
```
det(A - λI) = (3-λ)(2-λ) - 0
            = λ² - 5λ + 6
            = (λ-3)(λ-2)

λ₁ = 3, λ₂ = 2
```

**2. Find eigenvectors:**
```
For λ₁ = 3:
[[0, 1],   [[v₁],   [[0],
 [0, -1]] ×  [v₂]] =  [0]]

v₂ = 0, v₁ is free
Eigenvector: v₁ = [[1], [0]]

For λ₂ = 2:
[[1, 1],   [[v₁],   [[0],
 [0, 0]] ×  [v₂]] =  [0]]

v₁ + v₂ = 0  →  v₁ = -v₂
Eigenvector: v₂ = [[-1], [1]]
```

**3. Form matrix P:**
```
P = [[1, -1],
     [0,  1]]
```

**4. Form diagonal matrix D:**
```
D = [[3, 0],
     [0, 2]]
```

**5. Verify A = PDP⁻¹:**
```
First find P⁻¹:
P⁻¹ = [[1, 1],
       [0, 1]]

PD = [[1, -1],   [[3, 0],
      [0,  1]] ×  [0, 2]]

   = [[3, -2],
      [0,  2]]

PDP⁻¹ = [[3, -2],   [[1, 1],
         [0,  2]] ×  [0, 1]]

      = [[3, 3-2],
         [0,  2]]

      = [[3, 1],
         [0, 2]]

PDP⁻¹ = A ✓ Verified!
```

---

## Part 4: Singular Value Decomposition (SVD)

### Solution 4.1: Understanding SVD Concepts

Given A = [[3, 0], [4, 5]]

**1. Calculate AᵀA:**
```
Aᵀ = [[3, 4],
      [0, 5]]

AᵀA = [[3, 4],   [[3, 0],
       [0, 5]] ×  [4, 5]]

    = [[3(3)+4(4), 3(0)+4(5)],
       [0(3)+5(4), 0(0)+5(5)]]

    = [[9+16,  0+20],
       [0+20,  0+25]]

    = [[25, 20],
       [20, 25]]
```

**2. Calculate AAᵀ:**
```
AAᵀ = [[3, 0],   [[3, 4],
       [4, 5]] ×  [0, 5]]

    = [[3(3)+0(0), 3(4)+0(5)],
       [4(3)+5(0), 4(4)+5(5)]]

    = [[9,  12],
       [12, 41]]
```

**3. Find eigenvalues of AᵀA:**
```
det(AᵀA - λI) = det([[25-λ,   20],
                      [  20, 25-λ]])

              = (25-λ)² - 400
              = 625 - 50λ + λ² - 400
              = λ² - 50λ + 225
              = (λ - 45)(λ - 5)

λ₁ = 45, λ₂ = 5

These are σ²ᵢ (squared singular values)
```

**4. Calculate singular values:**
```
σ₁ = √45 = √(9×5) = 3√5 ≈ 6.708
σ₂ = √5 ≈ 2.236
```

### Solution 4.2: SVD Components

For 3×2 matrix A with SVD: A = UΣVᵀ

**1. Dimensions of U:**
```
U is 3×3 (m×m where m=3)
U contains left singular vectors
```

**2. Dimensions of Σ:**
```
Σ is 3×2 (same as A)
Σ is diagonal matrix with singular values
Σ = [[σ₁,  0],
     [ 0, σ₂],
     [ 0,  0]]
```

**3. Dimensions of V:**
```
V is 2×2 (n×n where n=2)
V contains right singular vectors
```

**4. Properties of U and V:**
```
- Both U and V are orthogonal matrices
- Columns are orthonormal (perpendicular unit vectors)
- UᵀU = I and VᵀV = I
- U⁻¹ = Uᵀ and V⁻¹ = Vᵀ
```

### Solution 4.3: Rank and SVD

Given singular values: σ₁ = 5, σ₂ = 3, σ₃ = 0, σ₄ = 0

**1. Rank of matrix:**
```
Rank = number of non-zero singular values = 2
```

**2. How many non-zero singular values determine rank:**
```
Exactly the number of non-zero singular values = rank

In general: rank(A) = number of non-zero σᵢ

This is a fundamental property of SVD!
The zero singular values indicate linear dependence
in the rows/columns of the matrix.
```

---

## Part 5: Practical Applications

### Solution 5.1: Rotation Matrix

**1. R(90°):**
```
θ = 90° = π/2 radians

cos(90°) = 0
sin(90°) = 1

R(90°) = [[cos(90°), -sin(90°)],
          [sin(90°),  cos(90°)]]

       = [[0, -1],
          [1,  0]]
```

**2. Apply to [1, 0]:**
```
R(90°) [[1],   [[0, -1],   [[1],
        [0]] =  [1,  0]] ×  [0]]

             = [[0(1) + (-1)(0)],
                [1(1) +  0(0)]]

             = [[0],
                [1]]

[1, 0] rotates to [0, 1] ✓
```

**3. Apply to [0, 1]:**
```
R(90°) [[0],   [[0, -1],   [[0],
        [1]] =  [1,  0]] ×  [1]]

             = [[0(0) + (-1)(1)],
                [1(0) +  0(1)]]

             = [[-1],
                [ 0]]

[0, 1] rotates to [-1, 0] ✓
```

**4. Verification:**
```
Original:    [1,0] (right) → [0,1] (up)
90° CCW:     [0,1] (up) → [-1,0] (left)

This is indeed a 90° counterclockwise rotation! ✓

Draw it:
        [0,1] up
          ^
          |
[-1,0] ←--+--→ [1,0] right
          |
          v
       [0,-1] down
```

### Solution 5.2: Projection

Given **a** = [3, 4], **b** = [1, 0]

proj_b(a) = (a·b / b·b) × b

**Calculate:**
```
a · b = 3(1) + 4(0) = 3
b · b = 1(1) + 0(0) = 1

proj_b(a) = (3/1) × [1, 0]
          = 3 × [1, 0]
          = [3, 0]
```

**Sketch:**
```
    a = [3,4]
     /|
    / |
   /  |
  /   | (perpendicular)
 /    |
+-----+----→ b = [1,0] (x-axis)
0  proj_b(a)=[3,0]

The projection is the "shadow" of a onto b.
It's the point on the b-axis closest to a.
```

### Solution 5.3: Linear System

System:
```
2x + y = 5
x - y = 1
```

**1. Matrix form:**
```
Ax = b

[[2,  1],   [[x],   [[5],
 [1, -1]] ×  [y]] =  [1]]

where A = [[2, 1], [1, -1]], x = [[x], [y]], b = [[5], [1]]
```

**2. Solve using A⁻¹:**
```
Find A⁻¹:

det(A) = 2(-1) - 1(1) = -2 - 1 = -3

A⁻¹ = (1/det(A)) × [[-1, -1],
                      [-1,  2]]

    = (-1/3) × [[-1, -1],
                 [-1,  2]]

    = [[1/3,  1/3],
       [1/3, -2/3]]

x = A⁻¹b

  = [[1/3,  1/3],   [[5],
     [1/3, -2/3]] ×  [1]]

  = [[(1/3)(5) + (1/3)(1)],
     [(1/3)(5) + (-2/3)(1)]]

  = [[5/3 + 1/3],
     [5/3 - 2/3]]

  = [[6/3],
     [3/3]]

  = [[2],
     [1]]

Solution: x = 2, y = 1
```

**3. Verify:**
```
2(2) + 1 = 4 + 1 = 5 ✓
2 - 1 = 1 ✓
```

---

## Challenge Problems

### Challenge 1: Matrix Powers

A = [[1, 1], [0, 1]]

**A²:**
```
A² = [[1, 1],   [[1, 1],
      [0, 1]] ×  [0, 1]]

   = [[1, 2],
      [0, 1]]
```

**A³:**
```
A³ = A² × A

   = [[1, 2],   [[1, 1],
      [0, 1]] ×  [0, 1]]

   = [[1, 3],
      [0, 1]]
```

**A⁴:**
```
A⁴ = A³ × A

   = [[1, 3],   [[1, 1],
      [0, 1]] ×  [0, 1]]

   = [[1, 4],
      [0, 1]]
```

**Pattern for Aⁿ:**
```
Aⁿ = [[1, n],
      [0, 1]]

Proof by induction would show this holds for all n ≥ 1.
```

**NumPy verification for n=10:**
```python
import numpy as np
A = np.array([[1, 1], [0, 1]])
A10 = np.linalg.matrix_power(A, 10)
# Should give [[1, 10], [0, 1]]
```

### Challenge 2: Gram-Schmidt Process

Given: **v₁** = [1, 1, 0], **v₂** = [1, 0, 1], **v₃** = [0, 1, 1]

**Step 1: Normalize v₁**
```
||v₁|| = √(1² + 1² + 0²) = √2

u₁ = v₁ / ||v₁||
   = [1, 1, 0] / √2
   = [1/√2, 1/√2, 0]
   = [√2/2, √2/2, 0]
```

**Step 2: Make v₂ orthogonal to u₁**
```
proj_u₁(v₂) = (v₂ · u₁) u₁
            = ([1,0,1] · [√2/2, √2/2, 0]) [√2/2, √2/2, 0]
            = (√2/2) [√2/2, √2/2, 0]
            = [1/2, 1/2, 0]

w₂ = v₂ - proj_u₁(v₂)
   = [1, 0, 1] - [1/2, 1/2, 0]
   = [1/2, -1/2, 1]

||w₂|| = √((1/2)² + (-1/2)² + 1²)
       = √(1/4 + 1/4 + 1)
       = √(3/2)
       = √6/2

u₂ = w₂ / ||w₂||
   = [1/2, -1/2, 1] / (√6/2)
   = [1/√6, -1/√6, 2/√6]
   = [√6/6, -√6/6, 2√6/6]
```

**Step 3: Make v₃ orthogonal to u₁ and u₂**
```
proj_u₁(v₃) = (v₃ · u₁) u₁
            = ([0,1,1] · [√2/2, √2/2, 0]) [√2/2, √2/2, 0]
            = (√2/2) [√2/2, √2/2, 0]
            = [1/2, 1/2, 0]

proj_u₂(v₃) = (v₃ · u₂) u₂
            = ([0,1,1] · [√6/6, -√6/6, 2√6/6]) [√6/6, -√6/6, 2√6/6]
            = (√6/6) [√6/6, -√6/6, 2√6/6]
            = [1/6, -1/6, 2/6]

w₃ = v₃ - proj_u₁(v₃) - proj_u₂(v₃)
   = [0, 1, 1] - [1/2, 1/2, 0] - [1/6, -1/6, 2/6]
   = [0 - 1/2 - 1/6, 1 - 1/2 + 1/6, 1 - 0 - 1/3]
   = [-2/3, 2/3, 2/3]

||w₃|| = √(4/9 + 4/9 + 4/9) = √(12/9) = 2/√3 = 2√3/3

u₃ = w₃ / ||w₃||
   = [-2/3, 2/3, 2/3] / (2√3/3)
   = [-1/√3, 1/√3, 1/√3]
   = [-√3/3, √3/3, √3/3]
```

**Orthonormal basis:**
```
u₁ = [√2/2, √2/2, 0]
u₂ = [√6/6, -√6/6, 2√6/6]
u₃ = [-√3/3, √3/3, √3/3]

Verify orthonormality:
- u₁ · u₂ = 0 ✓
- u₁ · u₃ = 0 ✓
- u₂ · u₃ = 0 ✓
- ||u₁|| = ||u₂|| = ||u₃|| = 1 ✓
```

### Challenge 3: Image Compression with SVD

Create 4×4 grayscale image (simplified example):
```
A = [[255, 200, 150, 100],
     [200, 180, 120,  80],
     [150, 120,  90,  60],
     [100,  80,  60,  40]]
```

This is a computationally intensive problem by hand. The approach would be:
1. Compute AᵀA and AAᵀ
2. Find eigenvalues and eigenvectors
3. Construct U, Σ, V
4. Reconstruct using only largest σ₁

**Recommend using NumPy for this challenge:**
```python
import numpy as np

A = np.array([[255, 200, 150, 100],
              [200, 180, 120,  80],
              [150, 120,  90,  60],
              [100,  80,  60,  40]])

U, sigma, Vt = np.linalg.svd(A)

# Reconstruct with only largest singular value
A_compressed = sigma[0] * np.outer(U[:, 0], Vt[0, :])

print("Original:\n", A)
print("\nSingular values:", sigma)
print("\nCompressed (rank-1):\n", A_compressed.astype(int))
print("\nCompression error:", np.linalg.norm(A - A_compressed))
```

---

## Key Takeaways

1. **Vector operations** are element-wise arithmetic
2. **Dot product = 0** means vectors are orthogonal
3. **Matrix multiplication** requires matching inner dimensions
4. **(AB)ᵀ = BᵀAᵀ** - reverse order for transpose
5. **Eigenvectors** satisfy Av = λv
6. **Eigenvalues** come from det(A - λI) = 0
7. **SVD** decomposes any matrix: A = UΣVᵀ
8. **Rank** = number of non-zero singular values

Well done completing these exercises! 🎯
