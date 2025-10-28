# Linear Algebra Exercises - Module 1

**Time:** 3-4 hours
**Difficulty:** Beginner to Intermediate
**Materials needed:** Paper, pencil, calculator (optional)

Complete these exercises by hand first, then verify with NumPy. Solutions are in `guides/solutions/linear_algebra_solutions.md`

---

## Part 1: Vectors and Vector Spaces (30 min)

### Exercise 1.1: Vector Operations
Given vectors:
- **a** = [2, 3, -1]
- **b** = [1, -2, 4]
- **c** = [-3, 0, 2]

Calculate by hand:
1. **a** + **b**
2. 2**a** - 3**b**
3. **a** · **b** (dot product)
4. ||**a**|| (L2 norm/magnitude)
5. ||**b**||₁ (L1 norm)

### Exercise 1.2: Linear Combinations
Can you express **c** = [-3, 0, 2] as a linear combination of **a** and **b**?

Find scalars α and β such that: **c** = α**a** + β**b**

Show your work.

### Exercise 1.3: Unit Vectors
1. Calculate the unit vector in the direction of **a** = [3, 4]
2. Verify that its magnitude is 1

### Exercise 1.4: Orthogonality
Given vectors **u** = [1, 2, -1] and **v** = [2, -1, 0]:
1. Calculate **u** · **v**
2. Are these vectors orthogonal? Why or why not?
3. If not, find a vector **w** that is orthogonal to **u**

---

## Part 2: Matrix Operations (45 min)

### Exercise 2.1: Matrix Addition and Scalar Multiplication
Given matrices:

A = [[2, -1, 3],
     [0,  4, -2]]

B = [[1,  2, -1],
     [3, -1,  0]]

Calculate by hand:
1. A + B
2. 3A - 2B

### Exercise 2.2: Matrix Multiplication
Given:

A = [[1, 2, 3],
     [4, 5, 6]]

B = [[2,  1],
     [0, -1],
     [3,  2]]

Calculate by hand:
1. AB (show work for each element)
2. What is the dimension of AB?
3. Can you compute BA? Why or why not?

### Exercise 2.3: Matrix-Vector Multiplication
Given:

A = [[2, -1,  3],
     [1,  0, -2],
     [0,  4,  1]]

x = [[1],
     [2],
     [-1]]

Calculate Ax by hand.

### Exercise 2.4: Transpose
Given:

A = [[1, 2, 3],
     [4, 5, 6]]

1. Calculate Aᵀ
2. Calculate (Aᵀ)ᵀ
3. Verify that (AB)ᵀ = BᵀAᵀ using matrices from Exercise 2.2

### Exercise 2.5: Identity and Inverse
1. Write the 3×3 identity matrix I₃
2. Verify that AI = A for any 3×3 matrix A (use an example)
3. Given A = [[2, 1], [1, 1]], find A⁻¹ by solving AA⁻¹ = I

---

## Part 3: Eigenvalues and Eigenvectors (60 min)

### Exercise 3.1: Understanding Eigenvectors
Given matrix:

A = [[3, 1],
     [0, 2]]

And vector **v** = [[1], [0]]

1. Calculate A**v**
2. Is **v** an eigenvector of A? If so, what is its eigenvalue?
3. Try **w** = [[1], [1]]. Is it an eigenvector?

### Exercise 3.2: Finding Eigenvalues
Find the eigenvalues of:

A = [[4, 2],
     [1, 3]]

Steps:
1. Set up characteristic equation: det(A - λI) = 0
2. Solve for λ
3. Show all your work

### Exercise 3.3: Finding Eigenvectors
For the matrix A and eigenvalues from Exercise 3.2:
1. For each eigenvalue λ, solve (A - λI)**v** = **0**
2. Find the corresponding eigenvectors

### Exercise 3.4: Diagonalization
Given:

A = [[3, 1],
     [0, 2]]

1. Find eigenvalues
2. Find corresponding eigenvectors
3. Form matrix P with eigenvectors as columns
4. Form diagonal matrix D with eigenvalues on diagonal
5. Verify that A = PDP⁻¹

---

## Part 4: Singular Value Decomposition (SVD) (45 min)

### Exercise 4.1: Understanding SVD Concepts
For matrix:

A = [[3, 0],
     [4, 5]]

1. Calculate AᵀA
2. Calculate AAᵀ
3. Find eigenvalues of AᵀA (these are σ²ᵢ)
4. The singular values σᵢ are the square roots - calculate them

### Exercise 4.2: SVD Components
Given that A has SVD: A = UΣVᵀ

For a 3×2 matrix A:
1. What are the dimensions of U?
2. What are the dimensions of Σ?
3. What are the dimensions of V?
4. What properties do U and V have? (orthogonal? orthonormal?)

### Exercise 4.3: Rank and SVD
Given singular values σ₁ = 5, σ₂ = 3, σ₃ = 0, σ₄ = 0:
1. What is the rank of the matrix?
2. How many non-zero singular values determine the rank?

---

## Part 5: Practical Applications (30 min)

### Exercise 5.1: Rotation Matrix
The 2D rotation matrix for angle θ is:

R(θ) = [[cos(θ), -sin(θ)],
        [sin(θ),  cos(θ)]]

1. Write the rotation matrix for θ = 90° (π/2 radians)
2. Apply it to vector [1, 0]
3. Apply it to vector [0, 1]
4. Verify that R(90°) rotates vectors counterclockwise by 90°

### Exercise 5.2: Projection
Given vector **a** = [3, 4] and **b** = [1, 0]:

Calculate the projection of **a** onto **b** using:

proj_b(a) = (a·b / b·b) * b

1. Calculate by hand
2. Draw a sketch showing **a**, **b**, and proj_b(a)

### Exercise 5.3: Linear System
Solve by hand using matrix form Ax = b:

2x + y = 5
x - y = 1

1. Write in matrix form
2. Solve using inverse matrix method: x = A⁻¹b
3. Verify your solution

---

## NumPy Verification Exercises

After solving by hand, verify your answers using NumPy:

```python
import numpy as np

# Example for Exercise 1.1
a = np.array([2, 3, -1])
b = np.array([1, -2, 4])

# 1. Vector addition
print("a + b =", a + b)

# 2. Linear combination
print("2a - 3b =", 2*a - 3*b)

# 3. Dot product
print("a · b =", np.dot(a, b))

# 4. L2 norm
print("||a|| =", np.linalg.norm(a))

# 5. L1 norm
print("||b||_1 =", np.linalg.norm(b, ord=1))
```

Write NumPy code to verify all your hand calculations.

---

## Challenge Problems (Optional)

### Challenge 1: Matrix Powers
Given A = [[1, 1], [0, 1]]:
1. Calculate A², A³, A⁴ by hand
2. Find a pattern for Aⁿ
3. Verify with NumPy for n=10

### Challenge 2: Gram-Schmidt Process
Given vectors:
- **v₁** = [1, 1, 0]
- **v₂** = [1, 0, 1]
- **v₃** = [0, 1, 1]

Apply Gram-Schmidt orthogonalization by hand to create orthonormal basis {**u₁**, **u₂**, **u₃**}

### Challenge 3: Image Compression with SVD
Create a simple 4×4 grayscale image matrix (values 0-255).
1. Perform SVD by hand (use calculator for arithmetic)
2. Reconstruct using only largest singular value
3. Compare with original

---

## Submission Checklist

- [ ] All calculations shown step-by-step
- [ ] Answers verified with NumPy
- [ ] Solutions compared with guides/solutions/linear_algebra_solutions.md
- [ ] Understanding of concepts, not just correct answers
- [ ] Challenge problems attempted (optional)

---

## Tips for Success

1. **Show your work** - Don't just write the answer
2. **Check dimensions** - Make sure matrix operations are valid
3. **Verify with NumPy** - Catch arithmetic errors
4. **Draw diagrams** - Visualize vectors and transformations
5. **Ask "why"** - Understand the geometry behind the algebra

Good luck! 🎯
